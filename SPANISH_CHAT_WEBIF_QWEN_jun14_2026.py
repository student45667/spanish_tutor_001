from pathlib import Path
from llama_cpp import Llama
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
import pandas as pd
import os as os
import numpy as np
import re

# =============================================================================
# CONFIG
# =============================================================================

# --- model settings ---
#MODEL_PATH = Path.home() / "hugging_face_rag/models/qwen2.5-coder-7b-q4/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
#MODEL_PATH = Path.home() / "hugging_face_rag/models/mistral-7b-q5/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
MODEL_PATH = Path.home() / "hugging_face_rag/models/qwen3.5-9b-q4/Qwen3.5-9B-Q4_K_M.gguf"
CONTEXT_LEN    = 16384 #16384 for quen , 10000 mistral # max tokens model can hold (prompt + reply)
N_THREADS      = 16     # CPU threads
N_GPU_LAYERS   = -1     # -1=all layers on GPU

# --- generation settings ---
# Spanish tutor profile: warm, natural, conversational
MAX_TOKENS     = 1024    # keep replies short and conversational — not essays
TEMPERATURE    = 0.75   # slightly creative — natural word choice, not robotic
TOP_P          = 0.92   # tighter than default — reduces nonsense vocabulary
TOP_K          = 50     # wider pool — more natural phrasing variety
REPEAT_PENALTY = 1.05   # light penalty — Spanish naturally repeats some words

# Profile reference:
# Coding:   temperature=0.2, top_p=0.95, top_k=40,  repeat_penalty=1.1
# Chat:     temperature=0.7, top_p=0.95, top_k=40,  repeat_penalty=1.1
# Spanish:  temperature=0.75,top_p=0.92, top_k=50,  repeat_penalty=1.05
# Creative: temperature=1.0, top_p=0.9,  top_k=50,  repeat_penalty=1.0

MAX_SESSIONS = 10

# ChatML tokens
S = "<|im_start|>"
E = "<|im_end|>\n"

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """/no_think
You are a warm, patient Spanish language tutor for Alex, an A2-level English/Russian speaker learning Spanish.
Your only mission is to help Alex improve Spanish through natural interaction.

════════════════════════════════════════
STUDENT PROFILE
════════════════════════════════════════
Name:            Alex
Level:           A2 (can handle present, past, near-future tenses; basic vocabulary)
Native languages: English
Goal:            Conversational Spanish, natural reading and listening

════════════════════════════════════════
LANGUAGE RULES
════════════════════════════════════════
1. Always respond in Spanish at A2 level — short, clear sentences.
2. Grammar notes and word explanations always in English (clearer for learner).
3. Introduce max 1-2 new words per reply. Use them in context first,
   then give the English meaning in parentheses on the same line.
4. Never use grammar above A2 without a brief English explanation.
5. Always end your reply with a question to keep the conversation going.

════════════════════════════════════════
MIXED INPUT HANDLING
════════════════════════════════════════
Alex may write mostly in Spanish but mix in English  words.
This is normal at A2 — do NOT treat it as an error.
- Continue replying naturally in Spanish.
- At the very end of your reply add one compact line:
  Words to remember: <english_word> → <spanish_word> | <another> → <otra>
- Keep tone encouraging. Never say "you made a mistake."




════════════════════════════════════════
VOCABULARY SHORTCUT  ( /// prefix )
════════════════════════════════════════
When you receive a message starting with "///" — everything after 
the three slashes is a new vocabulary word or phrase Alex wants to learn.

Example input:  ///madrugada

You must:
1. Acknowledge the word warmly in one short line in Spanish.
2. Give the English meaning in parentheses.
3. Use the word naturally in one example sentence at A2 level.
4. Add it to your active context — use this word in future replies
   when natural, to reinforce it.
5. End with a short question to keep conversation going.

Format:
  ✅ *la madrugada* (the early hours / early morning)
  "No me gusta levantarme en la madrugada." — I don't like getting up in the early hours.
  ¿A qué hora te despiertas normalmente?

Do NOT explain grammar at length. Keep it to 3-4 lines total.






════════════════════════════════════════
CORRECTION STYLE
════════════════════════════════════════
- Minor error (missing accent, gender agreement):
  Use the correct form naturally in your reply.
  Add a single soft note in English at the end:
  *Note: it's "Canadá" — accent on the last syllable.*
- Significant error (wrong tense, wrong verb):
  Echo the corrected sentence in italics before your reply:
  *"Fui al mercado ayer."*
  Then continue naturally. Never list more than one correction per reply.

════════════════════════════════════════
ACTIVITY MODES  (triggered by user or frontend)
════════════════════════════════════════

/story [topic]
  Generate a story (about 200 words) at A2 level on the given topic.
  Use present tense primarily. Everyday vocabulary.
  After the story add:
  -📖 New words: word → translation | word → translation
  -🔤 Grammar note: one point in English, max 2 sentences.
  -Translation

/translate [text]
  Translate the given text Spanish ↔ Russian or  English.
  If Spanish→English: also note 1-2 interesting grammar points.
  If English→Spanish: use A2 vocabulary, note any tricky words.

/drill
  Give Alex 3 English phrases to translate into Spanish.
  Wait for responses. Correct gently. Show the correct version.
  Track which words were missed — add them to the Words to remember line.

/vocab
  List the words Alex has looked up or struggled with this session.
  You will scan all prevous context and list words that are usuaaly beyond A2 level. (as model produces conversation this list will grow)  
  Format: Spanish | English | example sentence (A2 level)

/explain [grammar topic]
  Explain the grammar topic in simple Russian  or English with 2-3 Spanish examples at A2 level.
  For example tenses , conjugation  -you can use table as well

════════════════════════════════════════
RESPONSE FORMAT
════════════════════════════════════════
- You can use markdown, tables for tences, colour as needed.
- Story mode: use the 📖 and 🔤 emoji markers as shown above.
- Corrections and notes in plain English after the Spanish reply.
- Words to remember line only when Alex used English/Russian words.
- Keep replies short: 3-5 sentences of Spanish + any notes. Not essays.


════════════════════════════════════════
EXTERNAL COMMANDS
════════════════════════════════════════
/save  — save conversation to .md file
/load  — load text/context from file, format: /load filename.md
"""

#- Plain text only — no markdown, no bullet points, no headers in regular chat.

# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=CONTEXT_LEN,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)
print("Model loaded.\n")



# =============================================================================
# SESSION STORE
# =============================================================================

sessions: dict[str, dict] = {}

def get_or_create_session(session_id: str) -> dict:
    """Return existing session or create new one."""
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.pop(next(iter(sessions)))
        sessions[session_id] = {
            "history": [],
            "system": SYSTEM_PROMPT,
        }
    return sessions[session_id]

# =============================================================================
# TOKEN COUNTER
# =============================================================================

def count_tokens(text: str) -> int:
    return len(llm.tokenize(text.encode()))

# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(history: list, user_input: str ) -> str:
    system_block = f"{S}system\n{SYSTEM_PROMPT}{E}"
    user_block = f"{S}user\n{user_input}{E}{S}assistant\n"

    budget = CONTEXT_LEN - MAX_TOKENS - count_tokens(system_block  + user_block) - 64

    trimmed = list(history)
    while trimmed:
        hist_text = "".join(
            f"{S}user\n{t['user']}{E}{S}assistant\n{t['bot']}{E}"
            for t in trimmed
        )
        if count_tokens(hist_text) <= budget:
            break
        trimmed.pop(0)

    prompt = system_block 
    for t in trimmed:
        prompt += f"{S}user\n/no_think\n{t['user']}{E}{S}assistant\n{t['bot']}{E}"
    prompt += user_block

    return prompt

# =============================================================================
# Special commands
# =============================================================================

"""Append /// words to vocab.md with timestamp."""
def save_vocab_entry(words_str: str):  
    vocab_filename = "vocabulary.md"
    if words_str.strip().startswith("///"):
        words_str = words_str.strip()[3:]   # strip the ///  
    with open(vocab_filename, 'a') as f:
        f.write(f"{words_str}\n")




def save_session(session_id: str):
    sess = sessions.get(session_id)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{session_id}_{ts}.md"
    
    with open(filename, 'w') as f:
        for entry in sess["history"]:
            f.write(f"\n\n>>> USER: {entry['user']}:\n")
            f.write(f"\n\n>>> BOT: {entry['bot']}:\n")


def load_data(session_id: str, filename: str) -> str:
    """Load text or CSV file and return as string for prompt injection"""
    
    if not os.path.exists(filename):
        return f"[ERROR: file not found: {filename}]"
    
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        # Plain text / markdown / code
        if ext in ['.txt', '.md', '.csv', '.py' , '.c', '.cpp' , '.json' ]:
            with open(filename, 'r') as f:
                content = f.read()
            return f"[LOADED FILE: {filename}]\n{content}\n[END FILE]"
        
        else:
            return f"[ERROR: unsupported file type: {ext}]"
    
    except Exception as e:
        return f"[ERROR loading {filename}: {e}]"




    
# =============================================================================
# STREAMING INFERENCE
# =============================================================================

def stream_chat(session_id: str, user_input: str):
    sess = get_or_create_session(session_id)


    #========== SPECIAL COMMANDS =======================

    if user_input.strip().startswith("///"):
         save_vocab_entry (user_input)
         print(f"dictionary added: {user_input}")

    
    if user_input.strip().startswith("/save"):
         save_session(session_id)  # Just call it 
         user_input = f"\n\nConversation is now saved."
        
    if user_input.strip().startswith("/load"):
        parts = user_input.strip().split() 
        if len(parts) < 2:
            yield "[ERROR: Usage: /load filename.csv]"
            return
        filename = parts[1].strip()
        file_content = load_data(session_id, filename)
        user_input = f"{file_content}\n\nJust say loaded if new info appeared. dont analyze new info till asked"


    #===================================================
  
    # Build prompt with RAG context
    prompt = build_prompt(sess["history"], user_input )

    full_reply = []
    token_count = 0

    for chunk in llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repeat_penalty=REPEAT_PENALTY,
        stop=[E, S],
        echo=False,
        stream=True   
    ):
        token = chunk["choices"][0]["text"]
        full_reply.append(token)
        token_count += 1
        yield token

    reply = "".join(full_reply).strip()
    print(f">>>> User:{session_id}  generated:{token_count} tokens", flush=True)
    

    reply_no_internal_thoughts =  re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    
    sess["history"].append({"user": user_input, "bot": reply_no_internal_thoughts})

    

# =============================================================================
# FASTAPI
# =============================================================================

app = FastAPI(title="RAG Assistant")

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str = ""

@app.post("/stream")
def stream_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    return StreamingResponse(
        stream_chat(req.session_id, req.message),
        media_type="text/plain"
    )

@app.post("/clear")
def clear_endpoint(req: ChatRequest):
    sessions.pop(req.session_id, None)
    return {"status": "cleared"}




# In the FastAPI section, change:
@app.get("/")
def root():
    return FileResponse("ES_TUTOR_chat_jun12_2026.html")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
