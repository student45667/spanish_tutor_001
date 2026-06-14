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

# [QWEN] comment out when using Mistral
#MODEL_PATH = Path.home() / "hugging_face_rag/models/qwen3.5-9b-q4/Qwen3.5-9B-Q4_K_M.gguf"

# [MISTRAL] comment out when using Qwen
MODEL_PATH = Path.home() / "hugging_face_rag/models/mistral-7b-q5/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

# [QWEN]    CONTEXT_LEN = 16384
# [MISTRAL] reduced — safe on 8GB VRAM at Q5 quant
CONTEXT_LEN    = 10000   # max tokens model can hold (prompt + reply)
N_THREADS      = 16     # CPU threads
N_GPU_LAYERS   = -1     # -1=all layers on GPU

# --- generation settings ---
# Spanish tutor profile: warm, natural, conversational
MAX_TOKENS     = 1024    # keep replies short and conversational — not essays
TEMPERATURE    = 0.3   # slightly creative — natural word choice, not robotic
TOP_P          = 0.92   # tighter than default — reduces nonsense vocabulary
TOP_K          = 50     # wider pool — more natural phrasing variety
REPEAT_PENALTY = 1.05   # light penalty — Spanish naturally repeats some words

# Profile reference:
# Coding:   temperature=0.2, top_p=0.95, top_k=40,  repeat_penalty=1.1
# Chat:     temperature=0.7, top_p=0.95, top_k=40,  repeat_penalty=1.1
# Spanish:  temperature=0.75,top_p=0.92, top_k=50,  repeat_penalty=1.05
# Creative: temperature=1.0, top_p=0.9,  top_k=50,  repeat_penalty=1.0

MAX_SESSIONS = 10

# [QWEN] ChatML tokens — not used for Mistral
#S = "<|im_start|>"
#E = "<|im_end|>\n"

# [MISTRAL] instruct tokens
BOS        = "<s>"
INST_START = "[INST]"
INST_END   = "[/INST]"
# keep S/E as aliases so nothing else in the file breaks
S = INST_START
E = INST_END

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

# [QWEN] /no_think directive removed — Mistral does not support it



SYSTEM_PROMPT = """ You are a warm, patient Spanish language tutor for User, 
an A2-level English speaker learning Latin America Spanish. (Chile ,Mexico, Peru, Panama dialects)
Your only mission is to help User to improve Spanish through natural interaction.


GENERAL LANGUAGE RULES FOR CONVERSATION
    1. Always respond in Spanish at A2 level — short, clear sentences.
    2. Grammar notes and word explanations always in English (clearer for learner).
    3. Introduce max 1-2 new words per reply. 
       then give the English meaning in parentheses on the same line.
    4. Never use grammar above A2 without a brief English explanation.



MIXED INPUT HANDLING FROM USER
    Alex may write mostly in Spanish but mix in English  words.
    This is normal at A2 — do NOT treat it as an error.
    - Continue replying naturally in Spanish.
    - Keep tone encouraging. Never say "you made a mistake."

                

CORRECTION STYLE WHEN NEEDED DURING CONVERSATION   
    - Minor error (missing accent, gender agreement):
      Use the correct form naturally in your reply.
    - Significant error (wrong tense, wrong verb):
      Echo the corrected sentence in italics before your reply:
      *"Fui al mercado ayer."*
      Then continue naturally. Never list more than one correction per reply.



RESPONSE TEXT FORMAT
    - Use markdown,colours, emojies,  tables as needed.
    - Example1: vocabilary, conjugation, tenses  - use table
    - Example2: Difrent colour story in spanish and English (dimmer)



EXTERNAL COMMANDS
    /save  — save conversation to .md file
    /load  — load text/context from file, format: /load filename.md




ACTIVITY MODES  (triggered by user or frontend) 

    /// [text1], [text2] ,[text3].....
        When you receive a message starting with "///" — everything after 
        the three slashes is a new vocabulary word or phrase Alex wants to learn.
          
        You must:
                1. Give the English meaning, to each word in Spanish after "///" 
                2. Just Spanish word and english meaning, nothing more.
                3. Do NOT explain grammar. Keep it to 2 lines total.
               
            Example input:  
                ///madrugada , agua
            Answer Format:
              ✅  la madrugada  - the early hours / early morning.
              ✅  agua  - water.


    /story [topic]
      -Generate a story in Spanish (about 200 words) at A2 level on the given topic.
      -After each sentence in Spanish, place English translation.
      -Use present/past tense primarily. 
      -short sentences 8-15 words
      -In case no [topic] is given, create story from following topics: nature,travel, 
              south america history, tecnology, health, medicine


    /translate [text]
      -Translate the given text Spanish ↔ English.
      -Maximum 3 lines answer
    
    /drill
      -Give Alex 3 English phrases to translate into Spanish.
      -Wait for responses. Correct gently. Show the correct version.
    
    /vocab
      -List all the words Alex has looked up, added with "///" or struggled with this session.
      -You will scan all prevous context and list words that are usuaaly beyond A2 level. 
      -Format example: correr - to run 
      -Use table 
    
    /explain [grammar topic]
      -Explain the grammar topic in simple  English with 2 Spanish examples at A2 level.
      -For example tenses , conjugation  -you can use table as well

"""

# =============================================================================
# LOAD context starter and add it to system prompt to give AI example of desiered output
# =============================================================================
filename = "context_starter.md"
with open(filename, 'r') as f:
    context_starter = f.read()

SYSTEM_PROMPT += f"{context_starter}"

# =============================================================================




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
    # [QWEN]    enable_thinking=False   ← Qwen3 only, remove for Mistral
    # [MISTRAL] enable_thinking not supported — omitted
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

# [QWEN] original build_prompt using ChatML format — keep for reference
# def build_prompt(history: list, user_input: str) -> str:
#     system_block = f"{S}system\n{SYSTEM_PROMPT}{E}"
#     user_block   = f"{S}user\n{user_input}{E}{S}assistant\n"
#     budget = CONTEXT_LEN - MAX_TOKENS - count_tokens(system_block + user_block) - 64
#     trimmed = list(history)
#     while trimmed:
#         hist_text = "".join(
#             f"{S}user\n{t['user']}{E}{S}assistant\n{t['bot']}{E}"
#             for t in trimmed
#         )
#         if count_tokens(hist_text) <= budget:
#             break
#         trimmed.pop(0)
#     prompt = system_block
#     for t in trimmed:
#         prompt += f"{S}user\n/no_think\n{t['user']}{E}{S}assistant\n{t['bot']}{E}"
#     prompt += user_block
#     return prompt

# [MISTRAL] build_prompt using Mistral instruct format:
#   <s>[INST] system + user [/INST] reply </s>[INST] user [/INST] ...
def build_prompt(history: list, user_input: str) -> str:
    """
    Assembles the full prompt string for Mistral instruct format.

    Mistral instruct layout:
        <s>[INST] system + user_turn_1 [/INST] bot_reply_1 </s>
            [INST] user_turn_2         [/INST] bot_reply_2 </s>
            [INST] user_turn_N         [/INST]              <- model completes here

    System prompt is injected ONCE — prepended to the very first
    user turn only. Mistral has no dedicated <system> token.
    """

    # ── build the two fixed blocks that are always present ──────────────────
    # current user message wrapped in Mistral end-of-instruction marker
    user_block = f"{user_input} {INST_END}"

    # opening of the prompt: BOS token + [INST] + system prompt
    # this is the template start — always the same regardless of history
    sys_block  = f"{BOS}{INST_START} {SYSTEM_PROMPT}\n\n"

    # ── calculate how many history tokens we can afford ─────────────────────
    # total context window
    #   minus  MAX_TOKENS  (reserved for the model's reply)
    #   minus  fixed blocks (sys_block + user_block are always included)
    #   minus  64 token safety margin (for special tokens, rounding)
    # = budget available for historical turns
    budget = CONTEXT_LEN - MAX_TOKENS - count_tokens(sys_block + user_block) - 64

    # ── trim oldest history turns until they fit within budget ───────────────
    # start with the full history, drop oldest turns from the front
    # until the remaining turns fit inside the token budget.
    # this implements a sliding context window — recent turns are preserved,
    # old turns are sacrificed when context fills up.
    trimmed = list(history)
    while trimmed:
        hist_text = "".join(
            f"{INST_START} {t['user']} {INST_END} {t['bot']} </s>"
            for t in trimmed
        )
        if count_tokens(hist_text) <= budget:
            break                  # fits — stop trimming
        trimmed.pop(0)             # drop oldest turn and try again

    # ── assemble the final prompt ────────────────────────────────────────────
    if trimmed:
        # history exists: system prompt goes into the FIRST historical turn
        # format:  <s>[INST] SYSTEM + user_0 [/INST] bot_0 </s>
        #              [INST] user_1         [/INST] bot_1 </s>
        #              ...
        prompt  = f"{BOS}{INST_START} {SYSTEM_PROMPT}\n\n{trimmed[0]['user']} {INST_END} {trimmed[0]['bot']} </s>"

        # append remaining turns without repeating system prompt
        for t in trimmed[1:]:
            prompt += f"{INST_START} {t['user']} {INST_END} {t['bot']} </s>"
    else:
        # no history at all (first ever message in session):
        # open with BOS + [INST] + system prompt only
        # current user message is appended below
        prompt = f"{BOS}{INST_START} {SYSTEM_PROMPT}\n\n"

    # ── append current user turn — model generates what comes after ──────────
    # no bot reply here — this is the open end the model completes
    prompt += f"{user_input} {INST_END}"

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
        # [QWEN]    stop=[E, S]  i.e. ["<|im_end|>\n", "<|im_start|>"]
        # [MISTRAL] stop tokens match Mistral instruct format
        stop=["</s>", "[INST]"],
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
    
    # [QWEN+MISTRAL] bug fix: save cleaned reply (no <think> blocks) to history
    # was: sess["history"].append({"user": user_input, "bot": reply})
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
