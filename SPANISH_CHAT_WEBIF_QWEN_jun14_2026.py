# 🇪🇸 Tutor de Español — Local AI Spanish Tutor

A personal Spanish language learning web app powered by a **local LLM** (Qwen3.5-9B running via llama.cpp). No cloud, no subscriptions, no data leaving your machine. Designed for A2-level conversational practice with a warm, patient AI tutor.

---

## Screenshots

| Chat & Conversation | Story Mode |
|---|---|
| ![Chat](images/Screenshot%202026-06-13%20at%2000.39.45.png) | ![Story](images/Screenshot%202026-06-13%20at%2000.40.56.png) |

| Word Click — Vocab Selection | Vocab Toolbar |
|---|---|
| ![Word click](images/Screenshot%202026-06-13%20at%2000.41.31.png) | ![Toolbar](images/Screenshot%202026-06-13%20at%2000.41.52.png) |

| Drill Mode | Grammar / Explain |
|---|---|
| ![Drill](images/Screenshot%202026-06-13%20at%2000.42.40.png) | ![Explain](images/Screenshot%202026-06-13%20at%2000.44.16.png) |

---

## What it does

A single-page chat interface that connects to a locally-running LLM tutor. The AI knows your level, your native language, and your vocabulary — and uses that context to teach naturally through conversation rather than dry exercises.

**Core features:**

- **Free conversation** — write in Spanish, mix in English words freely. The tutor replies in Spanish at A2 level and notes any mixed words at the end of each reply without interrupting the flow
- **Story mode** — generates fresh short stories (200–250 words) on any topic at A2 level, with new vocabulary highlighted and one grammar note
- **Phrase drill** — AI gives 3 English phrases, you translate to Spanish, it corrects gently
- **Translate** — Spanish ↔ English/Russian with grammar observations
- **Vocab list** — AI scans the session context and lists all words above A2 level it has used
- **Grammar explain** — explain any grammar topic in English with Spanish examples and conjugation tables
- **Click-to-save vocabulary** — click any word in a bot reply to select it, select multiple, hit "Add N words to vocab" — sends them to the AI for definitions and saves them to `vocabulary.md`
- **`///word` shortcut** — type `///madrugada` to instantly add a word to the vocab file and get a definition without breaking conversation flow
- **Dark / light theme toggle**
- **Mobile compatible** — works on phone browsers, safe area aware, horizontal-scroll command bar

---

## How it works

```
Browser (HTML/JS)
      │
      │  POST /stream   (streaming text)
      │  POST /clear
      ▼
FastAPI server  ─── SPANISH_CHAT_WEBIF_MISTRAL_jun14_2026.py
      │
      │  Mistral instruct prompt:
      │  <s>[INST] system + context_starter + user_0 [/INST] bot_0 </s>
      │     [INST] user_1 [/INST] bot_1 </s>
      │     [INST] user_N [/INST]   ← model completes here
      ▼
llama_cpp  (Mistral-7B-v0.2-Q5_K_M.gguf, RTX 3070, ~67 tok/sec)
      │
      ▼
Streaming tokens → back to browser → rendered as markdown
```

**Session memory** — the backend keeps conversation history per session in RAM. The prompt builder trims oldest turns from the front when context fills up — recent turns are always preserved.

**Vocabulary persistence** — `///word` entries are appended to `vocabulary.md` on disk. Load this file into any session with `/load vocabulary.md` to give the AI full context of your known words.

**System prompt** — the tutor's personality, level calibration, correction style, and all activity modes are defined in `SYSTEM_PROMPT`. At startup, `context_starter.md` is loaded and appended to the system prompt automatically — giving the model output format examples before any user interaction.

**Context starter** — `context_starter.md` contains real examples of desired model output for each command (`/story`, `/vocab`, `///word`, `/translate`). Injected once at startup into the system prompt so the model mirrors this format consistently across all sessions.

---

## Project structure

```
spanish_tutor_001/
├── SPANISH_CHAT_WEBIF_MISTRAL_jun14_2026.py  # FastAPI backend + LLM inference (Mistral)
├── ES_TUTOR_chat_jun12_2026.html             # Single-file frontend (served by FastAPI)
├── context_starter.md                         # Few-shot output examples injected into system prompt
├── spanish_conjugation.md                     # Verb conjugation reference (load into context)
├── spanish_conjugation-2.html                 # Same reference as interactive HTML
├── vocabulary.md                              # Auto-saved vocab words (/// shortcut)
├── git_push.sh                                # Git push helper script
├── images/                                    # Screenshots for README
└── README.md
```

---

## Requirements

```
Python 3.10+
llama-cpp-python  (with CUDA for GPU offload)
fastapi
uvicorn
```

Install:
```bash
pip install fastapi uvicorn llama-cpp-python --break-system-packages
```

---

## Running

```bash
python SPANISH_CHAT_WEBIF_jun12_2026-2.py
```

Open in Chrome:
```
http://localhost:8000
```

From another device on the network:
```
http://<server-ip>:8000
```

---

## Model configuration

Supports two models — switch by commenting/uncommenting `MODEL_PATH` at the top of the Python file.

### Active: Mistral-7B-Instruct-v0.2-Q5_K_M

| Setting | Value | Why |
|---|---|---|
| `CONTEXT_LEN` | 10000 | Safe on 8GB VRAM at Q5 quant |
| `MAX_TOKENS` | 1024 | Longer than Qwen — Mistral needs room for structured output |
| `TEMPERATURE` | 0.3 | Low — consistent, predictable format |
| `TOP_P` | 0.92 | Reduces nonsense vocabulary |
| `TOP_K` | 50 | Wider candidate pool for natural phrasing |
| `REPEAT_PENALTY` | 1.05 | Light — Spanish naturally repeats some words |
| Prompt format | `[INST]...[/INST]` | Mistral instruct tokens — no ChatML |
| `context_starter.md` | injected at startup | Few-shot examples lock in output format |

### Available (comment in): Qwen3.5-9B-Q4_K_M

| Setting | Value | Why |
|---|---|---|
| `CONTEXT_LEN` | 16384 | Larger context window |
| `MAX_TOKENS` | 512 | Shorter — Qwen more concise by default |
| `TEMPERATURE` | 0.75 | More creative |
| Prompt format | `<\|im_start\|>...<\|im_end\|>` | ChatML tokens |
| `/no_think` | in system prompt | Disables Qwen3 thinking mode |

---

## Commands

| Command | Action |
|---|---|
| `/story [topic]` | Generate A2-level story on topic |
| `/drill` | 3-phrase translation exercise |
| `/translate [text]` | Translate Spanish ↔ English/Russian |
| `/vocab` | List session vocabulary above A2 |
| `/explain [topic]` | Grammar explanation with examples |
| `/save` | Save conversation to timestamped `.md` file |
| `/load filename.md` | Inject file content into session context |
| `///word` | Add word to vocab file + get definition |

Quick command buttons in the UI handle Story, Drill, Translate, Vocab, and Explain — no typing needed for the common ones.

---

## Vocabulary workflow

Words accumulate in `vocabulary.md` across sessions:

```markdown
madrugada
mercado
trabajar
caminar
```

At the start of a new session, load them back:
```
/load vocabulary.md
```

The AI then has your full personal dictionary in context and will use and reinforce those words naturally throughout the session.

---

## Student profile

Configured for **Alex — A2 level, English/Russian native**.
To adapt for a different student edit the `STUDENT PROFILE` section in `SYSTEM_PROMPT`:

```python
Name:            Alex
Level:           A2
Native languages: English
Goal:            Conversational Spanish
```

---

## Tech stack

| Component | Tech |
|---|---|
| LLM inference | llama-cpp-python (CUDA) |
| Model (active) | Mistral-7B-Instruct-v0.2-Q5_K_M |
| Model (available) | Qwen3.5-9B-Q4_K_M |
| Prompt format | Mistral instruct `[INST]...[/INST]` / Qwen ChatML |
| Output consistency | `context_starter.md` injected into system prompt at startup |
| API server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Markdown rendering | marked.js |
| Code highlighting | highlight.js |
| Math rendering | MathJax 3 |
| Diagrams | mermaid.js |
| Hardware | RTX 3070 8GB, Ryzen 7 5800X |

---

## Updates — jun 14 2026

- **Switched to Mistral-7B-Instruct-v0.2-Q5_K_M** — both models remain in the script, toggled by commenting/uncommenting `MODEL_PATH`
- **Prompt format rewritten** — ChatML (`<|im_start|>`) replaced with Mistral instruct format (`<s>[INST]...[/INST]`) in `build_prompt()`
- **`context_starter.md`** — new file containing real few-shot output examples for all commands. Loaded at startup and appended to `SYSTEM_PROMPT` to give the model consistent output format before any conversation starts
- **Latin American Spanish** — system prompt updated to target Chilean, Mexican, Peruvian, Panamanian dialects
- **Student profile generalised** — renamed from "Alex" to "User" for easier reuse
- **`CONTEXT_LEN` reduced to 10000** — safe on 8GB VRAM with Q5 quant + context_starter overhead
- **`TEMPERATURE` lowered to 0.3** — improves format consistency for structured outputs (tables, vocab lists, stories)
- **`MAX_TOKENS` raised to 1024** — Mistral needs more room to produce full structured responses
- **Bug fix** — history now saves `reply_no_internal_thoughts` (cleaned reply) instead of raw reply with `<think>` blocks
- **`enable_thinking` removed** from `Llama()` call — Qwen3-only parameter, not supported by Mistral
- **`/no_think` removed** from system prompt — Qwen3-only directive
