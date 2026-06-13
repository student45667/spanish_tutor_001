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
FastAPI server  ─── SPANISH_CHAT_WEBIF_jun12_2026-2.py
      │
      │  ChatML prompt: system + history + user input
      ▼
llama_cpp  (Qwen3.5-9B-Q4_K_M.gguf, RTX 3070, ~67 tok/sec)
      │
      ▼
Streaming tokens → back to browser → rendered as markdown
```

**Session memory** — the backend keeps conversation history per session in RAM. The prompt builder trims old turns to fit within the 16K context window automatically.

**Vocabulary persistence** — `///word` entries are appended to `vocabulary.md` on disk. Load this file into any session with `/load vocabulary.md` to give the AI full context of your known words.

**System prompt** — the tutor's personality, level calibration, correction style, and all activity modes are defined in a single `SYSTEM_PROMPT` string in the Python file. Easy to edit.

---

## Project structure

```
spanish_tutor_001/
├── SPANISH_CHAT_WEBIF_jun12_2026-2.py   # FastAPI backend + LLM inference
├── ES_TUTOR_chat_jun12_2026.html        # Single-file frontend (served by FastAPI)
├── spanish_conjugation.md               # Verb conjugation reference (load into context)
├── spanish_conjugation-2.html           # Same reference as interactive HTML
├── vocabulary.md                        # Auto-saved vocab words (/// shortcut)
├── git_push.sh                          # Git push helper script
├── images/                              # Screenshots for README
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

Configured for **Qwen3.5-9B-Q4_K_M** — change `MODEL_PATH` at the top of the Python file for any GGUF model.

| Setting | Value | Why |
|---|---|---|
| `MAX_TOKENS` | 512 | Short conversational replies, not essays |
| `TEMPERATURE` | 0.75 | Natural word variety, not robotic |
| `TOP_P` | 0.92 | Reduces nonsense vocabulary |
| `TOP_K` | 50 | Wider candidate pool for natural phrasing |
| `REPEAT_PENALTY` | 1.05 | Light — Spanish naturally repeats some words |
| `CONTEXT_LEN` | 16384 | Fits full session + conjugation reference |
| `/no_think` | in system prompt | Disables Qwen3 thinking mode — faster responses |

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
| Model | Qwen3.5-9B-Q4_K_M |
| API server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Markdown rendering | marked.js |
| Code highlighting | highlight.js |
| Math rendering | MathJax 3 |
| Diagrams | mermaid.js |
| Hardware | RTX 3070 8GB, Ryzen 7 5800X |
