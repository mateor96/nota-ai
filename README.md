# Nota.ai

A fully local, on-device audio transcription and speaker diarization web app for Apple Silicon Macs. No cloud, no subscriptions, no audio leaving your machine.

Built on [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for transcription and [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker separation.

Handles any audio — meetings, phone calls, voice memos, interviews — without a bot in the call, a cloud upload, or a subscription.

![Nota.ai — a speaker-labeled, click-to-seek transcript with one-click summary and export](docs/screenshot.png)

*Transcript text blurred for privacy.*

---

## Features

- **Transcription** via Whisper Large v3 Turbo — runs on the M-series GPU through Apple MLX
- **Robust on quiet & noisy audio** — Whisper decodes the recording in one continuous pass, and voice-activity detection (pyannote segmentation-3.0) guards it at both ends: the decode starts and stops at the real speech, so minutes of dead air can't seed a hallucinated "Thank you / Yeah" filler loop, and any loop that still forms over silence is detected and re-decoded from the point speech actually resumes. Language is detected once, from the middle of the speech, and pinned across the recording
- **Speaker diarization** via pyannote community-1 — separates and labels each speaker, with phantom (sub-5%-share) clusters auto-pruned so the speaker count stays right
- **LLM summarization** — one-click meeting summaries via your choice of model:
  - *Local:* LM Studio, Ollama
  - *Cloud:* Anthropic, OpenAI, Gemini
- **Live recording** — record directly from any audio input (mic, BlackHole, etc.)
- **Archive** — all past transcriptions saved locally in SQLite, accessible anytime
- **Speaker renaming** — rename Speaker 00 / Speaker 01 to real names; persists per transcript
- **Export** — download as TXT, Markdown, or PDF
- **Click-to-seek** — click any line in the transcript to jump to that moment in the audio

Everything runs at `localhost:8765`. No internet required after first setup.

---

## Requirements

- Apple Silicon Mac (M1 or later) — transcription runs on the M-series GPU via Apple MLX, so this is macOS / Apple Silicon only
- macOS 13+
- Python 3.11 — pinned because `pyannote.audio` isn't compatible with 3.12+ yet; `uv` installs the right version for you, no manual setup
- [uv](https://github.com/astral-sh/uv) — `brew install uv`
- [ffmpeg](https://ffmpeg.org) — `brew install ffmpeg`
- A [Hugging Face](https://huggingface.co) account (free)

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/mateor96/nota-ai.git
cd nota-ai
uv sync
```

**2. Authenticate with Hugging Face**

```bash
uv run huggingface-cli login
```

Then visit these two pages while logged in and click **Agree** on each:

- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

> Your token needs **"Read access to public gated repositories"** enabled under token settings.

**3. Run**

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The first transcription will download the Whisper model (~1.5 GB) and the pyannote models (~30 MB; diarization + the segmentation model used for voice-activity detection). After that, everything runs offline.

---

## Recording calls (optional)

To record both sides of a call with headphones, install [BlackHole](https://github.com/ExistentialAudio/BlackHole) — a free virtual audio device for macOS.

**One-time setup (~10 min):**

1. Install BlackHole 2ch from [existential.audio](https://existential.audio/blackhole/)
2. Open **Audio MIDI Setup** (in `/Applications/Utilities/`)
3. Click **+** → **Create Multi-Output Device**
4. Check both your headphones and **BlackHole 2ch**
5. In Google Meet (or any call tool): set the speaker output to this Multi-Output Device

After that, select **BlackHole 2ch** as the audio source in the app's Record tab and hit Start. You'll hear the call normally through your headphones; the app captures everything in the background.

---

## Summaries (optional)

Click **✦ Summarize** on any transcript to get a bullet-point summary of the meeting. Configure the LLM provider via the **⚙** icon in the header.

**Local providers** — no API key, runs entirely offline:

- **LM Studio** *(default)* — install [LM Studio](https://lmstudio.ai), load a chat model, start its local server. URL: `http://localhost:1234`
- **Ollama** — install [Ollama](https://ollama.com), pull a model (e.g. `ollama pull llama3.2:3b`). URL: `http://localhost:11434`

**Cloud providers** — require an API key and one extra install:

```bash
uv add anthropic     # Claude
uv add openai        # GPT
uv add google-genai  # Gemini
```

API keys are stored in the macOS Keychain (service: `nota.ai`). Audio and transcripts never leave your machine unless you point Nota.ai at a cloud LLM — and even then, only the transcript text is sent.

---

## Tech stack

| Layer | Choice |
|---|---|
| Transcription | `mlx-whisper` + `mlx-community/whisper-large-v3-turbo`, one continuous pass, VAD-bounded |
| Speech detection (VAD) | `pyannote/segmentation-3.0` |
| Diarization | `pyannote/speaker-diarization-community-1` |
| Backend | FastAPI + uvicorn + SSE |
| Storage | SQLite via `aiosqlite` (`~/.transcribe/archive.db`) |
| Frontend | Vanilla HTML/JS |
| Audio pipeline | ffmpeg (loudnorm → 16 kHz mono WAV) → VAD (decode bounds) → Whisper |

---

## Development

```bash
uv run pytest                         # run the test suite
uv run python -m app.eval             # score archived transcripts for quality issues
uv run python -m app.eval --vad       # ...also measure VAD coverage (reads the audio)
uv run python -m app.eval --baseline quality.json   # save a baseline, then flag regressions on later runs
```

`app.eval` is a reference-free quality check: with no ground-truth transcript it
can't grade word accuracy, but it flags failure *signatures* — repetition loops,
filler-only segments, phantom speakers, words transcribed over silence — so a
pipeline change that degrades a previously-good recording is caught.

---

## Notes

- Models are cached in `~/.cache/huggingface/hub`
- Transcripts are stored in `~/.transcribe/archive.db`
- Neither folder is tracked by git
