![CCMI](ccmi_logo.png)


# CCMI — Customized Consecutive Machine Interpreter

> **🎛️ We Customized Translation, Why Not Interpreting?**

**CCMI** is a desktop PyQt5 app that turns your microphone into a *customizable consecutive interpreter*:
**mic → Whisper ASR → GPT (translate with brief & termlist) → optional TTS.**
It adapts to the session (one‑way, two‑party, or two‑party + audience), keeps terminology consistent, and speaks back in the voice you choose.

---

## Why interpreting needs customization

- **🧵 Customization doesn’t cross the mic.**
  Traditional tools treat interpreting as one-size-fits-all: no pre‑briefs, no termlists, no tone or audience intent.
- **⏱️ Too many steps, too much lag.**
  Speech → text → translation → voice. Each hop adds delay and loses detail.
- **🧩 Sessions are not identical.**
  A sales call, a lecture, and a panel get treated the same. No session modes or memory per party.

## ⭐ Why CCMI?
- **🧭 Fits your setup.**
  Solo talk? Two‑person call? Conversation with listeners? Pick the right mode so roles and direction are clear.
- **🎙️ Tell it once, CCMI prepares.**
  Describe your session and CCMI fills a minimal brief: purpose, roles, tone, and rules.
- **📚 Your terms, locked in.**
  Import a list or add your own. Names and phrases stay consistent across the whole session.
- **🎧 Voices for every tone.**
  Choose among built‑in voice styles and test them anytime.
- **🧠 Context that grows.**
  CCMI remembers the session: new segments adapt to your brief, prior translations, and termlist.

---

## Features
- **Session modes**: One‑way, Two‑party, Two‑party + Audience
- **Briefs** for each direction, plus Audience fields when relevant
- **Termlist** (CSV/XLSX import, in‑app edit, clear/add/delete)
- **Review table** (copy last; **XLSX export**)
- **Device picker & meters**, swap languages (Ctrl/Cmd+Enter), **Shift+Space** to record
- **Voice tester** and multiple TTS styles
- **Privacy**: API key lives in memory only; temp audio files are removed
- **Online**: uses OpenAI APIs for ASR, translation, and TTS

## How it works
1. **Record** a segment (Shift+Space)
2. **ASR**: audio → text (OpenAI `whisper-1`)
3. **Translate** with GPT using:
   - current **session brief** (purpose, roles, tone…),
   - rolling **context** (recent translations),
   - enforced **terminology** (`source = target` pairs).
4. **TTS** (optional) with `gpt-4o-mini-tts` in the selected voice.

---

## Quick start

### Option A — Windows executable (no Python required)
1. Go to the repo’s **Releases** page and download the latest `CCMI_Windows_x64.zip`.
2. Unzip and run **`ccmi.exe`**.
3. When prompted, paste your **OpenAI API key** (stored only in memory for this session).

### Option B — Run from source (Windows/macOS/Linux)
- Requirements: **Python 3.9+**, microphone/speaker devices.
- Install:
  ```bash
  python -m venv .venv
  # Windows
  .venv\Scripts\activate
  # macOS / Linux
  source .venv/bin/activate

  pip install -r requirements.txt
  python ccmi.py
  ```
  On first launch, click **“Set API Key”** and enter an `sk-…` key.

### Keyboard shortcuts
- **Shift+Space** – Start/Stop recording
- **Ctrl+Enter** (Windows/Linux) or **⌘ Return** (macOS) – Swap Source ↔ Target

---

## Session modes
- **🎯 One‑Way (A → B)** — for announcements/lectures; has Audience fields.
- **🤝 Two‑Party (A ↔ B)** — for conversations/calls; separate briefs for each direction.
- **🎤 Two‑Party + Audience (A ↔ B + 👥)** — conversations with listeners; adds Audience fields.

## Terminology (CSV/XLSX)
- Two columns: **Source Term**, **Target Term** (header names are not required).
- Import via **Termlist → “Import Termlist (XLSX/CSV)”**.
- During translation, exact pairs are enforced (format used in prompt: `source = target`).  
  *(Excel support requires `openpyxl` — optional dependency.)*

## Export & review
- **Review → Export XLSX** saves the segment table with wrapped cells.
- **Copy Last** puts the newest translation on your clipboard.

---

## Build the Windows executable yourself
This repo ships with **`ccmi.spec`** for PyInstaller.

```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller ccmi.spec
```

- The executable will be in `dist/`. Don’t commit `dist/` or `build/`; publish the zip as a **Release asset**.
- Keep `ccmi.spec` tracked in git.

## Tech notes
- GUI: **PyQt5**
- Audio I/O: **sounddevice / soundfile**, 16 kHz mono capture
- ASR: **OpenAI `whisper-1`** (`client.audio.transcriptions.create`)
- Translation: **Chat Completions** (model configurable, e.g., `gpt-4.1-2025-04-14`)
- TTS: **`gpt-4o-mini-tts`** with selectable voices (`alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`)

## Privacy & security
- API key is requested at runtime and **not written to disk**.
- Temporary audio files are **deleted** after use.
- No telemetry, analytics, or background upload.
- You are responsible for complying with OpenAI API Terms in your region.

---

## Troubleshooting
- **“No OpenAI API key set.”** → Click **Set API Key** in the top‑right.
- **“No usable audio devices were found.”** → Plug in a mic/speaker and reopen CCMI.
- **Import XLSX fails** → Install `openpyxl` or import a CSV instead.
- **No sound on playback** → Pick an output device (🔊 badge) and test a voice with **🧪 Hear**.
- **Large prompts** → CCMI caps rolling context to ~5,000 chars to keep latency reasonable.

## Contributing
PRs and issues are welcome! See **CONTRIBUTING.md** for guidance.

## License
**MIT** — see `LICENSE`.

<p align="center">
  <a href="../../releases/latest">
    <img alt="Download for Windows" src="https://img.shields.io/badge/Download-Windows%20ZIP-blue">
  </a>
  <a href="../../releases/latest">
    <img alt="Latest release" src="https://img.shields.io/github/v/release/pasabayramoglu/ccmi?label=latest">
  </a>
</p>

**➡️ Quick download:** Get the ready-to-use Windows build from the
**[latest release](../../releases/latest)**. Unzip and run `ccmi.exe`.
