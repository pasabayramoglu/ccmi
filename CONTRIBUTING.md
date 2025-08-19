# Contributing to CCMI

Thanks for your interest!

## Development setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python ccmi.py
```

## Pull requests
- Keep PRs focused and small where possible.
- If adding features, update **README.md** and mention them in **RELEASE_NOTES.md**.
- No secrets in code or tests. API keys must be provided at runtime only.

## Issues
- Provide steps to reproduce and your OS/Python version.
- Attach screenshots if the problem is visual (UI/Qt).
