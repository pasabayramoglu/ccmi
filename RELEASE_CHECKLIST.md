# Release Checklist

## Before publishing
- [ ] Update **RELEASE_NOTES.md** date and bullets
- [ ] README verified on a fresh machine
- [ ] `requirements.txt` installs cleanly
- [ ] No secrets in repo (`git grep -n 'sk-'` should be empty)
- [ ] Build Windows exe with PyInstaller

## Build (Windows)
```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller ccmi.spec
```

## Draft the GitHub Release
- Tag: `vX.Y.Z`
- Title: `CCMI vX.Y.Z`
- Notes: copy section from **RELEASE_NOTES.md**
- Assets: attach `CCMI_Windows_x64.zip` (contains `ccmi.exe`)

## After publishing
- [ ] Add topics: `python`, `pyqt5`, `openai`, `asr`, `tts`, `interpreter`
- [ ] Pin the repo (optional)
