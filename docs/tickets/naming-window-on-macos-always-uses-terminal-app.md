# On macOS the naming window is always Terminal.app, and two first-use points are unchecked

Status: OPEN · Priority: LOW · Type: design limit and unverified behaviour · Cost: none

> Found while building the naming window on 2026-11-27 (see `docs/PROGRESS.md`,
> decision row 6).

## Symptom
1. The window opens in Terminal.app even when the user works in another
   terminal (iTerm2, Ghostty, the VS Code terminal).
2. It was not checked whether Terminal asks for file access the first time
   it opens a transcript that lives under `~/Library/CloudStorage` (Google
   Drive, OneDrive, iCloud Drive).
3. Step 7 of the skill has not been run from an installed plugin. The macOS
   proof ran the repo's `plugin_runner.py` directly.

## Evidence
- `claude_plugin/window_commands.py::macos_open_command` returns
  `["/usr/bin/open", "-a", "Terminal", <command file>]`. The app name is a
  constant.
- The real test used a throwaway transcript in a scratch folder whose path
  only imitated a Google Drive path.
- The real 2026-11-27 call transcript, which does live under
  `~/Library/CloudStorage`, was named by other means before this was built.

## Why it matters
Point 1 is a surprise at most. Point 2 could put a macOS privacy prompt, or
a "cannot open" error, in front of the user in exactly the common case,
since recordings are often kept in a synced drive. Point 3 is the last link
between the tested runner and what the user actually gets.

## Scope to decide
- Run one real `/transcribe` with speakers on a file under
  `~/Library/CloudStorage` after the plugin update, and record the result.
- Terminal choice: keep Terminal.app always (it is on every Mac)
  OR
  read a `TRANSCRIBE_TERMINAL_APP` setting and pass it to `open -a`.
