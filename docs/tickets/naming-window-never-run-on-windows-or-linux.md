# The speaker-naming window has never been opened on Windows or Linux

Status: OPEN · Priority: MEDIUM · Type: unverified platform path · Cost: none

> Found while building the naming window on 2026-11-27 (see `docs/PROGRESS.md`,
> decision rows 5 and 6). Only the macOS path was run for real.

## Symptom
`plugin_runner.py name` has a Windows path and a Linux path that exist only
as unit-tested command construction. Nobody has seen either window open.

## Evidence
- Windows: `Popen([python, "-B", "name_speakers.py", path], creationflags=0x10)`
  (`CREATE_NEW_CONSOLE`), in `claude_plugin/naming_window.py::launch_on_windows`.
- Linux: `x-terminal-emulator -e`, `gnome-terminal --`, `konsole -e`,
  `xfce4-terminal -x`, `xterm -e`, tried in that order when `DISPLAY` or
  `WAYLAND_DISPLAY` is set (`claude_plugin/window_commands.py`).
- `tests/test_naming_window.py` covers the argument lists and the fallback
  reasons with a fake host. No test starts a real window.

Known risks, none of them observed:
1. If Claude Code runs commands inside a Windows Job object that kills child
   processes, the new console may close when the runner exits.
   `CREATE_BREAKAWAY_FROM_JOB` was not added because it could not be tested.
2. With Windows Terminal as the default terminal the console is handed over
   to it. Expected to work, not seen.
3. A Linux launcher that exits 0 and then fails later (possible with
   gnome-terminal's server model) is reported as `opened` true.
4. Hebrew in the new Windows console relies on `terminal_bidi.should_reorder`
   treating it like today's consoles.

## Why it matters
On these systems the skill tells the user "a window has opened". If it did
not, or it closed at once, the user is left with unnamed speakers and no
command to paste, because the fallback only fires when the runner knows the
launch failed.

## Scope to decide
- Run `/transcribe <file> he speakers 2` from the installed plugin on the
  Windows machine used for the 2026-11-25 entries, and on one Linux desktop,
  and record what happened in `docs/PROGRESS.md` and the README's
  "What has been run for real" note.
- If risk 1 is real: add `CREATE_BREAKAWAY_FROM_JOB`, or start the window
  through `cmd /c start`.
- If risk 3 is real: confirm the window by a marker the naming program
  writes, not by the launcher's exit code.
