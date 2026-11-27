---
name: transcribe
description: Transcribe a Hebrew or English audio file on this machine with Whisper, optionally labelling who spoke, and report where the transcript was saved. Use when the user wants a recording, call, meeting or voice memo turned into text. The audio never leaves the computer.
argument-hint: <audio file> [he|en] [speakers [N] | nospeakers]
allowed-tools:
  - Bash(python3 "${CLAUDE_PLUGIN_ROOT}/plugin_runner.py" *)
  - Bash(py "${CLAUDE_PLUGIN_ROOT}/plugin_runner.py" *)
  - PowerShell(py "${CLAUDE_PLUGIN_ROOT}/plugin_runner.py" *)
  - AskUserQuestion
---

# Transcribe an audio file

The user typed:

<arguments>
$ARGUMENTS
</arguments>

Everything goes through one script, which prints one JSON object per call. Run it exactly as written below, with both paths in double quotes, because the pre-approved command must match character for character and a Windows home folder often has a space in it. On Windows write `py` where it says `python3`; on macOS and Linux keep `python3`.

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/plugin_runner.py" status --data-dir "${CLAUDE_PLUGIN_DATA}"
python3 "${CLAUDE_PLUGIN_ROOT}/plugin_runner.py" install --data-dir "${CLAUDE_PLUGIN_DATA}" [--with-diarization] [--gpu]
python3 "${CLAUDE_PLUGIN_ROOT}/plugin_runner.py" run --data-dir "${CLAUDE_PLUGIN_DATA}" -- --file '<path>' --lang <he|en> [--speakers auto|N]
python3 "${CLAUDE_PLUGIN_ROOT}/plugin_runner.py" name --data-dir "${CLAUDE_PLUGIN_DATA}" -- --file '<transcript path>'
```

## Privacy comes first

Do not open, read, preview, summarise or quote the transcript file or the log files, and do not `cat`, `tail` or `grep` them, unless the user asks for that in a later message. The point of this tool is that the audio stays on this machine; reading the transcript would send its text to Anthropic. The script already keeps transcript text out of its output, so its JSON is all you need. If the user later asks for a summary, read the transcript then, and only then.

## 1. Read the arguments

From the text above take the audio path and up to two options, which can only appear as trailing words:

- `he` or `en`: the language.
- `speakers` (count detected automatically), `speakers N` with N a positive whole number, or `nospeakers`.

Everything before those trailing words is the path, spaces included. Remove surrounding quotes. On macOS and Linux a path dragged into the terminal arrives with backslash escapes such as `My\ Call.m4a`; turn `\ ` back into a space and drop the other escaping backslashes. On Windows a backslash is the folder separator, so leave it alone.

If there is no path, ask the user for one in plain text and stop.

When you put the path in the command, wrap it in single quotes and write each `'` inside it as `'\''`. If the command runs through PowerShell rather than Bash, write each `'` as `''` instead.

## 2. Check the machine

Run `status` (it only reports, so it can run every time). Then:

- `ffmpeg` false: show `ffmpeg_hint` as the command that fixes it and stop.
- `python_ok` false: Python 3.10 or newer is needed to install. If an install is needed (step 3), say so and stop. A working install from earlier can still run.
- `installed` true but `capabilities` null, or `capabilities.engine` null: the last install did not finish. Treat it as not installed.

## 3. Install on first use, only after a yes

When `installed` is false, or `requirements_changed` is true (a plugin update changed the packages), explain what happens before anything is downloaded:

- The base install is Whisper and its packages, into a private folder for this plugin: about 1.3 GB on an Apple Silicon Mac (mlx-whisper brings PyTorch with it), smaller on Windows and Linux.
- Speaker labels add about 1.2 GB (PyTorch and pyannote).
- The first transcription in each language downloads its Whisper model, 1.6 to 3 GB, once.

Then ask with AskUserQuestion: install the base only (recommended unless they want speaker labels), install with speaker labels, or cancel. Never install without that yes; these are large downloads the user did not ask for by typing a file name.

On a yes, run `install`, adding `--with-diarization` if they chose speaker labels, and `--gpu` when `platform` is `win32` or starts with `linux` and `nvidia_gpu_detected` is true. Run it with `run_in_background: true`, because it can take longer than the Bash tool's time limit, and wait for the notification. If `ok` is false, show `log_tail` and the `log` path and stop. Otherwise run `status` again.

## 4. Language and speakers

Language: use the one in the arguments, else `default_language`. If neither exists, ask Hebrew or English. Never guess from the file name or anything else. The Hebrew model's own language detection is unreliable, so the tool always needs to be told, and a wrong pick wastes the whole run.

Speakers: `speakers` means `--speakers auto`, `speakers N` means `--speakers N`, `nospeakers` means no flag. If none of these was given, ask: no speaker labels, yes with the count detected automatically, or yes with a count they type.

When both are unknown, ask both in one AskUserQuestion call.

If speaker labels are wanted and `capabilities.diarization` is false, ask whether to install them now (about 1.2 GB; run `install --with-diarization`, plus `--gpu` under the rule above, in the background as in step 3) or to continue without labels.

## 5. Run

Run `run` with `run_in_background: true`; a long recording takes many minutes. Tell the user it has started and that the progress log is `${CLAUDE_PLUGIN_DATA}/logs/last-run.log`, so they can follow it themselves:

- macOS and Linux: `tail -f '${CLAUDE_PLUGIN_DATA}/logs/last-run.log'`
- Windows PowerShell: `Get-Content -Wait '${CLAUDE_PLUGIN_DATA}/logs/last-run.log'`

Then wait for the notification. Do not read the log yourself while it runs.

## 6. Report

Use only the JSON. Its text is ASCII, so a Hebrew file name arrives as `\u05..` escapes; show it decoded. On success (`ok` true) give `result.output` (the transcript path), `result.elapsed_s` as the time taken, `result.language`, and `result.speakers` when it is not null.

On failure give `message` and what the exit code means:

- 2: the input was refused (missing or unreadable file, unknown language, output already exists).
- 3: speaker labels were asked for but are not installed. The message's install commands are for someone running the script in a terminal, so leave them out and offer the plugin's own install from step 4.
- 4: ffmpeg is missing. Show the install hint from `status`.
- 1 or anything else: an error during transcription. Give the `log` path; the user can open it.

An object with an `error` field (`not_installed`, `usage`) comes from the script itself, not from a transcription: show its `message`, and for `not_installed` go back to step 3.

## 7. Naming the speakers

After a success where `result.speakers` is not null and is above 0, run `name` on `result.output`, quoting the path the same way as in step 1. It returns at once; do not run it in the background. It opens a terminal window on the user's screen that shows a few lines per speaker and asks for their names. You never see those lines, and you must not read the transcript to help.

- `opened` true: tell the user a window has opened where they can name the speakers, and that the sample lines it shows stay on their machine.
- `opened` false: the window could not be opened (`reason` says why, for example `remote_session` or `no_display`). Give them `manual_command` to paste into their own terminal; it asks the same questions there. If `manual_command` is null, skip naming.

Do not wait for the naming to finish or ask whether it went well; the window handles it.

After a success you may add one line: they can ask you for a summary of the transcript, which sends its text to Anthropic.
