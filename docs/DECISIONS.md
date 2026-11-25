# Decisions

Append-only. These are not preferences: each row records why, and reversing
one means revisiting those reasons in a new row that names the old one.

| # | Date | Who | Decision |
|---|------|-----|----------|
| 1 | 2026-11-25 | Amit Raz with Claude | **Bidi control characters (U+200E, U+200F, U+061C, U+202A–U+202E, U+2066–U+2069) are stripped from every transcript at render time, in `transcript_output.clean_text()`; a segment that is only such characters is dropped.** Why: the Hebrew model (`ivrit-ai/whisper-large-v3-ct2`) emitted U+202B on 159 of 458 lines of a real 28-minute call, sometimes mid-line; they are invisible but break searching and later processing of the text, and a plain-text editor lays out Hebrew right to left without them. Rendering is the one point that covers both engines (mlx-whisper and faster-whisper) and both transcript kinds. Rejected: stripping inside each engine — two places to keep in sync, and the mlx path could not be run on this machine; stripping in `diarization.merge` — it would only cover labelled transcripts, and diarization cannot import `transcript_output` (which imports diarization) without a cycle; leaving them in — the original behaviour, with the search breakage above. Revisit if a consumer needs the model's raw text (it would need a separate raw output, not this file). Proof: 9 tests in `tests/test_transcript_output.py`; the real file re-run gave 0 controls in 458 lines. Files: `transcript_output.py`, `tests/test_transcript_output.py`, `README.md`. |
| 2 | 2026-11-25 | Amit Raz with Claude | **The two tests that failed on Windows were fixed to run on every platform, not skipped on Windows.** Why: both were test bugs and the code under test was correct on Windows. The cp1252 stream test now passes `newline="\n"` so the platform line ending does not enter the bytes it compares; the macOS drag-and-drop test builds its path with forward slashes so its own folder separators are not unescaped. Rejected: `skipIf(win32)` — it would hide the behaviour on the platform where the UTF-8 stream fix matters most. Proof: full suite 178 tests OK on Windows. Files: `tests/test_platform_support.py`. |

## Open questions

None yet.
