# audio_transcriber

Local Hebrew audio transcription on Apple Silicon. A single interactive Python
script that runs [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
against the [ivrit-ai](https://huggingface.co/ivrit-ai) Whisper large-v3
finetune. Nothing is uploaded — transcription runs entirely on-device.

## Requirements

- Apple Silicon Mac (MLX is Metal-only; this will not run on Intel or Linux)
- Python 3.10 or newer
- `ffmpeg` on your `PATH`, used by Whisper to decode audio:
  ```
  brew install ffmpeg
  ```

## Setup

```
git clone <your-repo-url>
cd audio_transcriber
./setup.sh
```

`setup.sh` verifies ffmpeg is installed, creates `.venv`, and installs
`mlx-whisper` from `requirements.txt`.

## Usage

```
.venv/bin/python transcribe.py
```

It prompts for a path:

```
Path to audio file:
```

Drag the audio file from Finder into the Terminal window and press Enter, or
paste a path. Surrounding quotes, drag-and-drop backslash escaping, and `~` are
all handled. A bad path re-prompts up to three times.

The transcript is written next to the source audio as `<name>.txt` in UTF-8. An
existing file is never overwritten — the script falls back to `<name>_2.txt`,
`<name>_3.txt`, and so on.

On first run the model (~3GB) downloads from HuggingFace into
`~/.cache/huggingface`. The script prints a heads-up before this so a long
first startup doesn't look like a hang.

## Output format

One line per Whisper segment, prefixed with an `HH:MM:SS` timestamp. A blank
line is inserted wherever the gap between segments exceeds two seconds, which
gives rough paragraph breaks and loosely tracks speaker turns:

```
[00:00:00] שלום, שמי כרמית ואני בודקת את מערכת התמלול.
[00:00:03] היום יום נעים מאוד בתל אביב.

[00:00:09] זהו המשפט השני של הבדיקה, אחרי הפסקה ארוכה.
```

This is deliberate — `result["text"]` from Whisper is a single unbroken blob
that is unpleasant to read for anything longer than a minute.

## Notes on the transcription settings

- **`language="he"` is mandatory, not an optimization.** The finetune's
  language detection was degraded during training, so autodetection is
  unreliable. This model is for Hebrew only.
- **`condition_on_previous_text=False`** avoids Whisper's repetition-loop
  failure mode, where the decoder gets stuck emitting the same phrase for
  minutes on long recordings.
- **`hallucination_silence_threshold=2.0` is currently inert.** mlx-whisper
  0.4.3 only consults it when `word_timestamps=True`, and unlike
  openai-whisper it does not raise an error for the combination. Setting
  `word_timestamps=True` in the `transcribe()` call would activate it, at the
  cost of per-word cross-attention alignment on every segment — a meaningful
  slowdown on long files.

## Scope

Transcription only. No diarization, no summarization, no API calls, no CLI
flags. There is no CI: the test path needs Apple Silicon hardware and a 3GB
model download, which no hosted runner provides.

## License

MIT — see [LICENSE](LICENSE).
