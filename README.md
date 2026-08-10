# audio_transcriber

Local Hebrew and English audio transcription on Apple Silicon. A single
interactive Python script that runs [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
against a Whisper model chosen to match the language. Nothing is uploaded —
transcription runs entirely on-device.

> **Apple Silicon only.** This runs on M-series Macs and nowhere else. It is
> built on Apple's MLX framework, which is Metal-backed and has no CPU or CUDA
> fallback, so it will not run on an Intel Mac, on Linux, or on Windows —
> including under Docker or a VM on those hosts.

## Requirements

- **Apple Silicon Mac (M1 or newer).** Hard requirement, not a recommendation —
  see the note above.
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

It prompts for a language, then a path:

```
Language: [1] Hebrew  [2] English  (Enter = Hebrew):
Path to audio file:
```

Drag the audio file from Finder into the Terminal window and press Enter, or
paste a path. Surrounding quotes, drag-and-drop backslash escaping, and `~` are
all handled. A bad path re-prompts up to three times.

The transcript is written next to the source audio as `<name>.txt` in UTF-8. An
existing file is never overwritten — the script falls back to `<name>_2.txt`,
`<name>_3.txt`, and so on.

Each language has its own model, downloaded from HuggingFace into
`~/.cache/huggingface` the first time you pick it. The script prints a heads-up
before this so a long first startup doesn't look like a hang.

To skip the prompt — useful if you mostly work in one language — set
`TRANSCRIBE_LANG` to a language key:

```
TRANSCRIBE_LANG=en .venv/bin/python transcribe.py
```

An unrecognized value is an error rather than a silent fallback, so a typo
can't quietly transcribe an hour of audio with the wrong model.

## Languages

| Key | Language | Model | Download |
|-----|----------|-------|----------|
| `he` | Hebrew (default) | [`ivrit-ai-whisper-large-v3`](https://huggingface.co/mlx-community/ivrit-ai-whisper-large-v3-mlx) | ~3 GB |
| `en` | English | [`whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | ~1.6 GB |

The model and the language code are one choice, not two — they live paired in
`profiles.py`. The Hebrew entry is an [ivrit-ai](https://huggingface.co/ivrit-ai)
finetune that is *only* good at Hebrew, and the English entry is stock
large-v3-turbo, whose distilled decoder makes it several times faster than full
large-v3 at effectively equal quality on clear conversational English. Pointing
either model at the other's language degrades the output with no error to warn
you, which is why they cannot be set independently.

Adding a language is one entry in `PROFILES`. For difficult English audio —
heavy accents, background noise, persistent crosstalk — swap the English entry's
repo for `mlx-community/whisper-large-v3-mlx` (3 GB), which is slower but more
robust.

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

- **An explicit `language` is mandatory, not an optimization.** The Hebrew
  finetune's language detection was degraded during training, so autodetection
  is unreliable. Every profile therefore pins its language rather than letting
  Whisper guess — which also means a file must be transcribed in the language
  you picked, and mixed-language audio is rendered into that one language.
- **`condition_on_previous_text=False`** avoids Whisper's repetition-loop
  failure mode, where the decoder gets stuck emitting the same phrase for
  minutes on long recordings.
- **`hallucination_silence_threshold=2.0` is currently inert.** mlx-whisper
  0.4.3 only consults it when `word_timestamps=True`, and unlike
  openai-whisper it does not raise an error for the combination. Setting
  `word_timestamps=True` in the `transcribe()` call would activate it, at the
  cost of per-word cross-attention alignment on every segment — a meaningful
  slowdown on long files.

## What runs locally

All transcription and diarization runs on your Mac. Audio is never uploaded,
and there is no inference API involved.

Network access is needed **once**, to download model weights from HuggingFace
into `~/.cache/huggingface`. After that both scripts run fully offline — you
can verify this yourself by cutting the network, or by forcing it:

```
HF_HUB_OFFLINE=1 .venv/bin/python transcribe.py
```

The HuggingFace token that `diarize.py` requires is a **download credential**,
not an API key. It authenticates the one-time fetch of gated model files. It is
not used at inference time and nothing is metered or billed.

One caveat worth knowing: pyannote.audio 4.x bundles OpenTelemetry metrics
that are **enabled by default** and post to `https://otel.pyannote.ai` on every
run. No audio, text or filename is sent — it is the model id, library version,
a random session id, the audio duration and the speaker-count hint. `diarize.py`
opts out of this by setting `PYANNOTE_METRICS_ENABLED=false` before importing
pyannote. Export `PYANNOTE_METRICS_ENABLED=true` if you would rather send them.

## Speaker labels (who said what)

`diarize.py` is an optional second script that adds speaker attribution.
Whisper cannot do this alone — it has no concept of speakers — so this runs
[pyannote.audio](https://github.com/pyannote/pyannote-audio) to find *who spoke
when*, runs Whisper to find *what was said*, and merges the two by timestamp.
Each Whisper segment is assigned to whichever speaker overlaps it most.

### Extra setup

```
./setup.sh --with-diarization
```

The pyannote models are gated, so a HuggingFace token is required:

1. Create a free account at <https://huggingface.co/join>
2. Accept the conditions on **both** model pages:
   - <https://hf.co/pyannote/speaker-diarization-community-1>
   - <https://hf.co/pyannote/segmentation-3.0>
3. Create a **Read** token at <https://hf.co/settings/tokens>
4. Store it once: `.venv/bin/hf auth login`

### Usage

```
.venv/bin/python diarize.py
```

It prompts for the language, then the audio path, then a speaker count — leave
the count blank to auto-detect, or enter a number if you know it. Supplying it
is the single biggest accuracy lever. pyannote itself is language-agnostic, so
the language choice only affects the Whisper half of the merge. Output goes to
`<name>_speakers.txt`:

```
2 speaker(s) detected:
  SPEAKER_00  00:00:04  (61%)
  SPEAKER_01  00:00:02  (39%)

[00:00:00] SPEAKER_00: שלום, מה שלומך היום?

[00:00:04] SPEAKER_01: אני בסדר גמור, תודה ששאלת.
```

### Limitations

- **Speakers are anonymous.** You get `SPEAKER_00`, `SPEAKER_01` — the model
  separates voices, it cannot name them. Map them to real names yourself.
- **Crosstalk degrades it.** When two people talk over each other, the segment
  is attributed to whoever overlaps more; the other person's words are still
  in the text but under the wrong label.
- **Boundaries are approximate.** Whisper's segments and pyannote's turns are
  produced independently, so a speaker change mid-Whisper-segment cannot be
  split — that whole segment gets one label.
- Segments matching no speaker turn are labelled `UNKNOWN` rather than being
  silently dropped or guessed at.
- It roughly doubles the runtime, since the audio is processed by two models.

## Scope

Transcription and optional speaker attribution. No summarization, no API
calls, no CLI flags — both scripts are interactive prompts only, with a single
`TRANSCRIBE_LANG` environment override.

Because each profile pins its language, speech in any *other* language is not
transcribed in its own language; Whisper renders it into the selected one. One
language per run — a bilingual recording needs two passes, and neither will be
right about the other half.

There is no CI: the test path needs Apple Silicon hardware and multi-GB gated
model downloads, which no hosted runner provides.

## License

MIT — see [LICENSE](LICENSE).
