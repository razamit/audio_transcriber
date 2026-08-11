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

This is the only script you run. It prompts for a language, a file, and whether
you want to know who said what:

```
Language: [1] Hebrew  [2] English  (Enter = Hebrew): 2
Path to audio file: call.m4a
Label speakers? [y/N]: y
Number of speakers (press Enter to auto-detect): 3
```

Drag the audio file from Finder into the Terminal window and press Enter, or
paste a path. Surrounding quotes, drag-and-drop backslash escaping, and `~` are
all handled. A bad path re-prompts up to three times.

Answering `n` to speakers skips diarization entirely — no pyannote, no second
model, no extra dependencies needed. See
[Speaker labels](#speaker-labels-who-said-what) for the rest.

The transcript is written next to the source audio in UTF-8, as `<name>.txt` or
`<name>_speakers.txt`. An existing file is never overwritten — the script falls
back to `<name>_2.txt`, `<name>_3.txt`, and so on.

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

## Project layout

`transcribe.py` is the only entry point; everything else is a module it drives.
`name_speakers.py` is the one exception — see
[Naming the speakers](#naming-the-speakers).

| File | Responsibility |
|------|----------------|
| `transcribe.py` | Prompts, ordering, error handling. No model code. |
| `profiles.py` | Which Whisper model goes with which language |
| `paths.py` | Drag-and-drop path cleanup, output naming, ffmpeg check |
| `transcription.py` | Running Whisper |
| `diarization.py` | Running pyannote; merging speakers into segments |
| `transcript_output.py` | Rendering and writing both transcript formats |
| `speaker_names.py` | Replacing `SPEAKER_00` with real names |
| `name_speakers.py` | Entry point for naming an existing transcript |

`diarization.py` keeps its `torch` and `pyannote` imports *inside* functions on
purpose. The module has to be importable on a machine with only the base
install, so the speakers prompt can be offered and explained rather than
crashing the script at startup.

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
into `~/.cache/huggingface`. After that everything runs fully offline — you
can verify this yourself by cutting the network, or by forcing it:

```
HF_HUB_OFFLINE=1 .venv/bin/python transcribe.py
```

Neither script requires a HuggingFace account. If you happen to have a token
configured, `transcribe.py` passes it along purely to get the higher rate limit on
downloads — it is a **download credential**, never an API key, is not used at
inference time, and nothing is metered or billed.

One caveat worth knowing: pyannote.audio 4.x bundles OpenTelemetry metrics
that are **enabled by default** and post to `https://otel.pyannote.ai` on every
run. No audio, text or filename is sent — it is the model id, library version,
a random session id, the audio duration and the speaker-count hint. `diarization.py`
opts out of this by setting `PYANNOTE_METRICS_ENABLED=false` before importing
pyannote. Export `PYANNOTE_METRICS_ENABLED=true` if you would rather send them.

## Speaker labels (who said what)

Answering `y` to `Label speakers?` adds speaker attribution. Whisper cannot do
this alone — it has no concept of speakers — so this runs
[pyannote.audio](https://github.com/pyannote/pyannote-audio) to find *who spoke
when*, runs Whisper to find *what was said*, and merges the two by timestamp.
Each Whisper segment is assigned to whichever speaker overlaps it most.

### Extra setup

The prompt is always offered, but the models behind it install separately:

```
./setup.sh --with-diarization
```

pyannote pulls in PyTorch and friends — roughly 1.2 GB on top of the base
install, which plain transcription never touches. That is the only reason it is
a separate step. Answering `y` without it exits immediately with this command
rather than failing partway through a run.

Beyond that there is nothing to configure: no account, no token, no terms to
accept — the diarization model (~33 MB) downloads on first run like any other
dependency.

This uses [`pyannote-community/speaker-diarization-community-1`](https://huggingface.co/pyannote-community/speaker-diarization-community-1),
an **ungated re-upload** of pyannote's Community-1 pipeline, pinned to a fixed
revision. pyannote's own copy is identical in function but gated behind a
HuggingFace account, which would put a signup between you and a tool that
otherwise needs no accounts at all.

The tradeoff is provenance, and it is worth stating plainly: the mirror is
uploaded by a third party, not the pyannote org. Every filename and byte size
matches the official repo's listing, but HuggingFace does not serve content
hashes for a gated repo without auth, so the weights **cannot be verified as
byte-identical to upstream**. Pinning `DIARIZATION_REVISION` is the mitigation —
the bytes are fixed at one reviewed commit and cannot change under a later run.

If you would rather have first-party weights, set both constants at the top of
`diarization.py` back to the official repo and follow its gated flow:

```python
DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
DIARIZATION_REVISION = None
```

1. Create a free account at <https://huggingface.co/join>
2. Accept the conditions at <https://hf.co/pyannote/speaker-diarization-community-1>
3. Create a **Read** token at <https://hf.co/settings/tokens>
4. Store it once: `.venv/bin/hf auth login`

`transcribe.py` already passes a token when one is configured, so nothing else
changes.

### Usage

Answering `y` adds one more prompt — a speaker count. Leave it blank to
auto-detect, or enter a number if you know it; supplying it is the single
biggest accuracy lever pyannote exposes. pyannote itself is language-agnostic,
so the language choice only affects the Whisper half of the merge. Output goes
to `<name>_speakers.txt`:

```
2 speaker(s) detected:
  SPEAKER_00  00:00:04  (61%)
  SPEAKER_01  00:00:02  (39%)

[00:00:00] SPEAKER_00: שלום, מה שלומך היום?

[00:00:04] SPEAKER_01: אני בסדר גמור, תודה ששאלת.
```

### Naming the speakers

Diarization separates voices but cannot name them, so it labels them
`SPEAKER_00`, `SPEAKER_01`... The numbering reflects clustering order and
changes between runs, which makes a raw diarized transcript tiring to read.

When a run finishes, `transcribe.py` writes the file and *then* offers to name
them. It shows each speaker's longest lines, because the label alone is not
something anyone can answer:

```
SPEAKER_01 — 235 segment(s)
  [00:04:18] I'm guessing it will be easier for you now to really give me...
  [00:10:10] So, you see most stages are quite quick. The things that take...
Name (Enter to keep SPEAKER_01):
```

Press Enter to leave a speaker anonymous — useful when you recognise two voices
out of three. The transcript is rewritten in place, header included. Since the
file is written before the prompt, abandoning it with Ctrl-C costs nothing.

To name a transcript produced earlier, without re-running both models:

```
.venv/bin/python name_speakers.py
```

It takes an existing `<name>_speakers.txt` and rewrites it the same way. This is
the one place the tool overwrites a file rather than bumping to `_2` — renaming
labels is reversible by re-running diarization, and a stack of near-identical
transcripts is worse than editing the one you have.

### Limitations

- **The model cannot identify anyone.** It separates voices; putting names to
  them is a manual step you drive, from listening or from context in the text.
- **Crosstalk degrades it.** When two people talk over each other, the segment
  is attributed to whoever overlaps more; the other person's words are still
  in the text but under the wrong label.
- **Boundaries are approximate.** Whisper's segments and pyannote's turns are
  produced independently, so a speaker change mid-Whisper-segment cannot be
  split — that whole segment gets one label.
- **Short clips collapse to one speaker.** The clustering needs material to
  work with; under roughly ten seconds pyannote warns that the audio is *"too
  short to contain 2 or more speakers"* and returns a single label, even when
  you asked for two. This is a floor on the audio, not a bug.
- Segments matching no speaker turn are labelled `UNKNOWN` rather than being
  silently dropped or guessed at.
- It roughly doubles the runtime, since the audio is processed by two models.

## Scope

Transcription, optional speaker attribution, and naming those speakers. No
summarization, no API calls, no CLI flags — every script is interactive prompts
only, with a single `TRANSCRIBE_LANG` environment override.

Because each profile pins its language, speech in any *other* language is not
transcribed in its own language; Whisper renders it into the selected one. One
language per run — a bilingual recording needs two passes, and neither will be
right about the other half.

There is no CI: the test path needs Apple Silicon hardware and multi-GB model
downloads, which no hosted runner provides.

## License

MIT — see [LICENSE](LICENSE).
