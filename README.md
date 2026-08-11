# audio_transcriber

Local Hebrew and English audio transcription on Apple Silicon, with optional
speaker labels. One interactive Python script: point it at a recording, get a
timestamped transcript — and, if you want, one that says who said what.

Nothing is uploaded. Every model runs on your Mac, and the only network access
is a one-time download of model weights.

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
./setup.sh                      # transcription only
./setup.sh --with-diarization   # transcription + speaker labels
```

`setup.sh` verifies ffmpeg is installed, creates `.venv`, and installs
`mlx-whisper`. The `--with-diarization` flag adds `pyannote.audio` on top.

That flag is the only real decision at setup time. Speaker labelling needs
PyTorch, which is **~1.2 GB** of dependencies that plain transcription never
touches, so it is kept separate rather than imposed on everyone. You can add it
later by re-running with the flag; nothing else changes.

## Usage

```
.venv/bin/python transcribe.py
```

This is the only script you run. It asks up to four questions:

```
Language: [1] Hebrew  [2] English  (Enter = Hebrew): 2
Path to audio file: call.m4a
Label speakers? [y/N]: y
Number of speakers (press Enter to auto-detect): 3
```

Drag the audio file from Finder into the Terminal window and press Enter, or
paste a path. Surrounding quotes, drag-and-drop backslash escaping, and `~` are
all handled. A bad path re-prompts up to three times.

Answering `n` to speakers skips diarization entirely — no second model, and no
need for the diarization extras. Answering `y` without those extras installed
exits immediately with the install command, before touching your audio.

The transcript is written next to the source audio in UTF-8, as `<name>.txt` or
`<name>_speakers.txt`. An existing file is never overwritten — the script falls
back to `<name>_2.txt`, `<name>_3.txt`, and so on.

To skip the language prompt, set `TRANSCRIBE_LANG`:

```
TRANSCRIBE_LANG=en .venv/bin/python transcribe.py
```

An unrecognized value is an error rather than a silent fallback, so a typo
can't quietly transcribe an hour of audio with the wrong model.

## How it works

Two models run over the same audio and their results are merged by timestamp.
Neither can do the job alone: **Whisper has no concept of speakers, and pyannote
has no concept of words.**

```
                     audio file (any format ffmpeg reads)
                                   |
                     ffmpeg decodes to 16 kHz mono
                                   |
                 +-----------------+-----------------+
                 |                                   |
         pyannote  Community-1                mlx-whisper
         "who spoke when"                     "what was said"
                 |                                   |
   (0.0-4.1, SPEAKER_01)              [00:00:00] Hello, my name is...
   (7.1-11.2, SPEAKER_00)             [00:00:06] Today is a very...
                 |                                   |
                 +-----------------+-----------------+
                                   |
                     merge: each text segment gets the
                     speaker who overlaps it the longest
                                   |
                     [00:00:00] SPEAKER_01: Hello, my name is...
                                   |
                     you name the speakers (optional)
                                   |
                     [00:00:00] Samantha: Hello, my name is...
```

Step by step:

1. **Decode.** Whatever you hand it — `.m4a`, `.mp3`, `.wav` — is decoded to the
   16 kHz mono audio both models expect. This is why ffmpeg is a hard
   requirement: without it Whisper cannot open anything.
2. **Diarize** (only if you asked for speakers). pyannote finds speech regions,
   detects speaker changes, extracts a voice fingerprint per region, and
   clusters those fingerprints into N speakers. It outputs time ranges with
   anonymous labels and no text at all.
3. **Transcribe.** Whisper produces text segments with start/end times, using
   the model paired with your chosen language.
4. **Merge.** Each Whisper segment is assigned to whichever speaker turn
   overlaps it most in time. A segment overlapping no turn is labelled
   `UNKNOWN` rather than dropped or guessed at.
5. **Render.** Lines get `HH:MM:SS` timestamps, with a blank line at every long
   pause and at every speaker change.
6. **Name.** The file is written, and *then* you are offered the chance to
   replace `SPEAKER_00` with real names.

Diarization runs *before* transcription so that a missing model or unreadable
audio fails in seconds rather than after minutes of transcribing.

### Speed

Measured on a 41-minute recording, Apple Silicon, English:

| Mode | Time |
|------|------|
| Transcription only | 29 s |
| Transcription + diarization | 87 s |

Both run far above realtime. Diarization roughly triples the total, but the
total is still under 90 seconds for a 41-minute call — so the old advice to
skip it for speed no longer applies. Short clips are dominated by model loading
rather than audio length.

## Models

Four models across three jobs, all downloaded from HuggingFace into
`~/.cache/huggingface` on first use and cached forever after.

| Job | Model | Size | License |
|-----|-------|------|---------|
| Hebrew speech → text | [`mlx-community/ivrit-ai-whisper-large-v3-mlx`](https://huggingface.co/mlx-community/ivrit-ai-whisper-large-v3-mlx) | ~3 GB | Apache-2.0 upstream |
| English speech → text | [`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | ~1.6 GB | MIT upstream |
| Who spoke when | [`pyannote-community/speaker-diarization-community-1`](https://huggingface.co/pyannote-community/speaker-diarization-community-1) | ~33 MB | CC-BY-4.0 |

### Where the Whisper models come from

Both are **MLX format conversions** published by
[mlx-community](https://huggingface.co/mlx-community), the community org that
republishes models in Apple's MLX format. MLX is what makes these run on the
Mac's GPU; the conversions are re-encodings of existing weights, not retrained
models.

The lineage:

```
openai/whisper-large-v3  (MIT)
   |
   +-- ivrit-ai/whisper-large-v3  (Apache-2.0, finetuned on Hebrew)
   |        |
   |        +-- mlx-community/ivrit-ai-whisper-large-v3-mlx   <- Hebrew profile
   |
   +-- openai/whisper-large-v3-turbo  (MIT, distilled decoder)
            |
            +-- mlx-community/whisper-large-v3-turbo          <- English profile
```

[ivrit-ai](https://huggingface.co/ivrit-ai) is an Israeli open-source project
building Hebrew speech models; their large-v3 finetune is what makes Hebrew
transcription here markedly better than stock Whisper. It declares
`openai/whisper-large-v3` as its base model and is Apache-2.0.

The English model is OpenAI's `large-v3-turbo` — same encoder as large-v3 with a
distilled 4-layer decoder, which is why it is roughly 4–5× faster at half the
size while staying essentially as accurate on clear conversational English.

One caveat: the mlx-community conversions declare **no license of their own**.
The upstream licenses above are permissive, and the conversions are
straightforward format changes, but the conversion repos themselves are silent
on licensing.

### Where the diarization model comes from

[pyannote](https://github.com/pyannote/pyannote-audio) Community-1 is not one
model but a three-stage pipeline, all inside one ~33 MB repo:

| Stage | Weights | Job |
|-------|---------|-----|
| Segmentation | 5.9 MB | Where is there speech, where do speakers change, where do they overlap |
| Embedding | 26.6 MB | A voice fingerprint per speech region |
| VBx clustering + PLDA | 0.2 MB | Group the fingerprints into N speakers |

It ships tuned parameters — clustering `threshold: 0.6`, and
`embedding_exclude_overlap: true`, meaning it deliberately ignores crosstalk
when building fingerprints, since two mixed voices make a useless one.

**This is an ungated re-upload, and that deserves a straight explanation.**
pyannote's own `pyannote/speaker-diarization-community-1` is identical in
function but *gated*: downloading it requires a HuggingFace account and clicking
through a terms page. Both copies are CC-BY-4.0, a license that explicitly
permits redistribution, so a third party lawfully re-uploaded it without the
gate. This project uses that copy so `git clone && ./setup.sh` works with no
accounts anywhere.

The tradeoff is provenance. The mirror is uploaded by a third party, not the
pyannote org. Every filename and byte size matches the official repo's listing,
and it produces correct output, but HuggingFace does not serve content hashes
for a gated repo without auth — so the weights **cannot be verified as
byte-identical to upstream**. The mitigation is that `DIARIZATION_REVISION`
pins a specific commit, so the bytes are fixed and cannot change under a later
run.

If you would rather have first-party weights, edit the two constants at the top
of `diarization.py`:

```python
DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
DIARIZATION_REVISION = None
```

then create a free account, accept the conditions at
<https://hf.co/pyannote/speaker-diarization-community-1>, make a **Read** token
at <https://hf.co/settings/tokens>, and store it once with
`.venv/bin/hf auth login`. The code already passes a token when one is
configured, so nothing else changes.

## Output format

One line per Whisper segment, prefixed with an `HH:MM:SS` timestamp. A blank
line is inserted wherever the gap between segments exceeds two seconds, giving
rough paragraph breaks:

```
[00:00:00] שלום, שמי כרמית ואני בודקת את מערכת התמלול.
[00:00:03] היום יום נעים מאוד בתל אביב.

[00:00:09] זהו המשפט השני של הבדיקה, אחרי הפסקה ארוכה.
```

This is deliberate — `result["text"]` from Whisper is a single unbroken blob
that is unpleasant to read for anything longer than a minute.

With speakers, the file gains a talk-time header, and a blank line appears at
every speaker change as well as every long pause:

```
2 speaker(s) detected:
  SPEAKER_00  00:00:04  (61%)
  SPEAKER_01  00:00:02  (39%)

[00:00:00] SPEAKER_00: שלום, מה שלומך היום?

[00:00:04] SPEAKER_01: אני בסדר גמור, תודה ששאלת.
```

## Naming the speakers

Diarization separates voices but cannot name them, so it labels them
`SPEAKER_00`, `SPEAKER_01`... The numbering reflects clustering order and
changes between runs, which makes a raw diarized transcript tiring to read.

When a run finishes, the file is written and *then* you are offered the chance
to name them. The prompt shows each speaker's longest lines, because the label
alone is not something anyone can answer:

```
SPEAKER_01 — 235 segment(s)
  [00:04:18] I'm guessing it will be easier for you now to really give me...
  [00:10:10] So, you see most stages are quite quick. The things that take...
Name (Enter to keep SPEAKER_01):
```

Press Enter to leave a speaker anonymous — useful when you recognise two voices
out of three. Names work in Hebrew as well as English. The transcript is
rewritten in place, header included. Since the file is written before the
prompt, abandoning it with Ctrl-C costs nothing.

To name a transcript produced earlier, without re-running either model:

```
.venv/bin/python name_speakers.py
```

It takes an existing `<name>_speakers.txt` and rewrites it the same way. This is
the one place the tool overwrites a file rather than bumping to `_2` — renaming
labels is reversible by re-running diarization, and a stack of near-identical
transcripts is worse than editing the one you have.

## Limitations

- **The model cannot identify anyone.** It separates voices; putting names to
  them is a manual step you drive, from listening or from context in the text.
- **Crosstalk degrades it.** When two people talk over each other, the segment
  is attributed to whoever overlaps more; the other person's words are still
  in the text but under the wrong label.
- **Boundaries are approximate.** Whisper's segments and pyannote's turns are
  produced independently, so a speaker change mid-Whisper-segment cannot be
  split — that whole segment gets one label.
- **Short clips collapse to one speaker.** The clustering needs material to work
  with; under roughly ten seconds pyannote warns that the audio is *"too short
  to contain 2 or more speakers"* and returns a single label, even when you
  asked for two. This is a floor on the audio, not a bug.
- **Supplying the speaker count forces it.** Asking for 3 produces exactly 3,
  so a wrong guess splits one person in two rather than being corrected. When
  unsure, auto-detect.
- Segments matching no speaker turn are labelled `UNKNOWN` rather than being
  silently dropped or guessed at.

## What runs locally

All transcription and diarization runs on your Mac. Audio is never uploaded,
and there is no inference API involved.

Network access is needed **once**, to download model weights from HuggingFace
into `~/.cache/huggingface`. After that everything runs fully offline — you can
verify this yourself by cutting the network, or by forcing it:

```
HF_HUB_OFFLINE=1 .venv/bin/python transcribe.py
```

No HuggingFace account is required for anything. If you happen to have a token
configured it is passed along purely to get the higher rate limit on downloads —
it is a **download credential**, never an API key, is not used at inference
time, and nothing is metered or billed.

One caveat worth knowing: pyannote.audio 4.x bundles OpenTelemetry metrics that
are **enabled by default** and post to `https://otel.pyannote.ai` on every run.
No audio, text or filename is sent — it is the model id, library version, a
random session id, the audio duration and the speaker-count hint. `diarization.py`
opts out by setting `PYANNOTE_METRICS_ENABLED=false` before importing pyannote.
Export `PYANNOTE_METRICS_ENABLED=true` if you would rather send them.

## Project layout

`transcribe.py` is the only entry point; everything else is a module it drives.
`name_speakers.py` is the one exception, for relabelling a transcript you
already have.

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

Two details that look odd but are deliberate:

- **`diarization.py` keeps `torch` and `pyannote` imports inside functions.** The
  module must be importable on a machine with only the base install, so the
  speakers prompt can be offered and explained rather than crashing at startup.
- **`profiles.py` pairs the model and the language code in one frozen object.**
  They are one choice, not two — see below.

## Notes on the transcription settings

- **An explicit `language` is mandatory, not an optimization.** The Hebrew
  finetune's language detection was degraded during training, so autodetection
  is unreliable. Every profile pins its language rather than letting Whisper
  guess — which also means a file is transcribed in the language you picked, and
  mixed-language audio is rendered into that one language.
- **The model and the language cannot be set independently.** Pointing the
  Hebrew finetune at English produces degraded output, sometimes transliterated
  into Hebrew script, with no error to warn you. That is why they live paired in
  `profiles.py`; adding a language is one entry there.
- **`condition_on_previous_text=False`** avoids Whisper's repetition-loop
  failure mode, where the decoder gets stuck emitting the same phrase for
  minutes on long recordings.
- **`hallucination_silence_threshold=2.0` is currently inert.** mlx-whisper
  0.4.3 only consults it when `word_timestamps=True`, and unlike openai-whisper
  it does not raise an error for the combination. Setting `word_timestamps=True`
  would activate it, at the cost of per-word cross-attention alignment on every
  segment — a meaningful slowdown on long files.
- For difficult English audio — heavy accents, background noise, persistent
  crosstalk — swap the English profile's repo for
  `mlx-community/whisper-large-v3-mlx` (3 GB), which is slower but more robust.

## Scope

Transcription, optional speaker attribution, and naming those speakers. No
summarization, no API calls, no CLI flags — interactive prompts only, with a
single `TRANSCRIBE_LANG` environment override.

One language per run. A bilingual recording needs two passes, and neither will
be right about the other half.

There is no CI: the test path needs Apple Silicon hardware and multi-GB model
downloads, which no hosted runner provides.

## License

MIT — see [LICENSE](LICENSE). The models carry their own licenses; see
[Models](#models).
