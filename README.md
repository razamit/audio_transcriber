# audio_transcriber

Local Hebrew and English audio transcription on macOS, Windows and Linux, with
optional speaker labels. One interactive Python script: point it at a
recording, get a timestamped transcript, and, if you want, one that says who
said what.

Nothing is uploaded. Every model runs on your computer, and the only network
access is a one-time download of model weights.

Whisper runs on one of two engines, picked automatically:

| Machine | Engine | Runs on |
|---------|--------|---------|
| Apple Silicon Mac (M1 or newer) | [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | The Mac's GPU, through MLX |
| Windows (x64), Linux, Intel Mac | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) | An NVIDIA GPU if set up, otherwise the CPU |

Both engines use the same decoding settings, both decode greedily, and both
use the same model family per language. Transcripts from different machines
are close but not identical: the engines are separate implementations, the CPU
runs 8-bit weights, and Hebrew on the CPU uses the smaller turbo model (see
[Models](#models)). Windows on ARM is not supported: CTranslate2 publishes no
package for it.

Intel Macs get faster-whisper on the CPU, but that combination has not been
tested. Speaker labelling cannot be installed on an Intel Mac at all: PyTorch
no longer publishes Intel Mac builds new enough for pyannote.

> **What has been run for real.** On Windows 11 with an NVIDIA RTX 4090 and
> an Intel i9-14900K: `py install.py --gpu --with-diarization`, and a Hebrew
> transcription on the GPU, on the CPU, and with speaker labels on the GPU
> (see [Speed](#speed)). The CPU path of faster-whisper has also been run on a
> Mac, forced with `TRANSCRIBE_ENGINE=faster-whisper`. Not yet run: anything on
> Linux, and speaker labelling on a Windows machine without an NVIDIA GPU.
> Those are built from the library documentation and covered by unit tests
> with fakes.

## Requirements

- Python 3.10 or newer
- `ffmpeg` on your `PATH`, used to decode audio for both models:
  ```
  brew install ffmpeg        # macOS
  winget install ffmpeg      # Windows, then open a new terminal
  sudo apt install ffmpeg    # Debian and Ubuntu; use your distribution's package manager
  ```
- For GPU transcription on Windows or Linux: an NVIDIA GPU with a current
  driver. The CUDA libraries themselves come from pip (see `--gpu` below).

## Setup

macOS and Linux:

```
git clone <your-repo-url>
cd audio_transcriber
./setup.sh                      # transcription only
./setup.sh --with-diarization   # transcription + speaker labels
```

Windows (PowerShell or Command Prompt):

```
git clone <your-repo-url>
cd audio_transcriber
py install.py                      # transcription only
py install.py --with-diarization   # transcription + speaker labels
```

`py` starts the newest Python installed. faster-whisper needs onnxruntime,
which has no prebuilt package for Python 3.15 yet, so on 3.15 or newer the
installer warns that the install may fail. If it does, pick an older Python,
for example `py -3.13 install.py`.

`install.py` does the work on every platform; `setup.sh` just runs it with the
same arguments. It checks the Python version and ffmpeg, creates `.venv`, and
installs `requirements.txt`, whose markers pick `mlx-whisper` on Apple Silicon
and `faster-whisper` everywhere else. The `--with-diarization` flag adds
`pyannote.audio` on top.

`--venv-dir PATH` creates the environment in that folder instead of `.venv`;
the requirements files are still read from the project folder. The
[Claude Code plugin](#claude-code-plugin) uses it to keep its environment
outside the plugin folder, which Claude Code replaces on every update. Without
the flag nothing changes.

`--with-diarization` is the main decision at setup time. Speaker labelling
needs PyTorch, which is **~1.2 GB** of dependencies that plain transcription
never touches, so it is kept separate rather than imposed on everyone. You can
add it later by re-running with the flag; nothing else changes.

### NVIDIA GPU (Windows and Linux)

Add `--gpu` to use an NVIDIA GPU:

```
py install.py --gpu                      # Windows
./setup.sh --gpu --with-diarization      # Linux, with speaker labels
```

It installs cuBLAS for CUDA 12 and cuDNN 9 from pip (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`), which faster-whisper needs on the GPU. With
`--with-diarization` it also installs a CUDA 12 build of PyTorch from
`https://download.pytorch.org/whl/cu126` before pyannote, so speaker labelling
uses the GPU too. Any torch already installed is removed first, because pip
would otherwise keep an existing CPU build as "already satisfied".

On Windows the CUDA build of torch carries its own cuDNN 9, with the same file
names as `nvidia-cudnn-cu12` but a different version. A process can hold only
one of them, so when torch is installed faster-whisper uses torch's copy too;
otherwise speaker labelling fails at `import torch` with `WinError 127` on
`cudnn_cnn64_9.dll`.

At startup the tool checks that a GPU exists *and* that those libraries load.
If the GPU is there but the libraries are not, it prints one line saying so and
runs on the CPU instead of crashing partway through. It always prints which
device it uses, for example `Transcription running on: cuda (float16)` or
`Transcription running on: cpu (int8, 8 threads)`. If the GPU fails later, such
as running out of memory on a small card, it prints one line and transcribes
the file again on the CPU, with the CPU model, and stays on the CPU after that.

## Usage

```
.venv/bin/python transcribe.py          # macOS, Linux
.venv\Scripts\python transcribe.py      # Windows
```

This is the only script you run. The examples below use the macOS and Linux
form; on Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`.
It asks up to four questions:

```
Language: [1] Hebrew  [2] English  (Enter = Hebrew): 2
Path to audio file: call.m4a
Label speakers? [y/N]: y
Number of speakers (press Enter to auto-detect): 3
```

Drag the audio file from Finder (or File Explorer on Windows) into the
terminal window and press Enter, or paste a path. Surrounding quotes,
drag-and-drop backslash escaping on macOS and Linux, and `~` are all handled.
On Windows a backslash is kept as the path separator it is. A bad path
re-prompts up to three times.

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

`TRANSCRIBE_ENGINE=mlx` or `TRANSCRIBE_ENGINE=faster-whisper` overrides the
engine choice, as long as that engine is installed. It exists mainly to try
faster-whisper on an Apple Silicon Mac after
`.venv/bin/python -m pip install faster-whisper==1.2.1`. An unknown value, or an
engine that is not installed, is an error for the same reason.

## Scripting

Give `transcribe.py` any argument and it runs without prompts, for use from
other tools and scripts. With no arguments it behaves exactly as above.

```
.venv/bin/python transcribe.py --file PATH --lang he|en
                               [--speakers auto|N]
                               [--out PATH]
                               [--json]
.venv/bin/python transcribe.py --capabilities
```

| Flag | Meaning |
|------|---------|
| `--file PATH` | The audio file. Required. |
| `--lang he\|en` | Required unless `TRANSCRIBE_LANG` is set. The flag wins if both are given. |
| `--speakers auto\|N` | Label speakers. `auto` lets the model count them; `N` fixes the count. Without it there is no diarization. |
| `--out PATH` | Write exactly here instead of next to the audio. If the file already exists the run stops with exit code 2; it is never overwritten. |
| `--json` | Print the result as one JSON line instead of the bare path. |
| `--capabilities` | Print what this install can do, then exit. |

stdout carries a single line: the transcript path, or with `--json`:

```
{"output": "/path/call.txt", "language": "en", "speakers": null, "elapsed_s": 87.1}
```

`speakers` is the number of speakers found, or `null` without `--speakers`.
Everything else, including Whisper's running output and model download
progress, goes to stderr. A flag run never prompts, so the speaker-naming step
is skipped. To add names afterwards, run `.venv/bin/python name_speakers.py`
and give it the transcript when it asks, or pass the transcript as its
argument (see [Naming the speakers](#naming-the-speakers)). The Claude Code
plugin does this for you: it opens a window for the naming step.

`--capabilities` prints
`{"diarization": true, "engine": "mlx", "languages": ["he", "en"], "version": "1.1.0"}`.
`diarization` is false when the extras from `./setup.sh --with-diarization` are
not installed. `engine` is `"mlx"`, `"faster-whisper"`, or `null` when neither
is installed. Version 1.1.0 added the `engine` field.

| Exit code | Meaning |
|-----------|---------|
| 0 | Transcript written |
| 1 | Error during transcription, including an input file ffmpeg cannot decode |
| 2 | Bad arguments, unreadable input file, unknown language, or `--out` already exists |
| 3 | `--speakers` given but the diarization extras are not installed |
| 4 | ffmpeg not found |

Tests cover both modes and both engines, and need no model download:

```
.venv/bin/python -m unittest discover -s tests -t .
```

## Claude Code plugin

This repo is also a [Claude Code](https://code.claude.com) plugin, so you can
transcribe from inside a Claude Code session by typing `/transcribe` and a
file. Install it from GitHub with two commands in Claude Code:

```
/plugin marketplace add razamit/audio_transcriber
/plugin install audio-transcriber@audio-transcriber
```

The plugin keeps its own environment, separate from the `.venv` of any clone
you already have, so the first `/transcribe` offers to install it. Model
weights are shared with the clone through `~/.cache/huggingface` and are not
downloaded again.

Do not add a clone that has a `.venv` as a local marketplace
(`/plugin marketplace add /path/to/audio_transcriber`). Claude Code copies the
whole folder into its plugin cache on install and on every update, `.venv`
included, which was 1.5 GB per copy when tried. To run the plugin straight
from a clone, with that clone's `.venv` and no copy, start Claude Code with:

```
claude --plugin-dir /path/to/audio_transcriber
```

Usage:

```
/transcribe ~/Downloads/call.m4a
/transcribe ~/Downloads/call.m4a he
/transcribe ~/Downloads/call.m4a en speakers
/transcribe ~/Downloads/call.m4a he speakers 3
/transcribe ~/Downloads/call.m4a en nospeakers
```

You can also drag a file into the terminal after `/transcribe `. Whatever the
command leaves out, Claude asks: Hebrew or English (never guessed, for the
reason in [Notes on the transcription settings](#notes-on-the-transcription-settings)),
and whether to label speakers. `TRANSCRIBE_LANG` set in your shell answers the
language question. When the run ends Claude reports the transcript path, the
time taken, the language and the number of speakers.

After a run with speaker labels, the plugin opens a new terminal window on
your screen, already running the naming prompt on the new transcript: it shows
each speaker's longest lines and asks for a name, as the script does at the
end of an interactive run. Claude only learns whether the window opened. Type
the names there, then press Enter to close the window. How the window is
opened:

| System | How |
|--------|-----|
| macOS | Writes a small `.command` script into the plugin data folder and opens it in Terminal with `open -a Terminal`. The script deletes itself when it starts. This needs no Automation permission, unlike asking Terminal through AppleScript, which makes macOS ask once whether the app running Claude Code may control Terminal. |
| Windows | Starts the naming script in a new console window (`CREATE_NEW_CONSOLE`). Hebrew shows the same way as in any Windows console (see [Hebrew in the terminal](#hebrew-in-the-terminal)). |
| Linux | The first terminal it finds of `x-terminal-emulator`, `gnome-terminal`, `konsole`, `xfce4-terminal` and `xterm`, and only when `DISPLAY` or `WAYLAND_DISPLAY` is set. |

Over SSH (`SSH_CONNECTION` set), with no display on Linux, with no terminal
found, or when the launch fails, no window opens. Claude then gives you the
naming command to paste into your own terminal instead.

> **What has been run for real.** The macOS window: Terminal opened with the
> naming prompt showing, for a throwaway transcript whose path had spaces, an
> `@`, a single quote and Hebrew letters. The Windows and Linux windows have not
> been opened on a real machine yet. Their commands are built from the Python,
> Microsoft, Debian, GNOME, KDE, Xfce and xterm documentation and covered by
> unit tests with fakes.

The first `/transcribe` on a machine finds nothing installed. Claude says what
would be downloaded (the base packages, 1.3 GB on disk on an Apple Silicon Mac
because mlx-whisper brings PyTorch; speaker labels add about 1.2 GB;
each language's Whisper model, 1.6 to 3 GB, comes on its first transcription)
and installs only after you say yes. With an NVIDIA GPU on Windows or Linux it
installs the GPU support too.

Where things live:

| What | Where |
|------|-------|
| The environment, plugin installed from GitHub | `~/.claude/plugins/data/<plugin id>/venv`, which survives plugin updates |
| The environment, plugin loaded with `claude --plugin-dir` from a clone | The clone's own `.venv`, when it exists |
| Install log | `<data folder>/logs/install.log` |
| Progress of the current or last run | `<data folder>/logs/last-run.log`, replaced by each run |
| The script that opens the naming window on macOS | `<data folder>/naming/`, deleted as soon as Terminal runs it |
| Transcripts | Next to the audio, as with the script |

Claude gives you the log path when a run starts. Follow it with
`tail -f '<log path>'` on macOS and Linux, or `Get-Content -Wait '<log path>'`
in PowerShell. When a plugin update changes the requirements files, the next
`/transcribe` offers to update the environment.

**Privacy.** The audio is transcribed on your computer and never uploaded. By
default Claude sees only paths, timings and counts, not the transcript:
Whisper prints every transcribed line as it goes, so `plugin_runner.py` writes
that output to the log file and prints only a JSON summary, and the skill tells
Claude not to open the transcript or the logs. If you ask Claude to summarise
or quote the transcript, it reads the file then, and that text is sent to
Anthropic like anything else you show Claude. Naming the speakers has to show
transcript lines, so it happens in the window the plugin opens on your screen,
never in Claude's output: `plugin_runner.py name` reports only whether the
window opened, how, and the fallback command.

`plugin_runner.py` can be run by hand, which helps when something goes wrong:

```
python3 plugin_runner.py status --data-dir /some/folder
python3 plugin_runner.py run --data-dir /some/folder -- --file call.m4a --lang en
python3 plugin_runner.py name --data-dir /some/folder -- --file call_speakers.txt
```

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
         pyannote  Community-1         mlx-whisper or faster-whisper
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
   pause and at every speaker change. Invisible text-direction characters the
   Hebrew model sometimes emits (such as U+202B) are removed, so the
   transcript searches cleanly.
6. **Name.** The file is written, and *then* you are offered the chance to
   replace `SPEAKER_00` with real names.

Diarization runs *before* transcription so that a missing model or unreadable
audio fails in seconds rather than after minutes of transcribing.

### Speed

Measured on a 41-minute recording, Apple Silicon (mlx-whisper), English:

| Mode | Time |
|------|------|
| Transcription only | 29 s |
| Transcription + diarization | 87 s |

Both run far above realtime. Diarization roughly triples the total, but the
total is still under 90 seconds for a 41-minute call — so the old advice to
skip it for speed no longer applies. Short clips are dominated by model loading
rather than audio length.

Measured on a 28-minute recording, Windows 11, NVIDIA RTX 4090
(faster-whisper, `cuda (float16)`, `ivrit-ai/whisper-large-v3-ct2`), Hebrew,
transcription only:

| Run | Time |
|-----|------|
| First run, including the one-time 3.1 GB model download | 132 s |
| Model already cached | 98 s |

That is about 17 times faster than realtime, with the full large-v3 model
rather than turbo.

The same file and machine, other modes (models already cached):

| Mode | Device | Time | Speed |
|------|--------|------|-------|
| Transcription + speaker labels (5 speakers found) | `cuda (float16)` for both | 119 s | 14× realtime |
| Transcription only, CPU | `cpu (int8, 16 threads)`, i9-14900K, turbo model | 271 s | 6.2× realtime |
| Transcription only, CPU, simulating an 8-core laptop | `cpu (int8, 4 threads)`, same chip | 379 s | 4.4× realtime |

Speaker labels add about 20 seconds on the GPU. The first speaker-labelled
run took 123 s, including the one-time pyannote model download. The CPU was
forced with `CUDA_VISIBLE_DEVICES=-1`; the 4-thread row is the same chip
limited to the thread count the tool picks on an 8-core machine, so a real
laptop, with slower cores, will be slower than that row.

For a 1-hour podcast that projects to about 4 minutes on this GPU, 10 minutes
on this CPU, and 13–14 minutes at 4 threads. On an ordinary laptop CPU expect
roughly 15–30 minutes. That is an estimate, not a measurement.

There are no measurements yet for faster-whisper on Linux.

## Models

Each language has one model per engine and device, and only the ones your
machine uses are downloaded. They come from HuggingFace into
`~/.cache/huggingface` (on Windows, `C:\Users\<you>\.cache\huggingface`) on
first use and are cached forever after.

| Job | Engine and device | Model | Size | License |
|-----|-------------------|-------|------|---------|
| Hebrew speech → text | mlx-whisper | [`mlx-community/ivrit-ai-whisper-large-v3-mlx`](https://huggingface.co/mlx-community/ivrit-ai-whisper-large-v3-mlx) | ~3 GB | Apache-2.0 upstream |
| Hebrew speech → text | faster-whisper, NVIDIA GPU | [`ivrit-ai/whisper-large-v3-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-ct2) | ~3.1 GB | Apache-2.0 |
| Hebrew speech → text | faster-whisper, CPU | [`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2) | ~1.6 GB | Apache-2.0 |
| English speech → text | mlx-whisper | [`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | ~1.6 GB | MIT upstream |
| English speech → text | faster-whisper, GPU or CPU | [`mobiuslabsgmbh/faster-whisper-large-v3-turbo`](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) | ~1.6 GB | MIT |
| Who spoke when | either | [`pyannote-community/speaker-diarization-community-1`](https://huggingface.co/pyannote-community/speaker-diarization-community-1) | ~33 MB | CC-BY-4.0 |

**Hebrew on the CPU uses the turbo finetune, not full large-v3.** Full large-v3
on a CPU is several times slower, too slow to be the default for long
recordings. The turbo finetune is the same ivrit-ai training on the
large-v3-turbo base, so it stays a Hebrew model. To trade speed for accuracy,
point the `ct2-cpu` entry of the Hebrew profile in `profiles.py` at
`ivrit-ai/whisper-large-v3-ct2`.

The English faster-whisper repo is the one faster-whisper itself uses for the
name `large-v3-turbo`. Its owner has since been renamed to `dropbox-dash`;
HuggingFace redirects the old name, and the download is cached under it.

### Where the Whisper models come from

On Apple Silicon both are **MLX format conversions** published by
[mlx-community](https://huggingface.co/mlx-community), the community org that
republishes models in Apple's MLX format. MLX is what makes these run on the
Mac's GPU. Everywhere else they are **CTranslate2 conversions**, the format
faster-whisper reads. Either way the conversions are re-encodings of existing
weights, not retrained models.

The lineage:

```
openai/whisper-large-v3  (MIT)
   |
   +-- ivrit-ai/whisper-large-v3  (Apache-2.0, finetuned on Hebrew)
   |        |
   |        +-- mlx-community/ivrit-ai-whisper-large-v3-mlx   <- Hebrew, mlx-whisper
   |        +-- ivrit-ai/whisper-large-v3-ct2                 <- Hebrew, NVIDIA GPU
   |
   +-- openai/whisper-large-v3-turbo  (MIT, distilled decoder)
            |
            +-- ivrit-ai/whisper-large-v3-turbo  (Apache-2.0, finetuned on Hebrew)
            |        |
            |        +-- ivrit-ai/whisper-large-v3-turbo-ct2  <- Hebrew, CPU
            |
            +-- mlx-community/whisper-large-v3-turbo          <- English, mlx-whisper
            +-- mobiuslabsgmbh/faster-whisper-large-v3-turbo  <- English, faster-whisper
```

[ivrit-ai](https://huggingface.co/ivrit-ai) is an Israeli open-source project
building Hebrew speech models; their large-v3 finetune is what makes Hebrew
transcription here markedly better than stock Whisper. It declares
`openai/whisper-large-v3` as its base model and is Apache-2.0.

The English model is OpenAI's `large-v3-turbo` — same encoder as large-v3 with a
distilled 4-layer decoder, which is why it is roughly 4–5× faster at half the
size while staying essentially as accurate on clear conversational English.

The CTranslate2 conversions are published by ivrit-ai themselves (Apache-2.0)
and, for English, under MIT.

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
.venv/bin/python name_speakers.py '/path/to/call_speakers.txt'
```

It takes an existing `<name>_speakers.txt` and rewrites it the same way. With
no argument it asks for the file. Given the file as an argument it asks
nothing else first, and when it is done it waits for Enter before exiting, so
a window opened just for it stays up long enough to read the result. This is
the one place the tool overwrites a file rather than bumping to `_2` — renaming
labels is reversible by re-running diarization, and a stack of near-identical
transcripts is worse than editing the one you have.

## Hebrew in the terminal

Many terminals cannot display right-to-left text: they draw Hebrew letters
left to right in the order they are stored, so a sentence appears reversed.
Every Windows console does this, as do VS Code's terminal, Warp, Ghostty,
Alacritty, kitty and WezTerm. macOS Terminal, iTerm2 3.7 and newer, GNOME
Terminal and Konsole display Hebrew correctly.

On the terminals that cannot, the tool reorders Hebrew before printing it: the
running transcription lines and the sample lines shown when naming speakers.
It recognises the terminal from the environment variables it sets. The
transcript file is never reordered; it is always written in normal order, which
is what editors expect.

To override the detection, set `TRANSCRIBE_BIDI`:

```
TRANSCRIBE_BIDI=on .venv/bin/python transcribe.py     # always reorder
TRANSCRIBE_BIDI=off .venv/bin/python transcribe.py    # never reorder
```

`auto`, or leaving it unset, means detect. On Windows, set it first with
`set TRANSCRIBE_BIDI=on` (Command Prompt) or `$env:TRANSCRIBE_BIDI = "on"`
(PowerShell).

One thing the tool cannot fix: a speaker name typed in Hebrew still looks
reversed *while you type it* on such a terminal, because the terminal draws
your keystrokes itself. The name is saved correctly.

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

All transcription and diarization runs on your computer. Audio is never
uploaded, and there is no inference API involved.

Network access is needed **once**, to download model weights from HuggingFace
into `~/.cache/huggingface`, plus the packages the installer fetches. After
that everything runs fully offline. You can verify this yourself by cutting
the network, or by forcing it:

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
| `cli.py` | The no-prompt mode: flags, exit codes, the stdout result line |
| `version.py` | The version `--capabilities` reports |
| `profiles.py` | Which Whisper model goes with which language, per engine and device |
| `paths.py` | Drag-and-drop path cleanup, output naming, ffmpeg check |
| `transcription.py` | Running Whisper through the engine chosen for this machine |
| `engines/` | The two Whisper engines (`mlx_engine.py`, `faster_whisper_engine.py`), choosing one (`selection.py`), the shared decoding settings (`decoding.py`), GPU detection (`device_plan.py`, `cuda_libraries.py`) |
| `audio_decoding.py` | Decoding audio for pyannote and faster-whisper through ffmpeg, tolerating bad frames |
| `diarization.py` | Running pyannote; merging speakers into segments |
| `transcript_output.py` | Rendering and writing both transcript formats |
| `speaker_names.py` | Replacing `SPEAKER_00` with real names |
| `name_speakers.py` | Entry point for naming an existing transcript |
| `terminal_bidi.py` | Reordering Hebrew for terminals that cannot display right-to-left text |
| `console_streams.py` | UTF-8 output on Windows when stdout or stderr is redirected |
| `platform_hints.py` | Install advice that differs by operating system |
| `install.py` | The installer for every platform; `setup.sh` wraps it |
| `plugin_runner.py` | The one script the Claude Code skill calls: `status`, `install`, `run`, `name`, each printing one JSON object |
| `claude_plugin/` | The runner's parts: which venv to use (`venv_location.py`), the status report (`status_report.py`), the install step (`install_step.py`), a transcription run that keeps transcript text off stdout (`transcription_run.py`), requirements change detection (`requirements_snapshot.py`), logs (`run_logs.py`), the injectable machine probes (`host_system.py`), opening the speaker-naming window (`naming_window.py`) with one command builder per OS (`window_commands.py`) |
| `.claude-plugin/` | Claude Code plugin and marketplace manifests |
| `skills/transcribe/SKILL.md` | The `/transcribe` command: instructions Claude follows |
| `tests/` | Unit tests for both modes, both engines and the plugin runner; no model download |

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
- **Both engines use these settings.** They are defined once, in
  `engines/decoding.py`, and passed to mlx-whisper and faster-whisper alike.
  faster-whisper also gets `beam_size=1`, set in its engine because
  mlx-whisper's call does not accept it: faster-whisper defaults to a beam
  search of 5, and mlx-whisper decodes greedily, which a beam of 1 is. One
  difference remains: when a segment falls back to sampling at a higher
  temperature, faster-whisper keeps the best of 5 samples where mlx-whisper
  takes one. faster-whisper's optional voice-activity filter stays off, as
  mlx-whisper has none. faster-whisper is also handed audio decoded by the ffmpeg CLI the
  way mlx-whisper decodes it, rather than through its own PyAV loader, so both
  engines hear identical samples.
- **`hallucination_silence_threshold=2.0` is currently inert.** mlx-whisper
  0.4.3 only consults it when `word_timestamps=True`, and unlike openai-whisper
  it does not raise an error for the combination. faster-whisper 1.2.1 behaves
  the same way. Setting `word_timestamps=True` would activate it, at the cost
  of per-word cross-attention alignment on every segment, a meaningful slowdown
  on long files.
- For difficult English audio (heavy accents, background noise, persistent
  crosstalk), swap the English profile's models in `profiles.py` for full
  large-v3: `mlx-community/whisper-large-v3-mlx` (3 GB) for mlx-whisper, and
  `Systran/faster-whisper-large-v3` (3 GB) for faster-whisper. Slower, but
  more robust.

## Scope

Transcription, optional speaker attribution, and naming those speakers. No
summarization and no API calls. It runs interactively, without prompts
through the flags in [Scripting](#scripting), or from Claude Code through the
[plugin](#claude-code-plugin). A few environment variables
adjust it: `TRANSCRIBE_LANG`, `TRANSCRIBE_ENGINE` and `TRANSCRIBE_BIDI`, plus
HuggingFace's own `HF_TOKEN` and cache variables.

One language per run. A bilingual recording needs two passes, and neither will
be right about the other half.

There is no CI yet. The unit tests need neither Apple Silicon nor a model
download, so a hosted runner could run them; an end-to-end transcription still
needs multi-GB model downloads.

## License

MIT — see [LICENSE](LICENSE). The models carry their own licenses; see
[Models](#models).
