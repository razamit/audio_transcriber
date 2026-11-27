# There is no way to download a language's model ahead of time through the plugin

Status: OPEN · Priority: LOW · Type: missing capability · Cost: none

> Asked for in the first real plugin session, 2026-11-27 (see
> `docs/PROGRESS.md`).

## Symptom
Asked which language the recording was in, the user answered "it's Hebrew
but download both so it's ready for English". The plugin could not do that.

## Evidence
- `plugin_runner.py` has `status`, `install`, `run` and `name`. `install`
  fetches packages only.
- A Whisper model (1.6 to 3 GB) is downloaded by the first transcription in
  its language, inside `run`.
- Claude told the user the English model will download by itself on the
  first English file.

## Why it matters
The first run in a new language is much slower than later ones, and it
needs a network connection at a moment the user may not have one. Someone
setting the tool up once would rather pay both downloads then.

## Scope to decide
- Add `install --models he,en` (or a `fetch-model` subcommand) that
  downloads the model the engine on this machine would pick for each
  language, and let the first-run question offer it
  OR
- Leave it, and say in the skill's first-run text that each language's
  model arrives with its first transcription.
