# The first-run install question reads as "the plugin is not installed"

Status: OPEN · Priority: LOW · Type: confusing wording · Cost: none

> Seen in the first real session that installed the plugin from GitHub,
> 2026-11-27 (see `docs/PROGRESS.md`).

## Symptom
On the first `/transcribe`, Claude said the transcriber "is not installed
yet" and asked which install to run. The user dismissed the question, ran
`/plugin marketplace add` and `/plugin install` again (which answered
"already installed"), and typed `/transcribe` a second time.

## Evidence
- `status` returns `"installed": false` on first use, and step 3 of
  `skills/transcribe/SKILL.md` speaks of "install" throughout.
- The second time, the question said the plugin is installed and its Whisper
  engine has not been downloaded yet. The user answered it.
- That the wording caused the dismissal is Claude's reading of the sequence.
  The user did not say why.

Also in that session: the user expected to be asked how many speakers there
are before the run. The skill does ask, but Claude had folded the count
into the install options ("with labels, auto count" / "with labels, 2
speakers"), so it did not look like a question about speakers.

## Why it matters
A first-time user who has just run `/plugin install` hears "not installed"
as a failure of that command and repeats it. It costs a round trip on the
very first use.

## Scope to decide
- Reword step 3 so it says the plugin is installed and what is missing is
  the engine it downloads on first use.
- Say in step 4 that the speaker count is its own question, asked after the
  install choice and not merged into it.
- Optional: rename the status field (`engine_installed`). That changes the
  JSON contract and its tests.
