#!/usr/bin/env python3
"""Non-interactive mode: the same transcription, driven by flags instead of prompts.

transcribe.py hands control here only when it was given arguments. With no
arguments the interactive prompts run exactly as before, and nothing in this
module is touched.

The contract a calling script can rely on:

- No prompt, ever. Nothing here calls input().
- stdout carries one line: the transcript path, or a JSON object with --json.
  Every progress message, including Whisper's per-segment output and anything
  a native library prints, goes to stderr.
- Exit codes: 0 ok, 1 unexpected error, 2 bad arguments or input file,
  3 diarization requested but not installed, 4 ffmpeg missing.

The two transcribe functions are passed in rather than imported, so this
module does not import the entry script and tests can substitute fakes.
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass

import diarization
from paths import build_output_path, describe_path_problem
from profiles import PROFILE_ENV_VAR, PROFILES
from transcription import announce_first_run_download
from version import __version__

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_BAD_INPUT = 2
EXIT_NO_DIARIZATION = 3
EXIT_NO_FFMPEG = 4

AUTO_SPEAKERS = "auto"


class CliError(Exception):
    """A problem found before any model runs, with the exit code it maps to."""

    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class TranscriptionRequest:
    """Everything a run needs, validated, before any model loads."""

    audio_path: str
    profile: object
    with_speakers: bool
    speaker_count: object  # int, or None to auto-detect
    output_path: str


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def parse_speakers(value):
    """`auto` or a positive integer. Anything else is a usage error (exit 2)."""
    text = value.strip().lower()
    if text == AUTO_SPEAKERS:
        return AUTO_SPEAKERS
    if text.isdigit() and int(text) > 0:
        return int(text)
    raise argparse.ArgumentTypeError(f"expected 'auto' or a positive number, got {value!r}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="transcribe.py",
        allow_abbrev=False,  # --fi must not silently mean --file
        description="Transcribe one audio file without prompts. "
        "Run with no arguments for the interactive mode.",
    )
    parser.add_argument("--file", help="audio file to transcribe")
    parser.add_argument(
        "--lang", type=str.lower, choices=sorted(PROFILES),
        help=f"transcription language; required unless {PROFILE_ENV_VAR} is set",
    )
    parser.add_argument(
        "--speakers", type=parse_speakers, metavar="auto|N",
        help="label speakers; 'auto' detects how many. Absent means no diarization",
    )
    parser.add_argument("--out", help="exact output path; must not exist yet")
    parser.add_argument("--json", action="store_true", help="print the result as one JSON line")
    parser.add_argument(
        "--capabilities", action="store_true",
        help="print what this install can do as JSON, then exit",
    )
    return parser


def diarization_installed():
    """True if the diarization extras can be imported.

    diarization_is_available() asks find_spec for "pyannote.audio", which
    raises rather than returning None when the parent "pyannote" package is
    absent, as on a base install. Caught here so flag mode reports it cleanly;
    the interactive path is left as it was.
    """
    try:
        return diarization.diarization_is_available()
    except ImportError:
        return False


def capabilities():
    return {
        "diarization": diarization_installed(),
        "languages": list(PROFILES),
        "version": __version__,
    }


# --------------------------------------------------------------------------
# Validation: every check that can fail fast, in the order a user would fix them
# --------------------------------------------------------------------------

def resolve_profile(lang):
    """--lang wins; otherwise TRANSCRIBE_LANG; otherwise a usage error."""
    key = lang or (os.environ.get(PROFILE_ENV_VAR) or "").strip().lower()
    if not key:
        raise CliError(f"--lang is required unless {PROFILE_ENV_VAR} is set.", EXIT_BAD_INPUT)
    if key not in PROFILES:
        raise CliError(
            f"{PROFILE_ENV_VAR}={key!r} is not a known language. "
            f"Choose one of: {', '.join(PROFILES)}",
            EXIT_BAD_INPUT,
        )
    return PROFILES[key]


def resolve_output_path(out, audio_path, with_speakers):
    """--out exactly as given (never overwritten), else the usual next-to-audio name."""
    if not out:
        return build_output_path(audio_path, "_speakers" if with_speakers else "")
    path = os.path.abspath(os.path.expanduser(out))
    if os.path.isdir(path):
        raise CliError(f"--out must be a file path, not a folder: {path}", EXIT_BAD_INPUT)
    if os.path.exists(path):
        raise CliError(f"Output file already exists, not overwriting: {path}", EXIT_BAD_INPUT)
    if not os.path.isdir(os.path.dirname(path)):
        raise CliError(f"Output folder does not exist: {os.path.dirname(path)}", EXIT_BAD_INPUT)
    return path


def validate(args):
    """Turn parsed args into a request, or raise CliError with its exit code."""
    if shutil.which("ffmpeg") is None:
        raise CliError(
            "ffmpeg is required to decode audio but was not found on PATH.\n"
            "Install it with:  brew install ffmpeg",
            EXIT_NO_FFMPEG,
        )

    audio_path = os.path.abspath(os.path.expanduser(args.file))
    problem = describe_path_problem(audio_path)
    if problem:
        raise CliError(problem, EXIT_BAD_INPUT)

    profile = resolve_profile(args.lang)

    with_speakers = args.speakers is not None
    if with_speakers and not diarization_installed():
        raise CliError(diarization.INSTALL_HELP, EXIT_NO_DIARIZATION)
    speaker_count = None if args.speakers in (None, AUTO_SPEAKERS) else args.speakers

    output_path = resolve_output_path(args.out, audio_path, with_speakers)
    return TranscriptionRequest(audio_path, profile, with_speakers, speaker_count, output_path)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

@contextlib.contextmanager
def stdout_to_stderr():
    """Send everything written to stdout to stderr, Python-level and fd-level.

    The fd-level swap catches output from native code (MLX, torch) that
    bypasses sys.stdout. It is skipped when stdout has no real descriptor,
    as under a test harness that replaced it.
    """
    sys.stdout.flush()
    try:
        saved_fd = os.dup(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        saved_fd = None

    if saved_fd is not None:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        if saved_fd is not None:
            os.dup2(saved_fd, sys.stdout.fileno())
            os.close(saved_fd)


def reserve_output(path):
    """Create --out empty and exclusively, so a parallel run cannot claim it too.

    The existence check in validate() runs minutes before the transcript is
    written; without this, two runs given the same --out would both pass it
    and the second would overwrite the first.
    """
    try:
        with open(path, "x", encoding="utf-8"):
            pass
    except FileExistsError:
        raise CliError(f"Output file already exists, not overwriting: {path}", EXIT_BAD_INPUT)


def release_output(path):
    """Remove a reserved --out that a failed run left empty."""
    if os.path.isfile(path) and os.path.getsize(path) == 0:
        os.remove(path)


def execute(request, transcribe_plain, transcribe_speakers):
    """Run the transcription. Returns the number of speakers, or None."""
    announce_first_run_download(request.profile)
    if request.with_speakers:
        return transcribe_speakers(
            request.audio_path, request.profile, request.output_path, request.speaker_count
        )
    transcribe_plain(request.audio_path, request.profile, request.output_path)
    return None


def run_request(request, transcribe_plain, transcribe_speakers):
    """Execute with stdout diverted. Returns (exit_code, speaker_count)."""
    with stdout_to_stderr():
        try:
            return EXIT_OK, execute(request, transcribe_plain, transcribe_speakers)
        except SystemExit as exc:
            # The shared modules report fatal problems with sys.exit("message").
            if exc.code not in (None, 0):
                print(exc.code if isinstance(exc.code, str) else f"Exited with {exc.code}",
                      file=sys.stderr)
            return EXIT_UNEXPECTED, None
        except Exception:
            print("\nFailed:\n", file=sys.stderr)
            traceback.print_exc()
            return EXIT_UNEXPECTED, None


def result_line(request, speaker_count, elapsed, as_json):
    if not as_json:
        return request.output_path
    return json.dumps({
        "output": request.output_path,
        "language": request.profile.key,
        "speakers": speaker_count if request.with_speakers else None,
        "elapsed_s": round(elapsed, 1),
    }, ensure_ascii=False)


def run(argv, transcribe_plain, transcribe_speakers):
    """Entry point for non-interactive mode. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)  # exits 2 on bad arguments

    if args.capabilities:
        print(json.dumps(capabilities()))
        return EXIT_OK
    if not args.file:
        parser.error("--file is required (or use --capabilities)")

    try:
        request = validate(args)
        if args.out:
            reserve_output(request.output_path)
    except CliError as exc:
        print(str(exc).rstrip(), file=sys.stderr)
        return exc.exit_code

    started = time.monotonic()
    code, speaker_count = run_request(request, transcribe_plain, transcribe_speakers)
    if code != EXIT_OK:
        if args.out:
            release_output(request.output_path)
        return code

    elapsed = time.monotonic() - started
    print(f"Transcript written to: {request.output_path}", file=sys.stderr)
    print(result_line(request, speaker_count, elapsed, args.json))
    return EXIT_OK
