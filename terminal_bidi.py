"""Show right-to-left text (Hebrew, Arabic) correctly in terminals that cannot.

Text is stored in logical order: the order it is spoken and typed. Turning
that into the left-to-right sequence of cells a screen shows is the job of
the Unicode bidirectional algorithm. macOS Terminal.app, iTerm2 3.7+, GNOME
Terminal and Konsole run it. VS Code's terminal, Warp, Ghostty, Alacritty,
kitty, WezTerm (by default) and every Windows console do not: they draw the
characters left to right as stored, so a Hebrew sentence appears reversed.

This module reorders text into visual order before it is printed, and only
where that is needed. Files are never touched: the transcript stays in
logical order, which is what editors and other tools expect.

There is no reliable way to ask a terminal whether it reorders, so the
decision comes from environment variables the terminals set. It can be
forced with TRANSCRIBE_BIDI=on or turned off with TRANSCRIBE_BIDI=off.

Reordering a line only works if the terminal does not soft-wrap it
afterwards, because a wrap would cut the visual line in the wrong place.
So long lines are wrapped to the terminal width first, in logical order,
and each piece is reordered on its own.

The python-bidi package does the reordering. It is imported lazily and is
optional: without it, text is printed unchanged and nothing fails. Its
pure-Python bidi.algorithm module is used rather than the faster top-level
get_display: in python-bidi 0.6 the top-level one wraps a Rust crate that
does not mirror paired characters, so "(שלום)" came out as ")םולש(".
"""

import contextlib
import os
import re
import shutil
import sys
import textwrap
import unicodedata

OVERRIDE_VARIABLE = "TRANSCRIBE_BIDI"

# Hebrew, Hebrew presentation forms, Arabic, Arabic Supplement, Arabic
# Extended-A and the Arabic presentation forms.
RTL_LETTER_PATTERN = re.compile(
    "[\u0590-\u05ff\ufb1d-\ufb4f\u0600-\u06ff\u0750-\u077f"
    "\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]"
)

# "  [00:12:34] " or "[00:12.000 --> 00:15.000] ": stays on the left as is.
TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\s*\[[^\]\n]*\]\s*")

# Terminals known to draw characters in stored order. Matched on TERM_PROGRAM.
NON_REORDERING_TERM_PROGRAMS = {"vscode", "WarpTerminal", "ghostty"}

# Set by terminals that do not reorder, whatever TERM_PROGRAM says.
# WT_SESSION is Windows Terminal's; it is passed into WSL shells too, where
# sys.platform is "linux" but the drawing is still done by Windows Terminal.
NON_REORDERING_MARKER_VARIABLES = (
    "WT_SESSION",
    "KITTY_WINDOW_ID",
    "ALACRITTY_WINDOW_ID",
    "WEZTERM_EXECUTABLE",
)

# iTerm2 turned bidi rendering on by default in this release.
ITERM_FIRST_REORDERING_VERSION = (3, 7, 0)

# One column of slack so a line that exactly fills the width cannot wrap.
RIGHT_MARGIN_COLUMNS = 1
MIN_WRAP_WIDTH = 20


# --------------------------------------------------------------------------
# Policy: does this stream need text reordered before printing?
# --------------------------------------------------------------------------

def should_reorder(environ, platform, stream):
    """True when text printed to `stream` must be put in visual order first.

    Takes the environment mapping and sys.platform value as parameters so
    every terminal case can be tested without touching the real ones.
    """
    override = environ.get(OVERRIDE_VARIABLE, "").strip().lower()
    if override == "on":
        return True
    if override == "off":
        return False
    if not stream_is_terminal(stream):
        return False  # piped or redirected output stays in logical order
    if platform == "win32":
        return True
    return terminal_draws_in_stored_order(environ)


def stream_is_terminal(stream):
    """isatty() that treats a missing or closed stream as not a terminal."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def terminal_draws_in_stored_order(environ):
    """Identify the terminal from its environment variables.

    Unrecognised terminals count as reordering ones: printing reversed text
    into a terminal that reverses it again would be just as wrong, and the
    known non-reordering terminals all identify themselves.
    """
    term_program = environ.get("TERM_PROGRAM", "")
    if term_program in NON_REORDERING_TERM_PROGRAMS:
        return True
    if term_program == "iTerm.app":
        return iterm_predates_bidi(environ.get("TERM_PROGRAM_VERSION", ""))
    if term_program == "Apple_Terminal":
        return False
    return any(environ.get(name) for name in NON_REORDERING_MARKER_VARIABLES)


def iterm_predates_bidi(version_text):
    """True for iTerm2 versions before bidi rendering was on by default."""
    version = parse_version(version_text)
    if version is None:
        return False
    return version < ITERM_FIRST_REORDERING_VERSION


def parse_version(version_text):
    """A version such as 3.6.2 or 3.7.0beta1 as three ints; None if unreadable."""
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version_text.strip())
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


# --------------------------------------------------------------------------
# Reordering one line
# --------------------------------------------------------------------------

def contains_rtl(text):
    return RTL_LETTER_PATTERN.search(text) is not None


def to_visual_order(text, base_direction):
    """Visual order via python-bidi, or None when the package is missing."""
    try:
        from bidi.algorithm import get_display  # mirrors brackets; see module docstring
    except ImportError:
        return None
    return get_display(text, base_dir=base_direction)


def split_fixed_prefix(line):
    """(prefix kept on the left untouched, remainder to reorder).

    The prefix runs up to the first right-to-left letter, but stops after a
    leading timestamp so a number or English word opening a Hebrew sentence
    is reordered along with the sentence it belongs to.
    """
    first_rtl = RTL_LETTER_PATTERN.search(line).start()
    timestamp = TIMESTAMP_PREFIX_PATTERN.match(line)
    cut = min(first_rtl, timestamp.end()) if timestamp else first_rtl
    return line[:cut], line[cut:]


def paragraph_direction(text):
    """'R' when right-to-left letters outnumber left-to-right ones, else 'L'.

    A Hebrew sentence with an English word in it reads from its right end;
    an English sentence that mentions one Hebrew name still reads from its
    left end.
    """
    rtl_letters = len(RTL_LETTER_PATTERN.findall(text))
    all_letters = sum(1 for char in text if char.isalpha())
    return "R" if rtl_letters * 2 > all_letters else "L"


def wrap_logical(text, width):
    """Cut text into pieces no wider than `width`, in logical order."""
    if len(text) <= width:
        return [text]
    return textwrap.wrap(text, width=width, break_on_hyphens=False) or [text]


def display_width(text):
    """Columns used, ignoring combining marks such as Hebrew vowel points."""
    return sum(1 for char in text if not unicodedata.combining(char))


def align_right(visual_lines):
    """Pad a wrapped right-to-left block so its right edges line up."""
    block_width = max(display_width(line) for line in visual_lines)
    return [" " * (block_width - display_width(line)) + line for line in visual_lines]


def format_line(line, columns):
    """One logical line as one or more visual lines; None if bidi is missing."""
    prefix, remainder = split_fixed_prefix(line)
    direction = paragraph_direction(remainder)
    available = max(columns - RIGHT_MARGIN_COLUMNS - len(prefix), MIN_WRAP_WIDTH)

    visual_lines = []
    for piece in wrap_logical(remainder, available):
        visual = to_visual_order(piece, direction)
        if visual is None:
            return None
        visual_lines.append(visual)

    if direction == "R" and len(visual_lines) > 1:
        visual_lines = align_right(visual_lines)
    indent = " " * len(prefix)
    return "\n".join(
        (prefix if index == 0 else indent) + visual
        for index, visual in enumerate(visual_lines)
    )


def format_text(text, columns):
    """Reorder every line holding right-to-left letters; leave the rest."""
    output_lines = []
    for line in text.split("\n"):
        formatted = format_line(line, columns) if contains_rtl(line) else line
        if formatted is None:
            return text  # python-bidi is not installed
        output_lines.append(formatted)
    return "\n".join(output_lines)


def terminal_columns():
    return shutil.get_terminal_size().columns


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------

def for_display(text, stream=None):
    """Text ready to print() to `stream` (default: sys.stdout at call time).

    Returned unchanged when the stream does not need reordering or the text
    has no right-to-left letters.
    """
    if stream is None:
        stream = sys.stdout
    if not contains_rtl(text):
        return text
    if not should_reorder(os.environ, sys.platform, stream):
        return text
    return format_text(text, terminal_columns())


class LineReorderingStream:
    """Stand-in for a text stream that reorders each complete line it is given.

    Holds back a partial line until its newline arrives, because reordering
    half a sentence would put its words in the wrong place. Everything other
    than write, writelines and flush is passed to the wrapped stream.
    """

    def __init__(self, wrapped_stream, format_lines):
        self.wrapped_stream = wrapped_stream
        self.format_lines = format_lines
        self.pending_text = ""

    def write(self, text):
        self.pending_text += text
        complete, newline, partial = self.pending_text.rpartition("\n")
        if newline:
            self.wrapped_stream.write(self.format_lines(complete) + "\n")
            self.pending_text = partial
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        """Emit any partial line (a prompt waits on it) and flush the stream."""
        if self.pending_text:
            self.wrapped_stream.write(self.format_lines(self.pending_text))
            self.pending_text = ""
        self.wrapped_stream.flush()

    def __getattr__(self, name):
        return getattr(self.wrapped_stream, name)


@contextlib.contextmanager
def reordering_stdout():
    """Reorder every line printed to sys.stdout inside the block.

    Decides once, on the stream that is sys.stdout on entry. When that
    stream needs no reordering, nothing is wrapped. sys.stdout is restored
    on exit, even after an exception, with any partial line flushed.
    """
    original_stdout = sys.stdout
    if not should_reorder(os.environ, sys.platform, original_stdout):
        yield original_stdout
        return

    def format_lines(text):
        return format_text(text, terminal_columns()) if contains_rtl(text) else text

    proxy = LineReorderingStream(original_stdout, format_lines)
    sys.stdout = proxy
    try:
        yield proxy
    finally:
        sys.stdout = original_stdout
        proxy.flush()
