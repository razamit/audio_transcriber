"""Right-to-left text is put in visual order only for terminals that need it."""

import io
import os
import sys
import unittest
from unittest import mock

import terminal_bidi
from terminal_bidi import for_display, reordering_stdout, should_reorder

HEBREW = "שלום עולם"
HEBREW_VISUAL = "םלוע םולש"


class FakeStream(io.StringIO):
    """A text stream that can claim to be a terminal."""

    def __init__(self, is_terminal=True):
        super().__init__()
        self.is_terminal = is_terminal

    def isatty(self):
        return self.is_terminal


def tty():
    return FakeStream(is_terminal=True)


class ReorderPolicyTests(unittest.TestCase):
    def decide(self, environ, platform="darwin", stream=None):
        return should_reorder(environ, platform, stream if stream is not None else tty())

    def test_terminals_that_draw_in_stored_order(self):
        for environ in (
            {"TERM_PROGRAM": "vscode"},
            {"TERM_PROGRAM": "WarpTerminal"},
            {"TERM_PROGRAM": "ghostty"},
            {"KITTY_WINDOW_ID": "1"},
            {"ALACRITTY_WINDOW_ID": "12345"},
            {"WEZTERM_EXECUTABLE": "/Applications/WezTerm.app/wezterm-gui"},
            {"WT_SESSION": "5c0d5f0e-1a2b-4c3d-9e8f-0a1b2c3d4e5f"},  # Windows Terminal, e.g. in WSL
            {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.6.4"},
        ):
            with self.subTest(environ=environ):
                self.assertTrue(self.decide(environ))

    def test_terminals_that_reorder_themselves(self):
        for environ in (
            {"TERM_PROGRAM": "Apple_Terminal", "TERM_PROGRAM_VERSION": "455"},
            {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.7.0"},
            {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.8.1beta2"},
            {"TERM_PROGRAM": "iTerm.app"},
        ):
            with self.subTest(environ=environ):
                self.assertFalse(self.decide(environ))

    def test_windows_terminal_through_wsl_needs_it(self):
        self.assertTrue(self.decide({"WT_SESSION": "abc", "TERM": "xterm-256color"},
                                    platform="linux"))

    def test_unrecognised_terminal_is_left_alone(self):
        self.assertFalse(self.decide({}))
        self.assertFalse(self.decide({"TERM_PROGRAM": "SomethingNew"}))

    def test_windows_console_always_needs_it(self):
        self.assertTrue(self.decide({}, platform="win32"))

    def test_override_on_forces_even_when_piped(self):
        self.assertTrue(self.decide({"TRANSCRIBE_BIDI": "on"}, stream=FakeStream(False)))
        self.assertTrue(self.decide({"TRANSCRIBE_BIDI": " ON ",
                                     "TERM_PROGRAM": "Apple_Terminal"}))

    def test_override_off_disables(self):
        self.assertFalse(self.decide({"TRANSCRIBE_BIDI": "off", "TERM_PROGRAM": "vscode"}))
        self.assertFalse(self.decide({"TRANSCRIBE_BIDI": "off"}, platform="win32"))

    def test_auto_and_unknown_values_detect(self):
        for value in ("auto", "", "maybe"):
            with self.subTest(value=value):
                self.assertTrue(self.decide({"TRANSCRIBE_BIDI": value, "TERM_PROGRAM": "vscode"}))
                self.assertFalse(self.decide({"TRANSCRIBE_BIDI": value,
                                              "TERM_PROGRAM": "Apple_Terminal"}))

    def test_piped_output_is_never_reordered_in_auto_mode(self):
        piped = FakeStream(is_terminal=False)
        self.assertFalse(self.decide({"TERM_PROGRAM": "vscode"}, stream=piped))
        self.assertFalse(self.decide({}, platform="win32", stream=piped))

    def test_stream_without_isatty_counts_as_piped(self):
        self.assertFalse(self.decide({"TERM_PROGRAM": "vscode"}, stream=object()))


class ForcedReorderingTestCase(unittest.TestCase):
    """Runs with TRANSCRIBE_BIDI=on and a known terminal width."""

    columns = "200"

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"TRANSCRIBE_BIDI": "on", "COLUMNS": self.columns})
        patcher.start()
        self.addCleanup(patcher.stop)


class ForDisplayTests(ForcedReorderingTestCase):
    def test_pure_hebrew_is_reversed(self):
        self.assertEqual(for_display(HEBREW, tty()), HEBREW_VISUAL)

    def test_timestamp_prefix_stays_on_the_left(self):
        self.assertEqual(for_display("  [00:12:34] " + HEBREW, tty()),
                         "  [00:12:34] " + HEBREW_VISUAL)
        self.assertEqual(for_display("[00:12.000 --> 00:15.000] " + HEBREW, tty()),
                         "[00:12.000 --> 00:15.000] " + HEBREW_VISUAL)

    def test_embedded_english_word_and_number_keep_their_own_order(self):
        line = "[00:00:05] אני משתמש ב-Python 3 כל יום"
        self.assertEqual(for_display(line, tty()),
                         "[00:00:05] םוי לכ Python 3-ב שמתשמ ינא")

    def test_number_opening_a_hebrew_sentence_moves_to_its_right_end(self):
        self.assertEqual(for_display("[00:00:05] 3 אנשים", tty()), "[00:00:05] םישנא 3")

    def test_english_sentence_with_one_hebrew_name_reads_left_to_right(self):
        self.assertEqual(for_display("I met דני today", tty()), "I met ינד today")

    def test_plain_english_is_unchanged(self):
        line = "  [00:00:01] Nothing to reorder here."
        self.assertIs(for_display(line, tty()), line)

    def test_each_line_is_handled_on_its_own(self):
        self.assertEqual(for_display("SPEAKER_00\n  " + HEBREW, tty()),
                         "SPEAKER_00\n  " + HEBREW_VISUAL)

    def test_stream_defaults_to_current_stdout(self):
        with mock.patch.dict(os.environ, {"TRANSCRIBE_BIDI": "auto", "TERM_PROGRAM": "vscode"}):
            with mock.patch.object(sys, "stdout", tty()):
                self.assertEqual(for_display(HEBREW), HEBREW_VISUAL)
            with mock.patch.object(sys, "stdout", FakeStream(is_terminal=False)):
                self.assertEqual(for_display(HEBREW), HEBREW)

    def test_off_returns_text_unchanged(self):
        with mock.patch.dict(os.environ, {"TRANSCRIBE_BIDI": "off"}):
            self.assertEqual(for_display(HEBREW, tty()), HEBREW)

    def test_brackets_and_quotes_are_mirrored_inside_a_hebrew_sentence(self):
        cases = {
            "הוא אמר (שלום) לי": "יל (םולש) רמא אוה",
            "הוא אמר [שלום] לי": "יל [םולש] רמא אוה",
            'הוא אמר "שלום" לי': 'יל "םולש" רמא אוה',
        }
        for line, expected in cases.items():
            with self.subTest(line=line):
                self.assertEqual(for_display(line, tty()), expected)

    def test_brackets_are_mirrored_after_a_timestamp_that_keeps_its_own(self):
        self.assertEqual(for_display("[00:00:05] הוא אמר (שלום) לי", tty()),
                         "[00:00:05] יל (םולש) רמא אוה")
        self.assertEqual(for_display("[00:00.000 --> 00:02.000] אמר [שלום] לי", tty()),
                         "[00:00.000 --> 00:02.000] יל [םולש] רמא")

    def test_missing_python_bidi_returns_text_unchanged(self):
        with mock.patch.dict(sys.modules, {"bidi": None, "bidi.algorithm": None}):
            self.assertEqual(for_display(HEBREW, tty()), HEBREW)


class WrappingTests(ForcedReorderingTestCase):
    columns = "40"
    long_line = "  [00:00:01] " + " ".join(["מילה"] * 20)

    def test_long_line_is_wrapped_before_reordering(self):
        shown = for_display(self.long_line, tty()).split("\n")
        self.assertGreater(len(shown), 1)
        for row in shown:
            self.assertLessEqual(len(row), 39)  # one column of margin
        self.assertTrue(shown[0].startswith("  [00:00:01] "))
        for row in shown[1:]:
            self.assertTrue(row.startswith(" " * 13))

    def test_wrapped_rows_are_right_aligned_and_keep_every_word(self):
        shown = for_display(self.long_line, tty()).split("\n")
        text_rows = [row[13:] for row in shown]
        widths = {len(row) for row in text_rows}
        self.assertEqual(len(widths), 1)  # right edges line up
        words = " ".join(text_rows).split()
        self.assertEqual(words, ["הלימ"] * 20)


class ReorderingStdoutTests(ForcedReorderingTestCase):
    def test_complete_lines_are_reordered(self):
        terminal = tty()
        with mock.patch.object(sys, "stdout", terminal):
            with reordering_stdout():
                print("[00:00.000 --> 00:02.000] " + HEBREW)
                print("plain line")
        self.assertEqual(terminal.getvalue(),
                         "[00:00.000 --> 00:02.000] " + HEBREW_VISUAL + "\nplain line\n")

    def test_partial_line_waits_for_its_newline_and_is_flushed_on_exit(self):
        terminal = tty()
        with mock.patch.object(sys, "stdout", terminal):
            with reordering_stdout():
                sys.stdout.write("שלום ")
                self.assertEqual(terminal.getvalue(), "")
                sys.stdout.write("עולם\nאחרון")
                self.assertEqual(terminal.getvalue(), HEBREW_VISUAL + "\n")
        self.assertEqual(terminal.getvalue(), HEBREW_VISUAL + "\n" + "ןורחא")

    def test_stdout_is_restored_after_an_exception(self):
        terminal = tty()
        with mock.patch.object(sys, "stdout", terminal):
            with self.assertRaises(RuntimeError):
                with reordering_stdout():
                    sys.stdout.write("חצי")
                    raise RuntimeError("library failed")
            self.assertIs(sys.stdout, terminal)
        self.assertEqual(terminal.getvalue(), "יצח")

    def test_attributes_are_delegated(self):
        terminal = tty()
        with mock.patch.object(sys, "stdout", terminal):
            with reordering_stdout() as proxy:
                self.assertIsNot(proxy, terminal)
                self.assertTrue(sys.stdout.isatty())
                self.assertIs(proxy.getvalue.__self__, terminal)
                self.assertEqual(proxy.encoding, terminal.encoding)
                with self.assertRaises(io.UnsupportedOperation):
                    proxy.fileno()  # StringIO has none; the proxy must not invent one
                proxy.flush()

    def test_nothing_is_wrapped_when_reordering_is_off(self):
        terminal = tty()
        with mock.patch.dict(os.environ, {"TRANSCRIBE_BIDI": "off"}), \
                mock.patch.object(sys, "stdout", terminal):
            with reordering_stdout() as stream:
                self.assertIs(sys.stdout, terminal)
                self.assertIs(stream, terminal)
                print(HEBREW)
        self.assertEqual(terminal.getvalue(), HEBREW + "\n")


class VersionParsingTests(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(terminal_bidi.parse_version("3.7.0beta1"), (3, 7, 0))
        self.assertEqual(terminal_bidi.parse_version("3.6"), (3, 6, 0))
        self.assertIsNone(terminal_bidi.parse_version("nightly"))


if __name__ == "__main__":
    unittest.main()
