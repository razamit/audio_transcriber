#!/usr/bin/env python3
"""Making stdout and stderr able to carry Hebrew on Windows when redirected.

In a Windows console Python writes Unicode directly, and such a stream already
reports UTF-8 as its encoding. When stdout or stderr is redirected to a file or
a pipe, as when another program runs this tool in flag mode, Python falls back
to the legacy code page (cp1252 and the like), and the first Hebrew character
raises UnicodeEncodeError partway through a run. Switching those streams to
UTF-8 at startup prevents that. Other platforms already default to UTF-8 and
are left alone.

The test is the encoding, not isatty(): redirecting to the NUL device
(`> NUL`) gives a stream that isatty() calls a terminal but that still uses the
legacy code page.
"""

import sys


def is_utf8(stream):
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    return encoding == "utf8"


def use_utf8_when_redirected(platform_name=None, streams=None):
    """On Windows, reconfigure each stream whose encoding is not UTF-8 to UTF-8."""
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "win32":
        return
    for stream in streams if streams is not None else (sys.stdout, sys.stderr):
        if stream is None or is_utf8(stream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
