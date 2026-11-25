"""The running `[MM:SS.mmm --> MM:SS.mmm] text` lines printed while transcribing.

mlx-whisper prints these itself in verbose mode. faster-whisper prints
nothing, so its engine prints the same lines through here, and progress looks
the same on every platform.
"""


def format_progress_timestamp(seconds):
    """Seconds -> MM:SS.mmm, or HH:MM:SS.mmm past the first hour (mlx-whisper's format)."""
    milliseconds = round(max(0.0, seconds) * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def format_segment_line(start, end, text):
    """One progress line for one segment, exactly as mlx-whisper prints it."""
    return f"[{format_progress_timestamp(start)} --> {format_progress_timestamp(end)}] {text}"
