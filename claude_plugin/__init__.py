"""Support code for plugin_runner.py, the one script the Claude Code skill calls.

Standard library only, and no syntax newer than Python 3.8, because it runs
with the system Python before any virtual environment exists and must be able
to report an old Python as JSON rather than crash on import.
"""
