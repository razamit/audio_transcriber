"""Notice when a plugin update changed the requirements the data-folder venv was built from.

The venv in the plugin data folder outlives the code: an update replaces the
code but keeps the venv. After a successful install the runner saves a hash of
each requirements file next to it, and status compares the current files with
those hashes. A missing snapshot also counts as changed, because it means the
last install did not finish.
"""

import hashlib
import json
import os

REQUIREMENTS_FILES = ("requirements.txt", "requirements-diarize.txt")
SNAPSHOT_FILE = "requirements-snapshot.json"


def file_sha256(path):
    """The file's SHA-256, or None when it does not exist."""
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except FileNotFoundError:
        return None


def requirements_fingerprint(repo_folder):
    return {name: file_sha256(os.path.join(repo_folder, name)) for name in REQUIREMENTS_FILES}


def snapshot_path(data_dir):
    return os.path.join(data_dir, SNAPSHOT_FILE)


def save_snapshot(data_dir, repo_folder):
    os.makedirs(data_dir, exist_ok=True)
    with open(snapshot_path(data_dir), "w", encoding="utf-8") as handle:
        json.dump(requirements_fingerprint(repo_folder), handle, indent=2, sort_keys=True)


def load_snapshot(data_dir):
    """The saved hashes, or None when there is no readable snapshot."""
    try:
        with open(snapshot_path(data_dir), encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return None
    return saved if isinstance(saved, dict) else None


def requirements_changed(data_dir, repo_folder):
    return load_snapshot(data_dir) != requirements_fingerprint(repo_folder)
