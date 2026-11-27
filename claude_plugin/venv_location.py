"""Which virtual environment the plugin uses, and where a new one goes.

Two places are possible. A clone of the repo loaded in place with `claude
--plugin-dir` may already have `.venv` from `./setup.sh`, and reusing it saves
a second multi-GB install. A plugin installed from a marketplace lives in a
cache folder that Claude Code replaces on every update, so its environment
goes in the plugin data folder instead, which survives updates.

A marketplace added from a local folder is not loaded in place: Claude Code
copies the folder into its cache, `.venv` included, so that route is advised
against in the README rather than supported here.
"""

import os

from install import VENV_FOLDER, python_in_venv

DATA_VENV_FOLDER = "venv"
LOCATION_REPO = "repo"
LOCATION_DATA = "data"


def repo_venv_folder(repo_folder):
    return os.path.join(repo_folder, VENV_FOLDER)


def data_venv_folder(data_dir):
    return os.path.join(data_dir, DATA_VENV_FOLDER)


def resolve_venv(repo_folder, data_dir, platform_name):
    """(python, location) of the environment to use, or (None, None).

    The repo's own .venv wins: it is what an in-place clone already has.
    """
    candidates = (
        (repo_venv_folder(repo_folder), LOCATION_REPO),
        (data_venv_folder(data_dir), LOCATION_DATA),
    )
    for folder, location in candidates:
        python = python_in_venv(folder, platform_name)
        if os.path.isfile(python):
            return python, location
    return None, None


def install_target(repo_folder, data_dir, platform_name):
    """(venv_dir for install.py --venv-dir or None, location) for an install.

    An existing repo .venv is upgraded in place, so install.py runs without
    --venv-dir; anything else installs into the data folder.
    """
    if os.path.isfile(python_in_venv(repo_venv_folder(repo_folder), platform_name)):
        return None, LOCATION_REPO
    return data_venv_folder(data_dir), LOCATION_DATA
