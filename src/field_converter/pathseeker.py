import os
from pathlib import Path

# =============================================================================

# The default root dir is the root dir of the repo assuming the repo is like :
# field_converter <-------- ROOT DIRECTORY ---------
# ├── src                        <-------- SRC DIRECTORY ---------
# │   └── field_converter<-------- PACKAGE DIRECTORY ---------
# │       ├── api
# │       ├── cli
# │       ├── models
# │       ├── data
# │       ├── scripts 
# │       ├── visual
# │       ├── __init__.py
# │       ├── pathseeker.py  <<-------- YOU ARE HERE -------------------
# │       ├── settings.py
# │       └── typing.py
# ├── Readme.md
# ├── setup.cfg
# ├── pyproject.toml
# └── tests

# =============================================================================

# to call this patheeker file : from field_converter import pathseeker as ps

_THIS_FILE = Path(__file__)
_PACKAGE_DIR = _THIS_FILE.parent
SRC = _PACKAGE_DIR.parent
PROJECT_ROOT = SRC.parent

PROJECT_NAME = "field_converter"
# Default root dir for lemp project on your PC

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
