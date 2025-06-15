import project_name
from pathlib import Path


# Folders
SRC_FOLDER = Path(project_name.__file__).parent
MAIN_FOLDER = Path(SRC_FOLDER).parent
LOG_FOLDER = Path(MAIN_FOLDER, "logs")
STATIC_FOLDER = Path(MAIN_FOLDER, "static")
DATA_FOLDER = Path(MAIN_FOLDER, "data")
# Ensure that folders exists
FOLDERS = [LOG_FOLDER, DATA_FOLDER]
for fold in FOLDERS:
    if not fold.is_dir():
        fold.mkdir()


# Files
LOG_CONF_FILE =Path(SRC_FOLDER, 'utils', 'logging.conf')
