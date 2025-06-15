import logging
from pathlib import Path
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError

from project_name.utils.files_folders import LOG_FOLDER, LOG_CONF_FILE


logger = logging.getLogger(__name__)


def set_log_file_name_config(origin: str | None) -> None:
    """Set a log file to keep information from inverted_qualia runtime.

    Args:
        origin (str or None): The name to use to set the filename for the log file.
            If None, will only use the standard output, CLI or the one running the process.
    """
    if origin:
        log_file_name = Path(
            datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S.%Z") +
            f"inverted_qualia_{origin}.log"
        )
        log_file_name = Path.joinpath(LOG_FOLDER, log_file_name)
        logging.config.fileConfig(
            LOG_CONF_FILE,
            defaults={'logfilename': log_file_name},
            disable_existing_loggers=False,
        )
    else:
        logging.basicConfig(
            format="%(name)s - %(levelname)s : %(message)s"
        )
    # Track project and libs versions in the log
    libs = ['project_name']
    msg_l = []
    for lib in libs:
        try:
            msg_l.append(f"{lib}==`{version(lib)}`")
        except PackageNotFoundError:
            msg = f"The library: `{lib}` is missing from the local installation."
            logger.info(msg)
    msg_version = "(" + ", ".join(msg_l) + ")"
    if origin:
        logger.info("inverted_qualia library versions: "+msg_version)
