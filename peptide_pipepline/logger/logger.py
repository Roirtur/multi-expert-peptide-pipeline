import logging
import logging.config as logging_config
from logging.handlers import RotatingFileHandler
from logging import Logger
from typing import Optional
import os
_DEFAULT_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_DEFAULT_NAME = "peptide_pipeline"
_LOGGER: Optional[Logger] = None

def _ensure_log_dir(path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def _attach_handlers(logger: Logger, level: int, logfile: Optional[str]) -> None:
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(_DEFAULT_FORMATTER)
        logger.addHandler(sh)

        if logfile:
            _ensure_log_dir(logfile)
            fh = RotatingFileHandler(logfile, maxBytes=10_000_000, backupCount=5)
            fh.setLevel(level)
            fh.setFormatter(_DEFAULT_FORMATTER)
            logger.addHandler(fh)

def set_level(level: int) -> None:
    """
    Change logger and all handler levels on the fly.
    """
    global _LOGGER
    if _LOGGER is None:
        # create default logger if not yet created
        get_logger(level=level)
        return
    _LOGGER.setLevel(level)
    for h in _LOGGER.handlers:
        h.setLevel(level)

def get_logger(name: str = _DEFAULT_NAME, level: int = logging.INFO, logfile: Optional[str] = None) -> Logger:
    """
    Return a single shared logger instance. Can change its level with set_level().
    """
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger(name)
        _LOGGER.setLevel(level)
        _attach_handlers(_LOGGER, level, logfile)
    else:
        # update handlers/level if caller wants a different runtime level or logfile
        set_level(level)
        if logfile:
            # if logfile specified later, attach a file handler if missing
            has_file = any(isinstance(h, RotatingFileHandler) for h in _LOGGER.handlers)
            if not has_file:
                _attach_handlers(_LOGGER, _LOGGER.level, logfile)
    return _LOGGER

def debug_log(logger: Logger, message: str) -> None:
    """
    Log a debug message if the logger is set to DEBUG level.
    """
    set_level(logging.DEBUG)
    logger.debug(message)

def info_log(logger: Logger, message: str) -> None:
    """
    Log an info message if the logger is set to INFO level.
    """
    set_level(logging.INFO)
    logger.info(message)

def warning_log(logger: Logger, message: str) -> None:
    """
    Log a warning message if the logger is set to WARNING level.
    """
    set_level(logging.WARNING)
    logger.warning(message)

def configure_logging_from_yaml(yaml_path: Optional[str] = None, yaml_str: Optional[str] = None) -> None:
    """
    Configure logging using a YAML dictConfig. Provide either yaml_path or yaml_str.
    Requires PyYAML (pip install pyyaml).
    """
    if yaml_str is None and yaml_path is None:
        return

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML logging config. Install with: pip install pyyaml") from exc

    if yaml_str is None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML logging config not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as fh:
            yaml_str = fh.read()

    cfg = yaml.safe_load(yaml_str)
    if not isinstance(cfg, dict):
        raise ValueError("Loaded YAML logging config must be a mapping/dict")
    logging_config.dictConfig(cfg)
