import logging
import sys
import os
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime

# ---------------------------------------------------------------------------
# Custom level: NOTICE (between INFO=20 and WARNING=30)
# ---------------------------------------------------------------------------
NOTICE = 25
logging.addLevelName(NOTICE, "NOTICE")
logging.NOTICE = NOTICE

def notice(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTICE):
        self._log(NOTICE, message, args, **kwargs)

logging.Logger.notice = notice

# ---------------------------------------------------------------------------
# ANSI color codes — edit these to change output colors
# ---------------------------------------------------------------------------
class _Color:
    RESET      = "\033[0m"
    BOLD       = "\033[1m"

    BLACK      = "\033[30m"
    RED        = "\033[31m"
    GREEN      = "\033[32m"
    YELLOW     = "\033[33m"
    BLUE       = "\033[34m"
    MAGENTA    = "\033[35m"
    CYAN       = "\033[36m"
    WHITE      = "\033[37m"

    RED_BOLD   = "\033[1;31m"
    CYAN_BOLD  = "\033[1;36m"
    WHITE_BOLD = "\033[1;37m"

# ---------------------------------------------------------------------------
# Map each level to a color — change values here to restyle
# ---------------------------------------------------------------------------
_LEVEL_COLORS = {
    logging.DEBUG:   _Color.BLUE,
    logging.INFO:    _Color.GREEN,
    NOTICE:          _Color.CYAN,
    logging.WARNING: _Color.YELLOW,
    logging.ERROR:   _Color.RED_BOLD,
    logging.CRITICAL: _Color.WHITE_BOLD,
}

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DEFAULT_FORMATTER = logging.Formatter(_FORMAT)


class ColoredFormatter(logging.Formatter):
    """Applies per-level ANSI colors defined in _LEVEL_COLORS."""

    def format(self, record):
        color = _LEVEL_COLORS.get(record.levelno, "")
        formatter = logging.Formatter(f"{color}{_FORMAT}{_Color.RESET}")
        return formatter.format(record)

def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return logging.INFO


def configure_logging(
    level: int | str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """Configure project logging once.

    All handlers are attached to the **root** logger so every logger
    in the application (named or not) inherits them automatically.
    """
    root = logging.getLogger()
    root.setLevel(_coerce_level(level))

    # Guard against duplicate handlers on repeated calls
    has_console = any(getattr(h, "name", "") == "pipeline-console" for h in root.handlers)
    has_file = any(getattr(h, "name", "") == "pipeline-file" for h in root.handlers)

    if enable_console and not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.set_name("pipeline-console")
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter())
        root.addHandler(console_handler)

    if enable_file and not has_file:
        if not log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join("logs", f"{timestamp}_pipeline.log")

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
        file_handler.set_name("pipeline-file")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_DEFAULT_FORMATTER)
        root.addHandler(file_handler)
