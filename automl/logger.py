import logging
import os
import sys
from pathlib import Path
from typing import Optional

import colorlog
from colorlog import ColoredFormatter

_LOGGERS = {}


class TruncatingFormatter(ColoredFormatter):
    def __init__(self, *args, max_len=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = max_len

    def _truncate_head(self, value: str) -> str:
        """Hide start, keep end: …tail"""
        if not value:
            return value
        if len(value) <= self.max_len:
            return value
        return "…" + value[-(self.max_len - 1) :]

    def _truncate_tail(self, value: str) -> str:
        """Keep start, hide end: head…"""
        if not value:
            return value
        if len(value) <= self.max_len:
            return value
        return value[: self.max_len - 1] + "…"

    def format(self, record):
        original_filename = record.filename
        original_name = record.name

        try:
            # filename: hide start, keep end
            # record.filename = self._truncate_head(os.path.basename(record.filename))
            record.filename = os.path.basename(record.filename)

            # logger/app name: keep start, hide end
            record.name = self._truncate_tail(record.name)

            return super().format(record)

        finally:
            record.filename = original_filename
            record.name = original_name


def get_simple_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt=(
            "%(asctime)s  "
            "| %(levelname)-7s "
            "| %(name)-12s "
            "| %(filename)s:%(lineno)-4d "
            "| %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_colored_formatter() -> logging.Formatter:
    return colorlog.ColoredFormatter(
        fmt=(
            "%(log_color)s"
            "%(asctime)s  "
            "| %(levelname)-7s "
            "| %(name)-12s "
            "| %(filename)s:%(lineno)-4d "
            "| %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )


def get_logger(
    name: str = "app",
    log_file: Optional[str | Path] = None,
    level: int = logging.DEBUG,
    force_new=False,
):
    if (not force_new) and name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs

    formatter = get_colored_formatter()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_path)
        fh.setFormatter(get_simple_formatter())
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def setup_logging(
    output_path: Optional[str | Path] = None,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Configure application logging for stdout and an optional log file.

    Some modules create their logger at import time via ``get_logger()``. This
    function reconfigures those cached loggers as well, so the configured log
    level and log file apply consistently after CLI startup.
    """
    if isinstance(level, str):
        resolved_level = logging.getLevelName(level.upper())
        if not isinstance(resolved_level, int):
            raise ValueError(f"Invalid log level: {level}")
    else:
        resolved_level = level

    logger_names = set(_LOGGERS) | {""}
    configured_root = get_logger(
        name="",
        log_file=output_path,
        level=resolved_level,
        force_new=True,
    )

    for logger_name in logger_names - {""}:
        get_logger(
            name=logger_name,
            log_file=output_path,
            level=resolved_level,
            force_new=True,
        )

    return configured_root
