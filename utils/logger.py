"""Structured logs omit questions, filenames, secrets and exception bodies."""
import json
import logging
import os
import sys

class SafeFormatter(logging.Formatter):
    def format(self, record):
        # Call sites log only controlled messages and error class names.
        return json.dumps({"time": self.formatTime(record), "level": record.levelname,
                           "module": record.name, "event": record.getMessage()})

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SafeFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
    logger.propagate = False
    return logger
