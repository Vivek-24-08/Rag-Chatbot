# ==============================================================================
# utils/logger.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES:
#   Provides one standardized way to create a logger for any module in the
#   project, so log messages look consistent everywhere (timestamp, module
#   name, log level, message).
#
# WHY THIS MATTERS:
#   In a RAG pipeline, something can go wrong at many different stages
#   (PDF parsing, chunking, embedding API calls, vector search, LLM calls).
#   Good logging is how you figure out WHERE it went wrong without having to
#   re-run everything with print() statements sprinkled around.
# ==============================================================================

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Create (or reuse) a logger for the given module name.

    Args:
        name: Usually pass in __name__ from the calling module, so log lines
              show exactly which file produced them, e.g.
              "2026-09-02 10:00:00 [INFO] ingestion.pdf_loader: Loaded 12 pages"

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger() is called multiple
    # times for the same module (e.g., Streamlit re-runs scripts often).
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
