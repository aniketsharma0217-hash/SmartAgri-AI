"""
Utility functions and logging configuration for SmartAgri-AI.
Ensures secure, structured logging without leaking sensitive data.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str = "SmartAgri-AI", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a structured logger.

    Args:
        name: Logger name (module or component).
        level: Logging severity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def ensure_directory(path: Path) -> Path:
    """
    Safely creates a directory if it does not already exist.

    Args:
        path: Path object to ensure.

    Returns:
        Verified Path object.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Serializes a dictionary to a formatted JSON file safely.

    Args:
        data: Dictionary data to write.
        filepath: Target destination Path.
    """
    ensure_directory(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Safely loads and parses a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the target file does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration/metric file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_resolve_path(target_path: Path, allowed_base_dir: Path) -> Path:
    """
    Prevents path traversal attacks by verifying that target_path resolves strictly
    within allowed_base_dir.

    Args:
        target_path: Path to verify and resolve.
        allowed_base_dir: Permitted parent directory root.

    Returns:
        Path: Resolved safe Path.

    Raises:
        ValueError: If target_path escapes allowed_base_dir (traversal attempt).
    """
    resolved_target = target_path.resolve()
    resolved_base = allowed_base_dir.resolve()

    if resolved_base not in resolved_target.parents and resolved_target != resolved_base:
        raise ValueError(
            f"Security Violation: Target path '{target_path}' attempts directory traversal "
            f"outside permitted base '{allowed_base_dir}'."
        )

    return resolved_target


def safe_load_joblib(filepath: Path) -> Any:
    """
    Safely deserializes a joblib artifact with file existence and corruption checks.

    Args:
        filepath: Path to the joblib file.

    Returns:
        Any: Deserialized object.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If file is corrupted or unpickling fails.
    """
    import joblib

    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found at '{filepath}'.")

    if filepath.stat().st_size == 0:
        raise RuntimeError(f"Artifact at '{filepath}' is empty (0 bytes) and corrupted.")

    try:
        return joblib.load(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize artifact at '{filepath}': {str(e)}")

