"""
Security and Reliability Test Suite for SmartAgri AI.
Verifies:
- Path traversal defenses (safe_resolve_path)
- Deserialization safety and corruption defense (safe_load_joblib)
- Dataset loader fault tolerance (missing, empty, corrupt files)
- Zero-secrets verification across repository
- No sensitive keys or credentials in tracked files
"""

import os
import re
import tempfile
from pathlib import Path
import pytest

from src.utils import safe_resolve_path, safe_load_joblib
from src.data_loader import load_raw_data, DatasetValidationError
from src.config import PROJECT_ROOT


def test_safe_resolve_path_prevents_traversal():
    """Verify safe_resolve_path blocks path traversal escapes."""
    base_dir = PROJECT_ROOT / "models"
    malicious_path = base_dir / ".." / ".." / "etc" / "passwd"

    with pytest.raises(ValueError, match="Security Violation: Target path"):
        safe_resolve_path(malicious_path, base_dir)


def test_safe_resolve_path_allows_legitimate_subpaths():
    """Verify safe_resolve_path permits legitimate subpaths."""
    base_dir = PROJECT_ROOT / "models"
    valid_path = base_dir / "metadata.json"
    resolved = safe_resolve_path(valid_path, base_dir)
    assert resolved == valid_path.resolve()


def test_safe_load_joblib_missing_file():
    """Verify safe_load_joblib raises FileNotFoundError for missing files."""
    non_existent = PROJECT_ROOT / "models" / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError):
        safe_load_joblib(non_existent)


def test_safe_load_joblib_corrupt_empty_file():
    """Verify safe_load_joblib rejects 0-byte or corrupted files."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # File is empty (0 bytes)
        with pytest.raises(RuntimeError, match="empty .* and corrupted"):
            safe_load_joblib(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_safe_load_joblib_garbage_bytes():
    """Verify safe_load_joblib raises RuntimeError on corrupted byte stream."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp.write(b"NOT_A_VALID_JOBLIB_PICKLE_STREAM_12345")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(RuntimeError, match="Failed to deserialize artifact"):
            safe_load_joblib(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_data_loader_rejects_missing_dataset():
    """Verify load_raw_data raises FileNotFoundError when dataset is absent."""
    missing_file = PROJECT_ROOT / "data" / "raw" / "non_existent_dataset.csv"
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data(missing_file)


def test_data_loader_rejects_empty_dataset():
    """Verify load_raw_data raises DatasetValidationError for 0-byte file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(DatasetValidationError, match="empty \\(0 bytes\\)"):
            load_raw_data(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_repository_secrets_audit():
    """Scan source code, configs, and documentation for leaked credentials/API keys."""
    suspicious_patterns = [
        re.compile(r"(?i)(api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"])"),
        re.compile(r"(?i)(secret[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"])"),
        re.compile(r"(?i)(password\s*=\s*['\"][^'\"]{6,}['\"])"),
        re.compile(r"(?i)(bearer\s+[A-Za-z0-9_\-\.]{20,})"),
    ]

    target_extensions = {".py", ".json", ".yaml", ".yml", ".md", ".sh"}
    excluded_dirs = {".venv", ".git", "__pycache__", ".pytest_cache"}

    findings = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for f in files:
            file_path = Path(root) / f
            if file_path.suffix in target_extensions:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    for pattern in suspicious_patterns:
                        match = pattern.search(content)
                        if match:
                            findings.append(f"{file_path}: matched {match.group(0)}")
                except Exception:
                    pass

    assert len(findings) == 0, f"Found potential leaked secrets in repository: {findings}"
