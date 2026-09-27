"""
Basic environment, import, and configuration sanity tests for SmartAgri-AI.
Verifies that all core packages and configuration modules are operational.
"""

import sys
import importlib
import pytest


def test_python_version():
    """Verify that Python version is 3.10 or higher (recommended 3.12)."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"


@pytest.mark.parametrize("package_name", [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "joblib",
    "streamlit",
    "pytest"
])
def test_third_party_imports(package_name):
    """Verify that all pinned third-party dependencies can be imported."""
    mod = importlib.import_module(package_name)
    assert mod is not None, f"Failed to import package: {package_name}"


def test_internal_config_import():
    """Verify that internal configuration module loads and exports expected paths."""
    import src.config as cfg
    assert cfg.PROJECT_ROOT.exists(), "PROJECT_ROOT does not exist"
    assert cfg.DATA_DIR is not None
    assert len(cfg.NUMERICAL_FEATURES) == 7
    assert cfg.TARGET_COLUMN == "label"
    assert "N" in cfg.FEATURE_BOUNDS
    assert "rainfall" in cfg.FEATURE_BOUNDS


def test_internal_utils_import():
    """Verify that internal utils module provides working logger and helpers."""
    import src.utils as utils
    logger = utils.get_logger("SanityCheck")
    assert logger is not None
