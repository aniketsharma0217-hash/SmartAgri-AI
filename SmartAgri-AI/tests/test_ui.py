"""
Automated tests for Streamlit application structure, asset integrity, and UI imports.
"""

from pathlib import Path
from src.config import PROJECT_ROOT, FIGURES_DIR


def test_app_file_integrity():
    """Verify that main Streamlit application entrypoint exists and is populated."""
    app_file = PROJECT_ROOT / "app" / "app.py"
    assert app_file.exists(), f"App file missing at {app_file}"
    assert app_file.stat().st_size > 1000, "App file appears unexpectedly small"


def test_custom_stylesheet_integrity():
    """Verify that custom CSS stylesheet exists and defines core classes."""
    css_file = PROJECT_ROOT / "app" / "assets" / "style.css"
    assert css_file.exists(), f"CSS stylesheet missing at {css_file}"
    content = css_file.read_text(encoding="utf-8")
    assert ".hero-banner" in content
    assert ".result-card" in content
    assert ".disclaimer-box" in content


def test_ui_figures_exist():
    """Verify that all figures referenced by the application exist in reports/figures/."""
    expected_figures = [
        "model_comparison_bar.png",
        "confusion_matrix.png",
        "feature_importance.png",
        "crop_distribution.png",
        "feature_distributions.png",
        "correlation_heatmap.png",
        "npk_by_crop.png",
        "climate_clustering.png"
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"UI figure missing: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"UI figure {fig_name} is empty"
