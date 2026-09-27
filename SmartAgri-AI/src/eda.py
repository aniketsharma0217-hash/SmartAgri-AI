"""
Exploratory Data Analysis (EDA) Module for SmartAgri-AI.
Generates comprehensive statistical summaries and publishes publication-grade
visualizations to reports/figures/ based on the genuine dataset.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATASET_PATH,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FIGURES_DIR
)
from src.data_loader import load_raw_data
from src.utils import get_logger, ensure_directory

logger = get_logger("EDA")


def generate_eda_visualizations(df: pd.DataFrame) -> Dict[str, Path]:
    """
    Renders and exports core EDA figures using actual dataset values.

    Args:
        df: Loaded raw crop dataset DataFrame.

    Returns:
        Dict[str, Path]: Map of plot names to saved image paths.
    """
    ensure_directory(FIGURES_DIR)
    saved_figures = {}

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # 1. Target Crop Distribution Plot
    logger.info("Generating figure: Crop Distribution...")
    fig, ax = plt.subplots(figsize=(14, 6))
    crop_counts = df[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(
        x=crop_counts.index,
        y=crop_counts.values,
        ax=ax,
        palette="crest"
    )
    ax.set_title("Distribution of Samples per Target Crop (N=2,200)", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Agricultural Crop Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_ylim(0, 120)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points"
        )
    crop_dist_path = FIGURES_DIR / "crop_distribution.png"
    plt.savefig(crop_dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["crop_distribution"] = crop_dist_path

    # 2. Feature Distribution Histograms with KDE
    logger.info("Generating figure: Feature Distributions...")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
    axes = axes.flatten()

    units = {
        "N": "kg/ha",
        "P": "kg/ha",
        "K": "kg/ha",
        "temperature": "°C",
        "humidity": "%",
        "ph": "pH",
        "rainfall": "mm"
    }

    for i, col in enumerate(NUMERICAL_FEATURES):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color="#2b5c8f", bins=25, edgecolor="white")
        ax.set_title(f"{col.upper()} Distribution", fontsize=12, weight="bold")
        ax.set_xlabel(f"{col} ({units.get(col, '')})", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)

    # Hide 8th unused subplot
    axes[7].axis("off")
    fig.suptitle("Univariate Distributions of Soil Nutrients and Climatic Parameters", fontsize=15, weight="bold", y=0.98)
    feat_dist_path = FIGURES_DIR / "feature_distributions.png"
    plt.savefig(feat_dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["feature_distributions"] = feat_dist_path

    # 3. Correlation Heatmap
    logger.info("Generating figure: Correlation Matrix Heatmap...")
    fig, ax = plt.subplots(figsize=(9, 7))
    corr_matrix = df[NUMERICAL_FEATURES].corr(method="pearson")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Correlation Heatmap of Agricultural & Environmental Predictors", fontsize=13, weight="bold", pad=12)
    corr_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["correlation_heatmap"] = corr_path

    # 4. Nutrient Demands (NPK) by Crop Boxplots
    logger.info("Generating figure: NPK Distributions by Crop...")
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 14), sharex=True)

    sns.boxplot(x=TARGET_COLUMN, y="N", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Soil Nitrogen (N) Requirement Across Crops (kg/ha)", fontsize=12, weight="bold")
    axes[0].set_ylabel("Nitrogen (kg/ha)")

    sns.boxplot(x=TARGET_COLUMN, y="P", data=df, ax=axes[1], palette="Set2")
    axes[1].set_title("Soil Phosphorus (P) Requirement Across Crops (kg/ha)", fontsize=12, weight="bold")
    axes[1].set_ylabel("Phosphorus (kg/ha)")

    sns.boxplot(x=TARGET_COLUMN, y="K", data=df, ax=axes[2], palette="Set2")
    axes[2].set_title("Soil Potassium (K) Requirement Across Crops (kg/ha)", fontsize=12, weight="bold")
    axes[2].set_ylabel("Potassium (kg/ha)")
    axes[2].set_xlabel("Crop Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)

    fig.suptitle("Primary Macronutrient Profiles (N-P-K) Across 22 Crop Classes", fontsize=15, weight="bold", y=0.99)
    npk_path = FIGURES_DIR / "npk_by_crop.png"
    plt.savefig(npk_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["npk_by_crop"] = npk_path

    # 5. Climate Environmental Niche: Rainfall vs Temperature Clustering
    logger.info("Generating figure: Climate Clustering (Rainfall vs Temperature)...")
    fig, ax = plt.subplots(figsize=(12, 8))
    # Select subset of representative distinct crops for legible scatter visualization
    key_crops = ["rice", "coffee", "watermelon", "cotton", "chickpea", "apple", "banana"]
    df_subset = df[df[TARGET_COLUMN].isin(key_crops)]
    sns.scatterplot(
        x="temperature",
        y="rainfall",
        hue=TARGET_COLUMN,
        style=TARGET_COLUMN,
        s=90,
        alpha=0.85,
        data=df_subset,
        ax=ax,
        palette="tab10"
    )
    ax.set_title("Agro-Climatic Niches: Rainfall vs Temperature for Key Benchmark Crops", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Ambient Temperature (°C)", fontsize=11)
    ax.set_ylabel("Rainfall (mm)", fontsize=11)
    ax.legend(title="Crop Species", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    climate_path = FIGURES_DIR / "climate_clustering.png"
    plt.savefig(climate_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figures["climate_clustering"] = climate_path

    logger.info(f"Successfully generated and saved {len(saved_figures)} figures to {FIGURES_DIR}")
    return saved_figures


def compute_statistical_findings(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes key mathematical and agronomic findings from the dataset.

    Args:
        df: Loaded raw DataFrame.

    Returns:
        Dict[str, Any]: Key findings dictionary.
    """
    corr = df[NUMERICAL_FEATURES].corr()

    # Find highest positive and negative correlations
    corr_unstack = corr.unstack()
    corr_unstack = corr_unstack[corr_unstack < 0.9999]  # Exclude self-correlations
    top_pos_corr = corr_unstack.idxmax()
    top_pos_val = float(corr_unstack.max())

    # Crop with highest average rainfall
    rain_by_crop = df.groupby(TARGET_COLUMN)["rainfall"].mean()
    top_rain_crop = rain_by_crop.idxmax()
    top_rain_val = float(rain_by_crop.max())
    lowest_rain_crop = rain_by_crop.idxmin()
    lowest_rain_val = float(rain_by_crop.min())

    # Crop with highest N demand
    n_by_crop = df.groupby(TARGET_COLUMN)["N"].mean()
    top_n_crop = n_by_crop.idxmax()
    top_n_val = float(n_by_crop.max())

    # Crop with highest K demand
    k_by_crop = df.groupby(TARGET_COLUMN)["K"].mean()
    top_k_crop = k_by_crop.idxmax()
    top_k_val = float(k_by_crop.max())

    findings = {
        "dataset_shape": df.shape,
        "classes_count": len(df[TARGET_COLUMN].unique()),
        "top_positive_correlation": {
            "pair": list(top_pos_corr),
            "correlation": round(top_pos_val, 4)
        },
        "highest_rainfall_crop": {"crop": top_rain_crop, "mean_rainfall": round(top_rain_val, 2)},
        "lowest_rainfall_crop": {"crop": lowest_rain_crop, "mean_rainfall": round(lowest_rain_val, 2)},
        "highest_nitrogen_crop": {"crop": top_n_crop, "mean_n": round(top_n_val, 2)},
        "highest_potassium_crop": {"crop": top_k_crop, "mean_k": round(top_k_val, 2)}
    }
    return findings


if __name__ == "__main__":
    df = load_raw_data()
    figs = generate_eda_visualizations(df)
    findings = compute_statistical_findings(df)

    print("\n" + "=" * 55)
    print("SMARTAGRI-AI EXPLORATORY DATA ANALYSIS FINDINGS")
    print("=" * 55)
    print(f"Total Observations: {findings['dataset_shape'][0]} rows, {findings['dataset_shape'][1]} features")
    print(f"Strongest Correlation: {findings['top_positive_correlation']['pair']} (r = {findings['top_positive_correlation']['correlation']})")
    print(f"Highest Rainfall Demander: {findings['highest_rainfall_crop']['crop']} ({findings['highest_rainfall_crop']['mean_rainfall']} mm)")
    print(f"Drought-Tolerant (Lowest Rain): {findings['lowest_rainfall_crop']['crop']} ({findings['lowest_rainfall_crop']['mean_rainfall']} mm)")
    print(f"Top Nitrogen Consuming Crop: {findings['highest_nitrogen_crop']['crop']} ({findings['highest_nitrogen_crop']['mean_n']} kg/ha)")
    print(f"Top Potassium Consuming Crop: {findings['highest_potassium_crop']['crop']} ({findings['highest_potassium_crop']['mean_k']} kg/ha)")
    print(f"Figures Saved: {list(figs.keys())}")
    print("=" * 55)
