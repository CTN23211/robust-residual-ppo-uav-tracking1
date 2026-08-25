from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE6_RESULTS_DIR = RESULTS_DIR / "phase6"
PLOTS_DIR = PROJECT_ROOT / "plots" / "phase6"
TABLES_DIR = PROJECT_ROOT / "tables" / "phase6"

METHOD_ORDER = ["PD+FF", "Direct PPO", "Residual PPO"]
RESIDUAL_ORDER = ["Nominal Residual PPO", "Robust Residual PPO"]

# Colorblind-friendly palette. Kept in one place so paper figures are consistent.
METHOD_COLORS = {
    "PD+FF": "#4C78A8",
    "Direct PPO": "#F58518",
    "Residual PPO": "#54A24B",
    "Nominal Residual PPO": "#54A24B",
    "Robust Residual PPO": "#B279A2",
}

REQUIRED_SUMMARY_FILES = [
    "velocity_compensation_fault_summary.csv",
    "observation_noise_summary.csv",
    "observation_delay_summary.csv",
    "temporary_vision_loss_summary.csv",
    "unseen_trajectory_summary.csv",
    "robust_residual_alpha_sweep_summary.csv",
    "cross_robustness_summary.csv",
    "cross_robustness_paired_comparison.csv",
]

REQUIRED_RAW_FILES = [
    "velocity_compensation_fault_raw.csv",
    "observation_noise_raw.csv",
    "observation_delay_raw.csv",
    "temporary_vision_loss_raw.csv",
    "unseen_trajectory_raw.csv",
    "robust_residual_alpha_sweep_raw.csv",
    "cross_robustness_raw.csv",
]


def ensure_phase6_dirs() -> None:
    for path in (PHASE6_RESULTS_DIR, PLOTS_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def require_files(names: Iterable[str]) -> None:
    missing = [str(RESULTS_DIR / name) for name in names if not (RESULTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required Phase 5 result files:\n  " + "\n  ".join(missing))


def load_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def safe_float(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def percent_change(new: float, old: float) -> float:
    if not np.isfinite(old) or abs(old) < 1e-12:
        return np.nan
    return 100.0 * (new - old) / old


def bootstrap_mean_ci(values, confidence: float = 0.95, n_boot: int = 2000, seed: int = 20260820):
    """Return mean, lower CI, upper CI using deterministic non-parametric bootstrap."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    if x.size == 1:
        v = float(x[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    # Chunking keeps memory small while still being fast for n=100.
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        means[i] = sample.mean()

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(x.mean()), float(lower), float(upper)


def markdown_table(df: pd.DataFrame, float_formats: dict[str, str] | None = None) -> str:
    """Dependency-free Markdown table renderer (avoids requiring tabulate)."""
    float_formats = float_formats or {}
    display = df.copy()

    for col in display.columns:
        if col in float_formats:
            fmt = float_formats[col]
            display[col] = display[col].map(
                lambda x: "—" if pd.isna(x) else format(float(x), fmt)
            )
        else:
            display[col] = display[col].map(lambda x: "—" if pd.isna(x) else str(x))

    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in display.columns) + " |")
    return "\n".join(lines)


def save_png_svg(fig, stem: str, dpi: int = 300) -> tuple[Path, Path]:
    png = PLOTS_DIR / f"{stem}.png"
    svg = PLOTS_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return png, svg
