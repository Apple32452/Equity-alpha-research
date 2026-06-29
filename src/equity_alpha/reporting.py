"""CSV, JSON, and matplotlib report creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(frame: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    frame.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str, allow_nan=True)
    return path


def save_figures(baseline: pd.DataFrame, improved: pd.DataFrame, output_dir: Path) -> None:
    """Save default-style equity, drawdown, and turnover figures."""
    baseline = baseline.sort_values("date")
    improved = improved.sort_values("date")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(baseline["date"], baseline["net_equity"], label="baseline")
    axis.plot(improved["date"], improved["net_equity"], label="improved")
    axis.set_title("Net Equity Curves")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative wealth")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "equity_curve.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(baseline["date"], baseline["net_drawdown"], label="baseline")
    axis.plot(improved["date"], improved["net_drawdown"], label="improved")
    axis.set_title("Net Drawdown")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "drawdown.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(baseline["date"], baseline["one_way_turnover"], label="baseline")
    axis.plot(improved["date"], improved["one_way_turnover"], label="improved")
    axis.set_title("One-Way Daily Turnover")
    axis.set_xlabel("Date")
    axis.set_ylabel("Turnover")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "turnover.png", dpi=160)
    plt.close(figure)
