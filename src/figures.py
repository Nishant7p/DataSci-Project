"""Report figures. Sized for one column of the two-column template (83.5 mm wide).

Colour roles follow a validated palette: categorical slot 1 (blue) and slot 2 (orange) for
series, ink tokens for all text, hairline solid gridlines. A single series gets no legend box;
two series always get a legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COL_W = 83.5 / 25.4    # inches

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)
    return paths


def rating_ecdf(single: pd.Series, large: pd.Series, out_dir: Path, bar: float = 4.8) -> list[Path]:
    """ECDF of overall rating for single-host vs large-host established entire homes."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    for values, colour, label in [(single, SERIES_1, f"Single-listing hosts (n = {len(single):,})"),
                                  (large, SERIES_2, f"Hosts with 10+ listings (n = {len(large):,})")]:
        x = np.sort(values.to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=colour, linewidth=1.6, label=label)
    ax.axvline(bar, color=MUTED, linewidth=0.8)
    ax.text(bar + 0.012, 0.04, f"Superhost\nbar {bar}", color=INK_2, fontsize=6.3, ha="left", va="bottom")
    ax.set_xlim(4.0, 5.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Overall rating (stars); listings below 4.0 sit off the left edge")
    ax.set_ylabel("Share of listings at or below")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left", handlelength=1.6)
    return _save(fig, out_dir, "fig1_rating_ecdf")


def host_concentration(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Grouped horizontal bars: share of hosts vs share of listings by host portfolio class."""
    classes = list(table.index)
    y = np.arange(len(classes))
    h = 0.32
    fig, ax = plt.subplots(figsize=(COL_W, 1.45))
    for offset, col, colour, label in [(-h / 2 - 0.02, "share_hosts", SERIES_1, "Share of hosts"),
                                       (h / 2 + 0.02, "share_listings", SERIES_2, "Share of listings")]:
        vals = table[col].to_numpy()
        ax.barh(y + offset, vals, height=h, color=colour, label=label, zorder=2)
        for yi, v in zip(y + offset, vals):
            ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", ha="left", color=INK, fontsize=6.3)
    ax.set_yticks(y, classes)
    ax.invert_yaxis()
    ax.set_xlim(0, max(table[["share_hosts", "share_listings"]].max()) * 1.18)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Percent of exploration split")
    ax.legend(loc="lower right", handlelength=1.0)
    return _save(fig, out_dir, "fig2_host_concentration")


def large_share_by_borough(shares: pd.Series, out_dir: Path) -> list[Path]:
    """Share of entire homes run by 10+ listing hosts, per borough, sorted."""
    s = shares.sort_values()
    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    y = np.arange(len(s))
    ax.barh(y, s.to_numpy(), height=0.62, color=SERIES_1, zorder=2)
    ax.set_yticks(y, s.index)
    ax.tick_params(axis="y", labelsize=5.6, length=0)
    for idx in [0, len(s) - 1]:
        ax.text(s.iloc[idx] + 0.6, idx, f"{s.iloc[idx]:.1f}%", va="center", color=INK, fontsize=6.0)
    ax.set_xlim(0, s.max() * 1.15)
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share run by hosts with 10+ London listings\n(of entire homes run by single- or 10+-listing hosts)")
    return _save(fig, out_dir, "fig3_large_share_by_borough")


def accuracy_by_hype(data: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean accuracy score with 95% t-interval for each hype-word count bucket."""
    labels = list(data["bucket"].cat.categories)
    means, lows, highs, ns = [], [], [], []
    for b in labels:
        v = data.loc[data["bucket"] == b, "review_scores_accuracy"].to_numpy()
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        half = stats.t.ppf(0.975, len(v) - 1) * se
        means.append(m), lows.append(m - half), highs.append(m + half), ns.append(len(v))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    ax.vlines(x, lows, highs, color=SERIES_1, linewidth=1.6, zorder=2)
    ax.plot(x, means, "o", color=SERIES_1, markersize=4.5, markeredgecolor="white",
            markeredgewidth=1.0, zorder=3)
    ax.set_xticks(x, [f"{lab}\n(n = {n:,})" for lab, n in zip(labels, ns)])
    ax.set_xlabel("Hype words in the description")
    ax.set_ylabel("Mean accuracy score\n(95% interval)")
    ax.grid(axis="x", visible=False)
    # The axis spans +/- 0.10 star around the no-hype mean: the smallest gap a guest can see,
    # so the reader judges the differences on a scale that matters rather than a zoomed one.
    ax.set_ylim(means[0] - 0.10, means[0] + 0.10)
    ax.axhline(means[0], color=MUTED, linewidth=0.6, zorder=1)
    return _save(fig, out_dir, "fig4_accuracy_by_hype")
