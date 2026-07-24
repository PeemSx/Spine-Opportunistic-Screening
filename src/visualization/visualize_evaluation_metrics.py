from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


RAW_COLOR = "#4C78A8"
CHAIN_COLOR = "#F58518"
ACCENT_COLOR = "#2A9D8F"
DARK_COLOR = "#263238"
MUTED_COLOR = "#607D8B"
GRID_COLOR = "#D9E0E5"
PAGE_COLOR = "#F6F8FA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a multi-page PDF dashboard from evaluation metrics.csv."
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory for one PNG preview per PDF page.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    required = {"model", "scope", "source_dataset"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            "metrics.csv is missing required columns: " + ", ".join(sorted(missing))
        )
    return rows


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else float("nan")


def row_index(
    rows: list[dict[str, str]],
    *,
    model: str,
    scope: str,
    source: str = "",
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["model"] == model
        and row["scope"] == scope
        and row.get("source_dataset", "") == source
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for model={model}, scope={scope}, source={source!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def source_names(rows: list[dict[str, str]]) -> list[str]:
    return sorted(
        {
            row["source_dataset"]
            for row in rows
            if row["scope"] == "source" and row["source_dataset"]
        }
    )


def short_source(name: str) -> str:
    replacements = {
        "BUU AP": "BUU",
        "MICCAI-2019": "MICCAI",
        "Mendeley PA": "Mendeley",
        "NIH ChestX-ray14": "NIH",
    }
    return replacements.get(name, name)


def style_axis(ax: plt.Axes, *, x_grid: bool = False, y_grid: bool = True) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.tick_params(colors=DARK_COLOR, labelsize=9)
    ax.title.set_color(DARK_COLOR)
    ax.title.set_fontweight("bold")
    if y_grid:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    if x_grid:
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def page_title(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    page_number: int,
    source_path: Path,
) -> None:
    fig.suptitle(
        title,
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=DARK_COLOR,
    )
    fig.text(
        0.045,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=10.5,
        color=MUTED_COLOR,
    )
    fig.text(
        0.045,
        0.018,
        f"Source: {source_path}",
        ha="left",
        fontsize=7.5,
        color=MUTED_COLOR,
    )
    fig.text(
        0.955,
        0.018,
        f"Page {page_number}",
        ha="right",
        fontsize=7.5,
        color=MUTED_COLOR,
    )


def add_bar_labels(
    ax: plt.Axes,
    bars: Any,
    *,
    percent: bool = False,
    decimals: int = 2,
) -> None:
    for bar in bars:
        value = float(bar.get_height())
        if not np.isfinite(value):
            continue
        label = f"{value * 100:.1f}%" if percent else f"{value:.{decimals}f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=DARK_COLOR,
        )


def grouped_bars(
    ax: plt.Axes,
    labels: list[str],
    raw_values: list[float],
    chain_values: list[float],
    *,
    percent: bool = False,
    ylabel: str = "",
    title: str,
    ylim: tuple[float, float] | None = None,
    rotate: int = 0,
) -> None:
    positions = np.arange(len(labels))
    width = 0.36
    raw_bars = ax.bar(
        positions - width / 2,
        raw_values,
        width,
        label="Raw",
        color=RAW_COLOR,
    )
    chain_bars = ax.bar(
        positions + width / 2,
        chain_values,
        width,
        label="Spine chain",
        color=CHAIN_COLOR,
    )
    ax.set_xticks(positions, labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    style_axis(ax)
    add_bar_labels(ax, raw_bars, percent=percent)
    add_bar_labels(ax, chain_bars, percent=percent)


def rate_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0.0, 1.08)
    ticks = np.linspace(0.0, 1.0, 6)
    ax.set_yticks(ticks, [f"{tick * 100:.0f}%" for tick in ticks])


def new_page() -> plt.Figure:
    return plt.figure(figsize=(13.333, 7.5), facecolor=PAGE_COLOR)


def save_page(
    pdf: PdfPages,
    fig: plt.Figure,
    page_number: int,
    preview_dir: Path | None,
) -> None:
    pdf.savefig(fig, facecolor=PAGE_COLOR)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            preview_dir / f"page_{page_number:02d}.png",
            dpi=150,
            facecolor=PAGE_COLOR,
        )
    plt.close(fig)


def draw_card(
    ax: plt.Axes,
    title: str,
    value: str,
    note: str,
    *,
    color: str = DARK_COLOR,
) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.text(0.06, 0.90, title, fontsize=9, color=MUTED_COLOR, va="top")
    ax.text(0.06, 0.48, value, fontsize=21, fontweight="bold", color=color, va="center")
    ax.text(0.06, 0.10, note, fontsize=8, color=MUTED_COLOR, va="bottom")


def page_executive(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    raw = row_index(rows, model="raw", scope="overall_micro")
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    fig = new_page()
    page_title(
        fig,
        "Evaluation v2 — deployed test dashboard",
        "Overall technical performance • raw CenterNet candidates versus spine-chain output",
        1,
        source_path,
    )
    grid = fig.add_gridspec(
        3,
        6,
        left=0.045,
        right=0.965,
        top=0.865,
        bottom=0.075,
        height_ratios=[0.75, 2.3, 1.05],
        hspace=0.42,
        wspace=0.35,
    )

    cards = [
        ("Test images", f"{int(number(chain, 'images')):,}", "four source datasets"),
        (
            "GT vertebrae",
            f"{int(number(chain, 'gt_vertebrae')):,}",
            "all evaluation denominators",
        ),
        (
            "F1 @ 0.20D",
            f"{number(chain, 'center_f1_0.20d') * 100:.1f}%",
            f"{(number(chain, 'center_f1_0.20d') - number(raw, 'center_f1_0.20d')) * 100:+.1f} pp vs raw",
        ),
        (
            "Usable recall",
            f"{number(chain, 'usable_vertebra_recall') * 100:.1f}%",
            "match + valid quad + NME ≤ 0.10",
        ),
        (
            "Count MAE",
            f"{number(chain, 'count_mae'):.2f}",
            f"{number(chain, 'count_mae') - number(raw, 'count_mae'):+.2f} vs raw",
        ),
        (
            "Cobb MAE",
            f"{number(chain, 'cobb_mae_deg'):.2f}°",
            f"{number(chain, 'cobb_coverage') * 100:.0f}% coverage",
        ),
    ]
    for index, (title, value, note) in enumerate(cards):
        draw_card(
            fig.add_subplot(grid[0, index]),
            title,
            value,
            note,
            color=ACCENT_COLOR if index in (2, 3) else DARK_COLOR,
        )

    ax_rates = fig.add_subplot(grid[1, :4])
    labels = ["F1 @0.20D", "PCK @0.10", "Usable recall", "Cobb @5°", "Cobb @10°"]
    fields = [
        "center_f1_0.20d",
        "pck_0.10",
        "usable_vertebra_recall",
        "cobb_at_5deg",
        "cobb_at_10deg",
    ]
    grouped_bars(
        ax_rates,
        labels,
        [number(raw, field) for field in fields],
        [number(chain, field) for field in fields],
        percent=True,
        ylabel="Rate",
        title="Primary overall scorecard",
        ylim=(0.0, 1.10),
    )
    rate_axis(ax_rates)

    ax_errors = fig.add_subplot(grid[1, 4:])
    error_labels = ["FP / image", "Count MAE", "Cobb MAE (°)", "Mean NME ×10"]
    raw_errors = [
        number(raw, "false_positives_per_image"),
        number(raw, "count_mae"),
        number(raw, "cobb_mae_deg"),
        number(raw, "corner_nme_mean") * 10.0,
    ]
    chain_errors = [
        number(chain, "false_positives_per_image"),
        number(chain, "count_mae"),
        number(chain, "cobb_mae_deg"),
        number(chain, "corner_nme_mean") * 10.0,
    ]
    grouped_bars(
        ax_errors,
        error_labels,
        raw_errors,
        chain_errors,
        title="Error-oriented metrics (lower is better)",
        rotate=20,
    )

    raw_fp = number(raw, "center_fp_0.20d")
    chain_fp = number(chain, "center_fp_0.20d")
    fp_reduction = (raw_fp - chain_fp) / raw_fp if raw_fp > 0 else 0.0
    recall_loss = (
        number(raw, "center_recall_0.20d")
        - number(chain, "center_recall_0.20d")
    )
    observations = [
        (
            "False positives",
            f"{fp_reduction * 100:.1f}% reduction",
            f"{int(raw_fp)} raw → {int(chain_fp)} chain",
        ),
        (
            "Detection recall",
            f"{recall_loss * 100:.1f} pp loss",
            f"{number(raw, 'center_recall_0.20d') * 100:.1f}% → "
            f"{number(chain, 'center_recall_0.20d') * 100:.1f}%",
        ),
        (
            "Counting",
            f"{number(raw, 'count_mae') - number(chain, 'count_mae'):.2f} lower MAE",
            f"bias {number(chain, 'count_bias'):+.2f} vertebrae",
        ),
    ]
    for index, (title, value, note) in enumerate(observations):
        draw_card(
            fig.add_subplot(grid[2, index * 2 : index * 2 + 2]),
            title,
            value,
            note,
            color=ACCENT_COLOR,
        )
    save_page(pdf, fig, 1, preview_dir)


def page_detection(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    raw = row_index(rows, model="raw", scope="overall_micro")
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    sources = source_names(rows)
    fig = new_page()
    page_title(
        fig,
        "Detection performance",
        "One-to-one Hungarian matching normalized by GT vertebral diagonal D",
        2,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.965,
        top=0.86,
        bottom=0.08,
        hspace=0.40,
        wspace=0.28,
    )
    gates = ["0.10", "0.20", "0.25"]
    for column, (metric, label) in enumerate(
        (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
    ):
        ax = fig.add_subplot(grid[0, column])
        ax.plot(
            gates,
            [number(raw, f"center_{metric}_{gate}d") for gate in gates],
            marker="o",
            linewidth=2.3,
            color=RAW_COLOR,
            label="Raw",
        )
        ax.plot(
            gates,
            [number(chain, f"center_{metric}_{gate}d") for gate in gates],
            marker="o",
            linewidth=2.3,
            color=CHAIN_COLOR,
            label="Spine chain",
        )
        ax.set_title(f"Overall {label}", loc="left", fontsize=12)
        ax.set_xlabel("Center-match gate")
        ax.set_ylabel(label)
        ax.set_xticks(gates, [f"{gate}D" for gate in gates])
        ax.legend(frameon=False, fontsize=8)
        rate_axis(ax)
        style_axis(ax)

    source_labels = [short_source(source) for source in sources]
    for column, (metric, label) in enumerate(
        (("center_f1_0.20d", "F1 @0.20D"), ("center_recall_0.20d", "Recall @0.20D"))
    ):
        ax = fig.add_subplot(grid[1, column])
        raw_rows = [
            row_index(rows, model="raw", scope="source", source=source)
            for source in sources
        ]
        chain_rows = [
            row_index(rows, model="spine_chain", scope="source", source=source)
            for source in sources
        ]
        grouped_bars(
            ax,
            source_labels,
            [number(row, metric) for row in raw_rows],
            [number(row, metric) for row in chain_rows],
            percent=True,
            ylabel=label,
            title=f"Source-level {label}",
            ylim=(0.0, 1.10),
        )
        rate_axis(ax)

    ax_fp = fig.add_subplot(grid[1, 2])
    raw_rows = [
        row_index(rows, model="raw", scope="source", source=source)
        for source in sources
    ]
    chain_rows = [
        row_index(rows, model="spine_chain", scope="source", source=source)
        for source in sources
    ]
    grouped_bars(
        ax_fp,
        source_labels,
        [number(row, "false_positives_per_image") for row in raw_rows],
        [number(row, "false_positives_per_image") for row in chain_rows],
        ylabel="False positives per image",
        title="Source-level FP burden",
    )
    save_page(pdf, fig, 2, preview_dir)


def page_landmarks(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    raw = row_index(rows, model="raw", scope="overall_micro")
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    sources = source_names(rows)
    fig = new_page()
    page_title(
        fig,
        "Landmark localization",
        "Corner errors use the matched GT vertebra’s diagonal; end-to-end PCK penalizes missed detections",
        3,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.965,
        top=0.86,
        bottom=0.08,
        hspace=0.42,
        wspace=0.25,
    )
    ax_nme = fig.add_subplot(grid[0, 0])
    grouped_bars(
        ax_nme,
        ["Mean", "Median", "P95"],
        [number(raw, field) for field in ("corner_nme_mean", "corner_nme_median", "corner_nme_p95")],
        [number(chain, field) for field in ("corner_nme_mean", "corner_nme_median", "corner_nme_p95")],
        ylabel="Normalized mean error",
        title="NME distribution (lower is better)",
        ylim=(0.0, max(number(raw, "corner_nme_p95"), number(chain, "corner_nme_p95")) * 1.25),
    )

    thresholds = ["0.05", "0.10", "0.20"]
    ax_pck = fig.add_subplot(grid[0, 1])
    grouped_bars(
        ax_pck,
        [f"@{threshold}" for threshold in thresholds],
        [number(raw, f"pck_{threshold}") for threshold in thresholds],
        [number(chain, f"pck_{threshold}") for threshold in thresholds],
        percent=True,
        ylabel="PCK",
        title="Conditional PCK on matched vertebrae",
        ylim=(0.0, 1.10),
    )
    rate_axis(ax_pck)

    ax_e2e = fig.add_subplot(grid[1, 0])
    grouped_bars(
        ax_e2e,
        [f"@{threshold}" for threshold in thresholds],
        [number(raw, f"end_to_end_pck_{threshold}") for threshold in thresholds],
        [number(chain, f"end_to_end_pck_{threshold}") for threshold in thresholds],
        percent=True,
        ylabel="End-to-end PCK",
        title="End-to-end PCK including misses",
        ylim=(0.0, 1.10),
    )
    rate_axis(ax_e2e)

    raw_rows = [
        row_index(rows, model="raw", scope="source", source=source)
        for source in sources
    ]
    chain_rows = [
        row_index(rows, model="spine_chain", scope="source", source=source)
        for source in sources
    ]
    ax_usable = fig.add_subplot(grid[1, 1])
    grouped_bars(
        ax_usable,
        [short_source(source) for source in sources],
        [number(row, "usable_vertebra_recall") for row in raw_rows],
        [number(row, "usable_vertebra_recall") for row in chain_rows],
        percent=True,
        ylabel="Usable-vertebra recall",
        title="Usable output by source",
        ylim=(0.0, 1.10),
    )
    rate_axis(ax_usable)
    save_page(pdf, fig, 3, preview_dir)


def page_counting(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    raw = row_index(rows, model="raw", scope="overall_micro")
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    sources = source_names(rows)
    raw_rows = [
        row_index(rows, model="raw", scope="source", source=source)
        for source in sources
    ]
    chain_rows = [
        row_index(rows, model="spine_chain", scope="source", source=source)
        for source in sources
    ]
    fig = new_page()
    page_title(
        fig,
        "Counting and post-processing",
        "Vertebra-count accuracy and the operational effect of spine-chain selection",
        4,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.965,
        top=0.86,
        bottom=0.08,
        hspace=0.42,
        wspace=0.25,
    )
    ax_errors = fig.add_subplot(grid[0, 0])
    grouped_bars(
        ax_errors,
        ["MAE", "P90 abs. error", "Absolute bias"],
        [
            number(raw, "count_mae"),
            number(raw, "count_p90_absolute_error"),
            abs(number(raw, "count_bias")),
        ],
        [
            number(chain, "count_mae"),
            number(chain, "count_p90_absolute_error"),
            abs(number(chain, "count_bias")),
        ],
        ylabel="Vertebrae",
        title="Overall count error",
    )

    ax_rates = fig.add_subplot(grid[0, 1])
    grouped_bars(
        ax_rates,
        ["Exact count", "Within one", "Full coverage", "Clean output"],
        [
            number(raw, "count_exact_rate"),
            number(raw, "count_within_one_rate"),
            number(raw, "full_coverage_rate"),
            number(raw, "clean_chain_rate"),
        ],
        [
            number(chain, "count_exact_rate"),
            number(chain, "count_within_one_rate"),
            number(chain, "full_coverage_rate"),
            number(chain, "clean_chain_rate"),
        ],
        percent=True,
        ylabel="Image rate",
        title="Image success rates",
        ylim=(0.0, 1.10),
    )
    rate_axis(ax_rates)

    labels = [short_source(source) for source in sources]
    ax_source_mae = fig.add_subplot(grid[1, 0])
    grouped_bars(
        ax_source_mae,
        labels,
        [number(row, "count_mae") for row in raw_rows],
        [number(row, "count_mae") for row in chain_rows],
        ylabel="Count MAE",
        title="Count MAE by source",
    )

    ax_source_bias = fig.add_subplot(grid[1, 1])
    positions = np.arange(len(labels))
    width = 0.36
    raw_bars = ax_source_bias.bar(
        positions - width / 2,
        [number(row, "count_bias") for row in raw_rows],
        width,
        color=RAW_COLOR,
        label="Raw",
    )
    chain_bars = ax_source_bias.bar(
        positions + width / 2,
        [number(row, "count_bias") for row in chain_rows],
        width,
        color=CHAIN_COLOR,
        label="Spine chain",
    )
    ax_source_bias.axhline(0.0, color=DARK_COLOR, linewidth=1.0)
    ax_source_bias.set_xticks(positions, labels)
    ax_source_bias.set_ylabel("Signed count bias")
    ax_source_bias.set_title("Count bias by source", loc="left", fontsize=12)
    ax_source_bias.legend(frameon=False, fontsize=8, ncols=2)
    style_axis(ax_source_bias)
    add_bar_labels(ax_source_bias, raw_bars)
    add_bar_labels(ax_source_bias, chain_bars)
    save_page(pdf, fig, 4, preview_dir)


def page_cobb(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    raw = row_index(rows, model="raw", scope="overall_micro")
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    fig = new_page()
    page_title(
        fig,
        "Cobb geometric agreement",
        "Corner-derived agreement against GT corners; not radiologist-reference diagnostic accuracy",
        5,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.965,
        top=0.86,
        bottom=0.08,
        hspace=0.42,
        wspace=0.25,
    )
    ax_errors = fig.add_subplot(grid[0, 0])
    grouped_bars(
        ax_errors,
        ["MAE", "Median AE", "P95 AE", "RMSE"],
        [
            number(raw, "cobb_mae_deg"),
            number(raw, "cobb_median_absolute_error_deg"),
            number(raw, "cobb_p95_absolute_error_deg"),
            number(raw, "cobb_rmse_deg"),
        ],
        [
            number(chain, "cobb_mae_deg"),
            number(chain, "cobb_median_absolute_error_deg"),
            number(chain, "cobb_p95_absolute_error_deg"),
            number(chain, "cobb_rmse_deg"),
        ],
        ylabel="Degrees",
        title="Overall angle error (lower is better)",
    )

    ax_rates = fig.add_subplot(grid[0, 1])
    grouped_bars(
        ax_rates,
        ["Coverage", "Cobb @5°", "Cobb @10°", "Endpoints exact", "Endpoints ±1"],
        [
            number(raw, "cobb_coverage"),
            number(raw, "cobb_at_5deg"),
            number(raw, "cobb_at_10deg"),
            number(raw, "cobb_endpoint_exact_rate"),
            number(raw, "cobb_endpoint_within_one_rate"),
        ],
        [
            number(chain, "cobb_coverage"),
            number(chain, "cobb_at_5deg"),
            number(chain, "cobb_at_10deg"),
            number(chain, "cobb_endpoint_exact_rate"),
            number(chain, "cobb_endpoint_within_one_rate"),
        ],
        percent=True,
        ylabel="Rate",
        title="Coverage, tolerance, and endpoint agreement",
        ylim=(0.0, 1.10),
        rotate=12,
    )
    rate_axis(ax_rates)

    bands = [
        ("lt_5deg", "<5°"),
        ("5_to_lt_10deg", "5–<10°"),
        ("10_to_lt_20deg", "10–<20°"),
        ("ge_20deg", "≥20°"),
    ]
    ax_band_mae = fig.add_subplot(grid[1, 0])
    values = [number(chain, f"cobb_{key}_mae_deg") for key, _ in bands]
    bars = ax_band_mae.bar(
        np.arange(len(bands)),
        values,
        color=CHAIN_COLOR,
    )
    ax_band_mae.set_xticks(
        np.arange(len(bands)),
        [
            f"{label}\n(n={int(number(chain, f'cobb_{key}_images'))})"
            for key, label in bands
        ],
    )
    ax_band_mae.set_ylabel("Cobb MAE (degrees)")
    ax_band_mae.set_title("Spine-chain error by GT angle stratum", loc="left", fontsize=12)
    style_axis(ax_band_mae)
    add_bar_labels(ax_band_mae, bars)

    ax_band_rates = fig.add_subplot(grid[1, 1])
    positions = np.arange(len(bands))
    width = 0.25
    for offset, (field_suffix, label, color) in enumerate(
        (
            ("coverage", "Coverage", ACCENT_COLOR),
            ("at_5deg", "Cobb @5°", CHAIN_COLOR),
            ("at_10deg", "Cobb @10°", RAW_COLOR),
        )
    ):
        bars = ax_band_rates.bar(
            positions + (offset - 1) * width,
            [number(chain, f"cobb_{key}_{field_suffix}") for key, _ in bands],
            width,
            color=color,
            label=label,
        )
        add_bar_labels(ax_band_rates, bars, percent=True)
    ax_band_rates.set_xticks(positions, [label for _, label in bands])
    ax_band_rates.set_ylabel("Rate")
    ax_band_rates.set_title("Spine-chain stratum coverage and tolerance", loc="left", fontsize=12)
    ax_band_rates.legend(frameon=False, fontsize=8, ncols=3)
    rate_axis(ax_band_rates)
    style_axis(ax_band_rates)
    save_page(pdf, fig, 5, preview_dir)


def annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    *,
    title: str,
    delta: bool = False,
) -> None:
    if delta:
        limit = max(float(np.nanmax(np.abs(values))), 0.01)
        cmap = LinearSegmentedColormap.from_list(
            "delta",
            ["#B2182B", "#FFFFFF", "#2166AC"],
        )
        image = ax.imshow(
            values,
            aspect="auto",
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
    else:
        image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(column_labels)), column_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", fontsize=12)
    for row_index_value in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index_value, column_index]
            label = (
                f"{value * 100:+.1f} pp"
                if delta
                else f"{value * 100:.1f}%"
            )
            ax.text(
                column_index,
                row_index_value,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if (not delta and value > 0.70)
                    else DARK_COLOR
                ),
            )
    plt.colorbar(image, ax=ax, fraction=0.025, pad=0.02)


def page_sources(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    sources = source_names(rows)
    raw_rows = [
        row_index(rows, model="raw", scope="source", source=source)
        for source in sources
    ]
    chain_rows = [
        row_index(rows, model="spine_chain", scope="source", source=source)
        for source in sources
    ]
    fields = [
        ("center_f1_0.20d", "F1 @0.20D"),
        ("center_recall_0.20d", "Recall"),
        ("pck_0.10", "PCK @0.10"),
        ("usable_vertebra_recall", "Usable recall"),
        ("cobb_at_5deg", "Cobb @5°"),
        ("cobb_at_10deg", "Cobb @10°"),
        ("cobb_coverage", "Cobb coverage"),
    ]
    chain_values = np.asarray(
        [[number(row, field) for field, _ in fields] for row in chain_rows],
        dtype=float,
    )
    raw_values = np.asarray(
        [[number(row, field) for field, _ in fields] for row in raw_rows],
        dtype=float,
    )

    fig = new_page()
    page_title(
        fig,
        "Source-level scorecard",
        "Absolute spine-chain performance and paired change from raw candidates",
        6,
        source_path,
    )
    grid = fig.add_gridspec(
        3,
        1,
        left=0.08,
        right=0.94,
        top=0.85,
        bottom=0.08,
        height_ratios=[1.4, 1.4, 0.55],
        hspace=0.48,
    )
    labels = [short_source(source) for source in sources]
    annotated_heatmap(
        fig.add_subplot(grid[0, 0]),
        chain_values,
        labels,
        [label for _, label in fields],
        title="Spine-chain performance",
    )
    annotated_heatmap(
        fig.add_subplot(grid[1, 0]),
        chain_values - raw_values,
        labels,
        [label for _, label in fields],
        title="Spine chain minus raw (percentage points)",
        delta=True,
    )

    ax_counts = fig.add_subplot(grid[2, 0])
    image_counts = [int(number(row, "images")) for row in chain_rows]
    gt_counts = [int(number(row, "gt_vertebrae")) for row in chain_rows]
    positions = np.arange(len(labels))
    bars = ax_counts.bar(positions, image_counts, color=ACCENT_COLOR)
    ax_counts.set_xticks(positions, labels)
    ax_counts.set_ylabel("Images")
    ax_counts.set_title("Evaluation support by source", loc="left", fontsize=12)
    style_axis(ax_counts)
    for bar, gt_count in zip(bars, gt_counts):
        ax_counts.annotate(
            f"{int(bar.get_height())} images\n{gt_count:,} GT vertebrae",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK_COLOR,
        )
    save_page(pdf, fig, 6, preview_dir)


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_csv.resolve(strict=True)
    output_path = args.output_pdf.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = args.preview_dir.resolve() if args.preview_dir else None
    rows = load_rows(metrics_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": DARK_COLOR,
            "text.color": DARK_COLOR,
            "figure.dpi": 120,
        }
    )
    with PdfPages(output_path) as pdf:
        page_executive(rows, pdf, metrics_path, preview_dir)
        page_detection(rows, pdf, metrics_path, preview_dir)
        page_landmarks(rows, pdf, metrics_path, preview_dir)
        page_counting(rows, pdf, metrics_path, preview_dir)
        page_cobb(rows, pdf, metrics_path, preview_dir)
        page_sources(rows, pdf, metrics_path, preview_dir)
        metadata = pdf.infodict()
        metadata["Title"] = "Evaluation v2 Metrics Dashboard"
        metadata["Subject"] = "Deployed spine CenterNet evaluation visualization"
        metadata["Keywords"] = "detection landmarks counting spine-chain Cobb"

    print(f"saved: {output_path}")
    if preview_dir is not None:
        print(f"previews: {preview_dir}")


if __name__ == "__main__":
    main()
