from __future__ import annotations

import argparse
import csv
import json
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
CAUTION_COLOR = "#9C2A2A"

FRACTURE_READINESS_NOTICE = (
    "No fracture labels are used here, and no validated fracture classifier is implemented. "
    "These are measurement-readiness metrics; clinical review is required."
)


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
    parser.add_argument(
        "--focus",
        choices=("general", "fracture-readiness", "one-page-summary"),
        default="general",
        help=(
            "Report composition. The default preserves the general evaluation "
            "dashboard; fracture-readiness emphasizes geometric measurement reliability; "
            "one-page-summary creates a concise best-model performance overview."
        ),
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_fracture_readiness_inputs(
    metrics_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    evaluation_dir = metrics_path.parent
    supporting_paths = {
        "metrics JSON": evaluation_dir / "metrics.json",
        "per-image CSV": evaluation_dir / "per_image_metrics.csv",
        "per-instance CSV": evaluation_dir / "per_instance_metrics.csv",
        "evaluation manifest": evaluation_dir.parent / "evaluation_manifest.json",
    }
    missing = [label for label, path in supporting_paths.items() if not path.is_file()]
    if missing:
        raise ValueError(
            "Fracture-readiness mode requires sibling evaluation artifacts: "
            + ", ".join(missing)
        )
    return (
        load_json(supporting_paths["metrics JSON"]),
        load_json(supporting_paths["evaluation manifest"]),
        load_csv(supporting_paths["per-image CSV"]),
        load_csv(supporting_paths["per-instance CSV"]),
    )


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
        "Lumos AP": "Lumos",
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


def add_fracture_readiness_notice(fig: plt.Figure) -> None:
    fig.text(
        0.045,
        0.887,
        FRACTURE_READINESS_NOTICE,
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=CAUTION_COLOR,
    )
    fig.text(
        0.045,
        0.866,
        "All displayed results use spine-chain output only.",
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=ACCENT_COLOR,
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
    *,
    output_png: Path | None = None,
) -> None:
    pdf.savefig(fig, facecolor=PAGE_COLOR)
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, facecolor=PAGE_COLOR)
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


def draw_summary_card(
    ax: plt.Axes,
    stage: str,
    title: str,
    value: str,
    supporting_lines: list[str],
    *,
    color: str = DARK_COLOR,
    hero: bool = False,
) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.text(
        0.06,
        0.91,
        stage.upper(),
        fontsize=7.8,
        fontweight="bold",
        color=ACCENT_COLOR,
        va="top",
    )
    ax.text(
        0.06,
        0.79,
        title,
        fontsize=12.5 if hero else 10.5,
        fontweight="bold",
        color=DARK_COLOR,
        va="top",
        wrap=True,
    )
    ax.text(
        0.06,
        0.52,
        value,
        fontsize=39 if hero else 27,
        fontweight="bold",
        color=color,
        va="center",
    )
    ax.text(
        0.06,
        0.26,
        "\n".join(supporting_lines),
        fontsize=8.4 if hero else 7.9,
        color=MUTED_COLOR,
        va="top",
        linespacing=1.45,
        wrap=True,
    )


def confidence_interval(
    metrics_payload: dict[str, Any], model: str, field: str
) -> tuple[float, float] | None:
    interval = (
        metrics_payload.get("scorecard", {})
        .get("confidence_intervals", {})
        .get(model, {})
        .get(field)
    )
    if not isinstance(interval, dict):
        return None
    lower = interval.get("lower_95")
    upper = interval.get("upper_95")
    if lower is None or upper is None:
        return None
    return float(lower), float(upper)


def checkpoint_label(
    metrics_payload: dict[str, Any], evaluation_manifest: dict[str, Any]
) -> str:
    checkpoint = Path(
        str(metrics_payload.get("metadata", {}).get("checkpoint", "unknown checkpoint"))
    ).name
    epoch = evaluation_manifest.get("checkpoints", {}).get(checkpoint, {}).get("epoch")
    return f"{checkpoint} (epoch {epoch})" if epoch is not None else checkpoint


def finite_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value in ("", None):
            continue
        parsed = float(value)
        if np.isfinite(parsed):
            values.append(parsed)
    return values


def fracture_source_rows(
    rows: list[dict[str, str]], model: str = "spine_chain"
) -> tuple[list[str], list[dict[str, str]]]:
    sources = source_names(rows)
    return sources, [
        row_index(rows, model=model, scope="source", source=source)
        for source in sources
    ]


def add_horizontal_rate_labels(ax: plt.Axes, bars: Any) -> None:
    for bar in bars:
        value = float(bar.get_width())
        ax.annotate(
            f"{value * 100:.1f}%",
            (value, bar.get_y() + bar.get_height() / 2.0),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=DARK_COLOR,
        )


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


def page_one_page_summary(
    rows: list[dict[str, str]],
    metrics_payload: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
    output_png: Path,
) -> None:
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    required_fields = (
        "images",
        "gt_vertebrae",
        "center_f1_0.20d",
        "center_precision_0.20d",
        "center_recall_0.20d",
        "pck_0.10",
        "end_to_end_pck_0.10",
        "corner_nme_mean",
        "corner_nme_median",
        "corner_nme_p95",
        "cobb_mae_deg",
        "cobb_within_5deg_rate",
        "cobb_within_10deg_rate",
        "cobb_coverage",
        "count_mae",
        "count_within_one_rate",
    )
    missing = [field for field in required_fields if not np.isfinite(number(chain, field))]
    if missing:
        raise ValueError(
            "One-page summary requires finite spine-chain metrics: "
            + ", ".join(missing)
        )

    checkpoint = Path(
        str(metrics_payload.get("metadata", {}).get("checkpoint", "unknown checkpoint"))
    ).name
    epoch = (
        evaluation_manifest.get("checkpoints", {})
        .get(checkpoint, {})
        .get("epoch")
    )
    backbone = (
        metrics_payload.get("metadata", {}).get("settings", {}).get("backbone", "")
    )
    backbone_labels = {
        "hrnet_w18": "HRNet-W18",
        "hrnet_w30": "HRNet-W30",
        "hrnet_w32": "HRNet-W32",
        "hrnet_w40": "HRNet-W40",
        "hrnet_w44": "HRNet-W44",
        "hrnet_w48": "HRNet-W48",
        "hrnet_w64": "HRNet-W64",
        "resnet34": "ResNet34",
    }
    model_label = backbone_labels.get(str(backbone), str(backbone) or "CenterNet")
    epoch_label = f"epoch {epoch}" if epoch is not None else "epoch unavailable"
    cohort = (
        f"{int(number(chain, 'images')):,} radiographs  |  "
        f"{int(number(chain, 'gt_vertebrae')):,} reference vertebrae  |  "
        f"{len(source_names(rows))} source datasets  |  "
        f"{model_label}  |  {epoch_label}  |  {checkpoint}"
    )

    fig = new_page()
    page_title(
        fig,
        "Spine Measurement Readiness - Best Model",
        cohort,
        1,
        source_path,
    )
    add_fracture_readiness_notice(fig)

    grid = fig.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.965,
        top=0.81,
        bottom=0.225,
        width_ratios=[1.0, 1.42, 1.05],
        hspace=0.24,
        wspace=0.20,
    )
    draw_summary_card(
        fig.add_subplot(grid[0, 0]),
        "Detection",
        "Vertebral detection F1",
        f"{number(chain, 'center_f1_0.20d') * 100:.1f}%",
        [
            f"Precision {number(chain, 'center_precision_0.20d') * 100:.1f}%  |  "
            f"Recall {number(chain, 'center_recall_0.20d') * 100:.1f}%",
            "Higher is better",
        ],
        color=CHAIN_COLOR,
    )
    draw_summary_card(
        fig.add_subplot(grid[1, 0]),
        "Detection",
        "Vertebral count MAE",
        f"{number(chain, 'count_mae'):.2f}",
        [
            "vertebrae per radiograph",
            f"{number(chain, 'count_within_one_rate') * 100:.1f}% of radiographs within +/-1",
            "Lower is better",
        ],
        color=DARK_COLOR,
    )
    draw_summary_card(
        fig.add_subplot(grid[0, 1]),
        "Corner localization",
        "Corner localization accuracy",
        f"{number(chain, 'pck_0.10') * 100:.1f}%",
        [
            "Within 10% of the reference vertebral diagonal",
            "among matched vertebrae",
            f"{number(chain, 'end_to_end_pck_0.10') * 100:.1f}% end-to-end when misses count as failures",
        ],
        color=ACCENT_COLOR,
        hero=True,
    )
    draw_summary_card(
        fig.add_subplot(grid[1, 1]),
        "Corner localization",
        "Mean corner error relative to vertebral size",
        f"{number(chain, 'corner_nme_mean') * 100:.2f}%",
        [
            f"Median {number(chain, 'corner_nme_median') * 100:.2f}%  |  "
            f"P95 {number(chain, 'corner_nme_p95') * 100:.2f}%",
            "Matched vertebrae; lower is better",
        ],
        color=ACCENT_COLOR,
    )
    draw_summary_card(
        fig.add_subplot(grid[:, 2]),
        "Curvature measurement",
        "Cobb-angle MAE",
        f"{number(chain, 'cobb_mae_deg'):.2f} degrees",
        [
            f"{number(chain, 'cobb_within_5deg_rate') * 100:.1f}% within 5 degrees absolute error",
            f"{number(chain, 'cobb_within_10deg_rate') * 100:.1f}% within 10 degrees absolute error",
            f"{number(chain, 'cobb_coverage') * 100:.1f}% valid-angle coverage",
            "Lower MAE is better",
        ],
        color=CHAIN_COLOR,
    )

    ax_notes = fig.add_axes([0.045, 0.072, 0.92, 0.105])
    ax_notes.set_facecolor("white")
    ax_notes.set_xticks([])
    ax_notes.set_yticks([])
    for spine in ax_notes.spines.values():
        spine.set_color(GRID_COLOR)
    ax_notes.text(
        0.018,
        0.78,
        "Definitions",
        fontsize=9.5,
        fontweight="bold",
        color=DARK_COLOR,
        va="top",
    )
    notes = (
        "Detection matching uses predicted and reference centers within 20% of the reference "
        "vertebral diagonal.\n"
        "Relative corner error is each TL/TR/BL/BR pixel distance divided by that reference "
        "vertebra's diagonal, then averaged across matched vertebrae."
    )
    ax_notes.text(
        0.018,
        0.55,
        notes,
        fontsize=8.1,
        color=MUTED_COLOR,
        va="top",
        linespacing=1.45,
        wrap=True,
    )
    save_page(
        pdf,
        fig,
        1,
        preview_dir,
        output_png=output_png,
    )


def page_fracture_executive(
    rows: list[dict[str, str]],
    metrics_payload: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    per_instance_rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    identity = checkpoint_label(metrics_payload, evaluation_manifest)
    matched_instances = [
        row
        for row in per_instance_rows
        if row.get("model") == "spine_chain" and row.get("status") == "matched"
    ]
    point_errors = [
        float(row[f"{corner}_error_px"])
        for row in matched_instances
        for corner in ("tl", "tr", "bl", "br")
        if row.get(f"{corner}_error_px") not in ("", None)
    ]
    mean_point_mae = float(np.mean(point_errors))

    fig = new_page()
    page_title(
        fig,
        "Corner landmarking accuracy and measurement readiness",
        f"{identity} | held-out test split | spine-chain output only",
        1,
        source_path,
    )
    grid = fig.add_gridspec(
        3,
        8,
        left=0.045,
        right=0.965,
        top=0.82,
        bottom=0.09,
        height_ratios=[0.68, 0.68, 1.75],
        hspace=0.38,
        wspace=0.32,
    )
    ax_priority = fig.add_subplot(grid[0:2, :4])
    ax_priority.set_facecolor("white")
    ax_priority.set_xticks([])
    ax_priority.set_yticks([])
    for spine in ax_priority.spines.values():
        spine.set_color(GRID_COLOR)
    ax_priority.text(
        0.06,
        0.88,
        "Corner localization accuracy",
        fontsize=14,
        fontweight="bold",
        color=DARK_COLOR,
        va="top",
    )
    ax_priority.text(
        0.06,
        0.57,
        f"{number(chain, 'pck_0.10') * 100:.1f}%",
        fontsize=42,
        fontweight="bold",
        color=ACCENT_COLOR,
        va="center",
    )
    ax_priority.text(
        0.06,
        0.29,
        "Corner falls within 10% of the GT vertebral diagonal\namong matched vertebrae",
        fontsize=10,
        color=MUTED_COLOR,
        va="center",
        linespacing=1.45,
    )
    ax_priority.text(
        0.06,
        0.08,
        f"{int(number(chain, 'images')):,} radiographs  |  "
        f"{int(number(chain, 'gt_vertebrae')):,} GT vertebrae  |  "
        f"{len(source_names(rows))} sources",
        fontsize=8.5,
        color=MUTED_COLOR,
        va="bottom",
    )

    cards = [
        (
            "Overall corner localization success",
            f"{number(chain, 'end_to_end_pck_0.10') * 100:.1f}%",
            "10% tolerance; misses count as failures",
        ),
        (
            "Mean corner error vs vertebral size",
            f"{number(chain, 'corner_nme_mean') * 100:.2f}%",
            "mean corner distance / GT diagonal",
        ),
        ("Mean point MAE", f"{mean_point_mae:.1f} px", "four corners, matched vertebrae"),
        (
            "Valid geometry",
            f"{(1.0 - number(chain, 'invalid_geometry_rate')) * 100:.1f}%",
            "among predicted quadrilaterals",
        ),
    ]
    for index, (title, value, note) in enumerate(cards):
        row = index // 2
        column = 4 + (index % 2) * 2
        draw_card(
            fig.add_subplot(grid[row, column : column + 2]),
            title,
            value,
            note,
            color=ACCENT_COLOR if index in (0, 1, 2) else DARK_COLOR,
        )

    ax_rates = fig.add_subplot(grid[2, :5])
    rate_labels = ["5% tolerance", "10% tolerance", "20% tolerance"]
    positions = np.arange(len(rate_labels))
    width = 0.36
    conditional_bars = ax_rates.bar(
        positions - width / 2,
        [number(chain, f"pck_{threshold}") for threshold in ("0.05", "0.10", "0.20")],
        width,
        color=ACCENT_COLOR,
        label="Matched vertebrae",
    )
    end_to_end_bars = ax_rates.bar(
        positions + width / 2,
        [number(chain, f"end_to_end_pck_{threshold}") for threshold in ("0.05", "0.10", "0.20")],
        width,
        color=CHAIN_COLOR,
        label="All GT vertebrae; misses included",
    )
    ax_rates.set_xticks(positions, rate_labels)
    ax_rates.set_ylabel("Corner accuracy rate")
    ax_rates.set_title("Corner localization accuracy by tolerance", loc="left", fontsize=12)
    ax_rates.legend(frameon=False, fontsize=8, ncols=2)
    rate_axis(ax_rates)
    style_axis(ax_rates)
    add_bar_labels(ax_rates, conditional_bars, percent=True)
    add_bar_labels(ax_rates, end_to_end_bars, percent=True)

    ax_errors = fig.add_subplot(grid[2, 5:])
    nme_fields = ["corner_nme_mean", "corner_nme_median", "corner_nme_p95"]
    nme_bars = ax_errors.bar(
        np.arange(3),
        [number(chain, field) for field in nme_fields],
        color=[ACCENT_COLOR, RAW_COLOR, CHAIN_COLOR],
    )
    ax_errors.axhline(
        0.10,
        color=CAUTION_COLOR,
        linewidth=1.2,
        linestyle="--",
        label="10% relative-error criterion",
    )
    ax_errors.set_xticks(np.arange(3), ["Mean", "Median", "P95"])
    ax_errors.set_ylim(0.0, 0.16)
    error_ticks = np.linspace(0.0, 0.16, 5)
    ax_errors.set_yticks(error_ticks, [f"{value * 100:.0f}%" for value in error_ticks])
    ax_errors.set_ylabel("Error / vertebral diagonal")
    ax_errors.set_title("Corner error relative to vertebral size", loc="left", fontsize=12)
    ax_errors.legend(frameon=False, fontsize=8)
    style_axis(ax_errors)
    add_bar_labels(ax_errors, nme_bars, percent=True)
    add_fracture_readiness_notice(fig)
    save_page(pdf, fig, 1, preview_dir)


def page_fracture_landmarks(
    rows: list[dict[str, str]],
    metrics_payload: dict[str, Any],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    fig = new_page()
    page_title(
        fig,
        "Landmark reliability for geometry measurements",
        "Relative corner error = corner distance / GT vertebral diagonal; lower is better",
        2,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.11,
        right=0.965,
        top=0.82,
        bottom=0.09,
        hspace=0.42,
        wspace=0.28,
    )

    thresholds = ["0.05", "0.10", "0.20"]
    ax_pck = fig.add_subplot(grid[0, 0])
    positions = np.arange(len(thresholds))
    width = 0.36
    conditional = [number(chain, f"pck_{threshold}") for threshold in thresholds]
    end_to_end = [number(chain, f"end_to_end_pck_{threshold}") for threshold in thresholds]
    bars_conditional = ax_pck.bar(
        positions - width / 2,
        conditional,
        width,
        color=RAW_COLOR,
        label="Conditional on matched vertebrae",
    )
    bars_e2e = ax_pck.bar(
        positions + width / 2,
        end_to_end,
        width,
        color=CHAIN_COLOR,
        label="End-to-end including misses",
    )
    ax_pck.set_xticks(positions, [f"{float(value) * 100:.0f}% tolerance" for value in thresholds])
    ax_pck.set_ylabel("Rate")
    ax_pck.set_title("Corner localization accuracy", loc="left", fontsize=12)
    ax_pck.legend(frameon=False, fontsize=8)
    rate_axis(ax_pck)
    style_axis(ax_pck)
    add_bar_labels(ax_pck, bars_conditional, percent=True)
    add_bar_labels(ax_pck, bars_e2e, percent=True)

    ax_nme = fig.add_subplot(grid[0, 1])
    nme_fields = ["corner_nme_mean", "corner_nme_median", "corner_nme_p95"]
    nme_labels = ["Mean", "Median", "P95"]
    bars = ax_nme.bar(
        np.arange(3),
        [number(chain, field) for field in nme_fields],
        color=[ACCENT_COLOR, RAW_COLOR, CHAIN_COLOR],
    )
    mean_ci = confidence_interval(metrics_payload, "spine_chain", "corner_nme_mean")
    if mean_ci is not None:
        mean_value = number(chain, "corner_nme_mean")
        ax_nme.errorbar(
            [0],
            [mean_value],
            yerr=[[mean_value - mean_ci[0]], [mean_ci[1] - mean_value]],
            fmt="none",
            ecolor=DARK_COLOR,
            capsize=5,
            linewidth=1.5,
            label="Mean 95% bootstrap CI",
        )
    ax_nme.axhline(
        0.10,
        color=CAUTION_COLOR,
        linewidth=1.2,
        linestyle="--",
        label="10% relative-error criterion",
    )
    ax_nme.set_xticks(np.arange(3), nme_labels)
    ax_nme.set_ylim(0.0, 0.16)
    error_ticks = np.linspace(0.0, 0.16, 5)
    ax_nme.set_yticks(error_ticks, [f"{value * 100:.0f}%" for value in error_ticks])
    ax_nme.set_ylabel("Error / vertebral diagonal")
    ax_nme.set_title("Relative corner error among matched vertebrae", loc="left", fontsize=12)
    ax_nme.legend(frameon=False, fontsize=8)
    style_axis(ax_nme)
    add_bar_labels(ax_nme, bars, percent=True)

    ax_ci = fig.add_subplot(grid[1, 0])
    ci_fields = [
        ("pck_0.10", "Matched corners"),
        ("end_to_end_pck_0.10", "All GT corners"),
        ("usable_vertebra_recall", "Usable geometry"),
    ]
    ci_values = [number(chain, field) for field, _ in ci_fields]
    ci_ranges = [confidence_interval(metrics_payload, "spine_chain", field) for field, _ in ci_fields]
    y_positions = np.arange(len(ci_fields))
    bars = ax_ci.barh(y_positions, ci_values, color=ACCENT_COLOR, height=0.5)
    for position, value, interval in zip(y_positions, ci_values, ci_ranges):
        if interval is None:
            continue
        ax_ci.errorbar(
            value,
            position,
            xerr=[[value - interval[0]], [interval[1] - value]],
            fmt="none",
            ecolor=DARK_COLOR,
            capsize=4,
            linewidth=1.4,
        )
    ax_ci.set_yticks(y_positions, [label for _, label in ci_fields])
    ax_ci.set_xlim(0.0, 1.04)
    ax_ci.set_xticks(np.linspace(0, 1, 6), [f"{value * 100:.0f}%" for value in np.linspace(0, 1, 6)])
    ax_ci.set_xlabel("Rate with patient-clustered 95% CI")
    ax_ci.set_title("Uncertainty on key readiness rates", loc="left", fontsize=12)
    style_axis(ax_ci, x_grid=True, y_grid=False)
    add_horizontal_rate_labels(ax_ci, bars)

    ax_explain = fig.add_subplot(grid[1, 1])
    ax_explain.set_facecolor("white")
    ax_explain.set_xticks([])
    ax_explain.set_yticks([])
    for spine in ax_explain.spines.values():
        spine.set_color(GRID_COLOR)
    ax_explain.set_title(
        "What 'relative to vertebral size' means",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_explain.text(
        0.06,
        0.78,
        f"{number(chain, 'corner_nme_mean') * 100:.2f}%",
        fontsize=30,
        fontweight="bold",
        color=ACCENT_COLOR,
        va="center",
    )
    explanation = (
        "For each TL/TR/BL/BR landmark:\n"
        "1. Measure the pixel distance between prediction and reference.\n"
        "2. Divide by that GT vertebra's diagonal length.\n"
        "3. Average across corners and matched vertebrae.\n\n"
        "So 6.43% means the average corner miss is 6.43% of the "
        "vertebra's diagonal. Normalization makes differently sized "
        "vertebrae comparable."
    )
    ax_explain.text(
        0.06,
        0.60,
        explanation,
        ha="left",
        va="top",
        fontsize=9.2,
        color=DARK_COLOR,
        linespacing=1.35,
        wrap=True,
    )

    add_fracture_readiness_notice(fig)
    save_page(pdf, fig, 2, preview_dir)


def page_fracture_morphology(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    fig = new_page()
    page_title(
        fig,
        "How corner quality supports morphology measurements",
        "Evaluated corner inputs and pipeline dependencies - not fracture predictions or clinical thresholds",
        3,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.965,
        top=0.82,
        bottom=0.09,
        height_ratios=[1.15, 1.0],
        hspace=0.38,
        wspace=0.28,
    )

    ax_shape = fig.add_subplot(grid[0, 0])
    ax_shape.set_facecolor("white")
    corners = {
        "TL": np.asarray([0.20, 0.78]),
        "TR": np.asarray([0.78, 0.72]),
        "BL": np.asarray([0.27, 0.20]),
        "BR": np.asarray([0.72, 0.25]),
    }
    links = [
        ("TL", "TR", RAW_COLOR, "Superior endplate"),
        ("BL", "BR", RAW_COLOR, "Inferior endplate"),
        ("TL", "BL", CHAIN_COLOR, "Left height"),
        ("TR", "BR", CHAIN_COLOR, "Right height"),
    ]
    for first, second, color, _ in links:
        points = np.vstack([corners[first], corners[second]])
        ax_shape.plot(points[:, 0], points[:, 1], color=color, linewidth=3)
    for label, point in corners.items():
        ax_shape.scatter(*point, s=90, color=ACCENT_COLOR, zorder=3)
        ax_shape.text(point[0], point[1] + 0.055, label, ha="center", fontweight="bold")
    ax_shape.annotate("Left height", (0.21, 0.48), xytext=(0.02, 0.48), color=CHAIN_COLOR, fontsize=10)
    ax_shape.annotate("Right height", (0.75, 0.47), xytext=(0.82, 0.47), color=CHAIN_COLOR, fontsize=10)
    ax_shape.text(0.50, 0.84, "Superior endplate", ha="center", color=RAW_COLOR, fontsize=10)
    ax_shape.text(0.50, 0.11, "Inferior endplate", ha="center", color=RAW_COLOR, fontsize=10)
    ax_shape.text(
        0.50,
        0.48,
        "Four ordered corners\nTL, TR, BL, BR",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=DARK_COLOR,
    )
    ax_shape.set_xlim(-0.08, 1.08)
    ax_shape.set_ylim(0.02, 0.98)
    ax_shape.set_xticks([])
    ax_shape.set_yticks([])
    for spine in ax_shape.spines.values():
        spine.set_color(GRID_COLOR)
    ax_shape.set_title(
        "Four-corner inputs from selected spine-chain vertebrae",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    ax_corner = fig.add_subplot(grid[0, 1])
    corner_names = ["TL", "TR", "BL", "BR"]
    x_fields = [f"{corner.lower()}_dx_normalized_mae" for corner in corner_names]
    y_fields = [f"{corner.lower()}_dy_normalized_mae" for corner in corner_names]
    positions = np.arange(len(corner_names))
    width = 0.36
    bars_x = ax_corner.bar(
        positions - width / 2,
        [number(chain, field) for field in x_fields],
        width,
        color=RAW_COLOR,
        label="x-coordinate MAE",
    )
    bars_y = ax_corner.bar(
        positions + width / 2,
        [number(chain, field) for field in y_fields],
        width,
        color=CHAIN_COLOR,
        label="y-coordinate MAE",
    )
    ax_corner.set_xticks(positions, corner_names)
    ax_corner.set_ylabel("Normalized absolute coordinate error")
    ax_corner.set_title("Constituent corner errors (lower is better)", loc="left", fontsize=12)
    ax_corner.set_ylim(0.0, 0.058)
    ax_corner.legend(frameon=False, fontsize=8, ncols=2, loc="upper center")
    style_axis(ax_corner)
    add_bar_labels(ax_corner, bars_x, decimals=4)
    add_bar_labels(ax_corner, bars_y, decimals=4)

    ax_table = fig.add_subplot(grid[1, :])
    ax_table.axis("off")
    columns = ["Measurement family", "Corner dependency", "Available evidence", "Not established"]
    cells = [
        [
            "Left/right height",
            "TL-BL and TR-BR",
            "Coordinate MAE, accuracy, relative error",
            "Height-measurement agreement or fracture status",
        ],
        [
            "Height asymmetry",
            "Both side heights",
            "All four corner errors",
            "A validated abnormality threshold",
        ],
        [
            "Endplate geometry",
            "TL-TR and BL-BR",
            "Ordered-corner validity and error",
            "Compression-fracture classification",
        ],
        [
            "Neighbor-relative height",
            "Current plus adjacent vertebrae",
            f"Usable recall {number(chain, 'usable_vertebra_recall') * 100:.1f}% and count error",
            "Clinical interpretation of a deviation",
        ],
    ]
    table = ax_table.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.16, 0.19, 0.27, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.8)
        cell.set_facecolor(DARK_COLOR if row == 0 else "white")
        cell.get_text().set_color("white" if row == 0 else DARK_COLOR)
        if row == 0:
            cell.get_text().set_fontweight("bold")
    ax_table.set_title(
        "What the evaluation supports - and what remains unvalidated",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    add_fracture_readiness_notice(fig)
    save_page(pdf, fig, 3, preview_dir)


def page_fracture_cobb_angle(
    rows: list[dict[str, str]],
    per_image_rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    chain_images = [row for row in per_image_rows if row.get("model") == "spine_chain"]
    fig = new_page()
    page_title(
        fig,
        "Cobb-angle measurement performance",
        "Spine-chain predictions compared with reference Cobb angles on 223 held-out radiographs",
        4,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.965,
        top=0.82,
        bottom=0.09,
        hspace=0.42,
        wspace=0.28,
    )

    ax_error = fig.add_subplot(grid[0, 0])
    error_fields = [
        ("cobb_mae_deg", "Mean absolute error"),
        ("cobb_median_absolute_error_deg", "Median absolute error"),
        ("cobb_p95_absolute_error_deg", "P95 absolute error"),
    ]
    error_bars = ax_error.bar(
        np.arange(len(error_fields)),
        [number(chain, field) for field, _ in error_fields],
        color=[ACCENT_COLOR, RAW_COLOR, CHAIN_COLOR],
    )
    ax_error.set_xticks(
        np.arange(len(error_fields)),
        [label for _, label in error_fields],
        rotation=8,
    )
    ax_error.set_ylabel("Degrees")
    ax_error.set_title("Absolute Cobb-angle error (lower is better)", loc="left", fontsize=12)
    ax_error.set_ylim(0.0, number(chain, "cobb_p95_absolute_error_deg") * 1.25)
    style_axis(ax_error)
    add_bar_labels(ax_error, error_bars, decimals=2)

    ax_rates = fig.add_subplot(grid[1, 0])
    rate_fields = [
        ("cobb_at_5deg", "Within 5 degrees"),
        ("cobb_at_10deg", "Within 10 degrees"),
        ("cobb_coverage", "Valid-angle coverage"),
    ]
    rate_bars = ax_rates.bar(
        np.arange(len(rate_fields)),
        [number(chain, field) for field, _ in rate_fields],
        color=ACCENT_COLOR,
    )
    ax_rates.set_xticks(
        np.arange(len(rate_fields)),
        [label for _, label in rate_fields],
        rotation=8,
    )
    ax_rates.set_ylabel("Image rate")
    ax_rates.set_title("Cobb-angle agreement and coverage", loc="left", fontsize=12)
    rate_axis(ax_rates)
    style_axis(ax_rates)
    add_bar_labels(ax_rates, rate_bars, percent=True)

    absolute_errors = finite_values(chain_images, "cobb_absolute_error_deg")
    histogram_edges = np.asarray([0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0])
    histogram_counts, _ = np.histogram(absolute_errors, bins=histogram_edges)
    ax_distribution = fig.add_subplot(grid[0, 1])
    distribution_bars = ax_distribution.bar(
        np.arange(len(histogram_counts)),
        histogram_counts,
        color=CHAIN_COLOR,
    )
    interval_labels = [
        f"{histogram_edges[index]:.0f}-{histogram_edges[index + 1]:.0f}"
        for index in range(len(histogram_counts))
    ]
    ax_distribution.set_xticks(np.arange(len(histogram_counts)), interval_labels)
    ax_distribution.set_xlabel("Absolute error interval (degrees)")
    ax_distribution.set_ylabel("Radiographs")
    ax_distribution.set_title("Distribution of absolute Cobb-angle error", loc="left", fontsize=12)
    style_axis(ax_distribution)
    add_bar_labels(ax_distribution, distribution_bars, decimals=0)

    sources, chain_rows = fracture_source_rows(rows)
    positions = np.arange(len(sources))
    width = 0.36
    ax_source = fig.add_subplot(grid[1, 1])
    source_mean_bars = ax_source.bar(
        positions - width / 2,
        [number(row, "cobb_mae_deg") for row in chain_rows],
        width,
        color=ACCENT_COLOR,
        label="Mean absolute error",
    )
    source_p95_bars = ax_source.bar(
        positions + width / 2,
        [number(row, "cobb_p95_absolute_error_deg") for row in chain_rows],
        width,
        color=CHAIN_COLOR,
        label="P95 absolute error",
    )
    ax_source.set_xticks(positions, [short_source(source) for source in sources])
    ax_source.set_ylabel("Degrees")
    ax_source.set_title("Cobb-angle error by source", loc="left", fontsize=12)
    ax_source.legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
    style_axis(ax_source)
    add_bar_labels(ax_source, source_mean_bars, decimals=2)
    add_bar_labels(ax_source, source_p95_bars, decimals=2)
    add_fracture_readiness_notice(fig)
    save_page(pdf, fig, 4, preview_dir)


def page_fracture_sources(
    rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    sources, chain_rows = fracture_source_rows(rows)
    labels = [short_source(source) for source in sources]
    fig = new_page()
    page_title(
        fig,
        "Cross-source robustness",
        "Spine-chain measurement readiness across BUU, Lumos, MICCAI, Mendeley, and NIH",
        5,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.965,
        top=0.82,
        bottom=0.09,
        hspace=0.42,
        wspace=0.28,
    )

    positions = np.arange(len(sources))
    width = 0.36
    ax_rates = fig.add_subplot(grid[0, 0])
    usable_bars = ax_rates.bar(
        positions - width / 2,
        [number(row, "usable_vertebra_recall") for row in chain_rows],
        width,
        color=ACCENT_COLOR,
        label="Usable recall",
    )
    e2e_bars = ax_rates.bar(
        positions + width / 2,
        [number(row, "end_to_end_pck_0.10") for row in chain_rows],
        width,
        color=CHAIN_COLOR,
        label="Overall corner success (10% tolerance)",
    )
    ax_rates.set_xticks(positions, labels)
    ax_rates.set_ylabel("Rate")
    ax_rates.set_title("Corner measurement readiness by source", loc="left", fontsize=12)
    rate_axis(ax_rates)
    ax_rates.set_ylim(0.0, 1.18)
    ax_rates.legend(frameon=False, fontsize=7.5, ncols=2, loc="upper center")
    style_axis(ax_rates)
    add_bar_labels(ax_rates, usable_bars, percent=True)
    add_bar_labels(ax_rates, e2e_bars, percent=True)

    ax_nme = fig.add_subplot(grid[0, 1])
    mean_bars = ax_nme.bar(
        positions - width / 2,
        [number(row, "corner_nme_mean") for row in chain_rows],
        width,
        color=RAW_COLOR,
        label="Mean relative error",
    )
    p95_bars = ax_nme.bar(
        positions + width / 2,
        [number(row, "corner_nme_p95") for row in chain_rows],
        width,
        color=CHAIN_COLOR,
        label="P95 relative error",
    )
    ax_nme.axhline(
        0.10,
        color=CAUTION_COLOR,
        linewidth=1.2,
        linestyle="--",
        label="10% relative-error criterion",
    )
    ax_nme.set_xticks(positions, labels)
    ax_nme.set_ylim(0.0, 0.19)
    error_ticks = np.linspace(0.0, 0.16, 5)
    ax_nme.set_yticks(error_ticks, [f"{value * 100:.0f}%" for value in error_ticks])
    ax_nme.set_ylabel("Error / vertebral diagonal")
    ax_nme.set_title("Relative corner error by source (lower is better)", loc="left", fontsize=12)
    ax_nme.legend(frameon=False, fontsize=7.3, ncols=3, loc="upper center")
    style_axis(ax_nme)
    add_bar_labels(ax_nme, mean_bars, percent=True)
    add_bar_labels(ax_nme, p95_bars, percent=True)

    ax_support = fig.add_subplot(grid[1, 0])
    support_bars = ax_support.bar(
        positions,
        [number(row, "gt_vertebrae") for row in chain_rows],
        color=ACCENT_COLOR,
    )
    ax_support.set_xticks(positions, labels)
    ax_support.set_ylabel("GT vertebrae")
    ax_support.set_title("Evaluation support by source", loc="left", fontsize=12)
    style_axis(ax_support)
    for bar, row in zip(support_bars, chain_rows):
        ax_support.annotate(
            f"{int(number(row, 'gt_vertebrae')):,} GT\n{int(number(row, 'images'))} images",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK_COLOR,
        )

    ax_completeness = fig.add_subplot(grid[1, 1])
    full_bars = ax_completeness.bar(
        positions - width / 2,
        [number(row, "full_coverage_rate") for row in chain_rows],
        width,
        color=RAW_COLOR,
        label="Full vertebra coverage",
    )
    count_bars = ax_completeness.bar(
        positions + width / 2,
        [number(row, "count_within_one_rate") for row in chain_rows],
        width,
        color=CHAIN_COLOR,
        label="Count within one",
    )
    ax_completeness.set_xticks(positions, labels)
    ax_completeness.set_ylabel("Image rate")
    ax_completeness.set_title("Per-image chain completeness", loc="left", fontsize=12)
    ax_completeness.legend(frameon=False, fontsize=8, ncols=2)
    rate_axis(ax_completeness)
    style_axis(ax_completeness)
    add_bar_labels(ax_completeness, full_bars, percent=True)
    add_bar_labels(ax_completeness, count_bars, percent=True)

    add_fracture_readiness_notice(fig)
    save_page(pdf, fig, 5, preview_dir)


def page_fracture_mae_limitations(
    rows: list[dict[str, str]],
    per_instance_rows: list[dict[str, str]],
    pdf: PdfPages,
    source_path: Path,
    preview_dir: Path | None,
) -> None:
    chain = row_index(rows, model="spine_chain", scope="overall_micro")
    fig = new_page()
    page_title(
        fig,
        "Corner MAE audit and interpretation limits",
        "Pixel and normalized errors for spine-chain TL/TR/BL/BR landmarks",
        6,
        source_path,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.965,
        top=0.82,
        bottom=0.09,
        height_ratios=[1.15, 1.0],
        hspace=0.40,
        wspace=0.28,
    )

    corner_names = ["TL", "TR", "BL", "BR"]
    positions = np.arange(len(corner_names))
    width = 0.36
    ax_pixel = fig.add_subplot(grid[0, 0])
    bars_x = ax_pixel.bar(
        positions - width / 2,
        [number(chain, f"{corner.lower()}_dx_px_mae") for corner in corner_names],
        width,
        color=RAW_COLOR,
        label="x-coordinate MAE",
    )
    bars_y = ax_pixel.bar(
        positions + width / 2,
        [number(chain, f"{corner.lower()}_dy_px_mae") for corner in corner_names],
        width,
        color=CHAIN_COLOR,
        label="y-coordinate MAE",
    )
    ax_pixel.set_xticks(positions, corner_names)
    ax_pixel.set_ylabel("Mean absolute error (pixels)")
    ax_pixel.set_title("Coordinate MAE by corner (lower is better)", loc="left", fontsize=12)
    ax_pixel.set_ylim(0.0, max(bar.get_height() for bar in (*bars_x, *bars_y)) * 1.25)
    ax_pixel.legend(frameon=False, fontsize=8, ncols=2, loc="upper center")
    style_axis(ax_pixel)
    add_bar_labels(ax_pixel, bars_x)
    add_bar_labels(ax_pixel, bars_y)

    ax_bias = fig.add_subplot(grid[0, 1])
    bias_x = ax_bias.bar(
        positions - width / 2,
        [number(chain, f"{corner.lower()}_dx_normalized_bias") for corner in corner_names],
        width,
        color=RAW_COLOR,
        label="x bias",
    )
    bias_y = ax_bias.bar(
        positions + width / 2,
        [number(chain, f"{corner.lower()}_dy_normalized_bias") for corner in corner_names],
        width,
        color=CHAIN_COLOR,
        label="y bias",
    )
    ax_bias.axhline(0.0, color=DARK_COLOR, linewidth=1.0)
    ax_bias.set_xticks(positions, corner_names)
    ax_bias.set_ylabel("Signed normalized error")
    ax_bias.set_title("Coordinate bias: prediction minus GT", loc="left", fontsize=12)
    bias_min = min(bar.get_height() for bar in (*bias_x, *bias_y))
    bias_max = max(bar.get_height() for bar in (*bias_x, *bias_y))
    ax_bias.set_ylim(min(-0.0015, bias_min * 1.6), bias_max * 1.25)
    ax_bias.legend(frameon=False, fontsize=8, ncols=2, loc="upper center")
    style_axis(ax_bias)
    add_bar_labels(ax_bias, bias_x, decimals=4)
    add_bar_labels(ax_bias, bias_y, decimals=4)

    sources = source_names(rows)
    means: list[float] = []
    p95_values: list[float] = []
    for source in sources:
        source_instances = [
            row
            for row in per_instance_rows
            if row.get("model") == "spine_chain"
            and row.get("source_dataset") == source
            and row.get("status") == "matched"
        ]
        errors = [
            float(row[f"{corner}_error_px"])
            for row in source_instances
            for corner in ("tl", "tr", "bl", "br")
            if row.get(f"{corner}_error_px") not in ("", None)
        ]
        means.append(float(np.mean(errors)))
        p95_values.append(float(np.percentile(errors, 95)))
    ax_source = fig.add_subplot(grid[1, 0])
    mean_bars = ax_source.bar(
        np.arange(len(sources)) - width / 2,
        means,
        width,
        color=ACCENT_COLOR,
        label="Mean point MAE",
    )
    p95_bars = ax_source.bar(
        np.arange(len(sources)) + width / 2,
        p95_values,
        width,
        color=CHAIN_COLOR,
        label="P95 point error",
    )
    ax_source.set_xticks(np.arange(len(sources)), [short_source(source) for source in sources])
    ax_source.set_ylabel("Euclidean corner error (pixels)")
    ax_source.set_title("Point-level landmark error by source", loc="left", fontsize=12)
    ax_source.set_ylim(0.0, max(p95_values) * 1.22)
    ax_source.legend(frameon=False, fontsize=8, ncols=2, loc="upper center")
    style_axis(ax_source)
    add_bar_labels(ax_source, mean_bars)
    add_bar_labels(ax_source, p95_bars)

    ax_limits = fig.add_subplot(grid[1, 1])
    ax_limits.set_facecolor("white")
    ax_limits.set_xticks([])
    ax_limits.set_yticks([])
    for spine in ax_limits.spines.values():
        spine.set_color(GRID_COLOR)
    ax_limits.set_title("Interpretation boundary", loc="left", fontsize=12, fontweight="bold")
    limitations = [
        "No fracture labels or validated fracture classifier are present in this evaluation.",
        "Corner MAE does not directly establish height-ratio or fracture-detection accuracy.",
        "Cobb angle evaluates curvature geometry; it is not a vertebral-fracture metric.",
        "No clinical decision threshold or diagnostic category is inferred.",
        "Results describe this held-out dataset and require external clinical validation.",
    ]
    ax_limits.text(
        0.06,
        0.88,
        "\n\n".join(f"- {item}" for item in limitations),
        ha="left",
        va="top",
        fontsize=9.2,
        color=DARK_COLOR,
        wrap=True,
    )
    add_fracture_readiness_notice(fig)
    save_page(pdf, fig, 6, preview_dir)


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_csv.resolve(strict=True)
    output_path = args.output_pdf.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = args.preview_dir.resolve() if args.preview_dir else None
    summary_png_path: Path | None = None
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
        metadata = pdf.infodict()
        if args.focus in {"fracture-readiness", "one-page-summary"}:
            metrics_payload, evaluation_manifest, per_image_rows, per_instance_rows = (
                load_fracture_readiness_inputs(metrics_path)
            )
            checkpoint = Path(
                str(metrics_payload.get("metadata", {}).get("checkpoint", ""))
            ).name
            recommended = evaluation_manifest.get("recommended_checkpoint")
            if recommended and checkpoint != recommended:
                raise ValueError(
                    f"Focused report received {checkpoint}, but the evaluation manifest "
                    f"recommends {recommended}"
                )

        if args.focus == "one-page-summary":
            summary_png_path = output_path.with_suffix(".png")
            page_one_page_summary(
                rows,
                metrics_payload,
                evaluation_manifest,
                pdf,
                metrics_path,
                preview_dir,
                summary_png_path,
            )
            metadata["Title"] = "Spine Measurement Readiness - Best Model"
            metadata["Subject"] = (
                "Spine-chain vertebral detection, corner localization, counting, "
                "and Cobb-angle measurement performance"
            )
            metadata["Keywords"] = (
                "vertebral detection corner localization count MAE Cobb angle "
                "measurement readiness spine-chain non-diagnostic"
            )
        elif args.focus == "fracture-readiness":
            page_fracture_executive(
                rows,
                metrics_payload,
                evaluation_manifest,
                per_instance_rows,
                pdf,
                metrics_path,
                preview_dir,
            )
            page_fracture_landmarks(
                rows,
                metrics_payload,
                pdf,
                metrics_path,
                preview_dir,
            )
            page_fracture_morphology(rows, pdf, metrics_path, preview_dir)
            page_fracture_cobb_angle(
                rows, per_image_rows, pdf, metrics_path, preview_dir
            )
            page_fracture_sources(rows, pdf, metrics_path, preview_dir)
            page_fracture_mae_limitations(
                rows, per_instance_rows, pdf, metrics_path, preview_dir
            )
            metadata["Title"] = "Vertebral Fracture-Readiness Metrics"
            metadata["Subject"] = (
                "Landmark and geometry measurement readiness for vertebral morphology research"
            )
            metadata["Keywords"] = (
                "corner localization accuracy MAE vertebral morphology Cobb angle "
                "measurement readiness spine-chain non-diagnostic"
            )
        else:
            page_executive(rows, pdf, metrics_path, preview_dir)
            page_detection(rows, pdf, metrics_path, preview_dir)
            page_landmarks(rows, pdf, metrics_path, preview_dir)
            page_counting(rows, pdf, metrics_path, preview_dir)
            page_cobb(rows, pdf, metrics_path, preview_dir)
            page_sources(rows, pdf, metrics_path, preview_dir)
            metadata["Title"] = "Evaluation v2 Metrics Dashboard"
            metadata["Subject"] = "Deployed spine CenterNet evaluation visualization"
            metadata["Keywords"] = "detection landmarks counting spine-chain Cobb"

    print(f"saved: {output_path}")
    if summary_png_path is not None:
        print(f"saved: {summary_png_path}")
    if preview_dir is not None:
        print(f"previews: {preview_dir}")


if __name__ == "__main__":
    main()
