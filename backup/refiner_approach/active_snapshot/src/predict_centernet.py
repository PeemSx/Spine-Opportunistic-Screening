from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.centernet_decode import decode_centernet_outputs
from src.evaluation.cobb_angle import CobbResult, calculate_cobb_angles
from src.analysis.vertebral_morphology import (
    MORPHOLOGY_FIELDNAMES,
    extract_chain_morphology,
    morphology_json_payload,
)
from src.models.centernet import SUPPORTED_BACKBONES, build_centernet_model
from src.inference.corner_refiner import (
    RefinerRuntimeConfig,
    load_corner_refiner,
    refine_centernet_prediction,
)
from src.postprocessing.spine_chain import (
    candidates_to_prediction,
    prediction_to_candidates,
    select_spine_chain,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CORNER_ORDER = [0, 1, 3, 2, 0]
CORNER_DISPLAY_MODES = ("no", "point", "cross", "box")
CONFIDENCE_LABEL_MODES = ("all", "none", "low")
OVERLAY_BACKGROUNDS = ("heatmap", "original")
VERTEBRA_POINT_COLORS = (
    "yellow",
    "lime",
    "deepskyblue",
    "magenta",
    "orange",
    "red",
    "cyan",
    "white",
)
COBB_LABEL_PLACEMENTS = ("apex", "panel", "none")
COBB_OVERLAY_STYLES = {
    "Cobb 1": ("yellow", "-"),
    "Cobb 2": ("orange", "--"),
    "Cobb 3": ("magenta", ":"),
}
CSV_FIELDNAMES = [
    "image",
    "rank",
    "score",
    "center_x",
    "center_y",
    "top_left_x",
    "top_left_y",
    "top_right_x",
    "top_right_y",
    "bottom_left_x",
    "bottom_left_y",
    "bottom_right_x",
    "bottom_right_y",
]
COBB_FIELDNAMES = [
    "model",
    "image",
    "original_width",
    "original_height",
    "prediction_count",
    "cobb_valid",
    "vertebra_count",
    "cobb_1_deg",
    "cobb_2_deg",
    "cobb_3_deg",
    "major_line_1_start_x",
    "major_line_1_start_y",
    "major_line_1_end_x",
    "major_line_1_end_y",
    "major_line_1_sorted_index",
    "major_line_1_original_index",
    "major_line_2_start_x",
    "major_line_2_start_y",
    "major_line_2_end_x",
    "major_line_2_end_y",
    "major_line_2_sorted_index",
    "major_line_2_original_index",
]


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_images(source: Path, recursive: bool = True) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source}")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(source)

    iterator = source.rglob("*") if recursive else source.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No image files found under {source}")
    return images


def resize_pad_image(image_rgb: np.ndarray, input_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    original_h, original_w = image_rgb.shape[:2]
    scale = float(input_size) / float(max(original_h, original_w))
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (input_size - resized_w) // 2
    pad_top = (input_size - resized_h) // 2
    padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    padded[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized

    meta = {
        "original_w": original_w,
        "original_h": original_h,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "scale": scale,
        "input_size": input_size,
    }
    return padded, meta


def image_to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0 - 0.5
    image = np.transpose(image, (2, 0, 1))[None]
    return torch.from_numpy(np.ascontiguousarray(image)).to(device)


def map_points_to_original(points: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    mapped = points.copy().astype(np.float32)
    mapped[..., 0] = (mapped[..., 0] - float(meta["pad_left"])) / float(meta["scale"])
    mapped[..., 1] = (mapped[..., 1] - float(meta["pad_top"])) / float(meta["scale"])
    return mapped


def valid_center_mask(centers: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (centers[:, 0] >= 0)
        & (centers[:, 0] < float(meta["original_w"]))
        & (centers[:, 1] >= 0)
        & (centers[:, 1] < float(meta["original_h"]))
    )


def heatmap_to_original(heatmap: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    input_size = int(meta["input_size"])
    heat = cv2.resize(heatmap, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    top = int(meta["pad_top"])
    left = int(meta["pad_left"])
    resized_h = int(meta["resized_h"])
    resized_w = int(meta["resized_w"])
    crop = heat[top : top + resized_h, left : left + resized_w]
    return cv2.resize(crop, (int(meta["original_w"]), int(meta["original_h"])), interpolation=cv2.INTER_CUBIC)


def sorted_valid_cobb_centers(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32)
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        return np.zeros((0, 2), dtype=np.float32)

    finite_mask = np.isfinite(corners).all(axis=(1, 2))
    corners = corners[finite_mask]
    if len(corners) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    centers = corners.mean(axis=1)
    order = np.argsort(centers[:, 1], kind="stable")
    corners = corners[order]
    left_midpoints = (corners[:, 0, :] + corners[:, 2, :]) * 0.5
    right_midpoints = (corners[:, 1, :] + corners[:, 3, :]) * 0.5
    valid_lines = np.linalg.norm(right_midpoints - left_midpoints, axis=1) > 1e-6
    return corners[valid_lines].mean(axis=1)


def cobb_line_midpoint(line: Any) -> np.ndarray:
    start = np.asarray(line.start, dtype=np.float32)
    end = np.asarray(line.end, dtype=np.float32)
    return (start + end) * 0.5


def point_line_distances(points: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> np.ndarray:
    vector = line_end - line_start
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.linalg.norm(points - line_start[None, :], axis=1)
    deltas = points - line_start[None, :]
    cross_values = vector[0] * deltas[:, 1] - vector[1] * deltas[:, 0]
    return np.abs(cross_values) / norm


def cobb_apex_point(lines: tuple[Any, Any], sorted_centers: np.ndarray) -> np.ndarray:
    first_index, second_index = sorted((int(lines[0].sorted_index), int(lines[1].sorted_index)))
    if len(sorted_centers) == 0:
        return (cobb_line_midpoint(lines[0]) + cobb_line_midpoint(lines[1])) * 0.5

    first_index = max(0, min(first_index, len(sorted_centers) - 1))
    second_index = max(0, min(second_index, len(sorted_centers) - 1))
    candidates = sorted_centers[first_index : second_index + 1]
    if len(candidates) == 0:
        return (cobb_line_midpoint(lines[0]) + cobb_line_midpoint(lines[1])) * 0.5

    line_start = cobb_line_midpoint(lines[0])
    line_end = cobb_line_midpoint(lines[1])
    distances = point_line_distances(candidates, line_start, line_end)
    if float(distances.max(initial=0.0)) <= 1e-6:
        return candidates[len(candidates) // 2]
    return candidates[int(np.argmax(distances))]


def cobb_label_position(apex: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    side = -1.0 if float(apex[0]) <= float(width) * 0.5 else 1.0
    position = apex + np.asarray([side * float(width) * 0.14, 0.0], dtype=np.float32)
    margin = max(18.0, max(float(width), float(height)) * 0.012)
    position[0] = float(np.clip(position[0], margin, float(width) - margin))
    position[1] = float(np.clip(position[1], margin, float(height) - margin))
    return position


def diagnostics_for_selected_candidates(
    refiner_diagnostics: list[dict[str, Any]],
    selected_candidates: list[Any],
) -> list[dict[str, Any]]:
    """Return refiner diagnostics for only the candidates retained downstream."""
    diagnostics_by_index = {
        int(item["candidate_index"]): item for item in refiner_diagnostics
    }
    return [
        diagnostics_by_index[int(candidate.index)]
        for candidate in selected_candidates
        if int(candidate.index) in diagnostics_by_index
    ]


def draw_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    show_heatmap: bool,
    show_corners: str,
    show_cobb: bool,
    cobb_result: CobbResult | None,
    cobb_label_placement: str,
    confidence_labels: str,
    low_confidence_thresh: float,
    max_draw: int,
    refiner_diagnostics: list[dict[str, Any]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cobb_measurements = []
    if show_cobb and cobb_result is not None:
        display_index = 1
        for label, angle, lines in cobb_result.measurements():
            if angle is None or not lines:
                continue
            color, linestyle = COBB_OVERLAY_STYLES[label]
            cobb_measurements.append(
                {
                    "display_label": f"Cobb {display_index}",
                    "angle": float(angle),
                    "lines": lines,
                    "color": color,
                    "linestyle": linestyle,
                }
            )
            display_index += 1

    use_cobb_panel = bool(cobb_measurements and cobb_label_placement == "panel")
    if use_cobb_panel:
        fig, (ax, panel_ax) = plt.subplots(
            1,
            2,
            figsize=(10.4, 10),
            gridspec_kw={"width_ratios": [8.0, 1.8]},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 10))
        panel_ax = None

    ax.imshow(image_rgb)
    if show_heatmap:
        ax.imshow(heatmap, cmap="jet", alpha=0.42, vmin=0, vmax=1)

    if refiner_diagnostics:
        for item in refiner_diagnostics[:max_draw]:
            points = np.asarray(item["final_corners_original"], dtype=np.float32)
            color = "deepskyblue"
            polygon = points[CORNER_ORDER]
            ax.plot(
                polygon[:, 0],
                polygon[:, 1],
                color=color,
                linewidth=1.6,
                alpha=0.85,
            )
            label_anchor = points[0]
            ax.text(
                float(label_anchor[0]),
                float(label_anchor[1]),
                f"R{item['candidate_index'] + 1} refined",
                color="white",
                fontsize=5,
                ha="left",
                va="bottom",
                bbox={"facecolor": color, "alpha": 0.65, "pad": 0.8, "linewidth": 0},
            )

    centers = predictions["centers"][:max_draw]
    scores = predictions["scores"][:max_draw]
    corners = predictions["corners"][:max_draw]

    if len(centers) > 0:
        if confidence_labels == "all":
            label_mask = np.ones(len(centers), dtype=bool)
        elif confidence_labels == "low":
            label_mask = scores < float(low_confidence_thresh)
        else:
            label_mask = np.zeros(len(centers), dtype=bool)

        visible_centers = centers[label_mask]
        if len(visible_centers) > 0:
            ax.scatter(
                visible_centers[:, 0],
                visible_centers[:, 1],
                s=18,
                c="cyan",
                edgecolors="black",
                linewidths=0.35,
            )
        for index, (center, score) in enumerate(zip(centers, scores), start=1):
            if not bool(label_mask[index - 1]):
                continue
            ax.text(
                float(center[0]),
                float(center[1]),
                f"{index}:{score:.2f}",
                color="white",
                fontsize=6,
                ha="left",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "linewidth": 0},
            )

    if show_corners == "box":
        for points in corners:
            polygon = points[CORNER_ORDER]
            ax.plot(polygon[:, 0], polygon[:, 1], color="lime", linewidth=0.9, alpha=0.9)
    elif show_corners == "point":
        for vertebra_index, points in enumerate(corners):
            color = VERTEBRA_POINT_COLORS[vertebra_index % len(VERTEBRA_POINT_COLORS)]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=9,
                c=color,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.95,
            )
    elif show_corners == "cross":
        for points in corners:
            for start_index, end_index in ((0, 3), (1, 2)):
                start = points[start_index]
                end = points[end_index]
                ax.annotate(
                    "",
                    xy=(float(end[0]), float(end[1])),
                    xytext=(float(start[0]), float(start[1])),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "lime",
                        "linewidth": 0.9,
                        "alpha": 0.9,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )

    if cobb_measurements:
        sorted_centers = sorted_valid_cobb_centers(predictions["corners"])
        for measurement in cobb_measurements:
            color = measurement["color"]
            linestyle = measurement["linestyle"]
            for line in measurement["lines"]:
                start_x, start_y = line.start
                end_x, end_y = line.end
                ax.plot(
                    [start_x, end_x],
                    [start_y, end_y],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.0,
                    alpha=0.95,
                )
                ax.scatter([start_x, end_x], [start_y, end_y], s=18, c=color, edgecolors="black", linewidths=0.35)
            if cobb_label_placement == "apex":
                apex = cobb_apex_point(measurement["lines"], sorted_centers)
                label_position = cobb_label_position(apex, image_rgb.shape)
                ax.plot(
                    [float(apex[0]), float(label_position[0])],
                    [float(apex[1]), float(label_position[1])],
                    color=color,
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax.text(
                    float(label_position[0]),
                    float(label_position[1]),
                    f"{measurement['display_label']}: {measurement['angle']:.1f}",
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox={"facecolor": color, "alpha": 0.78, "pad": 2, "linewidth": 0},
                )

        if use_cobb_panel and panel_ax is not None:
            panel_ax.set_facecolor("#f3f4f6")
            panel_ax.set_xlim(0, 1)
            panel_ax.set_ylim(0, 1)
            panel_ax.axis("off")
            panel_ax.text(0.08, 0.96, "Cobb angles", fontsize=10, fontweight="bold", ha="left", va="top")
            for index, measurement in enumerate(cobb_measurements):
                y_coord = 0.86 - index * 0.12
                color = measurement["color"]
                linestyle = measurement["linestyle"]
                panel_ax.plot([0.08, 0.28], [y_coord, y_coord], color=color, linestyle=linestyle, linewidth=2.4)
                panel_ax.scatter([0.08, 0.28], [y_coord, y_coord], s=20, c=color, edgecolors="black", linewidths=0.35)
                panel_ax.text(
                    0.35,
                    y_coord,
                    f"{measurement['display_label']}: {measurement['angle']:.1f}",
                    fontsize=9,
                    ha="left",
                    va="center",
                    color="black",
                )

    ax.set_title(f"predicted centers: {len(predictions['centers'])}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prediction_rows(image_path: Path, prediction: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for rank, (score, center, corners) in enumerate(
        zip(prediction["scores"], prediction["centers"], prediction["corners"]),
        start=1,
    ):
        rows.append(
            {
                "image": str(image_path),
                "rank": rank,
                "score": round(float(score), 6),
                "center_x": round(float(center[0]), 3),
                "center_y": round(float(center[1]), 3),
                "top_left_x": round(float(corners[0, 0]), 3),
                "top_left_y": round(float(corners[0, 1]), 3),
                "top_right_x": round(float(corners[1, 0]), 3),
                "top_right_y": round(float(corners[1, 1]), 3),
                "bottom_left_x": round(float(corners[2, 0]), 3),
                "bottom_left_y": round(float(corners[2, 1]), 3),
                "bottom_right_x": round(float(corners[3, 0]), 3),
                "bottom_right_y": round(float(corners[3, 1]), 3),
            }
        )
    return rows


def rounded_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def add_cobb_line_fields(row: dict[str, Any], result: CobbResult, prefix: str, line_index: int) -> None:
    if result.major_lines is None or line_index >= len(result.major_lines):
        row.update(
            {
                f"{prefix}_start_x": None,
                f"{prefix}_start_y": None,
                f"{prefix}_end_x": None,
                f"{prefix}_end_y": None,
                f"{prefix}_sorted_index": None,
                f"{prefix}_original_index": None,
            }
        )
        return

    line = result.major_lines[line_index]
    row.update(
        {
            f"{prefix}_start_x": round(float(line.start[0]), 3),
            f"{prefix}_start_y": round(float(line.start[1]), 3),
            f"{prefix}_end_x": round(float(line.end[0]), 3),
            f"{prefix}_end_y": round(float(line.end[1]), 3),
            f"{prefix}_sorted_index": int(line.sorted_index),
            f"{prefix}_original_index": int(line.original_index),
        }
    )


def cobb_output(
    image_path: Path,
    meta: dict[str, Any],
    model_name: str,
    prediction: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], CobbResult]:
    result = calculate_cobb_angles(prediction["corners"])
    row: dict[str, Any] = {
        "model": model_name,
        "image": str(image_path),
        "original_width": meta["original_w"],
        "original_height": meta["original_h"],
        "prediction_count": len(prediction["centers"]),
        "cobb_valid": bool(result.valid),
        "vertebra_count": int(result.vertebra_count),
        "cobb_1_deg": rounded_or_none(result.cobb_1_deg),
        "cobb_2_deg": rounded_or_none(result.cobb_2_deg),
        "cobb_3_deg": rounded_or_none(result.cobb_3_deg),
    }
    add_cobb_line_fields(row, result, "major_line_1", 0)
    add_cobb_line_fields(row, result, "major_line_2", 1)
    payload = {
        "model": model_name,
        "image": str(image_path),
        "original_width": meta["original_w"],
        "original_height": meta["original_h"],
        "prediction_count": len(prediction["centers"]),
        "cobb": result.to_dict(),
    }
    return row, payload, result


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    input_size: int,
    down_ratio: int,
    peak_thresh: float,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    padded_rgb, meta = resize_pad_image(image_rgb, input_size=input_size)
    tensor = image_to_tensor(padded_rgb, device)

    outputs = model(tensor)
    decoded = decode_centernet_outputs(outputs, down_ratio=down_ratio, peak_thresh=peak_thresh, topk=topk)[0]

    centers = map_points_to_original(decoded["centers"], meta)
    corners = map_points_to_original(decoded["corners"], meta)
    mask = valid_center_mask(centers, meta)
    prediction = {
        "scores": decoded["scores"][mask],
        "centers": centers[mask],
        "corners": corners[mask],
    }

    heatmap = outputs["hm"][0, 0].detach().cpu().numpy()
    heatmap_original = heatmap_to_original(heatmap, meta)
    return image_rgb, heatmap_original, prediction, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CenterNet vertebra center prediction.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/centernet_predictions"))
    parser.add_argument("--backbone", type=str, default=None, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--down-ratio", type=int, default=None)
    parser.add_argument("--peak-thresh", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument(
        "--refiner-checkpoint",
        type=Path,
        default=None,
        help="Optional single-crop corner refiner; its landmarks replace CenterNet corners.",
    )
    parser.add_argument("--refiner-batch-size", type=int, default=16)
    parser.add_argument("--refiner-crop-scale", type=float, default=None)
    parser.add_argument("--refiner-min-base-side", type=float, default=8.0)
    parser.add_argument("--refiner-no-amp", action="store_true")
    parser.add_argument(
        "--show-refiner-boxes",
        "--show-refiner-rois",
        dest="show_refiner_rois",
        action="store_true",
        help=(
            "Draw final refined vertebral quadrilaterals "
            "for spine-chain-selected candidates only. --show-refiner-rois is retained "
            "as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--save-morphology",
        action="store_true",
        help="Write measurement-only vertebral morphology CSV/JSON outputs.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument(
        "--overlay-background",
        type=str,
        default="heatmap",
        choices=OVERLAY_BACKGROUNDS,
        help="Use heatmap-blended overlay or original image with annotations only.",
    )
    parser.add_argument("--no-heatmap", action="store_true", help="Alias for --overlay-background original.")
    parser.add_argument(
        "--confidence-labels",
        type=str,
        default="all",
        choices=CONFIDENCE_LABEL_MODES,
        help="Center confidence label overlay mode: all labels, no labels, or only low-confidence labels.",
    )
    parser.add_argument(
        "--low-confidence-thresh",
        type=float,
        default=0.30,
        help="Confidence threshold used by --confidence-labels low.",
    )
    parser.add_argument(
        "--show-corners",
        nargs="?",
        const="cross",
        default="cross",
        choices=CORNER_DISPLAY_MODES,
        help="Corner overlay mode: no corners, corner points only, diagonal cross arrows, or full box.",
    )
    parser.set_defaults(spine_chain=True)
    parser.add_argument("--spine-chain", dest="spine_chain", action="store_true", help="Use spine-chain predictions.")
    parser.add_argument("--raw", dest="spine_chain", action="store_false", help="Show raw predictions instead of spine-chain predictions.")
    parser.set_defaults(show_cobb=True)
    parser.add_argument("--show-cobb", dest="show_cobb", action="store_true", help="Draw selected lines for non-null Cobb angles.")
    parser.add_argument("--no-show-cobb", dest="show_cobb", action="store_false", help="Hide Cobb angle overlay lines.")
    parser.add_argument(
        "--cobb-label-placement",
        type=str,
        default="apex",
        choices=COBB_LABEL_PLACEMENTS,
        help="Where to draw Cobb angle labels.",
    )
    parser.set_defaults(no_save_tables=True)
    parser.add_argument("--no-save-tables", dest="no_save_tables", action="store_true", help="Skip CSV/JSON outputs and only write overlays.")
    parser.add_argument("--save-tables", dest="no_save_tables", action="store_false", help="Write CSV/JSON outputs.")
    parser.add_argument("--chain-duplicate-iou", type=float, default=0.18)
    parser.add_argument("--chain-duplicate-center-scale", type=float, default=0.35)
    parser.add_argument("--chain-score-thresh", type=float, default=0.18)
    parser.add_argument("--chain-score-weight", type=float, default=3.0)
    parser.add_argument("--chain-min-len", type=int, default=3)
    parser.add_argument("--max-draw", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    train_args = checkpoint.get("args", {})
    backbone = args.backbone or train_args.get("backbone", "hrnet_w18")
    input_size = int(args.input_size or train_args.get("input_size", 1024))
    down_ratio = int(args.down_ratio or train_args.get("down_ratio", 4))
    peak_thresh = float(args.peak_thresh if args.peak_thresh is not None else train_args.get("peak_thresh", 0.05))
    topk = int(args.topk or train_args.get("eval_topk", 100))
    show_heatmap = args.overlay_background == "heatmap" and not args.no_heatmap

    model = build_centernet_model(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    refiner_model: torch.nn.Module | None = None
    refiner_config: RefinerRuntimeConfig | None = None
    if args.refiner_checkpoint is not None:
        refiner_model, _, checkpoint_runtime = load_corner_refiner(args.refiner_checkpoint, device)
        refiner_config = replace(
            checkpoint_runtime,
            crop_scale=float(
                args.refiner_crop_scale
                if args.refiner_crop_scale is not None
                else checkpoint_runtime.crop_scale
            ),
            batch_size=int(args.refiner_batch_size),
            min_base_side=float(args.refiner_min_base_side),
        )

    images = resolve_images(args.source, recursive=args.recursive)
    overlays_dir = args.output_dir / "overlays"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_chain_rows: list[dict[str, Any]] = []
    cobb_rows: list[dict[str, Any]] = []
    json_output: list[dict[str, Any]] = []
    chain_json_output: list[dict[str, Any]] = []
    cobb_json_output: list[dict[str, Any]] = []
    refiner_json_output: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    morphology_json_output: list[dict[str, Any]] = []
    print(f"device: {device}")
    print(f"backbone: {backbone}")
    print(f"images: {len(images)}")
    print(f"output dir: {args.output_dir}")
    if args.spine_chain:
        print("spine chain: enabled")
    if refiner_model is not None and refiner_config is not None:
        print(f"corner refiner: {args.refiner_checkpoint}")
        print(
            f"refiner ROI scale: {refiner_config.crop_scale} | "
            "policy: single crop, unconditional refinement"
        )
    if args.no_save_tables:
        print("table outputs: disabled")

    for image_index, image_path in enumerate(images, start=1):
        image_rgb, heatmap, prediction, meta = predict_image(
            model=model,
            image_path=image_path,
            device=device,
            input_size=input_size,
            down_ratio=down_ratio,
            peak_thresh=peak_thresh,
            topk=topk,
        )
        refiner_diagnostics: list[dict[str, Any]] = []
        if refiner_model is not None and refiner_config is not None:
            prediction, refiner_diagnostics = refine_centernet_prediction(
                model=refiner_model,
                image_rgb=image_rgb,
                prediction=prediction,
                device=device,
                config=refiner_config,
                use_amp=not args.refiner_no_amp,
            )
            refiner_json_output.append(
                {
                    "image": str(image_path),
                    "policy": "single_crop_unconditional",
                    "candidate_count": len(refiner_diagnostics),
                    "refined_count": len(refiner_diagnostics),
                    "candidates": refiner_diagnostics,
                }
            )
        if args.no_save_tables:
            raw_cobb = calculate_cobb_angles(prediction["corners"])
        else:
            rows = prediction_rows(image_path, prediction)
            raw_cobb_row, raw_cobb_payload, raw_cobb = cobb_output(image_path, meta, "raw", prediction)
            cobb_rows.append(raw_cobb_row)
            cobb_json_output.append(raw_cobb_payload)
            all_rows.extend(rows)
            json_output.append(
                {
                    "image": str(image_path),
                    "original_width": meta["original_w"],
                    "original_height": meta["original_h"],
                    "prediction_count": len(prediction["centers"]),
                    "cobb": raw_cobb.to_dict(),
                    "predictions": rows,
                }
            )

        overlay_prediction = prediction
        overlay_cobb = raw_cobb
        raw_candidates = prediction_to_candidates(prediction)
        selected_candidates = raw_candidates
        chain_count: int | None = None
        if args.spine_chain:
            _, chain_candidates, chain_debug = select_spine_chain(
                raw_candidates,
                duplicate_iou_threshold=args.chain_duplicate_iou,
                duplicate_center_scale=args.chain_duplicate_center_scale,
                score_threshold=args.chain_score_thresh,
                score_weight=args.chain_score_weight,
                min_chain_len=args.chain_min_len,
            )
            chain_prediction = candidates_to_prediction(chain_candidates)
            if args.no_save_tables:
                chain_cobb = calculate_cobb_angles(chain_prediction["corners"])
            else:
                chain_rows = prediction_rows(image_path, chain_prediction)
                chain_cobb_row, chain_cobb_payload, chain_cobb = cobb_output(
                    image_path,
                    meta,
                    "spine_chain",
                    chain_prediction,
                )
                cobb_rows.append(chain_cobb_row)
                cobb_json_output.append(chain_cobb_payload)
                all_chain_rows.extend(chain_rows)
            chain_count = len(chain_prediction["centers"])
            if not args.no_save_tables:
                chain_json_output.append(
                    {
                        "image": str(image_path),
                        "original_width": meta["original_w"],
                        "original_height": meta["original_h"],
                        "prediction_count": chain_count,
                        "raw_prediction_count": len(prediction["centers"]),
                        "spine_chain": chain_debug,
                        "cobb": chain_cobb.to_dict(),
                        "predictions": chain_rows,
                    }
                )
            overlay_prediction = chain_prediction
            overlay_cobb = chain_cobb
            selected_candidates = chain_candidates

        if args.save_morphology or not args.no_save_tables:
            image_morphology = extract_chain_morphology(
                str(image_path),
                selected_candidates,
                refiner_diagnostics,
            )
            morphology_rows.extend(image_morphology)
            morphology_json_output.append(
                morphology_json_payload(str(image_path), image_morphology)
            )

        overlay_refiner_diagnostics = None
        if args.show_refiner_rois:
            overlay_refiner_diagnostics = diagnostics_for_selected_candidates(
                refiner_diagnostics,
                selected_candidates,
            )

        output_name = f"{image_index:04d}_{image_path.stem}.png"
        draw_overlay(
            image_rgb=image_rgb,
            heatmap=heatmap,
            predictions=overlay_prediction,
            output_path=overlays_dir / output_name,
            show_heatmap=show_heatmap,
            show_corners=args.show_corners,
            show_cobb=args.show_cobb,
            cobb_result=overlay_cobb,
            cobb_label_placement=args.cobb_label_placement,
            confidence_labels=args.confidence_labels,
            low_confidence_thresh=args.low_confidence_thresh,
            max_draw=args.max_draw,
            refiner_diagnostics=overlay_refiner_diagnostics,
        )
        if args.spine_chain:
            refiner_text = ""
            if refiner_diagnostics:
                refiner_text = f" | refiner {len(refiner_diagnostics)} refined"
            print(
                f"[{image_index}/{len(images)}] {image_path.name}: "
                f"{len(prediction['centers'])} raw -> {chain_count} chain centers{refiner_text}"
            )
        else:
            refiner_text = ""
            if refiner_diagnostics:
                refiner_text = f" | refiner {len(refiner_diagnostics)} refined"
            print(
                f"[{image_index}/{len(images)}] {image_path.name}: "
                f"{len(prediction['centers'])} centers{refiner_text}"
            )

    if args.no_save_tables:
        print("skipped CSV/JSON outputs")
    else:
        csv_path = args.output_dir / "predictions.csv"
        json_path = args.output_dir / "predictions.json"
        cobb_csv_path = args.output_dir / "cobb_angles.csv"
        cobb_json_path = args.output_dir / "cobb_angles.json"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(all_rows)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(json_output, file, indent=2)
        with cobb_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COBB_FIELDNAMES)
            writer.writeheader()
            writer.writerows(cobb_rows)
        with cobb_json_path.open("w", encoding="utf-8") as file:
            json.dump(cobb_json_output, file, indent=2)

        print("saved:", csv_path)
        print("saved:", json_path)
        print("saved:", cobb_csv_path)
        print("saved:", cobb_json_path)
        if args.spine_chain:
            chain_csv_path = args.output_dir / "chain_predictions.csv"
            chain_json_path = args.output_dir / "chain_predictions.json"
            with chain_csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(all_chain_rows)
            with chain_json_path.open("w", encoding="utf-8") as file:
                json.dump(chain_json_output, file, indent=2)
            print("saved:", chain_csv_path)
            print("saved:", chain_json_path)
    if args.save_morphology or not args.no_save_tables:
        morphology_csv_path = args.output_dir / "morphology_features.csv"
        morphology_json_path = args.output_dir / "morphology_features.json"
        with morphology_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MORPHOLOGY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(morphology_rows)
        with morphology_json_path.open("w", encoding="utf-8") as file:
            json.dump(morphology_json_output, file, indent=2, allow_nan=False)
        print("saved:", morphology_csv_path)
        print("saved:", morphology_json_path)
    if refiner_model is not None:
        refiner_json_path = args.output_dir / "refiner_diagnostics.json"
        with refiner_json_path.open("w", encoding="utf-8") as file:
            json.dump(refiner_json_output, file, indent=2)
        print("saved:", refiner_json_path)
    print("saved overlays:", overlays_dir)


if __name__ == "__main__":
    main()
