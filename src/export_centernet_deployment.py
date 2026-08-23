from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.evaluate_centernet import (
    DEPLOYED_ARTIFACT_TYPE,
    DEPLOYED_OUTPUT_CONTRACT,
    DEPLOYED_PREPROCESSING_CONTRACT,
    file_sha256,
    load_checkpoint,
)
from src.evaluation.config import DEFAULT_PEAK_THRESHOLD
from src.models.centernet import build_centernet_model


DEPLOYED_FORMAT_VERSION = 1
DEPLOYED_POSTPROCESSING_CONTRACT = {
    "spine_chain_enabled": True,
    "duplicate_iou_threshold": 0.18,
    "duplicate_center_scale": 0.35,
    "score_threshold": 0.18,
    "score_weight": 3.0,
    "min_chain_len": 3,
}

INFERENCE_ARGUMENT_DEFAULTS = {
    "backbone": "hrnet_w18",
    "input_size": 1024,
    "down_ratio": 4,
    "max_objects": 64,
    "peak_thresh": DEFAULT_PEAK_THRESHOLD,
    "eval_topk": 50,
    "hm_weight": 1.0,
    "reg_weight": 1.0,
    "wh_weight": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a compact CenterNet inference artifact from a full training checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output if it already exists.",
    )
    return parser.parse_args()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint field {name!r} must be a mapping")
    return value


def _coerce_inference_args(train_args: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, default in INFERENCE_ARGUMENT_DEFAULTS.items():
        value = train_args.get(name, default)
        if name == "backbone":
            result[name] = str(value)
        elif name in {"input_size", "down_ratio", "max_objects", "eval_topk"}:
            result[name] = int(value)
        else:
            result[name] = float(value)

    if not np.isclose(result["peak_thresh"], DEFAULT_PEAK_THRESHOLD):
        raise ValueError(
            "Deployed artifacts require peak_thresh "
            f"{DEFAULT_PEAK_THRESHOLD:.2f}, got {result['peak_thresh']:.6f}"
        )
    return result


def build_deployment_artifact(
    source_checkpoint: Mapping[str, Any],
    *,
    source_checkpoint_path: Path,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    model_state_dict = _mapping(
        source_checkpoint.get("model_state_dict"),
        "model_state_dict",
    )
    train_args = _mapping(source_checkpoint.get("args", {}), "args")
    inference_args = _coerce_inference_args(train_args)

    provenance: dict[str, Any] = {
        "source_checkpoint": source_checkpoint_path.name,
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "source_epoch": (
            int(source_checkpoint["epoch"])
            if source_checkpoint.get("epoch") is not None
            else None
        ),
        "experiment_name": str(train_args.get("experiment_name", "")),
    }
    dataset_provenance = source_checkpoint.get("dataset_provenance")
    if isinstance(dataset_provenance, Mapping):
        if dataset_provenance.get("dataset_version") is not None:
            provenance["dataset_version"] = str(dataset_provenance["dataset_version"])
        if dataset_provenance.get("fingerprint") is not None:
            provenance["dataset_fingerprint"] = str(dataset_provenance["fingerprint"])

    return {
        "artifact_type": DEPLOYED_ARTIFACT_TYPE,
        "format_version": DEPLOYED_FORMAT_VERSION,
        "model_state_dict": dict(model_state_dict),
        "args": inference_args,
        "preprocessing": dict(DEPLOYED_PREPROCESSING_CONTRACT),
        "output_contract": {
            key: list(value) if isinstance(value, list) else value
            for key, value in DEPLOYED_OUTPUT_CONTRACT.items()
        },
        "postprocessing": dict(DEPLOYED_POSTPROCESSING_CONTRACT),
        "provenance": provenance,
    }


def validate_model_state(artifact: Mapping[str, Any]) -> None:
    inference_args = _mapping(artifact.get("args"), "args")
    model_state_dict = _mapping(artifact.get("model_state_dict"), "model_state_dict")
    model = build_centernet_model(
        backbone=str(inference_args["backbone"]),
        pretrained=False,
    )
    model.load_state_dict(model_state_dict, strict=True)


def export_checkpoint(checkpoint_path: Path, output_path: Path, *, force: bool) -> Path:
    checkpoint_path = checkpoint_path.resolve(strict=True)
    output_path = output_path.resolve()
    if checkpoint_path == output_path:
        raise ValueError("The compact output must differ from the source checkpoint")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --force to replace it."
        )

    source_sha256 = file_sha256(checkpoint_path)
    source_checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    artifact = build_deployment_artifact(
        source_checkpoint,
        source_checkpoint_path=checkpoint_path,
        source_checkpoint_sha256=source_sha256,
    )
    validate_model_state(artifact)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        torch.save(artifact, temporary_path)
        reloaded = load_checkpoint(temporary_path, torch.device("cpu"))
        if reloaded.get("artifact_type") != DEPLOYED_ARTIFACT_TYPE:
            raise ValueError("Saved artifact failed its type round-trip check")
        validate_model_state(reloaded)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def main() -> None:
    args = parse_args()
    source_size = args.checkpoint.resolve(strict=True).stat().st_size
    output_path = export_checkpoint(args.checkpoint, args.output, force=args.force)
    output_size = output_path.stat().st_size
    print(f"Compact artifact: {output_path}")
    print(f"Source size: {source_size / (1024 ** 2):.2f} MiB")
    print(f"Compact size: {output_size / (1024 ** 2):.2f} MiB")
    print(f"Reduction: {(1.0 - output_size / source_size) * 100.0:.2f}%")


if __name__ == "__main__":
    main()
