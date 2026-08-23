from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROVENANCE_SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def build_dataset_provenance(dataset_root: Path | str) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    relative_paths = [Path("split_summary.json")]
    relative_paths.extend(
        Path(split) / "_annotations.keypoints.coco.json" for split in SPLITS
    )

    file_hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset provenance file is missing: {path}")
        relative_name = relative_path.as_posix()
        file_hash = _sha256_file(path)
        file_hashes[relative_name] = file_hash
        combined.update(relative_name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(file_hash))

    summary = _load_json(root / "split_summary.json")
    split_counts: dict[str, dict[str, int]] = {}
    category_schemas: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        coco = _load_json(root / split / "_annotations.keypoints.coco.json")
        split_counts[split] = {
            "images": len(coco.get("images", [])),
            "annotations": len(coco.get("annotations", [])),
        }
        category_schemas[split] = list(coco.get("categories", []))

    first_schema = category_schemas["train"]
    if any(category_schemas[split] != first_schema for split in SPLITS[1:]):
        raise ValueError("Dataset category schemas differ across train/val/test")

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "fingerprint": combined.hexdigest(),
        "dataset_version": str(summary.get("version") or "unknown"),
        "split_counts": split_counts,
        "categories": first_schema,
        "file_sha256": file_hashes,
    }


def validate_resume_dataset(
    checkpoint: dict[str, Any],
    current_provenance: dict[str, Any],
    *,
    allow_unsafe_resume: bool = False,
) -> None:
    if allow_unsafe_resume:
        return
    checkpoint_provenance = checkpoint.get("dataset_provenance")
    if not isinstance(checkpoint_provenance, dict):
        raise RuntimeError(
            "Resume checkpoint has no dataset provenance. Start a new experiment or pass "
            "--allow-unsafe-resume only after manually verifying the dataset."
        )
    checkpoint_fingerprint = str(checkpoint_provenance.get("fingerprint") or "")
    current_fingerprint = str(current_provenance.get("fingerprint") or "")
    if checkpoint_fingerprint != current_fingerprint:
        raise RuntimeError(
            "Resume checkpoint dataset mismatch: checkpoint "
            f"{checkpoint_provenance.get('dataset_version', 'unknown')} "
            f"({checkpoint_fingerprint[:12]}) != current "
            f"{current_provenance.get('dataset_version', 'unknown')} "
            f"({current_fingerprint[:12]}). Use a new experiment name."
        )
