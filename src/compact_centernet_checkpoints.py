from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.evaluate_centernet import file_sha256, load_checkpoint
from src.export_centernet_deployment import (
    build_deployment_artifact,
    export_checkpoint,
)


SCHEMA_VERSION = 1
MANIFEST_NAME = "compaction_manifest.json"
ALIASES_NAME = "checkpoint_aliases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one compact CenterNet inference artifact per unique model state."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-output-root",
        type=Path,
        default=None,
        help=(
            "Final logical output path recorded in the manifest when building in "
            "a staging directory."
        ),
    )
    parser.add_argument(
        "--representatives",
        nargs="+",
        required=True,
        help="Checkpoint filenames to retain, exactly one per unique state.",
    )
    return parser.parse_args()


def _update_sized(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def model_state_sha256(state: Mapping[str, Any]) -> tuple[str, int]:
    digest = hashlib.sha256()
    tensor_bytes = 0
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"Model state entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        _update_sized(digest, str(name).encode("utf-8"))
        _update_sized(digest, str(value.dtype).encode("ascii"))
        _update_sized(digest, json.dumps(list(value.shape)).encode("ascii"))
        _update_sized(digest, raw)
        tensor_bytes += len(raw)
    return digest.hexdigest(), tensor_bytes


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint field {name!r} must be a mapping")
    return value


def _inference_contract(
    checkpoint: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    artifact = build_deployment_artifact(
        checkpoint,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha256,
    )
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"model_state_dict", "provenance"}
    }


def _write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _scan_checkpoint(path: Path) -> dict[str, Any]:
    source_sha256 = file_sha256(path)
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    state = _mapping(checkpoint.get("model_state_dict"), "model_state_dict")
    state_sha256, tensor_bytes = model_state_sha256(state)
    provenance = checkpoint.get("dataset_provenance")
    dataset_fingerprint = (
        str(provenance.get("fingerprint"))
        if isinstance(provenance, Mapping) and provenance.get("fingerprint") is not None
        else None
    )
    retained_source_fields = {
        "model_state_dict",
        "args",
        "dataset_provenance",
        "epoch",
    }
    return {
        "checkpoint": path.name,
        "source_path": str(path),
        "source_sha256": source_sha256,
        "source_bytes": path.stat().st_size,
        "source_epoch": (
            int(checkpoint["epoch"])
            if checkpoint.get("epoch") is not None
            else None
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "state_sha256": state_sha256,
        "tensor_bytes": tensor_bytes,
        "source_fields": sorted(checkpoint),
        "removed_top_level_fields": sorted(set(checkpoint) - retained_source_fields),
        "inference_contract": _inference_contract(
            checkpoint,
            source_path=path,
            source_sha256=source_sha256,
        ),
    }


def _validate_roots(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    input_root = input_root.resolve(strict=True)
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if input_root == output_root:
        raise ValueError("Input and output roots must differ")
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"Output root is not empty: {output_root}")
        raise FileExistsError(f"Output root already exists: {output_root}")
    return input_root, output_root


def _group_sources(scans: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for scan in scans:
        groups.setdefault(str(scan["state_sha256"]), []).append(scan)
    for state_sha256, group in groups.items():
        reference_contract = group[0]["inference_contract"]
        for scan in group[1:]:
            if scan["inference_contract"] != reference_contract:
                names = sorted(item["checkpoint"] for item in group)
                raise ValueError(
                    "Identical model state has conflicting inference contracts: "
                    f"{state_sha256} ({names})"
                )
    return groups


def _select_representatives(
    scans: Sequence[dict[str, Any]],
    groups: Mapping[str, list[dict[str, Any]]],
    representative_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    by_name = {str(scan["checkpoint"]): scan for scan in scans}
    requested = list(representative_names)
    if len(requested) != len(set(requested)):
        raise ValueError("Representative checkpoint names must be unique")
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise FileNotFoundError(f"Representative checkpoints not found: {missing}")
    representatives: dict[str, dict[str, Any]] = {}
    for name in requested:
        scan = by_name[name]
        state_sha256 = str(scan["state_sha256"])
        if state_sha256 in representatives:
            other = representatives[state_sha256]["checkpoint"]
            raise ValueError(
                f"Representatives {other!r} and {name!r} share one model state"
            )
        representatives[state_sha256] = scan
    missing_states = sorted(set(groups) - set(representatives))
    if missing_states:
        raise ValueError(
            "Representatives do not cover every unique model state: "
            f"{missing_states}"
        )
    return representatives


def compact_checkpoint_directory(
    input_root: Path,
    output_root: Path,
    *,
    representative_names: Sequence[str],
    manifest_output_root: Path | None = None,
) -> dict[str, Any]:
    input_root, output_root = _validate_roots(input_root, output_root)
    source_paths = sorted(
        path for path in input_root.glob("*.pt") if path.is_file() and not path.is_symlink()
    )
    if not source_paths:
        raise FileNotFoundError(f"No .pt checkpoints found in {input_root}")

    scans = [_scan_checkpoint(path) for path in source_paths]
    groups = _group_sources(scans)
    representatives = _select_representatives(
        scans,
        groups,
        representative_names,
    )
    by_name = {str(scan["checkpoint"]): scan for scan in scans}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        compact_records: dict[str, dict[str, Any]] = {}
        for representative_name in representative_names:
            source = by_name[representative_name]
            source_path = input_root / representative_name
            compact_path = temporary_root / representative_name
            export_checkpoint(source_path, compact_path, force=False)
            compact_checkpoint = load_checkpoint(compact_path, torch.device("cpu"))
            compact_state = _mapping(
                compact_checkpoint.get("model_state_dict"),
                "model_state_dict",
            )
            compact_state_sha256, compact_tensor_bytes = model_state_sha256(compact_state)
            if compact_state_sha256 != source["state_sha256"]:
                raise ValueError(
                    f"Compact state hash differs for {representative_name}: "
                    f"{compact_state_sha256} != {source['state_sha256']}"
                )
            if compact_tensor_bytes != source["tensor_bytes"]:
                raise ValueError(
                    f"Compact tensor byte count differs for {representative_name}"
                )
            compact_bytes = compact_path.stat().st_size
            compact_records[representative_name] = {
                "checkpoint": representative_name,
                "source_epoch": source["source_epoch"],
                "state_sha256": source["state_sha256"],
                "source_sha256": source["source_sha256"],
                "compact_sha256": file_sha256(compact_path),
                "source_bytes": source["source_bytes"],
                "compact_bytes": compact_bytes,
                "reduction_fraction": 1.0 - compact_bytes / source["source_bytes"],
                "tensor_bytes": source["tensor_bytes"],
                "dataset_fingerprint": source["dataset_fingerprint"],
                "removed_top_level_fields": source["removed_top_level_fields"],
            }

        checkpoint_to_representative: dict[str, str] = {}
        unique_states = []
        for state_sha256, group in sorted(groups.items()):
            representative = str(representatives[state_sha256]["checkpoint"])
            names = sorted(str(scan["checkpoint"]) for scan in group)
            for name in names:
                checkpoint_to_representative[name] = representative
            unique_states.append(
                {
                    "representative_checkpoint": representative,
                    "state_sha256": state_sha256,
                    "source_epoch": representatives[state_sha256]["source_epoch"],
                    "checkpoints": names,
                }
            )

        aliases = {
            "schema_version": SCHEMA_VERSION,
            "policy": (
                "Checkpoints with identical tensor names, dtypes, shapes, and bytes "
                "share one compact artifact and one evaluation."
            ),
            "checkpoint_to_representative": dict(
                sorted(checkpoint_to_representative.items())
            ),
            "unique_states": sorted(
                unique_states,
                key=lambda item: representative_names.index(
                    str(item["representative_checkpoint"])
                ),
            ),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_root": str(input_root),
            "output_root": str(
                manifest_output_root.resolve()
                if manifest_output_root is not None
                else output_root
            ),
            "source_checkpoint_count": len(scans),
            "unique_state_count": len(groups),
            "representatives": list(representative_names),
            "dataset_fingerprints": sorted(
                {
                    str(scan["dataset_fingerprint"])
                    for scan in scans
                    if scan["dataset_fingerprint"] is not None
                }
            ),
            "source_checkpoints": {
                str(scan["checkpoint"]): {
                    key: value
                    for key, value in scan.items()
                    if key not in {"checkpoint", "inference_contract"}
                }
                for scan in scans
            },
            "compact_artifacts": compact_records,
            "source_total_bytes": sum(int(scan["source_bytes"]) for scan in scans),
            "compact_total_bytes": sum(
                int(record["compact_bytes"])
                for record in compact_records.values()
            ),
        }
        manifest["total_reduction_fraction"] = (
            1.0
            - manifest["compact_total_bytes"] / manifest["source_total_bytes"]
        )
        _write_json(temporary_root / ALIASES_NAME, aliases)
        _write_json(temporary_root / MANIFEST_NAME, manifest)
        os.replace(temporary_root, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    manifest = compact_checkpoint_directory(
        args.input_root,
        args.output_root,
        representative_names=args.representatives,
        manifest_output_root=args.manifest_output_root,
    )
    source_mib = manifest["source_total_bytes"] / 1024**2
    compact_mib = manifest["compact_total_bytes"] / 1024**2
    print(
        f"Compacted {manifest['source_checkpoint_count']} checkpoints into "
        f"{manifest['unique_state_count']} unique artifacts"
    )
    print(f"Source size: {source_mib:.2f} MiB")
    print(f"Compact size: {compact_mib:.2f} MiB")
    print(f"Reduction: {manifest['total_reduction_fraction'] * 100.0:.2f}%")


if __name__ == "__main__":
    main()
