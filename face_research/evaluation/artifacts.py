"""Create-only artifacts and a durable record of when a run opened test data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def write_json_once(path: Path, value: dict) -> None:
    """Atomically link a fully flushed JSON file; never replace an old receipt."""

    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".receipt-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def new_result_directory(destination: Path) -> Iterator[Path]:
    """Stage all artifacts and publish together; serialize cooperating writers.

    An interrupted process may leave its exclusive lock. Do not remove it and
    reuse the run ID: use a new output name so the attempted run remains visible.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.publish-lock")
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("reserved result destination\n")
    try:
        if destination.exists():
            raise FileExistsError(f"结果目录已存在，不覆盖：{destination}")
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}.staging-") as temporary:
            staging = Path(temporary)
            yield staging
            if destination.exists():
                raise FileExistsError(f"结果目录已存在，不覆盖：{destination}")
            staging.rename(destination)
    finally:
        lock.unlink(missing_ok=True)


def artifact_hashes(directory: Path) -> dict[str, str]:
    """Hash generated files without recording source image paths or identities."""

    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir()) if path.is_file() and path.name != "artifacts.json"
    }


def verify_result_directory(directory: str | Path) -> dict:
    """Reject incomplete/changed artifact sets using their published checksums."""

    source = Path(directory).resolve()
    manifest = json.loads((source / "artifacts.json").read_text(encoding="utf-8"))
    requirements = {
        "matrix-artifacts-v1": {"experiment_config.json", "metrics.json", "metrics.csv",
                                "threshold_selection.csv", "roc_curve.png", "threshold_curve.png"},
        "quality-artifacts-v1": {"metrics.json", "metrics.csv", "threshold_selection.csv"},
        "benchmark-artifacts-v1": {"metrics.json", "metrics.csv", "scaling_curve.png"},
        "retrieval-artifacts-v1": {"metrics.json", "metrics.csv"},
        "verification-artifacts-v1": {"experiment_config.json", "metrics.json", "metrics.csv",
                                      "pairs.csv", "threshold_selection.csv", "roc_curve.png",
                                      "threshold_curve.png", "frozen_validation.json"},
        "model-comparison-artifacts-v1": {"experiment_config.json", "metrics.json", "metrics.csv",
                                          "pairs.csv", "threshold_selection.csv", "extraction_timing.csv",
                                          "roc_sface.png", "roc_arcface.png", "threshold_curve.png", "frozen_validation.json"},
        "video-data-audit-artifacts-v1": {"metadata.json", "sequence_inventory.csv"},
        "video-benchmark-artifacts-v1": {"metrics.json", "experiment_config.json", "timing.csv"},
        "diagnostic-artifacts-v1": {"metrics.json", "experiment_config.json", "diagnostic_plan.json", "images.csv", "pairs.csv"},
    }
    if manifest.get("schema") not in requirements or not isinstance(manifest.get("sha256"), dict):
        raise ValueError("不支持的产物校验清单")
    required = requirements[manifest["schema"]]
    if not required.issubset(manifest["sha256"]) or artifact_hashes(source) != manifest["sha256"]:
        raise ValueError("实验产物缺失或内容已变化，校验不通过")
    return {"status": "verified", "files": len(manifest["sha256"])}


class RunJournal:
    """Single-use sibling journal retained on success, failure or interruption.

    This is local audit evidence, not a tamper-proof registry or proof that the
    researcher never viewed a previous test result. It does not store face data.
    """

    def __init__(self, output: str | Path, *, kind: str = "matrix"):
        self.output = Path(output).resolve()
        self.directory = self.output.with_name(self.output.name + ".audit")
        self.kind = kind

    def _event(self, filename: str, values: dict) -> None:
        write_json_once(self.directory / filename, {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(), **values,
        })

    def __enter__(self) -> RunJournal:
        if self.output.exists():
            raise FileExistsError(f"结果目录已存在，请换新的run ID：{self.output}")
        self.directory.mkdir(parents=True, exist_ok=False)
        self._event("started.json", {"schema": "research-run-journal-v1", "kind": self.kind, "status": "started"})
        return self

    def freeze(self, receipt: dict) -> None:
        write_json_once(self.directory / "frozen_validation.json", receipt)

    def test_opening(self) -> None:
        if not (self.directory / "frozen_validation.json").is_file():
            raise RuntimeError("没有已落盘的冻结凭据，不能打开test")
        self._event("test_opening.json", {"status": "test_may_have_been_accessed"})

    def completed(self, artifacts: Path) -> None:
        verify_result_directory(artifacts)
        self._event("completed.json", {
            "status": "completed", "artifact_sha256": artifact_hashes(artifacts),
        })

    def __exit__(self, error_type, error, traceback) -> bool:
        if error_type is not None:
            # Exception text may contain private paths/names: store only its type.
            self._event("failed.json", {
                "status": "failed", "error_type": error_type.__name__,
                "test_may_have_been_accessed": (self.directory / "test_opening.json").exists(),
                "note": "不覆盖失败记录；重试须用新run ID，不得因此声称测试仍是首次盲测",
            })
        return False
