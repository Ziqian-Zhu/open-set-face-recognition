"""Metadata-only MeGlass pairs; no quality screening, no fabricated sessions."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from .meglass import _distinct_photo_pool, _sha256_file, locate_images, read_meglass_metadata
from ..evaluation.artifacts import new_result_directory
from ..evaluation.reporting import _write_json


def build_pair_protocol(
    metadata: str | Path, images: str | Path, output: str | Path, *,
    subjects_per_split: int = 200, seed: int = 20260926,
    exclude_manifests: tuple[str | Path, ...] = (),
) -> dict:
    """One cross-condition genuine per subject, disjoint adjacent impostors.

    Each split has N genuine + N/2 impostor pairs, 2N images. Subjects are
    disjoint between splits. Within the impostor class each subject occurs
    once; images also participate in genuine pairs, so metrics are correlated.
    Selection depends only on labels, source-photo IDs, existence and seed.
    """
    if type(subjects_per_split) is not int or subjects_per_split < 2 or subjects_per_split % 2:
        raise ValueError("subjects_per_split必须为至少2的偶数")
    if type(seed) is not int:
        raise ValueError("seed必须为整数")
    destination = Path(output).resolve()
    metadata = Path(metadata).resolve()
    metadata_hash = _sha256_file(metadata)
    rows = read_meglass_metadata(metadata)
    lookup = {row.filename: row for row in rows}
    excluded_subjects, used_photos = set(), set()
    exclusion_hashes = []
    for manifest in sorted({Path(path).resolve() for path in exclude_manifests}):
        before = _sha256_file(manifest)
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if raw.get("split") not in {"validation", "test"}:
            raise ValueError("排除项必须是已有validation/test数据清单")
        if "pairs" in raw:
            filenames = [Path(row[key]).name for row in raw["pairs"] for key in ("left_image", "right_image")]
        else:
            filenames = [Path(row["image"]).name for group in ("gallery", "probes") for row in raw[group]]
        for filename in filenames:
            if filename not in lookup:
                raise ValueError("排除清单图像未出现在MeGlass元数据中")
            excluded_subjects.add(lookup[filename].identity)
            used_photos.add(lookup[filename].source_photo_id)
        if _sha256_file(manifest) != before:
            raise ValueError("排除清单在读取期间发生变化")
        exclusion_hashes.append(before)
    available, missing = locate_images(images, rows)
    pool = _distinct_photo_pool([row for row in available if row.identity not in excluded_subjects
                                 and row.source_photo_id not in used_photos])
    eligible = sorted(identity for identity, conditions in pool.items() if all(conditions.values()))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = []
    collision_skips = 0
    for identity in eligible:
        conditions = pool[identity]
        chosen = [rng.choice(conditions[condition]) for condition in ("glasses", "no_glasses")]
        photo_ids = {row.source_photo_id for row in chosen}
        if len(photo_ids) != 2 or photo_ids & used_photos:
            collision_skips += 1
            continue
        used_photos.update(photo_ids)
        selected.append(chosen)
        if len(selected) == 2 * subjects_per_split:
            break
    if len(selected) < 2 * subjects_per_split:
        raise ValueError("元数据身份/源照片隔离后数量不足；不打开像素或用识别分数补选")
    if _sha256_file(metadata) != metadata_hash:
        raise ValueError("元数据在构建期间发生变化")

    protocol = {
        "schema": "meglass-cross-condition-verification-v1", "seed": seed,
        "subjects_per_split": subjects_per_split, "metadata_sha256": metadata_hash,
        "excluded_manifest_sha256": exclusion_hashes,
        "excluded_subjects": len(excluded_subjects), "metadata_missing_images": missing,
        "source_collision_skips": collision_skips,
        "target_empirical_far": .05, "session_metadata": "unavailable",
        "selection": "metadata-only seeded shuffle, one glasses + one no_glasses per subject; no pixel/quality/model prefilter",
        "pairing": "one genuine per subject; adjacent disjoint subjects form one impostor (left glasses, right no_glasses)",
        "limitations": [
            "非MeGlass官方评测；仅研究/教育/非商业用途，遵守上游数据使用条件",
            "源照片编号不等于拍摄会话；不生成session_id或独立attempt_id",
            "genuine和impostor共享图像，不能把所有pairs视为独立试验；低FAR不报告",
            "不使用像素筛选；若运行发现精确重复/标签冲突，失败留档，不按结果补选",
            "公开裁剪/源标签有偏差；预训练重叠未知，无法证明现场跨会话泛化",
        ],
    }
    with new_result_directory(destination) as staging:
        for split_index, split in enumerate(("validation", "test")):
            start = split_index * subjects_per_split
            split_rows = selected[start:start + subjects_per_split]

            def pair(left_index, right_index, genuine):
                left, right = split_rows[left_index][0], split_rows[right_index][1]
                return {
                    "left_image": os.path.relpath(left.path, destination),
                    "right_image": os.path.relpath(right.path, destination),
                    "left_subject_id": f"subject_{start + left_index:04d}",
                    "right_subject_id": f"subject_{start + right_index:04d}",
                    "left_source_photo_id": left.source_photo_id,
                    "right_source_photo_id": right.source_photo_id,
                    "left_condition": "glasses", "right_condition": "no_glasses",
                    "genuine": genuine,
                }

            pairs = [pair(index, index, True) for index in range(subjects_per_split)]
            pairs.extend(pair(index, index + 1, False) for index in range(0, subjects_per_split, 2))
            _write_json(staging / f"pairs.{split}.json", {
                "split": split, "session_metadata": "unavailable", "protocol": protocol, "pairs": pairs,
            })
        _write_json(staging / "protocol.json", {
            **protocol, "manifest_sha256": {
                split: _sha256_file(staging / f"pairs.{split}.json") for split in ("validation", "test")
            },
        })
    return {"output": str(destination), "subjects_per_split": subjects_per_split,
            "pairs_per_split": subjects_per_split * 3 // 2, "excluded_subjects": len(excluded_subjects)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects-per-split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--exclude-manifests", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(build_pair_protocol(args.metadata, args.images, args.output,
                                        subjects_per_split=args.subjects_per_split, seed=args.seed,
                                        exclude_manifests=tuple(args.exclude_manifests)), ensure_ascii=False))


if __name__ == "__main__":
    main()
