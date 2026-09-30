"""Pre-registered, balanced mixed-appearance MeGlass research protocol.

Each enrolled identity contributes three black-eyeglass and three no-eyeglass
images to one gallery. Validation and test each hold out one image per
appearance. Unknown people are disjoint across validation/test and contribute
one probe each, balanced by appearance. This is exploratory: MeGlass lacks
capture-session IDs, and conservative dHash curation reads held-out pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from itertools import combinations
from pathlib import Path

import cv2

from ..manifest import _dhash64
from .meglass import (
    NEAR_DUPLICATE_MAX_HAMMING,
    GalleryPreflight,
    MeGlassRow,
    _check_source_photo_isolation,
    _distinct_photo_pool,
    _entry,
    _sha256_file,
    isolated_gallery_checker,
    locate_images,
    read_meglass_metadata,
)


MAX_ACCEPTED_PER_CONDITION = 8
CONDITIONS = ("glasses", "no_glasses")
SPEC_PATH = Path(__file__).resolve().parents[1] / "protocols" / "meglass_mixed_v1.json"


def _has_cross_split_near_duplicate(
    gallery: list[MeGlassRow],
    validation: list[MeGlassRow],
    test: list[MeGlassRow],
    hash_cache: dict[str, int],
) -> bool:
    """Check only cross-split image similarity for one identity."""

    def dhash(row: MeGlassRow) -> int:
        if row.filename not in hash_cache:
            assert row.path is not None
            image = cv2.imread(str(row.path))
            if image is None:
                raise ValueError(f"近重复检查无法读取图片：{row.filename}")
            hash_cache[row.filename] = _dhash64(image)
        return hash_cache[row.filename]

    sections = (gallery, validation, test)
    for index, left_rows in enumerate(sections):
        for right_rows in sections[index + 1:]:
            if any(
                (dhash(left) ^ dhash(right)).bit_count() <= NEAR_DUPLICATE_MAX_HAMMING
                for left in left_rows for right in right_rows
            ):
                return True
    return False


def build_meglass_mixed_manifests(
    metadata_path: str | Path,
    images_root: str | Path,
    *,
    output_dir: str | Path,
    image_kind: str,
    preflight: GalleryPreflight,
    preflight_config_sha256: str,
    known_people: int = 12,
    unknown_validation_people: int = 20,
    unknown_test_people: int = 20,
    gallery_per_condition: int = 3,
    max_budget: int = 3,
    seed: int = 42,
) -> dict:
    """Generate a fixed mixed-gallery 1:N protocol without copying images.

    The enrollment gate can inspect only gallery candidates. A separate
    cross-split dHash screen reads selected probe pixels to reject an entire
    identity, never to select a model threshold or tune a recognition score.
    Consequently this is not a completely sealed blind test.
    """

    for name, count, minimum in (
        ("known_people", known_people, 3),
        ("unknown_validation_people", unknown_validation_people, 2),
        ("unknown_test_people", unknown_test_people, 2),
        ("gallery_per_condition", gallery_per_condition, 1),
        ("max_budget", max_budget, 1),
    ):
        if type(count) is not int or count < minimum:
            raise ValueError(f"{name}必须是不小于{minimum}的整数")
    if unknown_validation_people % 2 or unknown_test_people % 2:
        raise ValueError("未知验证与测试人数必须为偶数，以便两种外观各占一半")
    if 2 * gallery_per_condition <= max_budget:
        raise ValueError("混合gallery总样本数必须大于模板预算")
    if type(seed) is not int:
        raise ValueError("seed必须是整数")
    if image_kind not in {"original", "cropped"}:
        raise ValueError("image_kind必须明确指定original或cropped")

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"结果目录已存在，不覆盖：{destination}")
    metadata = read_meglass_metadata(metadata_path)
    available, missing = locate_images(images_root, metadata)
    pools = _distinct_photo_pool(available)
    eligible = sorted(
        identity for identity, pool in pools.items()
        if all(len(pool[condition]) >= gallery_per_condition + 2 for condition in CONDITIONS)
    )
    unknown_eligible = sorted(
        identity for identity, pool in pools.items()
        if all(pool[condition] for condition in CONDITIONS)
    )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if len(eligible) < known_people:
        raise ValueError(f"符合混合gallery基础数量条件的身份只有{len(eligible)}人")

    selected: dict[str, tuple[list[MeGlassRow], list[MeGlassRow], list[MeGlassRow]]] = {}
    examined: set[str] = set()
    hash_cache: dict[str, int] = {}
    images_checked = images_rejected = candidate_sets_checked = 0
    candidate_sets_rejected = cohort_rejected = near_duplicate_rejected = 0
    for identity in eligible:
        examined.add(identity)
        pools_for_identity: dict[str, list[MeGlassRow]] = {}
        heldout: dict[str, tuple[MeGlassRow, MeGlassRow]] = {}
        for condition in CONDITIONS:
            shuffled = rng.sample(pools[identity][condition], len(pools[identity][condition]))
            heldout[condition] = (shuffled[0], shuffled[1])
            accepted = []
            for row in shuffled[2:]:
                images_checked += 1
                if not preflight(row):
                    images_rejected += 1
                    continue
                accepted.append(row)
                if len(accepted) == MAX_ACCEPTED_PER_CONDITION:
                    break
            if len(accepted) < gallery_per_condition:
                break
            pools_for_identity[condition] = accepted
        if len(pools_for_identity) != 2:
            continue

        validation = [heldout[condition][0] for condition in CONDITIONS]
        test = [heldout[condition][1] for condition in CONDITIONS]
        chosen = None
        suspected_duplicate = False
        for glasses in combinations(pools_for_identity["glasses"], gallery_per_condition):
            for no_glasses in combinations(pools_for_identity["no_glasses"], gallery_per_condition):
                gallery = [*glasses, *no_glasses]
                candidate_sets_checked += 1
                if not preflight.check_set(gallery):
                    candidate_sets_rejected += 1
                    continue
                # Never try another gallery subset after the held-out pixels
                # indicate a possible duplicate; that would adapt to test.
                if _has_cross_split_near_duplicate(gallery, validation, test, hash_cache):
                    near_duplicate_rejected += 1
                    suspected_duplicate = True
                    break
                if not preflight.check_cohort(
                    f"known_{len(selected) + 1:03d}", {"mixed": gallery}
                ):
                    cohort_rejected += 1
                    continue
                chosen = gallery
                break
            if chosen is not None or suspected_duplicate:
                break
        if chosen is None:
            continue
        # Freeze a condition-agnostic unstructured baseline order. The same
        # six photos are supplied to every selection method and budget.
        rng.shuffle(chosen)
        selected[identity] = (chosen, validation, test)
        if len(selected) == known_people:
            break
    if len(selected) < known_people:
        raise ValueError(
            f"完整预筛后只有{len(selected)}名混合外观已知身份，需要{known_people}名；"
            "此预注册方案无法在当前裁剪图上运行，不能偷改门槛凑结果"
        )

    unknown_pool = [identity for identity in unknown_eligible if identity not in examined]
    rng.shuffle(unknown_pool)
    count = unknown_validation_people + unknown_test_people
    if len(unknown_pool) < count:
        raise ValueError(f"可用未知身份只有{len(unknown_pool)}人，需要{count}人")
    unknown_validation = unknown_pool[:unknown_validation_people]
    unknown_test = unknown_pool[unknown_validation_people:count]

    names = {identity: f"known_{index:03d}" for index, identity in enumerate(selected, 1)}
    gallery_entries = [
        _entry(row, destination, label=names[identity])
        for identity, (gallery, _, _) in selected.items() for row in gallery
    ]
    manifests = {}
    for split, unknown_ids, heldout_index in (
        ("validation", unknown_validation, 1),
        ("test", unknown_test, 2),
    ):
        known_probes = [
            _entry(row, destination, label=names[identity])
            for identity, rows in selected.items() for row in rows[heldout_index]
        ]
        unknown_probes = [
            _entry(
                rng.choice(pools[identity][CONDITIONS[index % 2]]),
                destination,
                label=None,
                subject_id=f"unknown_{split}_{index + 1:03d}",
            )
            for index, identity in enumerate(unknown_ids)
        ]
        manifests[f"mixed.{split}.json"] = {
            "split": split,
            "gallery": gallery_entries,
            "probes": known_probes + unknown_probes,
        }

    lookup = {row.filename: row for row in available}
    for manifest in manifests.values():
        _check_source_photo_isolation(manifest, lookup)
    val_photos = {
        lookup[Path(item["image"]).name].source_photo_id
        for item in manifests["mixed.validation.json"]["probes"]
    }
    test_photos = {
        lookup[Path(item["image"]).name].source_photo_id
        for item in manifests["mixed.test.json"]["probes"]
    }
    if val_photos & test_photos:
        raise ValueError("同一源照片跨验证/测试复用，请更换种子或身份子集")

    registered = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    frozen = registered["split"]
    matches_registration = (
        image_kind == "cropped"
        and seed == frozen["seed"]
        and known_people == frozen["known_identities"]
        and unknown_validation_people == frozen["unknown_validation_identities"]
        and unknown_test_people == frozen["unknown_test_identities"]
        and gallery_per_condition == frozen["gallery_per_known_identity"]["glasses"]
        and gallery_per_condition == frozen["gallery_per_known_identity"]["no_glasses"]
        and max_budget == registered["experiment"]["primary_comparison"]["budget"]
    )
    summary = {
        "protocol": "meglass-balanced-mixed-open-set-exploratory-v1",
        "protocol_spec_sha256": _sha256_file(SPEC_PATH),
        "matches_registered_default": matches_registration,
        "source": "MeGlass authors' meta.txt: 1=black eyeglasses, 0=no eyeglasses",
        "metadata_sha256": _sha256_file(Path(metadata_path).resolve()),
        "metadata_rows": len(metadata),
        "located_images": len(available),
        "available_filenames_sha256": hashlib.sha256(
            "\n".join(sorted(row.filename for row in available)).encode("utf-8")
        ).hexdigest(),
        "missing_images": missing,
        "image_kind": image_kind,
        "seed": seed,
        "known_people": known_people,
        "unknown_validation_people": unknown_validation_people,
        "unknown_test_people": unknown_test_people,
        "gallery_per_condition": gallery_per_condition,
        "gallery_images_per_known_identity": 2 * gallery_per_condition,
        "validation_known_probes_per_condition": known_people,
        "test_known_probes_per_condition": known_people,
        "unknown_probes_per_condition_per_split": unknown_validation_people // 2,
        "gallery_order": "condition-agnostic seeded random shuffle; First-K is not chronology",
        "primary_comparison_frozen_before_test": {
            "budget": max_budget,
            "selection": "coverage",
            "comparator": "first",
            "aggregation": "top3_median",
            "seed": seed,
            "target_empirical_validation_fpir": 0.05,
            "primary_test_metric": "known_correct/all_known_attempts",
            "safety_metric": "false_accepts/usable_unknown_attempts with Wilson interval",
        },
        "preflight": {
            "config_sha256": preflight_config_sha256,
            "known_identities_examined": len(examined),
            "candidate_images_checked": images_checked,
            "candidate_images_rejected": images_rejected,
            "accepted_images_cap_per_condition": MAX_ACCEPTED_PER_CONDITION,
            "candidate_sets_checked": candidate_sets_checked,
            "candidate_sets_rejected": candidate_sets_rejected,
            "cross_identity_cohorts_rejected": cohort_rejected,
            "near_duplicate_identities_rejected": near_duplicate_rejected,
            "near_duplicate_dhash_max_hamming": NEAR_DUPLICATE_MAX_HAMMING,
        },
        "manifests": sorted(manifests),
        "status": "exploratory_only; no capture-session IDs and dHash curation reads held-out pixels",
        "limitations": [
            "120x120裁剪图不能代表原始摄像头检测和画质表现",
            "录入门槛和近重复剔除改变身份分布，不能推断随机人群表现",
            "每个未知身份仅一张probe；验证/测试人数不足以保证低FPIR",
            "dHash跨集筛选读取测试像素，故不是完全封存盲测",
        ],
    }
    destination.mkdir(parents=True, exist_ok=False)
    for filename, manifest in manifests.items():
        (destination / filename).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (destination / "protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="生成平衡戴/摘镜混合录入的MeGlass研究清单")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--image-kind", required=True, choices=("original", "cropped"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-people", type=int, default=12)
    parser.add_argument("--unknown-validation-people", type=int, default=20)
    parser.add_argument("--unknown-test-people", type=int, default=20)
    parser.add_argument("--gallery-per-condition", type=int, default=3)
    parser.add_argument("--max-budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", help="默认课程SFace配置；真实课程库始终不参与")
    args = parser.parse_args()
    try:
        with isolated_gallery_checker(args.config) as (preflight, config_hash):
            summary = build_meglass_mixed_manifests(
                args.metadata, args.images, output_dir=args.output_dir,
                image_kind=args.image_kind, preflight=preflight,
                preflight_config_sha256=config_hash,
                known_people=args.known_people,
                unknown_validation_people=args.unknown_validation_people,
                unknown_test_people=args.unknown_test_people,
                gallery_per_condition=args.gallery_per_condition,
                max_budget=args.max_budget, seed=args.seed,
            )
    except (OSError, ValueError) as error:
        parser.exit(2, f"混合MeGlass清单未生成：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
