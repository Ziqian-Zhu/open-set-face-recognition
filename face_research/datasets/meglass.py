"""Build private, exploratory MeGlass manifests without copying face images.

MeGlass ``meta.txt`` contains ``<filename> <0-or-1>``. The identity is the
filename prefix before the second ``@``; the suffix before its final ``_`` is
the source photo ID. A source photo ID is *not* a capture-session ID. We keep
these concepts separate and never fabricate ``session_id`` metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterator

import cv2

from .._baseline import PROJECT, FaceComparisonSystem, StorageConfig, load_config
from ..manifest import _dhash64
from face_compare_system.face_compare.enrollment import validate_identity_set
from face_compare_system.face_compare.models import PreparedSample


DIRECTIONS = {
    "glasses_to_no_glasses": ("glasses", "no_glasses"),
    "no_glasses_to_glasses": ("no_glasses", "glasses"),
}
MAX_ACCEPTED_GALLERY_CANDIDATES = 24
NEAR_DUPLICATE_MAX_HAMMING = 4


@dataclass(frozen=True)
class MeGlassRow:
    filename: str
    identity: str
    source_photo_id: str
    condition: str
    path: Path | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_meglass_metadata(path: str | Path) -> list[MeGlassRow]:
    """Parse the authors' labels, rejecting ambiguous names and duplicate rows."""

    source = Path(path).resolve()
    result = []
    seen = set()
    for line_number, text in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        fields = text.split()
        if len(fields) != 2 or fields[1] not in {"0", "1"}:
            raise ValueError(f"meta.txt第{line_number}行应为<文件名> <0或1>")
        filename, flag = fields
        if (
            Path(filename).name != filename
            or "\\" in filename
            or filename.count("@") != 2
            or not filename.lower().endswith(".jpg")
            or filename in seen
        ):
            raise ValueError(f"meta.txt第{line_number}行文件名无效或重复")
        identity, image_part = filename.rsplit("@", 1)
        photo_id, separator, face_index = image_part[:-4].rpartition("_")
        identity_parts = identity.split("@")
        if (
            len(identity_parts) != 2
            or not all(identity_parts)
            or not separator
            or not photo_id.isdigit()
            or not face_index.isdigit()
        ):
            raise ValueError(f"meta.txt第{line_number}行缺少源照片编号")
        seen.add(filename)
        # The authors define flag 1 as black eyeglasses, not every glasses type.
        result.append(MeGlassRow(
            filename, identity, photo_id,
            "glasses" if flag == "1" else "no_glasses",
        ))
    if not result:
        raise ValueError("meta.txt没有有效图片记录")
    return result


def locate_images(
    images_root: str | Path, metadata: list[MeGlassRow]
) -> tuple[list[MeGlassRow], int]:
    """Match extracted files by unique basename; do not load or copy pixels."""

    root = Path(images_root).resolve()
    if not root.is_dir():
        raise ValueError("--images必须是已解压的MeGlass图片目录")
    wanted = {row.filename for row in metadata}
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.name not in wanted or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("图片路径指向数据目录外，拒绝符号链接")
        if path.name in found:
            raise ValueError(f"数据目录中存在重名图片：{path.name}")
        found[path.name] = resolved
    available = [
        MeGlassRow(row.filename, row.identity, row.source_photo_id,
                   row.condition, found[row.filename])
        for row in metadata if row.filename in found
    ]
    return available, len(metadata) - len(available)


def _distinct_photo_pool(rows: list[MeGlassRow]) -> dict[str, dict[str, list[MeGlassRow]]]:
    """Use at most one crop per source photo and discard conflicting labels."""

    by_identity: dict[str, dict[str, list[MeGlassRow]]] = {}
    grouped: dict[str, dict[str, list[MeGlassRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row.identity][row.source_photo_id].append(row)
    for identity, by_photo in grouped.items():
        conditions: dict[str, list[MeGlassRow]] = {name: [] for name in ("glasses", "no_glasses")}
        for photo_rows in by_photo.values():
            if len({row.condition for row in photo_rows}) != 1:
                continue
            chosen = min(photo_rows, key=lambda row: row.filename)
            conditions[chosen.condition].append(chosen)
        for condition in conditions:
            conditions[condition].sort(key=lambda row: row.filename)
        by_identity[identity] = conditions
    return by_identity


def _choose(rng: random.Random, rows: list[MeGlassRow], count: int) -> list[MeGlassRow]:
    return rng.sample(rows, count)


def _entry(row: MeGlassRow, output_dir: Path, *, label: str | None,
           subject_id: str | None = None) -> dict:
    assert row.path is not None
    item = {
        "image": os.path.relpath(row.path, output_dir),
        "label": label,
        "condition": row.condition,
    }
    if subject_id is not None:
        item["subject_id"] = subject_id
    # MeGlass contains no trustworthy capture-session annotation.
    return item


def _check_source_photo_isolation(manifest: dict, lookup: dict[str, MeGlassRow]) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    for split, rows in (("gallery", manifest["gallery"]), (manifest["split"], manifest["probes"])):
        for item in rows:
            filename = Path(item["image"]).name
            groups[lookup[filename].source_photo_id].add(split)
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("同一源照片跨录入/验证或录入/测试复用，请更换种子或身份子集")


def _has_cross_split_near_duplicate(
    per_condition: dict[str, tuple[list[MeGlassRow], MeGlassRow, MeGlassRow]],
    hash_cache: dict[str, int],
) -> bool:
    """Conservatively exclude suspicious same-ID photos across held-out splits.

    This checks image similarity only, never a model score or probe quality.
    dHash is an imperfect similarity cue, so the exclusion is recorded as
    dataset curation rather than proof of capture-session independence.
    """

    def dhash(row: MeGlassRow) -> int:
        if row.filename not in hash_cache:
            assert row.path is not None
            image = cv2.imread(str(row.path))
            if image is None:
                raise ValueError(f"近重复检查无法读取图片：{row.filename}")
            hash_cache[row.filename] = _dhash64(image)
        return hash_cache[row.filename]

    for gallery_condition, probe_condition in DIRECTIONS.values():
        sections = (
            per_condition[gallery_condition][0],
            [per_condition[probe_condition][1]],
            [per_condition[probe_condition][2]],
        )
        for index, left_rows in enumerate(sections):
            for right_rows in sections[index + 1:]:
                if any(
                    (dhash(left) ^ dhash(right)).bit_count() <= NEAR_DUPLICATE_MAX_HAMMING
                    for left in left_rows for right in right_rows
                ):
                    return True
    return False


class GalleryPreflight:
    """Cache gallery-only samples and check both enrollment directions in temp DBs."""

    def __init__(self, systems: dict[str, FaceComparisonSystem]) -> None:
        self.systems = systems
        self.cache: dict[str, PreparedSample | None] = {}

    def __call__(self, row: MeGlassRow) -> bool:
        if row.filename not in self.cache:
            assert row.path is not None
            image = cv2.imread(str(row.path))
            if image is None:
                self.cache[row.filename] = None
            else:
                try:
                    self.cache[row.filename] = self.systems["glasses"].prepare_sample(image)
                except ValueError:
                    self.cache[row.filename] = None
        return self.cache[row.filename] is not None

    def check_set(self, rows: list[MeGlassRow]) -> bool:
        samples = [self.cache[row.filename] for row in rows]
        if any(sample is None for sample in samples):
            return False
        # The research gallery rejects identical aligned crops even when their
        # source filenames differ; screen them before writing a manifest.
        aligned = [hashlib.sha256(sample.crop.tobytes()).digest() for sample in samples]
        if len(set(aligned)) != len(aligned):
            return False
        try:
            system = self.systems["glasses"]
            validate_identity_set(
                samples, system.extractor, system.config.engine.enrollment_consistency
            )
        except ValueError:
            return False
        return True

    def check_cohort(self, name: str, galleries: dict[str, list[MeGlassRow]]) -> bool:
        """Commit both candidate enrollments, or roll back both on rejection."""

        try:
            with ExitStack() as stack:
                for condition in galleries:
                    stack.enter_context(self.systems[condition].database.write_guard())
                for condition in galleries:
                    system = self.systems[condition]
                    samples = [self.cache[row.filename] for row in galleries[condition]]
                    if any(sample is None for sample in samples):
                        raise ValueError("候选录入图未通过单图检查")
                    system.enroll_samples(name, samples)
        except ValueError:
            return False
        return True


@contextmanager
def isolated_gallery_checker(config_path: str | Path | None) -> Iterator[tuple[GalleryPreflight, str]]:
    """Run the complete production enrollment gate in disposable SQLite DBs."""

    source = Path(config_path).resolve() if config_path else PROJECT / "config.json"
    config = load_config(source)
    if config.engine.backend != "sface" or config.storage.backend != "sqlite":
        raise ValueError("MeGlass预筛选要求YuNet+SFace和隔离SQLite配置")
    model_dir = (PROJECT / config.engine.model_directory).resolve()
    with tempfile.TemporaryDirectory(prefix="meglass-preflight-") as temporary:
        isolated = replace(
            config,
            engine=replace(config.engine, model_directory=str(model_dir)),
            storage=StorageConfig(
                str(Path(temporary) / "data"),
                str(Path(temporary) / "events.jsonl"),
                False,
                "sqlite",
            ),
        )
        systems = {}
        for condition in ("glasses", "no_glasses", "mixed"):
            condition_config = replace(
                isolated,
                storage=StorageConfig(
                    str(Path(temporary) / condition / "data"),
                    str(Path(temporary) / condition / "events.jsonl"),
                    False,
                    "sqlite",
                ),
            )
            systems[condition] = FaceComparisonSystem(condition_config, temporary)

        try:
            yield GalleryPreflight(systems), _sha256_file(source)
        finally:
            for system in systems.values():
                if hasattr(system.database, "close"):
                    system.database.close()


def build_meglass_manifests(
    metadata_path: str | Path,
    images_root: str | Path,
    *,
    output_dir: str | Path,
    image_kind: str,
    known_people: int = 12,
    unknown_validation_people: int = 20,
    unknown_test_people: int = 20,
    max_budget: int = 3,
    seed: int = 42,
    gallery_checker: Callable[[MeGlassRow], bool] | None = None,
    gallery_set_checker: Callable[[list[MeGlassRow]], bool] | None = None,
    gallery_cohort_checker: Callable[[str, dict[str, list[MeGlassRow]]], bool] | None = None,
    preflight_config_sha256: str | None = None,
) -> dict:
    """Create two directional 1:N protocols with identity/source-photo isolation.

    Every known identity needs ``max_budget + 1`` distinct source photos for
    gallery and one additional validation and test photo in *each* appearance.
    Both directions use the same selected identities, but separate galleries.
    Unknown validation and test identities are disjoint from each other and
    from known identities. Model/quality preflight inspects gallery candidates
    only after reserving probes. The separate dHash leakage screen subsequently
    reads selected probe pixels to curate identities, so this is exploratory
    dataset construction, not a wholly sealed blind test.
    """

    for name, count, minimum in (
        ("known_people", known_people, 3),
        ("unknown_validation_people", unknown_validation_people, 1),
        ("unknown_test_people", unknown_test_people, 1),
        ("max_budget", max_budget, 1),
    ):
        if type(count) is not int or count < minimum:
            raise ValueError(f"{name}必须是不小于{minimum}的整数")
    if type(seed) is not int:
        raise ValueError("seed必须是整数")
    if image_kind not in {"original", "cropped"}:
        raise ValueError("image_kind必须明确指定original或cropped")

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"结果目录已存在，不覆盖：{destination}")
    metadata = read_meglass_metadata(metadata_path)
    available, missing = locate_images(images_root, metadata)
    if not available:
        raise ValueError("数据目录中没有找到meta.txt所列图片")
    pools = _distinct_photo_pool(available)
    gallery_count = max_budget + 1
    required_known = gallery_count + 2
    known_eligible = sorted(
        identity for identity, conditions in pools.items()
        if all(len(conditions[name]) >= required_known for name in ("glasses", "no_glasses"))
    )
    unknown_eligible = sorted(
        identity for identity, conditions in pools.items()
        if all(conditions[name] for name in ("glasses", "no_glasses"))
    )
    rng = random.Random(seed)
    rng.shuffle(known_eligible)
    if len(known_eligible) < known_people:
        raise ValueError(
            f"符合双向K={max_budget}要求的已知身份只有{len(known_eligible)}人，"
            f"需要{known_people}人；可降低--max-budget或--known-people"
        )
    selected: dict[str, dict[str, tuple[list[MeGlassRow], MeGlassRow, MeGlassRow]]] = {}
    known = []
    screened = failed = checked_sets = rejected_sets = rejected_cohorts = capped = 0
    rejected_near_duplicate = 0
    near_hash_cache: dict[str, int] = {}
    examined_identities = 0
    examined_known_ids = set()
    for identity in known_eligible:
        examined_identities += 1
        examined_known_ids.add(identity)
        per_condition = {}
        for condition in ("glasses", "no_glasses"):
            # Reserve validation/test before inspecting gallery quality. This
            # prevents a quality-aware holdout split from cherry-picking easy
            # test probes. Preflight sees candidate enrollment photos only.
            shuffled = _choose(rng, pools[identity][condition], len(pools[identity][condition]))
            validation, test = shuffled[:2]
            gallery_candidates = shuffled[2:]
            accepted = []
            gallery = None
            for row in gallery_candidates:
                if gallery_checker is not None:
                    screened += 1
                    if not gallery_checker(row):
                        failed += 1
                        continue
                accepted.append(row)
                if len(accepted) < gallery_count:
                    continue
                if gallery_set_checker is None:
                    gallery = accepted[:gallery_count]
                    break
                # A failed pair can still belong to a coherent larger set:
                # check complete combinations, not a greedy valid prefix.
                for prior in combinations(accepted[:-1], gallery_count - 1):
                    proposal = [*prior, row]
                    checked_sets += 1
                    if gallery_set_checker(proposal):
                        gallery = proposal
                        break
                    rejected_sets += 1
                if gallery is not None:
                    break
                if len(accepted) >= MAX_ACCEPTED_GALLERY_CANDIDATES:
                    capped += 1
                    break
            if gallery is None:
                break
            per_condition[condition] = (gallery, validation, test)
        if len(per_condition) != 2:
            continue
        if _has_cross_split_near_duplicate(per_condition, near_hash_cache):
            rejected_near_duplicate += 1
            continue
        if gallery_cohort_checker is not None and not gallery_cohort_checker(
            f"known_{len(known) + 1:03d}",
            {condition: per_condition[condition][0] for condition in per_condition},
        ):
            rejected_cohorts += 1
            continue
        selected[identity] = per_condition
        known.append(identity)
        if len(known) == known_people:
            break
    if len(known) < known_people:
        raise ValueError(
            f"预筛选后仅有{len(known)}名可用已知人员，需要{known_people}名；"
            "可降低预算/人数，或使用原图，但不得放宽测试图过滤来凑指标"
        )
    # An identity whose candidate gallery was inspected and then rejected may
    # not silently re-enter as an unknown test subject.
    candidates = [identity for identity in unknown_eligible if identity not in examined_known_ids]
    rng.shuffle(candidates)
    needed_unknown = unknown_validation_people + unknown_test_people
    if len(candidates) < needed_unknown:
        raise ValueError(
            f"可用未知身份只有{len(candidates)}人，需要{needed_unknown}人"
        )
    unknown_validation = candidates[:unknown_validation_people]
    unknown_test = candidates[unknown_validation_people:needed_unknown]

    unknown_selected: dict[str, dict[str, MeGlassRow]] = {}
    for identity in (*unknown_validation, *unknown_test):
        unknown_selected[identity] = {
            condition: _choose(rng, pools[identity][condition], 1)[0]
            for condition in ("glasses", "no_glasses")
        }

    known_names = {identity: f"known_{index:03d}" for index, identity in enumerate(known, 1)}
    unknown_val_names = {
        identity: f"unknown_val_{index:03d}"
        for index, identity in enumerate(unknown_validation, 1)
    }
    unknown_test_names = {
        identity: f"unknown_test_{index:03d}"
        for index, identity in enumerate(unknown_test, 1)
    }
    manifests = {}
    for direction, (gallery_condition, probe_condition) in DIRECTIONS.items():
        gallery = [
            _entry(row, destination, label=known_names[identity])
            for identity in known
            for row in selected[identity][gallery_condition][0]
        ]
        for split, unknown_ids, unknown_names, probe_index in (
            ("validation", unknown_validation, unknown_val_names, 1),
            ("test", unknown_test, unknown_test_names, 2),
        ):
            probes = [
                _entry(selected[identity][probe_condition][probe_index], destination,
                       label=known_names[identity])
                for identity in known
            ] + [
                _entry(unknown_selected[identity][probe_condition], destination,
                       label=None, subject_id=unknown_names[identity])
                for identity in unknown_ids
            ]
            manifests[f"{direction}.{split}.json"] = {
                "split": split, "gallery": gallery, "probes": probes,
            }

    lookup = {row.filename: row for row in available}
    for manifest in manifests.values():
        _check_source_photo_isolation(manifest, lookup)
    for direction in DIRECTIONS:
        validation = manifests[f"{direction}.validation.json"]
        test = manifests[f"{direction}.test.json"]
        val_photos = {
            lookup[Path(item["image"]).name].source_photo_id for item in validation["probes"]
        }
        test_photos = {
            lookup[Path(item["image"]).name].source_photo_id for item in test["probes"]
        }
        if val_photos & test_photos:
            raise ValueError("同一源照片跨验证/测试复用，请更换种子或身份子集")

    summary = {
        "protocol": "meglass-bidirectional-open-set-exploratory-v1",
        "source": "MeGlass (Guo et al., 2018), authors' meta.txt",
        "condition_definitions": {
            "glasses": "meta.txt flag 1: black eyeglasses, not every eyewear type",
            "no_glasses": "meta.txt flag 0: no eyeglasses",
        },
        "metadata_sha256": _sha256_file(Path(metadata_path).resolve()),
        "metadata_rows": len(metadata),
        "located_images": len(available),
        "available_filenames_sha256": hashlib.sha256(
            "\n".join(sorted(row.filename for row in available)).encode("utf-8")
        ).hexdigest(),
        "missing_images": missing,
        "image_kind": image_kind,
        "seed": seed,
        "known_people": len(known),
        "unknown_validation_people": len(unknown_validation),
        "unknown_test_people": len(unknown_test),
        "gallery_images_per_known_identity": gallery_count,
        "gallery_order": "seeded random order; First-K is an unstructured baseline, not capture chronology",
        "max_comparable_budget": max_budget,
        "gallery_preflight": {
            "enabled": gallery_checker is not None,
            "config_sha256": preflight_config_sha256,
            "candidate_images_checked": screened,
            "candidate_images_rejected": failed,
            "candidate_sets_checked": checked_sets,
            "candidate_sets_rejected": rejected_sets,
            "candidate_sets_candidate_limit": MAX_ACCEPTED_GALLERY_CANDIDATES,
            "candidate_sets_limit_reached": capped,
            "cross_identity_cohorts_rejected": rejected_cohorts,
            "known_identities_examined": examined_identities,
        },
        "near_duplicate_screen": {
            "enabled": True,
            "algorithm": "64-bit decoded-frame dHash across splits for the same identity",
            "max_hamming": NEAR_DUPLICATE_MAX_HAMMING,
            "known_identities_rejected": rejected_near_duplicate,
            "note": "conservative curation, not proof of independent capture sessions",
        },
        "manifests": sorted(manifests),
        "session_metadata": "not_provided; do not invent session_id",
        "status": "exploratory_only; source-photo isolation is not capture-session isolation",
        "limitations": [
            "跨来源照片编号与身份隔离已检查，但无法证明拍摄批次独立",
            "120x120裁剪图不代表原图的YuNet检测与画质门控表现",
            "启用预筛时按生产录入门槛筛选gallery，会偏向可录入的身份与照片；不得代表随机总体",
            "跨集合近重复dHash筛选可能误判相似但不同的照片，并进一步改变身份分布",
            "每个单向协议的gallery只有一种眼镜外观，不能验证混合外观模板覆盖假设",
            "随机身份子集不是MeGlass论文的官方测试协议；反向戴镜录入是自定义协议",
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
    parser = argparse.ArgumentParser(description="生成MeGlass双向戴镜研究清单，不复制图片")
    parser.add_argument("--metadata", required=True, help="作者仓库的meta.txt")
    parser.add_argument("--images", required=True, help="已经合法获取并解压的图片目录")
    parser.add_argument("--image-kind", required=True, choices=("original", "cropped"))
    parser.add_argument("--output-dir", required=True, help="新建私有清单目录，不覆盖")
    parser.add_argument("--known-people", type=int, default=12)
    parser.add_argument("--unknown-validation-people", type=int, default=20)
    parser.add_argument("--unknown-test-people", type=int, default=20)
    parser.add_argument("--max-budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gallery-preflight", action="store_true",
                        help="仅对录入候选执行正式单图、组内一致性和跨人录入检查")
    parser.add_argument("--config", help="预筛选配置，默认课程SFace配置")
    args = parser.parse_args()
    try:
        if args.gallery_preflight:
            with isolated_gallery_checker(args.config) as (checker, config_hash):
                summary = build_meglass_manifests(
                    args.metadata, args.images, output_dir=args.output_dir,
                    image_kind=args.image_kind, known_people=args.known_people,
                    unknown_validation_people=args.unknown_validation_people,
                    unknown_test_people=args.unknown_test_people,
                    max_budget=args.max_budget, seed=args.seed,
                    gallery_checker=checker,
                    gallery_set_checker=checker.check_set,
                    gallery_cohort_checker=checker.check_cohort,
                    preflight_config_sha256=config_hash,
                )
        else:
            summary = build_meglass_manifests(
                args.metadata, args.images, output_dir=args.output_dir,
                image_kind=args.image_kind, known_people=args.known_people,
                unknown_validation_people=args.unknown_validation_people,
                unknown_test_people=args.unknown_test_people,
                max_budget=args.max_budget, seed=args.seed,
            )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"MeGlass清单未生成：{exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
