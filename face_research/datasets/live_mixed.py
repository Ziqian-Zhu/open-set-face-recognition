"""Prepare consented built-in-camera images for independent-session research.

The template and builder use pseudonymous metadata. Only the explicit capture
command opens a camera. No image, embedding, or personal name is uploaded.
The builder validates metadata and paths without decoding held-out pixels;
the experiment runner later checks hashes and opens test only after freezing
validation choices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .._baseline import PROJECT, load_config
from .meglass import _sha256_file
from face_compare_system.face_compare.camera import CameraManager


FIELDS = (
    "subject_id", "role", "split", "condition", "session_id", "image",
    "consent_confirmed", "captured_at_utc",
)
CONDITIONS = ("glasses", "no_glasses")
PROFILES = {
    "pilot": (3, 2, 2),
    "extended": (12, 20, 20),
}
SPEC_PATH = Path(__file__).resolve().parents[1] / "protocols" / "live_camera_mixed_v1.json"
SUBJECT_ID = re.compile(r"^[A-Za-z0-9_-]{3,64}$")
SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{3,80}$")


def _read_index(index_path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    source = Path(index_path).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("采集索引CSV列必须与模板完全一致")
        rows = list(reader)
    if any(set(row) != set(FIELDS) or any(value is None for value in row.values()) for row in rows):
        raise ValueError("采集索引CSV存在缺失或多余单元格")
    if not rows:
        raise ValueError("采集索引为空")
    return source, rows


def _image_path(index_path: Path, value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("图片路径必须是索引目录内的相对路径")
    path = (index_path.parent / relative).resolve()
    if not path.is_relative_to(index_path.parent) or path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png"
    }:
        raise ValueError("图片必须位于索引目录内，且为JPG或PNG")
    return path


def make_template(output_path: str | Path, *, profile: str = "pilot") -> dict:
    """Write metadata slots only; no consent is prefilled or camera opened."""

    if profile not in PROFILES:
        raise ValueError("profile必须为pilot或extended")
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"索引文件已存在，不覆盖：{destination}")
    known, unknown_val, unknown_test = PROFILES[profile]
    rows = []
    for number in range(1, known + 1):
        subject = f"known_{number:03d}"
        for split, count in (("gallery", 3), ("validation", 1), ("test", 1)):
            for condition in CONDITIONS:
                for index in range(1, count + 1):
                    rows.append({
                        "subject_id": subject,
                        "role": "known",
                        "split": split,
                        "condition": condition,
                        "session_id": "",
                        "image": f"images/{subject}/{split}/{condition}_{index:02d}.png",
                        "consent_confirmed": "",
                        "captured_at_utc": "",
                    })
    for split, count, role in (
        ("validation", unknown_val, "unknown_validation"),
        ("test", unknown_test, "unknown_test"),
    ):
        for index in range(1, count + 1):
            subject = f"{role}_{index:03d}"
            condition = CONDITIONS[(index - 1) % 2]
            rows.append({
                "subject_id": subject,
                "role": role,
                "split": split,
                "condition": condition,
                "session_id": "",
                "image": f"images/{subject}/{split}/{condition}_01.png",
                "consent_confirmed": "",
                "captured_at_utc": "",
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"profile": profile, "rows": len(rows), "index": str(destination)}


def _validate_rows(
    index_path: Path, rows: list[dict[str, str]], profile: str
) -> dict[str, list[dict]]:
    if profile not in PROFILES:
        raise ValueError("profile必须为pilot或extended")
    groups: dict[str, list[dict]] = defaultdict(list)
    roles: dict[str, str] = {}
    paths: set[Path] = set()
    for line_number, row in enumerate(rows, 2):
        subject = row["subject_id"].strip()
        role = row["role"].strip()
        split = row["split"].strip()
        condition = row["condition"].strip()
        session = row["session_id"].strip()
        if not SUBJECT_ID.fullmatch(subject):
            raise ValueError(f"第{line_number}行须使用匿名ASCII subject_id")
        if role not in {"known", "unknown_validation", "unknown_test"}:
            raise ValueError(f"第{line_number}行role无效")
        if split not in {"gallery", "validation", "test"} or condition not in CONDITIONS:
            raise ValueError(f"第{line_number}行split或condition无效")
        if (role == "known" and split not in {"gallery", "validation", "test"}) or (
            role == "unknown_validation" and split != "validation"
        ) or (role == "unknown_test" and split != "test"):
            raise ValueError(f"第{line_number}行人员角色与集合不匹配")
        if not SESSION_ID.fullmatch(session):
            raise ValueError(f"第{line_number}行必须填写真实拍摄批次session_id")
        if row["consent_confirmed"].strip().lower() != "yes":
            raise ValueError(f"第{line_number}行未确认参与者同意；CSV勾选不能替代实际同意")
        prior = roles.setdefault(subject, role)
        if prior != role:
            raise ValueError(f"{subject}在不同人员角色中重复")
        image_path = _image_path(index_path, row["image"].strip())
        if image_path in paths:
            raise ValueError("同一图片路径重复使用，拒绝跨集合泄漏")
        if not image_path.is_file():
            raise ValueError(f"图片不存在：{image_path}")
        paths.add(image_path)
        groups[subject].append({
            "path": image_path, "role": role, "split": split,
            "condition": condition, "session_id": session,
        })

    known_count, val_count, test_count = PROFILES[profile]
    counts = Counter(roles.values())
    if (counts["known"], counts["unknown_validation"], counts["unknown_test"]) != (
        known_count, val_count, test_count
    ):
        raise ValueError(f"{profile}需要{known_count}名已知、{val_count}+{test_count}名互异未知人员")
    for subject, entries in groups.items():
        role = roles[subject]
        if role == "known":
            expected = {("gallery", name): 3 for name in CONDITIONS}
            expected.update({(split, name): 1 for split in ("validation", "test") for name in CONDITIONS})
            actual = Counter((entry["split"], entry["condition"]) for entry in entries)
            if actual != expected:
                raise ValueError(f"{subject}应有每种外观3张gallery、各1张validation/test")
            sessions = {
                split: {entry["session_id"] for entry in entries if entry["split"] == split}
                for split in ("gallery", "validation", "test")
            }
            if any(len(value) != 1 for value in sessions.values()) or len(
                set.union(*sessions.values())
            ) != 3:
                raise ValueError(f"{subject}的gallery/validation/test须属于三个不同拍摄批次")
        elif len(entries) != 1:
            raise ValueError(f"{subject}作为未知人员只能有一张probe")
    for role in ("unknown_validation", "unknown_test"):
        by_condition = Counter(
            entries[0]["condition"] for entries in groups.values() if entries[0]["role"] == role
        )
        expected = (val_count if role == "unknown_validation" else test_count) // 2
        if any(by_condition[name] != expected for name in CONDITIONS):
            raise ValueError(f"{role}的戴镜/不戴镜probe人数必须各占一半")
    return groups


def build_manifests(
    index_file: str | Path, output_dir: str | Path, *, profile: str = "pilot"
) -> dict:
    """Create manifests from locked metadata without opening any face pixels."""

    index_path, rows = _read_index(index_file)
    groups = _validate_rows(index_path, rows, profile)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"结果目录已存在，不覆盖：{destination}")

    def manifest_entry(subject: str, entry: dict) -> dict:
        item = {
            "image": os.path.relpath(entry["path"], destination),
            "label": subject if entry["role"] == "known" else None,
            "condition": entry["condition"],
            "session_id": entry["session_id"],
        }
        if entry["role"] != "known":
            item["subject_id"] = subject
        return item

    known = sorted(subject for subject, entries in groups.items() if entries[0]["role"] == "known")
    gallery = [
        manifest_entry(subject, entry)
        for subject in known for entry in groups[subject] if entry["split"] == "gallery"
    ]
    manifests = {}
    for split in ("validation", "test"):
        probes = [
            manifest_entry(subject, entry)
            for subject in known for entry in groups[subject] if entry["split"] == split
        ]
        unknown_role = f"unknown_{split}"
        probes.extend(
            manifest_entry(subject, entry)
            for subject in sorted(groups)
            for entry in groups[subject]
            if entry["role"] == unknown_role
        )
        manifests[f"mixed.{split}.json"] = {
            "split": split, "gallery": gallery, "probes": probes,
        }
    summary = {
        "protocol": "consented-live-camera-mixed-open-set-v1",
        "profile": profile,
        "protocol_spec_sha256": _sha256_file(SPEC_PATH),
        "capture_index_sha256": _sha256_file(index_path),
        "known_people": len(known),
        "unknown_validation_people": PROFILES[profile][1],
        "unknown_test_people": PROFILES[profile][2],
        "gallery_images_per_known_identity": 6,
        "gallery_conditions": {"glasses": 3, "no_glasses": 3},
        "session_metadata": "provided_by_collector; distinct IDs checked, physical independence cannot be proven from CSV",
        "image_pixels_read_during_build": False,
        "consent": "each row attested yes; actual informed consent remains collector's responsibility",
        "manifests": sorted(manifests),
        "status": "ready_for_frozen_validation_test; not yet evaluated",
        "limitations": [
            "pilot仅检验流程，不构成人群准确率或低FPIR证据",
            "CSV无法独立证明同意、真实拍摄时间或参与者身份",
            "图片内容与近重复须由实验入口检查并人工复核",
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


def collection_status(index_file: str | Path, *, profile: str = "pilot") -> dict:
    """Report missing slots and session jobs without decoding held-out pixels."""

    if profile not in PROFILES:
        raise ValueError("profile必须为pilot或extended")
    source, rows = _read_index(index_file)
    jobs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not SUBJECT_ID.fullmatch(row["subject_id"].strip()):
            raise ValueError("采集索引须使用匿名ASCII subject_id")
        jobs[(row["subject_id"].strip(), row["split"].strip())].append({
            "exists": _image_path(source, row["image"].strip()).is_file(),
            "consent": row["consent_confirmed"].strip().lower() == "yes",
            "session": bool(SESSION_ID.fullmatch(row["session_id"].strip())),
        })
    validation_error = None
    try:
        _validate_rows(source, rows, profile)
    except ValueError as exc:
        validation_error = str(exc)
    pending = []
    for (subject, split), slots in jobs.items():
        if not all(all(slot.values()) for slot in slots):
            pending.append({
                "subject_id": subject, "split": split, "slots": len(slots),
                "missing_images": sum(not slot["exists"] for slot in slots),
                "missing_consent_attestations": sum(not slot["consent"] for slot in slots),
                "missing_session_ids": sum(not slot["session"] for slot in slots),
            })
    return {
        "profile": profile, "expected_people": sum(PROFILES[profile]),
        "total_image_slots": len(rows),
        "existing_images": sum(slot["exists"] for slots in jobs.values() for slot in slots),
        "ready_to_build": validation_error is None,
        "metadata_validation_error": validation_error,
        "pending_sessions": pending,
        "image_pixels_read": False, "accuracy_evaluated": False,
        "note": "本命令只检查本地文件和采集元数据；不证明参与者同意、真实批次独立或识别效果",
    }


def _write_index_atomically(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def capture_session(
    index_file: str | Path, *, subject_id: str, split: str,
    session_id: str, consent_confirmed: bool, camera_index: int | None = None,
) -> int:
    """Interactively save uncropped frames using only the computer camera."""

    if not consent_confirmed:
        raise ValueError("须先取得参与者同意，再显式传入--consent-confirmed")
    if not SESSION_ID.fullmatch(session_id):
        raise ValueError("请填写真实拍摄批次编号")
    index_path, rows = _read_index(index_file)
    pending = []
    for row in rows:
        if row["subject_id"] != subject_id or row["split"] != split:
            continue
        if row["session_id"] and row["session_id"] != session_id:
            raise ValueError("该人员此集合已有不同session_id，不可混合拍摄批次")
        path = _image_path(index_path, row["image"])
        if path.exists():
            if row["consent_confirmed"].lower() != "yes" or not row["session_id"]:
                raise ValueError(f"照片已存在但索引未完成，请人工核对：{path}")
            continue
        pending.append((row, path))
    if not pending:
        raise ValueError("未找到待拍摄的该人员/集合照片")

    config = load_config(PROJECT / "config.json")
    manager = CameraManager(config.camera)
    devices = manager.discover()
    built_in = [device for device in devices if device.built_in]
    if sys.platform == "darwin":
        if not built_in:
            raise RuntimeError("没有可用的电脑内置摄像头；不会切换到手机或外置摄像头")
        device = next(
            (item for item in built_in if item.index == camera_index), None
        ) if camera_index is not None else built_in[0]
        if device is None:
            raise ValueError("所选设备不是电脑内置摄像头")
    else:
        device = next(
            (item for item in devices if item.index == camera_index), None
        ) if camera_index is not None else (devices[0] if devices else None)
        if device is None:
            raise RuntimeError("未找到指定摄像头")
    window = "Face research - local capture"
    saved = 0
    window_opened = False
    try:
        with manager:
            manager.open(device.index, unique_id=device.unique_id)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_opened = True
            for row, path in pending:
                while True:
                    ok, frame = manager.read()
                    if not ok or frame is None:
                        raise RuntimeError("摄像头读取失败")
                    preview = cv2.flip(frame, 1)
                    title = f"{subject_id} | {split} | {row['condition']} | {saved + 1}/{len(pending)}"
                    cv2.putText(preview, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(preview, "SPACE save raw frame | ESC stop", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window, preview)
                    key = cv2.waitKey(20) & 0xFF
                    if key == 27:
                        return saved
                    if key != 32:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        raise FileExistsError(f"图片已存在，拒绝覆盖：{path}")
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f"无法保存照片：{path}")
                    row["session_id"] = session_id
                    row["consent_confirmed"] = "yes"
                    row["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
                    _write_index_atomically(index_path, rows)
                    saved += 1
                    break
    finally:
        if window_opened:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="独立真人相机混合外观研究数据的本地采集/清单工具")
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="创建不含照片和同意状态的采集索引")
    template.add_argument("--output", required=True)
    template.add_argument("--profile", choices=PROFILES, default="pilot")
    build = commands.add_parser("build", help="检查采集索引并生成研究清单，不读取图像像素")
    build.add_argument("--index", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--profile", choices=PROFILES, default="pilot")
    status = commands.add_parser("status", help="列出缺少的图片/同意/拍摄批次，不打开相机或图片")
    status.add_argument("--index", required=True)
    status.add_argument("--profile", choices=PROFILES, default="pilot")
    capture = commands.add_parser("capture", help="仅用电脑摄像头拍摄指定人员的一个批次")
    capture.add_argument("--index", required=True)
    capture.add_argument("--subject-id", required=True)
    capture.add_argument("--split", required=True, choices=("gallery", "validation", "test"))
    capture.add_argument("--session-id", required=True)
    capture.add_argument("--consent-confirmed", action="store_true")
    capture.add_argument("--camera-index", type=int)
    args = parser.parse_args()
    try:
        if args.command == "template":
            result = make_template(args.output, profile=args.profile)
        elif args.command == "build":
            result = build_manifests(args.index, args.output_dir, profile=args.profile)
        elif args.command == "status":
            result = collection_status(args.index, profile=args.profile)
        else:
            result = {"saved_images": capture_session(
                args.index, subject_id=args.subject_id, split=args.split,
                session_id=args.session_id,
                consent_confirmed=args.consent_confirmed,
                camera_index=args.camera_index,
            )}
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"真人相机协议未完成：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
