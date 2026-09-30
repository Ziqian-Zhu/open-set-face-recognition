"""Strict metadata-only inventory of the authors' LTFT ChokePoint annotations.

This does not download/decode videos, create a gallery, infer identities, or
declare the detector-assisted annotations to be exhaustive ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from ..evaluation.artifacts import artifact_hashes, new_result_directory
from ..evaluation.reporting import _csv, _write_json


AUTHOR_REPOSITORY = "https://github.com/hertasecurity/LTFT"
DATASET_SOURCE = "https://arma.sourceforge.net/chokepoint/"


@dataclass(frozen=True)
class DetectionAnnotation:
    subject: int
    box: tuple[float, float, float, float]
    face: bool
    confidence: float


@dataclass(frozen=True)
class FrameAnnotation:
    index: int
    detections: tuple[DetectionAnnotation, ...]


@dataclass(frozen=True)
class AnnotationFile:
    frames: tuple[FrameAnnotation, ...]
    sha256: str


def _integer(token: str, field: str, *, minimum: int = 0) -> int:
    # int('1.0') must fail; frame numbers/IDs cannot silently be rounded.
    try:
        value = int(token)
    except ValueError as error:
        raise ValueError(f"{field}必须为整数") from error
    if value < minimum:
        raise ValueError(f"{field}不能小于{minimum}")
    return value


def parse_annotations(content: bytes) -> AnnotationFile:
    """Validate the entire file, including face=0 rows (no confidence filtering).

    The first integer is the total frame count, followed by exactly one row
    per frame: index, count, and count repetitions of seven detection fields.
    face=0 means false positives OR multi-face regions in the source format;
    preserve it instead of treating it as an unknown person or dropping a frame.
    """
    lines = content.decode("utf-8-sig").splitlines()
    if not lines or len(lines[0].split()) != 1:
        raise ValueError("标注缺少单独的总帧数")
    count = _integer(lines[0].strip(), "总帧数", minimum=1)
    if len(lines) != count + 1:
        raise ValueError("标注总帧数与实际行数不一致，不能删掉无人帧或困难帧")
    frames = []
    for expected, line in enumerate(lines[1:]):
        tokens = line.split()
        if len(tokens) < 2:
            raise ValueError("每帧须有帧号及检测数")
        index = _integer(tokens[0], "帧号")
        if index != expected:
            raise ValueError("帧号须从0开始连续递增，不允许重复/漏帧/重新排序")
        detections = _integer(tokens[1], "检测数")
        if len(tokens) != 2 + detections * 7:
            raise ValueError("每个检测须包含7个字段，数量必须与声明一致")
        rows, subjects = [], set()
        for offset in range(2, len(tokens), 7):
            subject = _integer(tokens[offset], "subject", minimum=1)
            if subject in subjects:
                raise ValueError("同帧subject重复")
            subjects.add(subject)
            try:
                box = tuple(float(value) for value in tokens[offset+1:offset+5])
                confidence = float(tokens[offset+6])
            except ValueError as error:
                raise ValueError("标注框及置信度须为数值") from error
            if not all(math.isfinite(value) for value in box) or min(box[2:]) <= 0:
                raise ValueError("标注框须有限且宽高为正")
            face = _integer(tokens[offset+5], "face标记")
            if face not in (0, 1):
                raise ValueError("face标记只能为0或1")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence须在[0,1]且有限")
            rows.append(DetectionAnnotation(subject, box, bool(face), confidence))
        frames.append(FrameAnnotation(index, tuple(rows)))
    return AnnotationFile(tuple(frames), hashlib.sha256(content).hexdigest())


def annotation_inventory(annotation: AnnotationFile, *, sequence: str) -> dict:
    """Report what the source actually annotates, not model performance."""
    if sequence not in {"choke1", "choke2"}:
        raise ValueError("当前预检仅支持许可已核验的ChokePoint来源，不下载YouTube序列")
    expected_frames = {"choke1": 2526, "choke2": 2139}[sequence]
    if len(annotation.frames) != expected_frames:
        raise ValueError("帧数与作者公布的ChokePoint序列不一致，不能自动截断或补帧")
    valid = [item for frame in annotation.frames for item in frame.detections if item.face]
    excluded = [item for frame in annotation.frames for item in frame.detections if not item.face]
    counts = Counter(sum(item.face for item in frame.detections) for frame in annotation.frames)
    out_of_bounds = sum(x < 0 or y < 0 or x+w > 800 or y+h > 600
                        for x, y, w, h in (item.box for item in valid))
    return {
        "sequence": sequence, "annotation_sha256": annotation.sha256,
        "frames": len(annotation.frames), "source_fps": 30, "source_resolution": [800, 600],
        "face1_annotations": len(valid), "face0_annotations": len(excluded),
        "face1_subjects": len({item.subject for item in valid}),
        "frames_without_face1": counts[0], "frames_with_one_face1": counts[1],
        "frames_with_multiple_face1": sum(count for faces, count in counts.items() if faces > 1),
        "maximum_face1_per_frame": max(counts),
        "face1_out_of_image_bounds": out_of_bounds,
        "face1_count_histogram": {str(key): counts[key] for key in sorted(counts)},
        "video_bytes_available": None, "video_decode_verified": False,
        "open_set_gallery_ready": False,
        "session_id": None, "cross_sequence_identity_map_verified": False,
    }


def audit_annotations(choke1: str | Path, choke2: str | Path) -> dict:
    paths = [Path(choke1).resolve(), Path(choke2).resolve()]
    data = [path.read_bytes() for path in paths]
    inventories = [annotation_inventory(parse_annotations(content), sequence=sequence)
                   for sequence, content in zip(("choke1", "choke2"), data, strict=True)]
    if any(path.read_bytes() != content for path, content in zip(paths, data, strict=True)):
        raise ValueError("预检过程中标注文件发生变化")
    return {
        "schema": "ltft-chokepoint-metadata-audit-v1",
        "kind": "metadata_only_not_video_evaluation",
        "source_repository": AUTHOR_REPOSITORY, "dataset_source": DATASET_SOURCE,
        "parser_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sequences": inventories,
        "limitations": [
            "没有解码视频或运行人脸模型；这些是作者标注统计，不是本系统检测/识别结果",
            "Choke1/2是三视角拼接；同时拍摄的相机视角不能命名为独立采集会话",
            "数字subject在不同文件中的对应关系尚未核验，不自动跨序列构建同人gallery/test",
            "标注来自检测器候选经人工核验；可能缺少未检出脸，不能宣称穷尽所有人脸的真值",
            "face=0同时表示假检和包含多脸的框；没有区分信息，不自动当unknown或ignore区",
            "源视频仅限ChokePoint非商业研究等许可用途，保留NICTA及论文归属，不再分发原图/标注",
            "LTFT仓库未附通用LICENSE，不能把公开访问推定为可商用或任意再分发",
        ],
        "remaining_requirements": [
            "合法获取源视频并逐帧核对帧数、分辨率、拼接顺序和标注坐标",
            "固定face=0歧义区域的评估规则；必要时人工补标，不能由本模型生成真值",
            "有证据的独立录入来源、身份对应及capture-event隔离，保留未知人员",
            "先冻结阈值/抽帧/时序规则，再运行真实视频并报告完整失败分母和分阶段时延",
        ],
    }


def write_audit(report: dict, output: str | Path) -> Path:
    destination = Path(output).resolve()
    with new_result_directory(destination) as staging:
        _write_json(staging / "metadata.json", report)
        rows = report["sequences"]
        fields = [field for field in rows[0] if field != "face1_count_histogram"]
        _csv(staging / "sequence_inventory.csv", rows, fields)
        _write_json(staging / "artifacts.json", {
            "schema": "video-data-audit-artifacts-v1", "sha256": artifact_hashes(staging),
        })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choke1", required=True)
    parser.add_argument("--choke2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_annotations(args.choke1, args.choke2)
    destination = write_audit(report, args.output)
    print(json.dumps({"output": str(destination), "kind": report["kind"],
                      "sequences": report["sequences"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
