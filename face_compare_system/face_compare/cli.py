"""Headless commands for repeatable enrollment, recognition, and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import sqlite3
from pathlib import Path
from typing import Any

from .camera import CameraManager
from .config import load_config
from .service import FaceComparisonSystem


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="离线人脸比对系统命令行工具")
    parser.add_argument("--config", default="config.json", help="JSON 配置文件")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll = subparsers.add_parser("enroll", help="从图片批量录入或追加人员")
    enroll.add_argument("--name", required=True, help="人员姓名")
    enroll.add_argument("--allow-low-quality", action="store_true", help="允许低质量样本（不建议）")
    enroll.add_argument("images", nargs="+", help="包含单张人脸的图片")

    recognize = subparsers.add_parser("recognize", help="识别静态图片")
    recognize.add_argument("--json", action="store_true", help="输出 JSON Lines")
    recognize.add_argument("images", nargs="+", help="待识别图片")

    subparsers.add_parser("list", help="列出标准库人员")
    subparsers.add_parser("calibrate", help="基于库内正负样本对校准阈值")
    subparsers.add_parser("cameras", help="扫描可用摄像头")
    subparsers.add_parser("gui", help="启动桌面界面")
    evaluate = subparsers.add_parser("evaluate", help="独立图片清单评估（不修改现有标准库）")
    evaluate.add_argument("manifest", help="含gallery/probes的JSON清单")
    evaluate.add_argument("--output", required=True, help="新的JSON结果文件（不覆盖已有文件）")
    video = subparsers.add_parser("video", help="本地视频多人跟踪，导出逐帧JSONL（无标注不报告准确率）")
    video.add_argument("path")
    video.add_argument("--output", required=True)
    video.add_argument("--every", type=int, default=3)
    video.add_argument("--max-frames", type=int, default=0)
    operating = subparsers.add_parser("select-threshold", help="仅从validation报告选取经验FPIR约束阈值")
    operating.add_argument("report")
    operating.add_argument("--target-fpir", type=float, default=.01)
    operating.add_argument("--output", required=True)
    restore = subparsers.add_parser("restore", help="恢复SQLite软删除归档；同名冲突时拒绝")
    restore.add_argument("archive")
    video_eval = subparsers.add_parser("evaluate-video", help="对视频导出与逐帧真值进行评测")
    video_eval.add_argument("prediction")
    video_eval.add_argument("truth")
    video_eval.add_argument("--output", required=True)
    temporal = subparsers.add_parser("evaluate-temporal", help="同一视频导出中单帧与多帧确认的配对对照")
    temporal.add_argument("prediction")
    temporal.add_argument("truth")
    temporal.add_argument("--output", required=True)
    return parser


def _recognition_dict(image: str, face_index: int, observation: Any) -> dict[str, Any]:
    """Convert a face observation into stable CLI JSON."""

    result = observation.recognition
    return {
        "image": image,
        "face_index": face_index,
        "box": observation.box.as_tuple(),
        "quality": observation.quality.to_dict(),
        "known": result.known if result else False,
        "name": result.name if result else "未知人员",
        "similarity": round(result.similarity, 3) if result else 0.0,
        "distance": round(result.distance, 6) if result else None,
        "threshold": round(result.threshold, 6) if result else None,
        "reason": result.reason if result else "未执行识别",
    }


def main(argv: list[str] | None = None) -> int:
    """Run one CLI command and return a process status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"无法加载配置: {error}")
    project_root = config_path.parent

    if args.command in ("video", "select-threshold", "evaluate-video", "evaluate-temporal"):
        try:
            if args.command == "video":
                from .video import analyze_video
                report = analyze_video(config, project_root, args.path, args.output, every=args.every, max_frames=args.max_frames)
            else:
                from .operating_point import select_operating_point
                output = Path(args.output)
                if output.exists():
                    raise ValueError("结果文件已存在，拒绝覆盖")
                if args.command == "evaluate-video":
                    from .video_evaluation import evaluate_video
                    report = evaluate_video(args.prediction, args.truth)
                elif args.command == "evaluate-temporal":
                    from .video_evaluation import evaluate_temporal_ablation
                    report = evaluate_temporal_ablation(args.prediction, args.truth)
                else:
                    report = select_operating_point(json.loads(Path(args.report).read_text(encoding="utf-8")), args.target_fpir)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("x", encoding="utf-8") as handle:
                    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"结果已保存：{args.output}")
            return 0
        except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError, TypeError) as error:
            print(f"执行失败：{error}", file=sys.stderr)
            return 1

    if args.command == "restore":
        try:
            system = FaceComparisonSystem(config, project_root)
            if not hasattr(system.database, "restore_archive"):
                raise ValueError("该命令仅支持SQLite归档")
            system.database.restore_archive(args.archive)
            system.database.close()
            print("恢复完成；已运行的新版实例会在下次查询时刷新索引")
            return 0
        except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError) as error:
            print(f"恢复失败：{error}", file=sys.stderr)
            return 1

    if args.command == "evaluate":
        from .evaluation import evaluate
        try:
            output = Path(args.output)
            if output.exists():
                raise ValueError("输出文件已存在，请换一个名称保留实验记录")
            report = evaluate(config, project_root, args.manifest)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            print(f"完整结果：{output}")
            return 0
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            print(f"评估失败：{error}", file=sys.stderr)
            return 1

    if args.command == "cameras":
        manager = CameraManager(config.camera)
        devices = manager.discover()
        if not devices:
            print("未发现可读摄像头", file=sys.stderr)
            return 2
        for device in devices:
            preferred = " [默认]" if device.index == config.camera.preferred_index else ""
            print(f"{device.index}\t{device.label}{preferred}")
        return 0

    if args.command == "gui":
        from .gui_runtime import ensure_gui_runtime
        ensure_gui_runtime(project_root, ["-m", "face_compare.cli", *sys.argv[1:]])
        from .ui import launch_ui

        launch_ui(config, project_root)
        return 0

    try:
        system = FaceComparisonSystem(config, project_root)
        if args.command == "enroll":
            person, rejected = system.enroll_images(
                args.name,
                args.images,
                allow_low_quality=args.allow_low_quality,
            )
            calibration = system.calibrate_threshold()
            print(
                f"录入成功：{person.name}，总样本 {person.sample_count}；"
                f"阈值 {calibration.threshold:.3f}（{calibration.source}）"
            )
            for item in rejected:
                print(f"跳过：{item}", file=sys.stderr)
            return 0
        if args.command == "recognize":
            exit_code = 0
            for image in args.images:
                observations = system.recognize_image(image)
                if not observations:
                    record = {"image": image, "faces": 0, "error": "未检测到人脸"}
                    print(json.dumps(record, ensure_ascii=False) if args.json else f"{image}: 未检测到人脸")
                    exit_code = 3
                    continue
                for index, observation in enumerate(observations):
                    record = _recognition_dict(image, index, observation)
                    if args.json:
                        print(json.dumps(record, ensure_ascii=False))
                    else:
                        verdict = "通过" if record["known"] else "不通过"
                        print(
                            f"{image} [人脸 {index + 1}]: {record['name']} / {verdict} / "
                            f"相似度 {record['similarity']:.1f}% / 距离 {record['distance']:.3f}"
                        )
            return exit_code
        if args.command == "list":
            people = system.database.list_people()
            if not people:
                print("标准库为空")
            for person in people:
                print(f"{person.person_id}\t{person.name}\t{person.sample_count} 个样本")
            return 0
        if args.command == "calibrate":
            result = system.calibrate_threshold()
            print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
            return 0
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
