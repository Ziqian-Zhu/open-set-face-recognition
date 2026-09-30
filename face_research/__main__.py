"""Run with ``python -m face_research`` from the workspace root."""

from __future__ import annotations

import argparse

from .experiment import run_experiment, write_report
from .matrix import run_matrix_experiment
from .evaluation.reporting import write_matrix_results
from .quality_ablation import run_quality_ablation, write_quality_results
from .evaluation.artifacts import RunJournal


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="隔离验证/测试的人脸模板选样实验")
    parser.add_argument("--validation", required=True, help="validation清单JSON")
    parser.add_argument("--test", required=True, help="test清单JSON")
    parser.add_argument("--output", required=True, help="旧模式为结果JSON；--matrix/--quality-ablation时为新结果目录，不覆盖")
    parser.add_argument("--matrix", action="store_true", help="运行预算×选样×聚合矩阵及联合阈值/margin标定")
    parser.add_argument("--quality-ablation", action="store_true", help="固定完整质量gallery，对同一probe池比较四种画质门控")
    parser.add_argument("--budgets", default="1,2,3,5", help="--matrix预算列表，默认1,2,3,5")
    parser.add_argument("--random-seeds", default="42,43,44", help="--matrix随机基线种子列表")
    parser.add_argument("--margins", default="0,0.02,0.04,0.06", help="--matrix验证集候选距离间隔")
    parser.add_argument("--allow-missing-session-ids", action="store_true",
                        help="仅探索用：缺少采集批次信息时允许运行，但不能声称批次独立")
    parser.add_argument("--verification-test-pairs", help="--matrix可选：独立的test 1:1图片对清单JSON")
    parser.add_argument("--verification-validation-pairs", help="--matrix可选：validation 1:1图片对，用于阈值扫描的FAR/FRR/TAR列")
    parser.add_argument("--protocol-spec", help="--matrix固定协议JSON；核对主比较及已声明的网格/FPIR目标")
    parser.add_argument(
        "--config", help="课程配置文件，默认使用face_compare_system/config.json"
    )
    parser.add_argument("--budget", type=int, default=3, help="每人模板上限，默认3")
    parser.add_argument(
        "--target-fpir",
        type=float,
        default=0.05,
        help="验证集可用未知人脸经验误接受率上限，默认0.05",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机基线种子")
    args = parser.parse_args(argv)
    if args.matrix and args.quality_ablation:
        parser.error("--matrix和--quality-ablation不能同时使用")
    if not args.matrix and (args.verification_validation_pairs or args.verification_test_pairs):
        parser.error("1:1验证对参数仅支持--matrix；不能静默忽略验证对")
    if args.protocol_spec and not args.matrix:
        parser.error("--protocol-spec只支持--matrix")
    if args.allow_missing_session_ids and not (args.matrix or args.quality_ablation):
        parser.error("--allow-missing-session-ids仅支持--matrix/--quality-ablation")
    try:
        if args.quality_ablation:
            with RunJournal(args.output, kind="quality_ablation") as journal:
                report = run_quality_ablation(
                    args.validation, args.test, budget=args.budget,
                    target_fpir=args.target_fpir,
                    margins=tuple(float(value.strip()) for value in args.margins.split(",")),
                    config_path=args.config,
                    require_session_ids=not args.allow_missing_session_ids,
                    on_freeze=journal.freeze, on_test_open=journal.test_opening,
                )
                output = write_quality_results(report, args.output)
                journal.completed(output)
        elif args.matrix:
            with RunJournal(args.output) as journal:
                report = run_matrix_experiment(
                    args.validation,
                    args.test,
                    budgets=tuple(int(value.strip()) for value in args.budgets.split(",")),
                    random_seeds=tuple(int(value.strip()) for value in args.random_seeds.split(",")),
                    margins=tuple(float(value.strip()) for value in args.margins.split(",")),
                    target_fpir=args.target_fpir,
                    config_path=args.config,
                    require_session_ids=not args.allow_missing_session_ids,
                    verification_validation_pairs=args.verification_validation_pairs,
                    verification_test_pairs=args.verification_test_pairs,
                    protocol_spec=args.protocol_spec,
                    on_freeze=journal.freeze,
                    on_test_open=journal.test_opening,
                )
                output = write_matrix_results(report, args.output)
                journal.completed(output)
        else:
            report = run_experiment(
                args.validation,
                args.test,
                budget=args.budget,
                target_fpir=args.target_fpir,
                seed=args.seed,
                config_path=args.config,
            )
            output = write_report(report, args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"实验未完成：{exc}\n")
    print(f"已写入 {output}")
    if args.quality_ablation:
        print("画质门控消融：" + ", ".join(f"{name}={item['status']}" for name, item in report["variants"].items()))
        return
    if args.matrix:
        print(f"已冻结并评估 {len(report['variants'])} 个组合；1:1 verification：{report['verification']['status']}")
        print(f"冻结/测试访问记录：{journal.directory}")
        return
    for method, result in report["methods"].items():
        summary = result["test"]["summary"]
        print(
            f"{method:10} 阈值={result['threshold_from_validation']:.5f} "
            f"已知正确={summary['known_correct']['count']}/{summary['known_correct']['total']} "
            f"未知误接收={summary['unknown_false_accept_usable_faces']['count']}/"
            f"{summary['unknown_false_accept_usable_faces']['total']}"
        )


if __name__ == "__main__":
    main()
