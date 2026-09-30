"""Paired exact-search benchmark; synthetic vectors, never biometric accuracy.

Both implementations use the same SQLite snapshot, matrix multiply and tuple
tie-break. The reference retains the previous per-identity scalar aggregation.
Order alternates per query to limit simple warm-cache/order bias.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from face_compare_system.face_compare.aggregation import aggregate_identity_score
from face_compare_system.face_compare.decision import accept_identity
from face_compare_system.face_compare.models import PreparedSample, QualityReport
from face_compare_system.face_compare.vector_database import SQLiteVectorDatabase

from ._baseline import PROJECT
from .benchmark import _machine_info, _milliseconds, _thread_limit
from .evaluation.artifacts import artifact_hashes, new_result_directory
from .evaluation.reporting import _csv, _write_json
from .experiment import _source_hashes


PROFILES = {"uniform_3": (3,), "mixed_1_2_3_5_8": (1, 2, 3, 5, 8)}
DECISION_THRESHOLDS = (0.0, 0.275, 0.5, 1.0)
DECISION_MARGINS = (0.0, 0.02, 0.04, 0.06)


def reference_rank(database, feature, nearest_samples=3):
    """Pre-optimization scalar algorithm, including validation/revision checks."""

    query = database._validate_vector(feature)
    with database._lock:
        database.refresh_cache()
        distances = np.clip((1 - database._matrix @ (query / np.linalg.norm(query))) / 2, 0, 1)
        return sorted((aggregate_identity_score(distances[start:end], "topk_median", nearest_samples),
                       person.person_id, person.name)
                      for start, end, person in database._ranges)


def _assert_equivalent(reference, candidate) -> int:
    # No tolerance: equal rankings alone could conceal boundary decision drift.
    if reference != candidate:
        raise ValueError("精确检索对照的距离或完整身份排名改变，拒绝发布性能结果")
    if not reference:
        return 0
    first = reference[0][0]
    second = reference[1][0] if len(reference) > 1 else None
    # Include exact threshold/margin boundaries and one-ULP neighbours.
    thresholds = (*DECISION_THRESHOLDS, first, float(np.nextafter(first, -np.inf)))
    margins = (*DECISION_MARGINS, *((second - first,) if second is not None else ()))
    checked = 0
    for threshold in thresholds:
        for margin in margins:
            expected = accept_identity(first, second, threshold, margin)
            actual = accept_identity(candidate[0][0], candidate[1][0] if len(candidate) > 1 else None,
                                     threshold, margin)
            if actual != expected:
                raise ValueError("精确检索对照改变了开放集判定")
            checked += 1
    return checked


def _timing(values):
    return {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "raw_ms": values}


def run_retrieval_benchmark(*, gallery_sizes=(10, 100, 1000), profiles=tuple(PROFILES),
                            iterations=200, warmup=20, dimension=128, seed=42,
                            opencv_threads=1):
    """Compare all-identity exact ranking without touching any existing DB."""

    if (not gallery_sizes or any(type(n) is not int or n < 1 for n in gallery_sizes)
            or type(iterations) is not int or iterations < 1
            or type(warmup) is not int or warmup < 0
            or type(dimension) is not int or dimension < 1
            or type(seed) is not int or seed < 0):
        raise ValueError("基准规模/次数/维度必须为正整数，warmup和seed须为非负整数")
    if not profiles or any(name not in PROFILES for name in profiles) or len(set(profiles)) != len(profiles):
        raise ValueError("必须选择非空且不重复的已定义模板分布")
    if type(opencv_threads) is not int or opencv_threads < 1:
        raise ValueError("opencv_threads必须为正整数")
    sizes = sorted(set(gallery_sizes))
    def sources():
        return {"research": _source_hashes(Path(__file__).parent),
                "production": _source_hashes(PROJECT / "face_compare")}
    snapshot = sources()
    outcomes = {}
    with _thread_limit(opencv_threads), tempfile.TemporaryDirectory(prefix="face-exact-benchmark-") as folder:
        for name in profiles:
            rng = np.random.default_rng(seed)
            queries = np.random.default_rng(seed + 1).normal(size=(iterations, dimension)).astype(np.float32)
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            database = SQLiteVectorDatabase(Path(folder) / name, expected_dimension=dimension,
                                            feature_signature=f"SYNTHETIC-{dimension}-NOT-A-FACE-MODEL",
                                            save_face_images=False)
            count = 0
            rows = {}
            try:
                for size in sizes:
                    started = time.perf_counter_ns()
                    for identity in range(count, size):
                        templates = PROFILES[name][identity % len(PROFILES[name])]
                        vectors = rng.normal(size=(templates, dimension)).astype(np.float32)
                        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                        samples = [PreparedSample(np.zeros((1, 1, 3), np.uint8), vector,
                                                  QualityReport(True, 128., 40., 100., .1)) for vector in vectors]
                        database.add_samples(f"synthetic_{identity:05d}", samples)
                    build_ms = _milliseconds(started)
                    added = size - count
                    count = size
                    database.refresh_cache()
                    for index in range(warmup):
                        query = queries[index % iterations]
                        _assert_equivalent(reference_rank(database, query), database.rank_candidates(query))
                    baseline, optimized = [], []
                    decision_checks = 0
                    for index, query in enumerate(queries):
                        operations = (("reference", reference_rank), ("batched", SQLiteVectorDatabase.rank_candidates))
                        if index % 2:
                            operations = operations[::-1]
                        ranks = {}
                        for label, operation in operations:
                            tick = time.perf_counter_ns()
                            ranks[label] = operation(database, query)
                            elapsed = _milliseconds(tick)
                            (baseline if label == "reference" else optimized).append(elapsed)
                        decision_checks += _assert_equivalent(ranks["reference"], ranks["batched"])
                    scalar, batch = _timing(baseline), _timing(optimized)
                    rows[str(size)] = {
                        "identities": size, "templates": len(database._matrix),
                        "template_count_pattern": list(PROFILES[name]), "nearest_samples": 3,
                        "reference": scalar, "batched": batch,
                        "p50_speedup": scalar["p50_ms"] / batch["p50_ms"],
                        "paired_raw_speedup": [a / b for a, b in zip(baseline, optimized)],
                        "equivalence": {"queries": iterations, "full_rank_score_exact_matches": iterations,
                                        "open_set_decisions_checked": decision_checks,
                                        "note": "exact equality; not approximate candidate recall or biometric accuracy"},
                        "cache_index_bytes": sum(ids.nbytes + indices.nbytes
                                                 for ids, indices in database._aggregation_groups),
                        "single_vector_matrix_bytes": database._matrix.nbytes,
                        "incremental_build_ms": build_ms, "identities_added": added,
                        "memory_note": "array payloads only, not total process RSS",
                    }
                outcomes[name] = rows
            finally:
                database.close()
        machine = _machine_info()
        machine["opencv_thread_control"] = {"requested": opencv_threads, "reported": cv2.getNumThreads(),
                                            "request_matched_report": cv2.getNumThreads() == opencv_threads}
    if sources() != snapshot:
        raise ValueError("基准运行期间代码发生变化，拒绝混合版本结果")
    return {
        "protocol": "paired-exact-retrieval-v1", "evidence_kind": "synthetic_performance_and_equivalence_only",
        "seed": seed, "query_seed": seed + 1, "dimension": dimension, "gallery_sizes": sizes,
        "iterations": iterations, "warmup": warmup, "machine": machine, "source_sha256": snapshot,
        "query_order": "reference-first for even indices, batched-first for odd indices; one Python worker",
        "results": outcomes,
        "limitations": ["合成向量不评价人脸精度，所有得分/排名相等只验证算法实现等价",
                        "单机微基准不含检测/特征提取/UI/摄像头；不是产品端到端SLA或相机FPS",
                        "OpenCV请求可能未生效，BLAS原生线程不受该参数控制；无多进程并发测试",
                        "相同预算分组的收益受模板分布影响；不宣称所有库形状均同幅加速",
                        "仍为全库精确检索，不是ANN，也不是FAISS性能比较"],
    }


def write_retrieval_results(report, output_dir):
    destination = Path(output_dir).resolve()
    with new_result_directory(destination) as staging:
        _write_json(staging / "metrics.json", report)
        rows = []
        for profile, sizes in report["results"].items():
            for result in sizes.values():
                rows.append({"profile": profile, "identities": result["identities"],
                             "templates": result["templates"],
                             "reference_p50_ms": result["reference"]["p50_ms"],
                             "reference_p95_ms": result["reference"]["p95_ms"],
                             "batched_p50_ms": result["batched"]["p50_ms"],
                             "batched_p95_ms": result["batched"]["p95_ms"],
                             "p50_speedup": result["p50_speedup"],
                             "exact_queries": result["equivalence"]["full_rank_score_exact_matches"],
                             "decision_checks": result["equivalence"]["open_set_decisions_checked"],
                             "cache_index_bytes": result["cache_index_bytes"]})
        _csv(staging / "metrics.csv", rows, list(rows[0]))
        _write_json(staging / "artifacts.json", {"schema": "retrieval-artifacts-v1", "sha256": artifact_hashes(staging)})
    return destination


def main():
    parser = argparse.ArgumentParser(description="同图库/同查询的标量与分组批量精确检索对照（不测准确率）")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gallery-sizes", default="10,100,1000")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opencv-threads", type=int, default=1)
    args = parser.parse_args()
    try:
        if Path(args.output_dir).exists():
            raise FileExistsError("结果已存在，请换新run ID")
        report = run_retrieval_benchmark(gallery_sizes=tuple(int(v) for v in args.gallery_sizes.split(",")),
                                         profiles=tuple(args.profiles), iterations=args.iterations,
                                         warmup=args.warmup, seed=args.seed, opencv_threads=args.opencv_threads)
        result = write_retrieval_results(report, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"检索对照未完成：{exc}\n")
    print(json.dumps({"output": str(result), "gallery_sizes": report["gallery_sizes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
