import json

import pytest

from face_research.evaluation.artifacts import verify_result_directory
from face_research.retrieval_benchmark import _assert_equivalent, run_retrieval_benchmark, write_retrieval_results


def test_paired_retrieval_runs_real_isolated_sqlite_and_publishes_raw_timings(tmp_path):
    report = run_retrieval_benchmark(gallery_sizes=(2, 5), iterations=4, warmup=1, dimension=8)
    for sizes in report["results"].values():
        for size, result in sizes.items():
            assert result["identities"] == int(size)
            assert result["equivalence"]["full_rank_score_exact_matches"] == 4
            assert result["equivalence"]["open_set_decisions_checked"] == 4 * 6 * 5
            assert len(result["reference"]["raw_ms"]) == len(result["batched"]["raw_ms"]) == 4
            assert result["cache_index_bytes"] > 0
            assert result["p50_speedup"] > 0  # no environment-dependent speed assertion
    output = write_retrieval_results(report, tmp_path / "paired")
    assert verify_result_directory(output)["files"] == 2
    assert json.loads((output / "metrics.json").read_text())["evidence_kind"].startswith("synthetic")
    with pytest.raises(FileExistsError):
        write_retrieval_results(report, output)


def test_even_tiny_distance_drift_is_not_permitted():
    with pytest.raises(ValueError, match="距离或完整身份排名"):
        _assert_equivalent([(.275, "1", "A")], [(.2750000000000001, "1", "A")])
    assert _assert_equivalent([], []) == 0
    assert _assert_equivalent([(.275, "1", "A")], [(.275, "1", "A")]) == 24


@pytest.mark.parametrize("kwargs", [{"iterations": 0}, {"gallery_sizes": ()}, {"gallery_sizes": (True,)},
                                    {"profiles": ()}, {"profiles": ("bad",)}, {"seed": -1},
                                    {"profiles": ("uniform_3", "uniform_3")}, {"opencv_threads": 0}])
def test_invalid_benchmark_arguments_fail_before_work(kwargs):
    with pytest.raises(ValueError):
        run_retrieval_benchmark(**kwargs)
