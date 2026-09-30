import json

import pytest

from face_research import __main__ as cli
from face_research.evaluation import reporting
from face_research.evaluation.artifacts import RunJournal, write_json_once
from face_research.experiment import _source_hashes


def test_receipts_are_atomic_create_only(tmp_path):
    path = tmp_path / "frozen.json"
    write_json_once(path, {"threshold": 0.2})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"threshold": 0.9})
    assert json.loads(path.read_text()) == {"threshold": 0.2}
    with pytest.raises(ValueError):
        write_json_once(tmp_path / "nan.json", {"threshold": float("nan")})
    assert sorted(item.name for item in tmp_path.iterdir()) == ["frozen.json"]


def test_partial_results_are_never_published(tmp_path, monkeypatch):
    def fail_midway(report, staging):
        (staging / "metrics.json").write_text("partial")
        raise OSError("simulated plot write failure")
    monkeypatch.setattr(reporting, "_write_matrix_results", fail_midway)
    output = tmp_path / "run"
    with pytest.raises(OSError):
        reporting.write_matrix_results({}, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("after_freeze", [False, True])
def test_failure_journal_records_test_exposure_and_cannot_be_reused(tmp_path, after_freeze):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        with RunJournal(output) as journal:
            if after_freeze:
                journal.freeze({"threshold": 0.2})
                journal.test_opening()
            raise ValueError("private-name / private-path")
    failure_text = (journal.directory / "failed.json").read_text()
    assert "private-name" not in failure_text
    assert json.loads(failure_text)["test_may_have_been_accessed"] is after_freeze
    with pytest.raises(FileExistsError):
        with RunJournal(output):
            pytest.fail("an attempted run ID must never be reused")


def test_journal_refuses_test_without_a_saved_freeze(tmp_path):
    with pytest.raises(RuntimeError, match="冻结凭据"):
        with RunJournal(tmp_path / "run") as journal:
            journal.test_opening()
    assert not (journal.directory / "test_opening.json").exists()


def test_existing_destination_blocks_matrix_before_reading_test(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(cli, "run_matrix_experiment", lambda *a, **kw: pytest.fail("must not execute"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--matrix", "--validation", "v", "--test", "t", "--output", str(output)])
    assert error.value.code == 2


def test_cli_does_not_silently_ignore_pair_protocol(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(["--validation", "v", "--test", "t", "--output", str(tmp_path / "run"),
                  "--verification-test-pairs", "pairs.json"])
    assert error.value.code == 2


def test_code_provenance_includes_nested_modules_but_not_tests(tmp_path):
    for relative in ("main.py", "evaluation/metrics.py", "datasets/live.py", "tests/test_x.py",
                     "data/private_name.py", "results/private_run.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    first = _source_hashes(tmp_path)
    assert set(first) == {"main.py", "evaluation/metrics.py", "datasets/live.py"}
    (tmp_path / "evaluation/metrics.py").write_text("# changed\n")
    assert _source_hashes(tmp_path)["evaluation/metrics.py"] != first["evaluation/metrics.py"]
