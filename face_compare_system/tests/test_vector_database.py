import json
import sqlite3
import numpy as np
import pytest

from face_compare.database import FaceDatabase
from face_compare.deep_engine import SFaceExtractor
from face_compare.models import PreparedSample, QualityReport
from face_compare.vector_database import SQLiteVectorDatabase
from face_compare.aggregation import STRATEGIES, aggregate_identity_score
from face_compare.decision import accept_identity


def sample(vector=(1., 0.)):
    return PreparedSample(np.full((16, 16, 3), 120, np.uint8), np.array(vector, np.float32),
                          QualityReport(True, 120, 40, 100, .1))


def db(path):
    return SQLiteVectorDatabase(path, expected_dimension=2, feature_signature="test", save_face_images=True)


def test_migration_is_once_and_preserves_original_files(tmp_path):
    old = FaceDatabase(tmp_path, expected_dimension=2, feature_signature="test")
    old.add_samples("A", [sample()])
    before = (tmp_path/"people.json").read_bytes()
    current = db(tmp_path)
    assert current.sample_count() == 1
    assert (tmp_path/"people.json").read_bytes() == before
    archive = current.archive_people()
    current.close()
    reopened = db(tmp_path)
    assert reopened.sample_count() == 0  # never resurrect the old JSON entries
    reopened.restore_archive(archive)
    assert reopened.sample_count() == 1
    reopened.close()


def test_other_connection_updates_search_and_delete(tmp_path):
    a, b = db(tmp_path), db(tmp_path)
    assert b.rank_candidates(np.array([1., 0.])) == []
    person = a.add_samples("A", [sample()])
    assert b.rank_candidates(np.array([1., 0.]))[0][1] == person.person_id
    a.archive_people(person.person_id)
    assert b.rank_candidates(np.array([1., 0.])) == []
    a.close(); b.close()


def test_exact_search_matches_scalar_multi_template_distance(tmp_path):
    current = db(tmp_path)
    for name, vectors in (("A", [(1., 0.), (.8, .2), (.9, .1)]), ("B", [(0., 1.)])):
        current.add_samples(name, [sample(v) for v in vectors])
    query = np.array([.9, .1], np.float32)
    expected = sorted((float(np.median(sorted(SFaceExtractor.distance(query, v) for v in vectors)[:3])), p.person_id, p.name)
                      for p, vectors in current.feature_sets())
    actual = current.rank_candidates(query)
    assert [item[1:] for item in actual] == [item[1:] for item in expected]
    assert [item[0] for item in actual] == pytest.approx([item[0] for item in expected], abs=1e-7)
    current.close()


@pytest.mark.parametrize("vector", [(0., 0.), (float("nan"), 0.), (1., 2., 3.)])
def test_bad_batch_not_partially_persisted(tmp_path, vector):
    current = db(tmp_path)
    with pytest.raises(ValueError):
        current.add_samples("A", [sample(), sample(vector)])
    assert current.sample_count() == 0
    current.close()


def test_sql_failure_rolls_back_whole_batch(tmp_path):
    current = db(tmp_path)
    current._conn.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON samples
        WHEN (SELECT count(*) FROM samples) >= 1 BEGIN SELECT RAISE(ABORT,'injected'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        current.add_samples("A", [sample(), sample()])
    assert current.sample_count() == 0
    assert current._conn.execute("SELECT count(*) FROM people").fetchone()[0] == 0
    current.close()


def test_restore_name_conflict_rolls_back(tmp_path):
    current = db(tmp_path)
    current.add_samples("A", [sample()])
    archive = current.archive_people()
    current.add_samples("A", [sample((0., 1.))])
    with pytest.raises(ValueError, match="同名"):
        current.restore_archive(archive)
    assert current.sample_count() == 1
    current.close()


def test_changed_legacy_data_is_not_silently_overwritten(tmp_path):
    old = FaceDatabase(tmp_path, expected_dimension=2, feature_signature="test")
    current = db(tmp_path)
    old.add_samples("A", [sample()])
    with pytest.raises(RuntimeError, match="旧版"):
        current.list_people()
    current.close()


def test_model_mismatch_rejected_even_for_empty_sqlite(tmp_path):
    current = db(tmp_path)
    current.close()
    with pytest.raises(RuntimeError, match="不一致"):
        SQLiteVectorDatabase(tmp_path, expected_dimension=2, feature_signature="other")


def test_outer_transaction_rolls_back_cached_nested_write(tmp_path):
    current = db(tmp_path)
    with pytest.raises(RuntimeError, match="abort"):
        with current.write_guard():
            current.add_samples("A", [sample()])
            assert current.sample_count() == 1
            raise RuntimeError("abort")
    assert current.sample_count() == 0
    current.add_samples("B", [sample((0., 1.))])
    assert current.list_people()[0].name == "B"
    current.close()


def scalar_reference(database, query, strategy="topk_median", k=3):
    """Historical ranking on the same revision, vectors and BLAS operation."""
    database.refresh_cache()
    query = database._validate_vector(query)
    distances = np.clip((1 - database._matrix @ (query / np.linalg.norm(query))) / 2, 0, 1)
    return sorted((aggregate_identity_score(distances[start:end], strategy, k), person.person_id, person.name)
                  for start, end, person in database._ranges)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_batched_ranking_preserves_all_scores_ties_and_open_set_boundaries(tmp_path, strategy):
    rng = np.random.default_rng(20260926)
    database = SQLiteVectorDatabase(tmp_path, expected_dimension=16, feature_signature="synthetic", save_face_images=False)
    try:
        for i, count in enumerate((1, 2, 3, 5, 8, 17, 1, 3, 5)):
            samples = [sample(rng.normal(size=16)) for _ in range(count)]
            database.add_samples(f"synthetic-{i}", samples)
        # Identical identities must keep the old UUID tie-break, not bucket order.
        identical = sample(np.ones(16))
        database.add_samples("tie-a", [identical] * 3)
        database.add_samples("tie-b", [identical] * 3)
        for query in [np.ones(16), *rng.normal(size=(15, 16))]:
            for k in (1, 2, 3, 5, 99):
                actual = database.rank_candidates(query, k, strategy=strategy)
                expected = scalar_reference(database, query, strategy, k)
                assert actual == expected  # not approximate equality
                first, second = actual[0][0], actual[1][0]
                for threshold in (0., .275, first, np.nextafter(first, -np.inf), 1.):
                    for margin in (0., .04, second - first, np.nextafter(second - first, np.inf)):
                        assert accept_identity(first, second, threshold, margin) == accept_identity(
                            expected[0][0], expected[1][0], threshold, margin)
    finally:
        database.close()


def test_batch_groups_follow_append_archive_restore_and_transaction_rollback(tmp_path):
    writer, reader = db(tmp_path), db(tmp_path)
    query = np.array([.8, .2], np.float32)
    try:
        person = writer.add_samples("A", [sample()])
        writer.add_samples("B", [sample((0., 1.))] * 3)
        assert reader.rank_candidates(query) == scalar_reference(reader, query)
        writer.add_samples("A", [sample((.8, .2)), sample((.9, .1))])
        assert reader.rank_candidates(query) == scalar_reference(reader, query)
        assert len(reader._aggregation_groups) == 1  # both now have three templates
        archive = writer.archive_people(person.person_id)
        assert len(reader.rank_candidates(query)) == 1
        writer.restore_archive(archive)
        before = reader.rank_candidates(query)
        with pytest.raises(RuntimeError, match="rollback"):
            with writer.write_guard():
                writer.add_samples("A", [sample((.7, .3))])
                assert writer.rank_candidates(query) == scalar_reference(writer, query)
                raise RuntimeError("rollback")
        assert reader.rank_candidates(query) == before
        assert writer.rank_candidates(query) == before
        writer.archive_people()
        assert reader.rank_candidates(query) == []
        assert reader._aggregation_groups == ()
    finally:
        writer.close()
        reader.close()
