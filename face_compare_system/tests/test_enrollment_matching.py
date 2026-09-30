"""Synthetic vectors exercise enrollment policy, not real biometric accuracy."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from face_compare.config import EngineConfig, RecognitionConfig
from face_compare.database import FaceDatabase
from face_compare.deep_engine import SFaceExtractor
from face_compare.models import PreparedSample, QualityReport
from face_compare.recognizer import FaceRecognizer
from face_compare.service import FaceComparisonSystem
from face_compare.vector_database import SQLiteVectorDatabase


def sample(angle):
    radians = np.deg2rad(angle)
    return PreparedSample(np.zeros((112, 112, 3), np.uint8),
                          np.array([np.cos(radians), np.sin(radians)], np.float32),
                          QualityReport(True, 120, 40, 100, .2))


@pytest.fixture(params=["files", "sqlite"])
def system(tmp_path, request):
    system = FaceComparisonSystem.__new__(FaceComparisonSystem)
    system.config = SimpleNamespace(engine=EngineConfig(backend="sface"), recognition=RecognitionConfig())
    database = SQLiteVectorDatabase if request.param == "sqlite" else FaceDatabase
    system.database = database(tmp_path, save_face_images=False,
                               feature_signature="synthetic-test", expected_dimension=2)
    system.extractor = SFaceExtractor
    system.logger = SimpleNamespace(write=lambda *args, **kwargs: None)
    system.recognizer = FaceRecognizer(system.database, SFaceExtractor,
                                       replace(RecognitionConfig(), default_distance_threshold=.275, ambiguity_margin=.04))
    system.recognizer.use_calibration = False
    yield system
    if hasattr(system.database, "close"):
        system.database.close()


def seed_multiple_appearances(system):
    system.enroll_samples("A", [sample(0)]*5)
    system.enroll_samples("A", [sample(50)]*3)


def test_recognized_secondary_appearance_can_append_despite_old_majority(system):
    seed_multiple_appearances(system)
    assert system.recognizer.match(sample(80).feature).name == "A"
    system.enroll_samples("A", [sample(80)]*3)
    assert system.database.sample_count() == 11


def test_recognized_secondary_appearance_cannot_be_renamed_as_new_identity(system):
    seed_multiple_appearances(system)
    with pytest.raises(ValueError, match="原姓名"):
        system.enroll_samples("B", [sample(80)]*3)
    assert system.database.sample_count() == 8


def test_batch_cannot_bootstrap_distant_samples_from_other_new_samples(system):
    system.enroll_samples("A", [sample(0)]*3)
    with pytest.raises(ValueError, match="已有样本"):
        system.enroll_samples("A", [sample(50)]*3 + [sample(80)]*3)
    assert system.database.sample_count() == 3


def test_ambiguous_append_rejected(system):
    system.enroll_samples("A", [sample(0)]*3)
    system.enroll_samples("B", [sample(90)]*3)
    with pytest.raises(ValueError, match="歧义"):
        system.enroll_samples("A", [sample(45)]*3)
    assert system.database.sample_count() == 6


def test_distinct_person_still_can_enroll(system):
    system.enroll_samples("A", [sample(0)]*3)
    system.enroll_samples("B", [sample(90)]*3)
    assert len(system.database.list_people()) == 2
