import pytest

from face_research.manifest import check_manifest_splits, check_session_isolation, near_duplicate_warnings


def _groups(gallery_session="enroll", validation_session="day2", test_session="day3"):
    gallery = [{"label": "person_a", "session_id": gallery_session}]
    validation = [{"label": "person_a", "session_id": validation_session}]
    test = [{"label": "person_a", "session_id": test_session}]
    return {"gallery": gallery, "probes": validation}, {"gallery": gallery, "probes": test}


def test_subject_session_must_not_cross_gallery_validation_or_test():
    validation, test = _groups()
    check_session_isolation(validation, test, require_session_ids=True)
    for bad in (_groups(validation_session="enroll"), _groups(test_session="day2")):
        with pytest.raises(ValueError, match="会话级泄漏"):
            check_session_isolation(*bad, require_session_ids=True)


def test_strict_protocol_rejects_missing_session_metadata():
    validation, test = _groups(validation_session=None)
    check_session_isolation(validation, test)
    with pytest.raises(ValueError, match="不能证明"):
        check_session_isolation(validation, test, require_session_ids=True)


def test_session_names_are_scoped_to_subject():
    validation, test = _groups()
    test["probes"].append({"label": None, "subject_id": "stranger", "session_id": "day2"})
    check_session_isolation(validation, test, require_session_ids=True)


def test_perceptual_near_duplicates_are_warnings_not_automatic_identity_claims():
    gallery = [{"label": "a", "sha256": "g", "dhash64": 0b1010}]
    validation = {"gallery": gallery, "probes": [{"label": "a", "sha256": "v", "dhash64": 0b1011}]}
    test = {"gallery": gallery, "probes": [{"label": "b", "sha256": "t", "dhash64": 0b1010}]}
    warnings = near_duplicate_warnings(validation, test, max_hamming=1)
    assert len(warnings) == 1
    assert warnings[0]["left_split"] == "gallery"
    assert warnings[0]["right_split"] == "validation"
    assert warnings[0]["dhash_hamming"] == 1


@pytest.mark.parametrize("field", ["condition", "session_id"])
def test_gallery_metadata_cannot_change_between_manifest_copies(field):
    validation, test = _groups()
    test["gallery"] = [{**row, field: "changed"} for row in validation["gallery"]]
    for groups in (validation, test):
        for row in groups["gallery"] + groups["probes"]:
            row.setdefault("sha256", row["session_id"])
    with pytest.raises(ValueError, match="gallery"):
        check_manifest_splits(validation, test)
