import pytest

from face_research.evaluation.grouped import grouped_verification


def row(index, left, right, distance=.1, status="scored"):
    return {"pair_index": index, "left_subject": left, "right_subject": right,
            "genuine": left == right, "distance": distance, "status": status}


def test_shared_subject_chains_are_not_independent_pair_clusters():
    records = [row(0,"a","a"), row(1,"b","b"), row(2,"a","b",.9),
               row(3,"b","c",.9), row(4,"d","d",.4)]
    report = grouped_verification(records, .3, .2, iterations=100)
    assert report["subjects"] == 4 and report["components"] == 2
    assert report["max_component_pairs"] == 4
    assert report["metrics"]["tar_all_attempts"]["rate"] == 2/3
    assert report["metrics"]["tar_all_attempts"]["interval_status"] == "insufficient_components"


def test_macro_genuine_weights_subjects_and_failures_are_not_dropped():
    records = [row(i,"a","a") for i in range(10)] + [row(10,"b","b",None,"acquisition_failed")]
    report = grouped_verification(records, .3, .2, iterations=100)
    assert report["metrics"]["tar_all_attempts"]["rate"] == 10/11
    assert report["metrics"]["tar_usable_pairs"]["rate"] == 1
    assert report["macro_genuine_accept_rate"] == .5
    assert report["metrics"]["genuine_acquisition_failure_rate"]["rate"] == 1/11


def test_paired_bootstrap_reproducible_zero_error_never_yields_zero_risk_bound():
    records = []
    for i in range(40):
        records += [row(2*i,f"a{i}",f"a{i}",.1 if i < 20 else .28),
                    row(2*i+1,f"a{i}",f"b{i}",.9)]
    result = grouped_verification(records, .3, .2, iterations=500)
    assert result == grouped_verification(records, .3, .2, iterations=500)
    delta = result["metrics"]["tar_delta_vs_fixed_engineering_all_attempts"]
    assert delta["rate"] == .5 and delta["percentile95"][0] < .5 < delta["percentile95"][1]
    far = result["metrics"]["far_usable_pairs"]
    assert far["count"] == 0 and far["total"] == 40
    assert far["percentile95"] is None
    assert far["interval_status"] == "degenerate_resamples_no_risk_bound"


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"iterations": 99},
                                    {"minimum_components": 1}, {"iterations": 100001}])
def test_invalid_bootstrap_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        grouped_verification([], .3, .2, **kwargs)


@pytest.mark.parametrize("change", [{"genuine": False}, {"distance": float("nan")},
                                  {"status": "unknown"}, {"status": "acquisition_failed"},
                                  {"left_subject": ""}, {"pair_index": -1}])
def test_invalid_pair_records_fail(change):
    with pytest.raises(ValueError):
        grouped_verification([{**row(0,"a","a"), **change}], .3, .2)


def test_empty_or_all_failed_pairs_never_create_rates_or_intervals():
    result = grouped_verification([], .3, .2)
    assert result["components"] == 0
    assert all(metric["rate"] is None for metric in result["metrics"].values())
    result = grouped_verification([row(0,"a","b",None,"acquisition_failed")], .3, .2)
    assert result["metrics"]["far_usable_pairs"]["rate"] is None


def test_duplicate_pair_indices_fail():
    with pytest.raises(ValueError):
        grouped_verification([row(0,"a","a"), row(0,"b","b")], .3, .2)


def test_record_order_does_not_change_resampling_or_connected_components():
    records = [row(i, str(i), str(i), .1 if i % 2 else .4) for i in range(40)]
    records += [row(100, "1", "3", .9), row(101, "3", "5", .9)]
    assert grouped_verification(records, .3, .2) == grouped_verification(list(reversed(records)), .3, .2)


def test_bootstrap_does_not_silently_drop_empty_denominator_draws():
    records = [row(0, "a", "a"), row(1, "b", "c", .9)]
    result = grouped_verification(records, .3, .2, minimum_components=2, iterations=100)
    metric = result["metrics"]["tar_all_attempts"]
    assert metric["interval_status"] == "empty_denominator_resamples"
    assert metric["valid_resamples"] < 100 and metric["percentile95"] is None
