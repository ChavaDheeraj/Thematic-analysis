import pandas as pd
import pytest

from thematicml.analytics import (
    benjamini_hochberg,
    code_frequency_table,
    coverage_metrics,
    group_comparison,
    percentage,
    two_proportion,
    wilson_ci,
)


@pytest.fixture
def toy():
    segments = pd.DataFrame({
        "record_id": [1, 1, 2, 3, 4],
        "record_key": ["A", "A", "A", "B", "C"],
        "group_value": ["g1", "g1", "g1", "g2", "g2"],
    })
    codes = pd.DataFrame({"code_key": ["x", "y"], "label": ["X", "Y"], "method": ["m", "m"]})
    assigns = pd.DataFrame({"code_key": ["x", "x", "x", "y"], "segment_pos": [0, 1, 3, 1]})
    return segments, codes, assigns


def test_percentage_definition():
    p = percentage(3, 12, "a / b", "record", ci=True)
    assert p["value"] == 25.0 and p["numerator"] == 3 and p["denominator"] == 12 and p["level"] == "record"
    assert p["ci95"][0] < 25 < p["ci95"][1]
    assert percentage(1, 0, "x")["value"] is None


def test_wilson_bounds():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0 < hi < 40
    assert wilson_ci(5, 0) is None


def test_frequency_levels(toy):
    segments, codes, assigns = toy
    f = code_frequency_table(codes, assigns, segments).set_index("code_key")
    assert f.loc["x", "assignments"] == 3
    assert f.loc["x", "segments"] == 3
    assert f.loc["x", "records"] == 2          # records 1 and 3
    assert f.loc["x", "respondents"] == 2      # A and B
    assert f.loc["x", "segment_pct"] == 60.0   # 3 / 5 segments
    assert f.loc["x", "record_pct"] == 50.0    # 2 / 4 records
    assert f.loc["x", "respondent_pct"] == pytest.approx(66.67)  # 2 / 3 respondents
    assert code_frequency_table(codes, assigns, segments).attrs["denominators"] == {"segments": 5, "records": 4, "respondents": 3}


def test_coverage(toy):
    segments, _, assigns = toy
    c = coverage_metrics(assigns, segments)
    assert c["coded_segments"]["numerator"] == 3 and c["coded_segments"]["denominator"] == 5
    assert c["uncoded_segments"]["numerator"] == 2
    assert c["coded_records"]["value"] == 50.0
    assert c["codes_per_respondent"]["max"] == 2


def test_bh():
    adj = benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert adj[0] == pytest.approx(0.04)
    assert adj[3] == pytest.approx(0.2)
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03, 0.2]))


def test_two_proportion():
    r = two_proportion(10, 100, 30, 100)
    assert r["diff_pct"] == 20.0
    assert r["ci95"][0] < 20 < r["ci95"][1]
    assert r["p_value"] < 0.01


def test_group_comparison_requires_groups(toy):
    segments, codes, assigns = toy
    r = group_comparison(assigns, segments, {"x": "X"}, ["x"], min_group_size=1)
    assert r["available"]
    res = r["results"][0]
    assert res["groups"]["g1"] == {"records": 2, "with_code": 1, "pct": 50.0}
    assert res["test"] in ("fisher_exact", "chi_square")
    assert group_comparison(assigns, segments.assign(group_value=None), {}, ["x"])["available"] is False
