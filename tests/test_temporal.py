import pandas as pd
import pytest

from thematicml.temporal import choose_frequency, temporal_analysis
from thematicml.temporal.trends import change_points, exponential_smoothing, mann_kendall, rolling_mean, seasonal_decomposition


def test_mann_kendall_increasing():
    r = mann_kendall([1, 2, 3, 4, 5, 6, 7, 8])
    assert r["trend"] == "increasing" and r["p_value"] < 0.01 and r["sens_slope"] == 1.0
    assert mann_kendall([5, 5, 5, 5])["trend"] == "no trend"
    assert mann_kendall([1, 2])["available"] is False


def test_mann_kendall_no_trend():
    assert mann_kendall([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])["trend"] == "no significant trend"


def test_change_point_detects_step():
    y = [1.0] * 10 + [10.0] * 10
    assert change_points(y) == [10]
    assert change_points([1.0, 1.1, 0.9, 1.0, 1.05, 0.95]) == []


def test_smoothing():
    assert rolling_mean([1, 2, 3, 4], 3) == [None, None, 2.0, 3.0]
    assert exponential_smoothing([0, 10], 0.5) == [0.0, 5.0]


def test_seasonal_requires_two_cycles():
    assert seasonal_decomposition([1] * 12, "M")["available"] is False


def test_choose_frequency():
    ts = pd.Series(pd.to_datetime(["2024-01-01", "2024-12-31"]))
    assert choose_frequency(ts) == "M"


def test_temporal_unavailable_without_timestamps():
    segs = pd.DataFrame({"record_id": [1, 2], "timestamp": [None, None]})
    r = temporal_analysis(segs, pd.DataFrame({"code_key": [], "segment_pos": []}), {})
    assert r["available"] is False and "timestamp" in r["reason"].lower()


def test_temporal_detects_planted_trend():
    rows, assigns = [], []
    pos = 0
    for m in range(1, 13):
        for i in range(20):
            rows.append({"record_id": pos, "timestamp": f"2024-{m:02d}-10T10:00:00"})
            if i < m:  # code share increases each month
                assigns.append({"code_key": "perf", "segment_pos": pos})
            pos += 1
    segs = pd.DataFrame(rows)
    r = temporal_analysis(segs, pd.DataFrame(assigns), {"perf": "Performance"}, freq="M", n_windows=3)
    assert r["available"] and r["frequency"] == "M" and len(r["periods"]) == 12
    code = r["codes"][0]
    assert code["trend"]["trend"] == "increasing"
    assert code["share_pct"][0] == pytest.approx(5.0) and code["share_pct"][-1] == pytest.approx(60.0)
    assert len(r["dynamic_graph"]["windows"]) == 3
