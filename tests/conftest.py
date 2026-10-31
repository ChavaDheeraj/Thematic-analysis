import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_sample", ROOT / "sample_data" / "generate_sample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sample_rows():
    return _load_generator().generate(400)


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory, sample_rows):
    path = tmp_path_factory.mktemp("data") / "feedback.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        w.writeheader()
        w.writerows(sample_rows)
    return path


@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "ws"


@pytest.fixture(scope="session")
def analysed(tmp_path_factory, sample_csv):
    """One fully analysed project shared by read-only tests."""
    from thematicml import AnalysisConfig, Project

    ws = tmp_path_factory.mktemp("ws_shared")
    p = Project("shared", workspace=ws)
    p.load(sample_csv)
    cfg = AnalysisConfig(text_col="response_text", id_col="respondent_id", time_col="created_at", group_col="region",
                         n_codes=30, random_state=7)
    result = p.analyze(cfg)
    assert result.status == "completed", result.run
    return p, result
