"""Per-structure failure thresholds of plotting.failure_quadrants."""

import numpy as np
import pytest

from segevalkit import plotting as P
from segevalkit.plotting.quantitative import FAILURE_CLASSES, failure_thresholds


@pytest.mark.parametrize("label, cls", [
    ("liver", "large_organ"), ("kidney_left", "compact_organ"), ("pancreas", "elongated_organ"),
    ("gall_bladder", "small_organ"), ("aorta", "vessel"), ("postcava", "vessel"), ("veins", "small_vessel"),
    ("pancreatic_lesion", "lesion"), ("tumour", "lesion"),
])
def test_class_resolution(label, cls):
    t = failure_thresholds(label)
    assert t["class"] == cls
    assert t["dice"] == FAILURE_CLASSES[cls]["dice"] and t["hd95"] == t["error_mm"]


def test_unknown_structure_uses_default_and_overrides_win():
    assert failure_thresholds("organ")["class"].startswith("compact_organ")
    t = failure_thresholds("pancreas", {"pancreas": {"dice": 0.8}})
    assert t["dice"] == 0.8 and t["hd95"] == FAILURE_CLASSES["elongated_organ"]["hd95"]
    assert t["class"] == "user-defined"


def test_thresholds_are_consistent():
    for c in FAILURE_CLASSES.values():
        assert 0 < c["dice"] < 1 and c["hd95"] == c["error_mm"] > 0


def _title(ax):
    return " ".join(ax.get_title(loc=loc) for loc in ("left", "center", "right"))


class _Result:
    def __init__(self, df):
        self._df = df

    def wide(self):
        return self._df


def test_failure_quadrants_uses_structure_thresholds():
    import pandas as pd

    df = pd.DataFrame({"case_id": ["a", "b", "c"], "label": "pancreas", "dice": [0.9, 0.4, 0.8],
                       "hd95": [2.0, 3.0, 40.0]})
    fig = P.failure_quadrants(_Result(df), "pancreas")
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("< 0.5" in t for t in texts) and any("> 5 mm" in t for t in texts)
    assert "2/3 flagged" in _title(ax)
    fig = P.failure_quadrants(_Result(df), "pancreas", thresholds={"pancreas": {"dice": 0.95, "hd95": 100}})
    assert "3/3 flagged" in _title(fig.axes[0])   # every Dice is below 0.95


def test_failure_quadrants_explicit_override():
    import pandas as pd

    df = pd.DataFrame({"case_id": ["a", "b"], "label": "liver", "dice": [0.97, 0.9], "hd95": [1.0, 2.0]})
    fig = P.failure_quadrants(_Result(df), "liver", x_thr=0.95, y_thr=10)
    assert "1/2 flagged" in _title(fig.axes[0])
