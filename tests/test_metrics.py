"""Compare metric with scorer."""

import numpy as np
import polars as pl
import pytest

from possession.match import DEAD
from possession.metrics import macro_f1, spell_f1
from possession.paths import PUBLIC_GT, RESULTS_DIR

import possession_metrics as official  # isort: skip

OURS = {"H": 0, "A": 1, "D": 2, "X": 3}


def codes(path) -> np.ndarray:
    return pl.read_csv(path)["label"].replace_strict(OURS, return_dtype=pl.Int8).to_numpy()


@pytest.mark.parametrize("name", ["baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"])
def test_matches_official_scorer(name: str) -> None:
    truth, guess = codes(PUBLIC_GT), codes(RESULTS_DIR / f"{name}.csv")
    expected = official.evaluate(truth, guess)
    blocks = tuple(official.valid_runs(truth))
    graded = truth != 3
    assert spell_f1(truth, guess, blocks) == pytest.approx(expected["spell_f1_mean"])
    assert macro_f1(truth[graded], guess[graded]) == pytest.approx(expected["macro_f1"])


def test_constant_guess_scores_zero() -> None:
    truth = np.array([0, 0, 1, 1, DEAD, 0], dtype=np.int8)
    assert spell_f1(truth, np.full(6, DEAD, dtype=np.int8), ((0, 6),)) == 0.0
