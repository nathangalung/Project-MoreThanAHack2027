"""Acceptance scenarios for the file."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from pytest_bdd import given, scenarios, then, when
from test_pipeline import toy_match

from possession import adapt, predict
from possession.decode import adapted_teams
from possession.match import DEAD, Match
from possession.model import PossessionNet, predict_match

scenarios("features/submission.feature")

FRAMES = 1200
BLOCKS = ((100, 600), (700, 1150))
KICKOFFS = (130, 740)


@given("a match with two graded halves", target_fixture="match")
def match_with_halves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Match:
    match = replace(toy_match(FRAMES), blocks=BLOCKS, label=None)
    truth = np.full(FRAMES, "X")
    truth[150:400] = np.repeat(np.array(["H", "A", "D", "H", "A"]), 50)
    public = tmp_path / "public.csv"
    pl.DataFrame({"frame": np.arange(FRAMES), "label": truth}).write_csv(public)
    for module in (predict, adapt):
        monkeypatch.setattr(module, "target_match", lambda: match)
        monkeypatch.setattr(module, "KICKOFFS", KICKOFFS)
        monkeypatch.setattr(module, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict, "PUBLIC_GT", public)
    monkeypatch.setattr(adapt, "CROP", 64)
    monkeypatch.setattr(adapt, "STEPS", 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return match


@given("trained and adapted networks", target_fixture="folder")
def saved_networks(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    for name in ("model", "model_adapted"):
        torch.save([PossessionNet().state_dict() for _ in range(2)], tmp_path / f"{name}.pt")
    (tmp_path / "model_decoder.json").write_text(json.dumps({"penalty": 20.0, "min_spell": 0, "min_dead": 0}))
    return tmp_path


@when("the prediction runs", target_fixture="rows")
def prediction_runs(folder: Path, capsys: pytest.CaptureFixture[str]) -> pl.DataFrame:
    labels, prob = predict.run(out=folder / "out.csv")
    assert labels.shape == (FRAMES,) and prob.shape == (FRAMES, 3)
    assert "public window spellF1" in capsys.readouterr().out
    return pl.read_csv(folder / "out.csv")


@when("the prediction runs without the adapted networks", target_fixture="rows")
def prediction_runs_plain(folder: Path, monkeypatch: pytest.MonkeyPatch) -> pl.DataFrame:
    out = folder / "plain.csv"
    monkeypatch.setattr(sys, "argv", ["predict.py", "--adapted", "", "--out", str(out)])
    predict.main()
    return pl.read_csv(out)


@then("the file has one row per frame")
def one_row_per_frame(rows: pl.DataFrame) -> None:
    assert rows.columns == ["frame", "label"]
    assert rows["frame"].to_list() == list(range(FRAMES))


@then("frames outside the halves are X")
def outside_is_unlabelled(rows: pl.DataFrame) -> None:
    labels = rows["label"].to_numpy()
    inside = np.zeros(FRAMES, dtype=bool)
    for start, stop in BLOCKS:
        inside[start:stop] = True
    assert (labels[~inside] == "X").all()


@then("graded frames are H, A, or D")
def graded_frames_are_labelled(rows: pl.DataFrame) -> None:
    labels = rows["label"].to_numpy()
    for start, stop in BLOCKS:
        assert set(labels[start:stop]) <= {"H", "A", "D"}


@then("every graded frame up to a kickoff is D")
def dead_before_kickoff(rows: pl.DataFrame) -> None:
    labels = rows["label"].to_numpy()
    for (start, _), kickoff in zip(BLOCKS, KICKOFFS, strict=True):
        assert (labels[start : kickoff + 1] == "D").all()


@given("frame probabilities from the trained networks", target_fixture="trained")
def trained_probabilities() -> np.ndarray:
    return np.array([[0.5, 0.1, 0.4], [0.2, 0.2, 0.6], [0.1, 0.1, 0.8]])


@given("frame probabilities from the adapted networks", target_fixture="adapted")
def adapted_probabilities() -> np.ndarray:
    return np.array([[0.1, 0.3, 0.6], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])


@when("the team split is taken from the adapted networks", target_fixture="combined")
def combine(trained: np.ndarray, adapted: np.ndarray) -> np.ndarray:
    return adapted_teams(trained, adapted)


@then("the dead ball probability is unchanged")
def dead_is_unchanged(combined: np.ndarray, trained: np.ndarray) -> None:
    assert np.allclose(combined[:, DEAD], trained[:, DEAD])


@then("the live probability follows the adapted team split")
def live_follows_adapted(combined: np.ndarray) -> None:
    assert np.allclose(combined[0, :DEAD], [0.15, 0.45])
    assert np.allclose(combined[1, :DEAD], [0.36, 0.04])
    # No adapted team mass, none kept
    assert np.allclose(combined[2, :DEAD], 0.0)


@when("the networks adapt to the match", target_fixture="pseudo")
def networks_adapt(match: Match, folder: Path, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    models = predict.load_models(["model"], "cpu")
    settings = json.loads((folder / "model_decoder.json").read_text())
    confident = np.zeros((FRAMES, 3), dtype=np.float32)
    confident[:, DEAD] = 1.0
    monkeypatch.setattr(adapt, "predict_views", lambda *args: confident)
    pseudo = adapt.pseudo_labels(models, match, [], settings, "cpu")
    monkeypatch.undo()
    monkeypatch.setattr(sys, "argv", ["adapt.py", "--name", "model"])
    for module in (predict, adapt):
        monkeypatch.setattr(module, "target_match", lambda: match)
        monkeypatch.setattr(module, "KICKOFFS", KICKOFFS)
        monkeypatch.setattr(module, "MODEL_DIR", folder)
    monkeypatch.setattr(adapt, "CROP", 64)
    monkeypatch.setattr(adapt, "STEPS", 1)
    monkeypatch.setattr(adapt, "CONFIDENCE", 0.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (folder / "model_adapted.pt").unlink()
    adapt.main()
    return pseudo


@then("adapted weights are saved for every network")
def adapted_weights_saved(folder: Path, match: Match) -> None:
    students = predict.load_models(["model_adapted"], "cpu")
    assert len(students) == 2
    assert predict_match(students, match, "cpu").shape == (FRAMES, 3)


@then("frames before a kickoff are left out of the pseudo-labels")
def kickoff_frames_left_out(pseudo: np.ndarray) -> None:
    for (start, stop), kickoff in zip(BLOCKS, KICKOFFS, strict=True):
        assert (pseudo[start : kickoff + 1] == adapt.UNUSED).all()
        assert (pseudo[kickoff + 1 : stop] == DEAD).all()
