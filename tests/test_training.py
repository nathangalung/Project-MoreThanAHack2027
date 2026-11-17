"""Check training on toy matches."""

import json
import signal
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor
from pathlib import Path

import numpy as np
import pytest
import torch
from test_pipeline import toy_match

from possession import train
from possession.match import Match

FRAMES = 1200


class InlinePool(Executor):
    """Pool that runs in process."""

    def __init__(self, *args: object) -> None:
        self.args = args

    def map(
        self,
        fn: Callable,
        *iterables: Iterable,
        timeout: float | None = None,
        chunksize: int = 1,
        buffersize: int | None = None,
    ) -> Iterator:
        return map(fn, *iterables, strict=True)


@pytest.fixture
def toy_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    height = np.zeros(FRAMES, dtype=np.float32)
    monkeypatch.setattr(train, "load_dfl", lambda match_id: (toy_match(FRAMES), height))
    monkeypatch.setattr(train, "load_real", lambda match_id: toy_match(FRAMES))
    monkeypatch.setattr(train, "MATCHES", {"a": "x", "b": "x"})
    monkeypatch.setattr(train, "REAL_MATCHES", ("r",))
    monkeypatch.setattr(train, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(train, "CROP", 64)
    monkeypatch.setattr(train, "VARIANTS", 1)
    monkeypatch.setattr(train, "VALID_COPIES", 1)
    monkeypatch.setattr(train, "CHECK", 1)
    monkeypatch.setattr(train, "PENALTIES", (5.0, 20.0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return tmp_path


def test_samples_are_the_same_with_and_without_a_pool(toy_training: Path) -> None:
    alone = train.make_samples(["a", "b"], 1, 3)
    pooled = train.make_samples(["a", "b"], 1, 3, InlinePool())
    assert len(alone) == 2
    for (features, label, blocks), (other, other_label, other_blocks) in zip(alone, pooled, strict=True):
        assert np.array_equal(features, other) and np.array_equal(label, other_label)
        assert blocks == other_blocks == ((0, FRAMES),)
    assert train.copies(["a", "b"], 2, 1) == (["a", "a", "b", "b"], [10000, 10001, 10100, 10101])


def test_batches_are_crops_of_the_samples(toy_training: Path) -> None:
    samples = train.make_samples(["a"], 1, 0)
    x, y = train.draw_batch(samples, np.random.default_rng(0), "cpu")
    assert x.shape == (train.BATCH, 64, samples[0][0].shape[1])
    assert y.shape == (train.BATCH, 64) and y.dtype == torch.int64


def test_group_score_is_one_for_true_paths() -> None:
    match: Match = toy_match(900)
    assert match.label is not None
    assert train.group_score([match.label], [match], 0, 0) == 1.0


def test_run_trains_tunes_and_saves(toy_training: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scores = train.run(valid_ids=("a",), epochs=1, steps=1, seeds=(0,), name="toy")
    assert len(scores) == 2 and all(0.0 <= score <= 1.0 for score in scores)
    decoder = json.loads((toy_training / "models" / "toy_decoder.json").read_text())
    assert set(decoder) == {"penalty", "min_spell", "min_dead"}
    states = torch.load(toy_training / "models" / "toy.pt", weights_only=True)
    assert len(states) == 1
    assert (toy_training / "toy_validation.npz").is_file()
    printed = capsys.readouterr().out
    assert "valid macroF1" in printed and "step 2 of 2" in printed


def test_run_rejects_bad_arguments(toy_training: Path) -> None:
    with pytest.raises(ValueError, match="held-out ids"):
        train.run(valid_ids=("zzz",))
    with pytest.raises(ValueError, match="held-out ids"):
        train.run(valid_ids=())
    with pytest.raises(ValueError, match="seed"):
        train.run(valid_ids=("a",), seeds=())


def test_command_line_passes_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(train, "run", lambda *args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["train.py", "--epochs", "2", "--steps", "3", "--seeds", "7", "--name", "n"])
    train.main()
    assert seen == [(train.VALID_IDS, 2, 3, (7,), "n")]


def test_workers_ignore_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(signal, "signal", lambda number, handler: seen.append((number, handler)))
    train.ignore_interrupt()
    assert seen == [(signal.SIGINT, signal.SIG_IGN)]
