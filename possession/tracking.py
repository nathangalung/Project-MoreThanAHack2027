"""Load the automatic tracking data."""

import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from possession.paths import CACHE_DIR, asset

HOME, AWAY = 0, 1


@dataclass(frozen=True)
class Tracking:
    """Arrays indexed by video frame."""

    xy: np.ndarray
    speed: np.ndarray
    team: np.ndarray
    ball: np.ndarray
    ball_speed: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]


def slot_teams(meta_path: Path) -> np.ndarray:
    """Team code of each slot."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if slot == team_ids[0] else AWAY for slot in slots], dtype=np.int8)


def parse_raw(raw_path: Path, n_slots: int) -> np.ndarray:
    """Raw text to float matrix."""
    text = raw_path.read_bytes().replace(b":", b",").replace(b";", b",")
    n_columns = 1 + 3 * n_slots + 3
    table = pl.read_csv(
        io.BytesIO(text),
        has_header=False,
        new_columns=[f"c{i}" for i in range(n_columns)],
        schema_overrides=[pl.Float64] * n_columns,
        null_values=["NaN"],
    )
    if table.width != n_columns:
        raise ValueError(f"{raw_path}: expected {n_columns} columns, found {table.width}")
    return table.fill_null(float("nan")).to_numpy()


def load_tracking(source: str = "tactical", refresh: bool = False) -> Tracking:
    """Tracking arrays, cached on disk."""
    cache = CACHE_DIR / f"tracking_{source}.npz"
    if cache.is_file() and not refresh:
        with np.load(cache) as stored:
            return Tracking(**{name: stored[name] for name in stored.files})

    team = slot_teams(asset("atd_meta", source))
    matrix = parse_raw(asset("atd_raw", source), len(team))
    frame_ids = matrix[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, len(frame_ids) + 1)):
        raise ValueError("tracking frame ids are not 1..n in order")

    players = matrix[:, 1:-3].reshape(len(matrix), len(team), 3)
    tracking = Tracking(
        xy=players[:, :, :2].astype(np.float32),
        speed=players[:, :, 2].astype(np.float32),
        team=team,
        ball=matrix[:, -3:-1].astype(np.float32),
        ball_speed=matrix[:, -1].astype(np.float32),
    )
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, **tracking.__dict__)
    return tracking
