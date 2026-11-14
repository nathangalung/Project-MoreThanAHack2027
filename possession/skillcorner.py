"""Load the open SkillCorner matches."""

import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from possession.match import AWAY, DEAD, HOME, Match
from possession.paths import CACHE_DIR, FPS

# SkillCorner open data, MIT licence
REPOSITORY = "SkillCorner/opendata/4340d274572876239c154c90bc507a9b3250a656/data/matches"
SMALL_URL = f"https://raw.githubusercontent.com/{REPOSITORY}"
LARGE_URL = f"https://media.githubusercontent.com/media/{REPOSITORY}"
REAL_MATCHES = (
    "1874553",
    "1886347",
    "1899585",
    "1925299",
    "1927964",
    "1953632",
    "1959846",
    "1986691",
    "1996435",
    "1996436",
    "2006229",
    "2006363",
    "2007448",
    "2007721",
    "2010085",
    "2011166",
    "2013725",
    "2015213",
    "2016236",
    "2017461",
)
REAL_DIR = CACHE_DIR / "skillcorner"
SOURCE_FPS = 10
SLOTS = 20


def fetch(match_id: str, name: str) -> Path:
    """Download a file once."""
    path = REAL_DIR / match_id / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        base = LARGE_URL if name.endswith(".jsonl") else SMALL_URL
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{base}/{match_id}/{match_id}_{name}", partial)
        partial.rename(path)
    return path


def upsample(values: np.ndarray, length: int) -> np.ndarray:
    """Linear resample to target rate."""
    at = np.arange(length) * SOURCE_FPS / FPS
    low = np.minimum(at.astype(int), len(values) - 1)
    high = np.minimum(low + 1, len(values) - 1)
    part = (at - low).reshape(-1, *([1] * (values.ndim - 1)))
    return (values[low] * (1 - part) + values[high] * part).astype(np.float32)


def hold(values: np.ndarray, length: int) -> np.ndarray:
    """Nearest source frame value."""
    at = (np.arange(length) * SOURCE_FPS / FPS).round().astype(int)
    return values[np.minimum(at, len(values) - 1)]


def build(match_id: str) -> dict[str, np.ndarray]:
    """Parse one match into arrays."""
    meta = json.loads(fetch(match_id, "match.json").read_text(encoding="utf-8"))
    home_id = meta["home_team"]["id"]
    scale = np.array([105.0 / meta["pitch_length"], 68.0 / meta["pitch_width"]], dtype=np.float32)
    frames = pl.read_ndjson(fetch(match_id, "tracking_extrapolated.jsonl"), infer_schema_length=None).sort("frame")
    n = frames.height

    # Ball counts only when detected
    ball = frames.select(
        pl.col("ball_data").struct.field("x"),
        pl.col("ball_data").struct.field("y"),
        pl.col("ball_data").struct.field("is_detected").fill_null(False),
    ).to_numpy()
    seen = ball[:, 2].astype(bool)
    position = np.where(seen[:, None], ball[:, :2].astype(np.float32), np.nan) * scale
    in_view = frames["image_corners_projection"].struct.field("x_top_left").is_not_null().to_numpy()

    # One slot per player
    side = pl.DataFrame(
        {
            "player_id": [player["id"] for player in meta["players"]],
            "side": [HOME if player["team_id"] == home_id else AWAY for player in meta["players"]],
        }
    )
    players = (
        frames.select("frame", "player_data")
        .explode("player_data")
        .unnest("player_data")
        .drop_nulls("player_id")
        .join(side, on="player_id")
        .with_columns(pl.col("player_id").rank("dense").over("side").cast(pl.Int64).alias("slot") - 1)
    )
    teams = np.full((2, n, SLOTS, 2), np.nan, dtype=np.float32)
    index = (players["side"].to_numpy(), players["frame"].to_numpy(), players["slot"].to_numpy())
    teams[index] = players.select("x", "y").to_numpy().astype(np.float32) * scale

    # Possession phases give the label
    label = np.full(n, DEAD, dtype=np.int8)
    phases = pl.read_csv(fetch(match_id, "phases_of_play.csv"), infer_schema_length=None)
    for start, stop, team in phases.select("frame_start", "frame_end", "team_in_possession_id").iter_rows():
        label[start:stop] = HOME if team == home_id else AWAY

    length = n * FPS // SOURCE_FPS
    periods = meta["match_periods"][:2]
    return {
        "home": upsample(teams[HOME], length),
        "away": upsample(teams[AWAY], length),
        "ball": upsample(position, length),
        "cutaway": hold(~in_view, length),
        "label": hold(label, length),
        "blocks": np.array([[p["start_frame"], p["end_frame"]] for p in periods]) * FPS // SOURCE_FPS,
    }


def load_real(match_id: str, refresh: bool = False) -> Match:
    """Match arrays with labels."""
    cache = REAL_DIR / match_id / "arrays.npz"
    if refresh or not cache.is_file():
        np.savez_compressed(cache, **build(match_id))  # pyright: ignore[reportArgumentType]
    with np.load(cache) as stored:
        blocks = tuple((int(start), int(stop)) for start, stop in stored["blocks"])
        ball = stored["ball"]
        return Match(
            match_id,
            stored["home"],
            stored["away"],
            ball,
            np.full_like(ball, np.nan),
            blocks,
            stored["cutaway"],
            stored["label"],
        )
