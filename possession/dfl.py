"""Load the open DFL matches."""

import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from possession.match import AWAY, DEAD, HOME, Match
from possession.paths import CACHE_DIR

# DFL open data, CC-BY
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
MATCHES = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
DFL_DIR = CACHE_DIR / "dfl"
SECTIONS = ("firstHalf", "secondHalf")


def fetch(match_id: str, kind: str) -> Path:
    """Download a file once."""
    prefix = {"meta": "DFL_02_01_matchinformation", "tracking": "DFL_04_03_positions_raw_observed"}[kind]
    name = f"{prefix}_{MATCHES[match_id]}_DFL-MAT-{match_id}.xml"
    path = DFL_DIR / name
    if not path.is_file():
        DFL_DIR.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", partial)
        partial.rename(path)
    return path


def club_ids(text: str) -> tuple[str, str]:
    """Home and away club ids."""
    home, away = (text.split(f'{side}TeamId="')[1].split('"')[0] for side in ("Home", "Guest"))
    return home, away


def squads(frames: pl.DataFrame, home_id: str, away_id: str) -> dict[str, list[str]]:
    """Players per club, no referees."""
    return {
        side: frames.filter(pl.col("team") == club)["person"].unique(maintain_order=True).to_list()
        for side, club in (("home", home_id), ("away", away_id))
    }


def read_frames(path: Path) -> pl.DataFrame:
    """One row per object frame."""
    lines = pl.read_csv(path, has_header=False, separator="\x01", quote_char=None, new_columns=["line"])
    attribute = lambda name: pl.col("line").str.extract(f' {name}="([^"]*)"')  # noqa: E731
    is_set = pl.col("line").str.starts_with("<FrameSet")
    return (
        lines.filter(is_set | pl.col("line").str.starts_with("<Frame "))
        .with_columns(
            pl.when(is_set).then(attribute("GameSection")).forward_fill().alias("section"),
            pl.when(is_set).then(attribute("TeamId")).forward_fill().alias("team"),
            pl.when(is_set).then(attribute("PersonId")).forward_fill().alias("person"),
        )
        .filter(~is_set)
        .select(
            "section",
            "team",
            "person",
            attribute("N").cast(pl.Int32).alias("frame"),
            attribute("X").cast(pl.Float32).alias("x"),
            attribute("Y").cast(pl.Float32).alias("y"),
            attribute("Z").cast(pl.Float32).alias("z"),
            attribute("BallPossession").cast(pl.Int8).alias("possession"),
            attribute("BallStatus").cast(pl.Int8).alias("status"),
        )
    )


def team_array(frames: pl.DataFrame, persons: list[str], first: int, n: int) -> np.ndarray:
    """Player positions, absent as NaN."""
    out = np.full((n, len(persons), 2), np.nan, dtype=np.float32)
    for slot, person in enumerate(persons):
        rows = frames.filter(pl.col("person") == person)
        out[rows["frame"].to_numpy() - first, slot] = rows.select("x", "y").to_numpy()
    return out


def build(match_id: str) -> dict[str, np.ndarray]:
    """Parse one match into arrays."""
    frames = read_frames(fetch(match_id, "tracking"))
    persons = squads(frames, *club_ids(fetch(match_id, "meta").read_text(encoding="utf-8")))
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("home", "away", "ball", "height", "label")}
    lengths = []
    for section in SECTIONS:
        part = frames.filter(pl.col("section") == section)
        ball = part.filter(pl.col("team") == "BALL").sort("frame")
        first, n = int(ball["frame"][0]), ball.height
        if not np.array_equal(ball["frame"].to_numpy(), np.arange(first, first + n)):
            raise ValueError(f"{match_id} {section}: ball frames are not contiguous")
        players = part.filter(pl.col("frame").is_between(first, first + n - 1))
        parts["home"].append(team_array(players, persons["home"], first, n))
        parts["away"].append(team_array(players, persons["away"], first, n))
        parts["ball"].append(ball.select("x", "y").to_numpy())
        parts["height"].append(ball["z"].to_numpy())
        team = np.where(ball["possession"].to_numpy() == 1, HOME, AWAY)
        parts["label"].append(np.where(ball["status"].to_numpy() == 1, team, DEAD).astype(np.int8))
        lengths.append(n)
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    arrays["lengths"] = np.array(lengths)
    return arrays


def load_dfl(match_id: str) -> tuple[Match, np.ndarray]:
    """Match arrays and ball height."""
    cache = DFL_DIR / f"{match_id}.npz"
    if not cache.is_file():
        np.savez_compressed(cache, **build(match_id))  # pyright: ignore[reportArgumentType]
    with np.load(cache) as stored:
        stops = np.cumsum(stored["lengths"])
        blocks = tuple((int(stop - n), int(stop)) for stop, n in zip(stops, stored["lengths"], strict=True))
        none = np.zeros(len(stored["label"]), dtype=bool)
        ball = stored["ball"]
        match = Match(
            match_id, stored["home"], stored["away"], ball, np.full_like(ball, np.nan), blocks, none, stored["label"]
        )
        return match, stored["height"]
