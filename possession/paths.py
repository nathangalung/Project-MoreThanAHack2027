"""Project paths and match constants."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

import games  # noqa: E402

GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = games.GAMES[GAME]["pitch"]
HALF_BLOCKS = games.half_video_blocks(GAME)
# Kickoff frames published by organisers
KICKOFFS = tuple(games.anchor_video_frame(games.GAMES[GAME][key]) for key in ("kickoff", "kickoff_2nd"))

CACHE_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "evaluation" / "results"
TEAM = "BYTZ"
PUBLIC_GT = ROOT / "evaluation" / "gt" / f"{GAME}_possession_gt_public.csv"


def asset(name: str, source: str = "tactical") -> Path:
    """Path of one provided asset."""
    path = Path(games.asset_paths(source=source)[name])
    if path.is_file():
        return path
    # Delivered videos carry a suffix
    matches = sorted(path.parent.glob(f"{path.stem}*{path.suffix}"))
    if not matches:
        raise FileNotFoundError(f"{path} not found, see docs/DATA.md")
    return matches[0]
