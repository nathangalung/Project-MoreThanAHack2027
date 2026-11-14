"""Common match array format."""

import itertools
from dataclasses import dataclass, replace

import numpy as np

HOME, AWAY, DEAD = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


@dataclass(frozen=True)
class Match:
    """Positions in metres, centre origin."""

    name: str
    home: np.ndarray
    away: np.ndarray
    ball: np.ndarray
    second: np.ndarray
    blocks: tuple[tuple[int, int], ...]
    cutaway: np.ndarray
    label: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return self.ball.shape[0]


def mirrored(match: Match) -> list[tuple[Match, bool]]:
    """Flipped and team swapped copies."""
    views = []
    for flip_x, flip_y, swapped in itertools.product((1, -1), (1, -1), (False, True)):
        sign = np.array([flip_x, flip_y], dtype=np.float32)
        home, away = match.home * sign, match.away * sign
        if swapped:
            home, away = away, home
        views.append(
            (replace(match, home=home, away=away, ball=match.ball * sign, second=match.second * sign), swapped)
        )
    return views
