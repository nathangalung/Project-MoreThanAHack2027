"""Frame features from arrays."""

import numpy as np

from possession.match import Match
from possession.paths import FPS

HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
MAX_GAP = 250
MAX_SPEED = 40.0
NEAREST = 3
FAR = 60.0

# Same order as the columns
BALL_FEATURES = (
    "Ball Seen",
    "Time Since Ball Seen",
    "Time Until Ball Seen",
    "Ball Gap Length",
    "Ball X",
    "Ball Y",
    "Ball Margin X",
    "Ball Margin Y",
    "Ball Outside Pitch",
    "Ball Speed",
    "Ball Log Speed",
    "Ball Velocity X",
    "Ball Velocity Y",
    "Ball Acceleration",
    "Ball Slow",
)
CAMERA_FEATURES = (
    "Both Balls Seen",
    "Ball Disagreement",
    "Cutaway",
    "Time in Cutaway",
    "Time Until Cutaway Ends",
    "Time in Live View",
    "Time Until Live View Ends",
)
TEAM_FEATURES = (
    "Nearest Player",
    "Second Nearest Player",
    "Third Nearest Player",
    "Players Within 1.5 m",
    "Players Within 4 m",
    "Nearest Player Speed",
    "Mean Speed",
    "Players Seen",
    "Centroid X",
    "Centroid Y",
    "Ball Ahead of Centroid",
    "Spread",
    "Drift X",
    "Drift Y",
)
FEATURE_NAMES = (
    *BALL_FEATURES,
    *CAMERA_FEATURES,
    *(f"Home {name}" for name in TEAM_FEATURES),
    *(f"Away {name}" for name in TEAM_FEATURES),
)
N_FEATURES = len(FEATURE_NAMES)


def interpolate(values: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Linear fill across missing frames."""
    index = np.arange(len(values))
    if not seen.any():
        return np.zeros_like(values)
    return np.stack([np.interp(index, index[seen], column[seen]) for column in values.T], axis=1)


def gap_clocks(seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames since and until seen."""
    index = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, index, -MAX_GAP))
    upcoming = np.minimum.accumulate(np.where(seen, index, len(seen) + MAX_GAP)[::-1])[::-1]
    return np.minimum(index - last, MAX_GAP), np.minimum(upcoming - index, MAX_GAP)


def velocity(positions: np.ndarray) -> np.ndarray:
    """Central difference velocity."""
    out = np.full_like(positions, np.nan)
    out[1:-1] = (positions[2:] - positions[:-2]) * (FPS / 2)
    return out


def team_features(players: np.ndarray, ball: np.ndarray) -> list[np.ndarray]:
    """Features of one team."""
    seen = ~np.isnan(players[:, :, 0])
    count = seen.sum(axis=1)
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    distance = np.where(seen, distance, FAR)
    order = np.argsort(distance, axis=1)[:, :NEAREST]
    nearest = np.minimum(np.take_along_axis(distance, order, axis=1), FAR)

    speed = np.linalg.norm(velocity(players), axis=2)
    speed = np.where(np.isnan(speed), 0.0, np.minimum(speed, 12.0))
    moving = np.maximum((~np.isnan(velocity(players)[:, :, 0])).sum(axis=1), 1)
    closest_speed = np.take_along_axis(speed, order[:, :1], axis=1)[:, 0]

    safe = np.where(seen[:, :, None], players, 0.0)
    centroid = safe.sum(axis=1) / np.maximum(count, 1)[:, None]
    spread = np.sqrt(
        (np.where(seen, (players[:, :, 0] - centroid[:, None, 0]) ** 2, 0.0)).sum(axis=1) / np.maximum(count, 1)
    )
    drift = np.nan_to_num(velocity(centroid)).clip(-8, 8)
    return [
        *(nearest[:, k] / 10 for k in range(NEAREST)),
        (distance < 1.5).sum(axis=1).astype(np.float32),
        (distance < 4.0).sum(axis=1) / 3,
        closest_speed / 5,
        speed.sum(axis=1) / moving / 5,
        count / 11,
        centroid[:, 0] / HALF_LENGTH,
        centroid[:, 1] / HALF_WIDTH,
        (ball[:, 0] - centroid[:, 0]) / HALF_LENGTH,
        spread / 20,
        drift[:, 0] / 4,
        drift[:, 1] / 4,
    ]


def block_features(
    home: np.ndarray, away: np.ndarray, first: np.ndarray, second: np.ndarray, cutaway: np.ndarray
) -> np.ndarray:
    """Features of one block."""
    # Cameras disagree on raised balls
    both = ~np.isnan(first[:, 0]) & ~np.isnan(second[:, 0])
    apart = np.where(both, np.minimum(np.linalg.norm(first - second, axis=1), 30.0), 0.0)
    ball = np.where(np.isnan(first), second, first)
    seen = ~np.isnan(ball[:, 0])
    filled = interpolate(ball, seen)
    since, until = gap_clocks(seen)
    motion = velocity(filled)
    motion[[0, -1]] = motion[[1, -2]] if len(ball) > 2 else 0.0
    speed = np.minimum(np.linalg.norm(motion, axis=1), MAX_SPEED)
    change = np.zeros(len(ball), dtype=np.float32)
    change[1:-1] = np.minimum(np.linalg.norm(motion[2:] - motion[:-2], axis=1), MAX_SPEED)

    margin_x = HALF_LENGTH - np.abs(filled[:, 0])
    margin_y = HALF_WIDTH - np.abs(filled[:, 1])
    columns = [
        seen.astype(np.float32),
        np.log1p(since) / 5,
        np.log1p(until) / 5,
        np.log1p(since + until) / 5,
        filled[:, 0] / HALF_LENGTH,
        filled[:, 1] / HALF_WIDTH,
        np.clip(margin_x, -5, 10) / 10,
        np.clip(margin_y, -5, 10) / 10,
        ((margin_x < 0) | (margin_y < 0)).astype(np.float32),
        speed / 10,
        np.log1p(speed),
        motion[:, 0].clip(-MAX_SPEED, MAX_SPEED) / 10,
        motion[:, 1].clip(-MAX_SPEED, MAX_SPEED) / 10,
        change / 10,
        (speed < 0.5).astype(np.float32),
    ]
    cut_age, cut_left = gap_clocks(~cutaway)
    live_age, live_left = gap_clocks(cutaway)
    columns += [
        both.astype(np.float32),
        apart / 10,
        cutaway.astype(np.float32),
        np.log1p(cut_age) / 5,
        np.log1p(cut_left) / 5,
        np.log1p(live_age) / 5,
        np.log1p(live_left) / 5,
    ]
    columns += team_features(home, filled) + team_features(away, filled)
    return np.stack(columns, axis=1).astype(np.float32)


def match_features(match: Match) -> np.ndarray:
    """Feature matrix of a match."""
    out = np.zeros((match.n_frames, N_FEATURES), dtype=np.float32)
    for start, stop in match.blocks:
        part = slice(start, stop)
        out[part] = block_features(
            match.home[part], match.away[part], match.ball[part], match.second[part], match.cutaway[part]
        )
    return out
