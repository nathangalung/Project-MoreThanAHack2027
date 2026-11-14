"""Align broadcast to tactical time."""

import numpy as np
import polars as pl

from possession.match import Match
from possession.paths import GAME, HALF_BLOCKS, PITCH_LENGTH, PITCH_WIDTH, asset
from possession.tracking import AWAY, HOME, Tracking, load_tracking

MIN_SHARED = 5000
FINE = 40
MAX_ERROR = 3.0


def coarse_lag(tactical: Tracking, broadcast: Tracking, start: int, stop: int) -> int:
    """Lag from masked cross correlation."""
    seen_a = ~np.isnan(tactical.ball[start:stop, 0])
    seen_b = ~np.isnan(broadcast.ball[:, 0])
    a = np.where(seen_a, tactical.ball[start:stop, 0] - 0.5, 0.0)
    b = np.where(seen_b, broadcast.ball[:, 0] - 0.5, 0.0)
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))

    def correlate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(y, size)), size)

    shared = correlate(seen_a.astype(float), seen_b.astype(float))
    energy = correlate(a * a, seen_b.astype(float)) * correlate(seen_a.astype(float), b * b)
    score = np.where(shared > MIN_SHARED, correlate(a, b) / np.sqrt(np.maximum(energy, 1e-9)), 0.0)
    best = int(score.argmax())
    shift = best if best < size // 2 else best - size
    return start + shift


def ball_error(tactical: Tracking, broadcast: Tracking, start: int, stop: int, lag: int) -> float:
    """Median ball distance in metres."""
    frames = np.arange(start, stop)
    source = frames - lag
    inside = (source >= 0) & (source < broadcast.n_frames)
    delta = tactical.ball[frames[inside]] - broadcast.ball[source[inside]]
    distance = np.hypot(delta[:, 0] * PITCH_LENGTH, delta[:, 1] * PITCH_WIDTH)
    distance = distance[~np.isnan(distance)]
    return float(np.median(distance)) if len(distance) > MIN_SHARED else float("inf")


def half_lags() -> list[int]:
    """Frame lag of each half."""
    tactical, broadcast = load_tracking("tactical"), load_tracking("broadcast")
    lags = []
    for _, first, last in HALF_BLOCKS:
        guess = coarse_lag(tactical, broadcast, first, last + 1)
        errors = {
            lag: ball_error(tactical, broadcast, first, last + 1, lag) for lag in range(guess - FINE, guess + FINE)
        }
        lag = min(errors, key=lambda key: errors[key])
        if errors[lag] > MAX_ERROR:
            raise ValueError(f"broadcast alignment failed, ball error {errors[lag]:.1f} m")
        lags.append(lag)
    return lags


def broadcast_view() -> tuple[np.ndarray, np.ndarray]:
    """Broadcast ball and cutaway flag."""
    tactical, broadcast = load_tracking("tactical"), load_tracking("broadcast")
    table = pl.read_csv(asset("homography", "broadcast"), columns=["valid"])
    live = table["valid"].to_numpy().astype(bool)
    ball = np.full_like(tactical.ball, np.nan)
    cutaway = np.ones(tactical.n_frames, dtype=bool)
    for (_, first, last), lag in zip(HALF_BLOCKS, half_lags(), strict=True):
        frames = np.arange(first, last + 1)
        source = frames - lag
        inside = (source >= 0) & (source < broadcast.n_frames)
        ball[frames[inside]] = broadcast.ball[source[inside]]
        cutaway[frames[inside]] = ~live[source[inside]]
    ball[cutaway] = np.nan
    return ball, cutaway


def target_match() -> Match:
    """Provided match in common format."""
    tracking = load_tracking("tactical")
    extra_ball, cutaway = broadcast_view()
    size = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)
    metres = (tracking.xy - 0.5) * size
    return Match(
        name=GAME,
        home=metres[:, tracking.team == HOME],
        away=metres[:, tracking.team == AWAY],
        ball=(tracking.ball - 0.5) * size,
        second=(extra_ball - 0.5) * size,
        blocks=tuple((first, last + 1) for _, first, last in HALF_BLOCKS),
        cutaway=cutaway,
    )
