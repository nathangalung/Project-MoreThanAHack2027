"""Make open tracking look automatic."""

from dataclasses import replace

import numpy as np

from possession.match import DEAD, Match


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True))


def hide_ball(match: Match, height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ball with missing stretches."""
    ball = match.ball.copy()
    n = match.n_frames
    hidden = np.zeros(n, dtype=bool)

    # Stoppages often hide the ball
    dead_rate = rng.uniform(0.3, 0.95)
    for start, stop in runs(match.label == DEAD):
        if rng.random() < dead_rate:
            lead = int(rng.integers(0, 75))
            tail = int(rng.integers(-50, 25))
            hidden[min(start + lead, stop) : max(stop + tail, start)] = True

    # Occlusion gaps during play
    target = rng.uniform(0.03, 0.25)
    median = rng.uniform(100, 350)
    weight = 1.0 + rng.uniform(0, 2) * (height > 2.5)
    weight = weight / weight.sum()
    budget = int(target * (match.label != DEAD).sum())
    while budget > 0:
        length = int(np.clip(rng.lognormal(np.log(median), 1.0), 15, 2500))
        start = int(rng.choice(n, p=weight))
        hidden[start : start + length] = True
        budget -= length

    # Ball outside the pitch
    outside = (np.abs(ball[:, 0]) > 52.5) | (np.abs(ball[:, 1]) > 34.0)
    out_rate = rng.uniform(0.0, 0.8)
    for start, stop in runs(outside):
        if rng.random() < out_rate:
            hidden[start:stop] = True

    # Rare single frame flickers
    hidden |= rng.random(n) < rng.uniform(0, 0.0005)
    ball[hidden] = np.nan
    return ball


def confirm_tracks(ball: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drop ball tracks under minimum."""
    # Trackers report only confirmed tracks
    out = ball.copy()
    shortest = rng.uniform(15, 50)
    for start, stop in runs(~np.isnan(ball[:, 0])):
        if stop - start < shortest:
            out[start:stop] = np.nan
    return out


def hide_players(players: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Players with lost tracks."""
    out = players.copy()
    n, slots, _ = out.shape
    rate = rng.uniform(0.02, 0.12)
    for slot in range(slots):
        budget = int(rate * n)
        while budget > 0:
            length = int(np.clip(rng.lognormal(np.log(150), 1.2), 5, 5000))
            start = int(rng.integers(0, n))
            out[start : start + length, slot] = np.nan
            budget -= length
    return out


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur along time."""
    radius = int(3 * sigma) + 1
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    flat = values.reshape(len(values), -1)
    known = ~np.isnan(flat)

    def blur(table: np.ndarray) -> np.ndarray:
        padded = np.pad(table, ((radius, radius), (0, 0)), mode="edge")
        return np.stack([np.convolve(column, kernel, mode="valid") for column in padded.T], axis=1)

    # Missing frames carry no weight
    weight = blur(known.astype(np.float64))
    out = blur(np.where(known, flat, 0.0)) / np.maximum(weight, 1e-9)
    return np.where(known, out, np.nan).reshape(values.shape).astype(np.float32)


def seen_from_camera(ball: np.ndarray, height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ground position a camera reports."""
    # Raised balls project further away
    camera = np.array([rng.uniform(-15, 15), -rng.uniform(45, 80)], dtype=np.float32)
    away = ball - camera
    away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1.0)
    shift = np.minimum(rng.uniform(0.0, 6.0) * np.clip(height, 0, None), rng.uniform(6.0, 20.0))

    # Calibration error drifts slowly
    knots = rng.normal(0.0, rng.uniform(0.2, 1.2), (len(ball) // 250 + 2, 2))
    frames = np.arange(len(ball)) / 250
    drift = np.stack([np.interp(frames, np.arange(len(knots)), knots[:, axis]) for axis in (0, 1)], axis=1)
    return (ball + shift[:, None] * away + drift).astype(np.float32)


def camera_loss(match: Match, rng: np.random.Generator) -> np.ndarray:
    """Frames where tracks vanish."""
    n = match.n_frames
    lost = np.zeros(n, dtype=bool)
    weight = np.where(match.label == DEAD, 5.0, 1.0)
    weight /= weight.sum()
    for _ in range(int(rng.integers(5, 40))):
        length = int(np.clip(rng.lognormal(np.log(70), 1.2), 10, 2000))
        start = int(rng.choice(n, p=weight))
        lost[start : start + length] = True
    return lost


def cutaways(match: Match, rng: np.random.Generator) -> np.ndarray:
    """Simulated broadcast replay stretches."""
    n = match.n_frames
    cut = np.zeros(n, dtype=bool)
    shortest = rng.uniform(50, 200)
    rate = rng.uniform(0.6, 1.0)
    for start, stop in runs(match.label == DEAD):
        if stop - start > shortest and rng.random() < rate:
            lead = int(np.clip(rng.lognormal(np.log(75), 0.5), 15, 300))
            tail = int(rng.integers(-40, 90))
            cut[min(start + lead, n) : max(stop + tail, 0)] = True

    # Close-ups during live play
    for _ in range(int(rng.integers(0, 30))):
        start = int(rng.integers(0, n))
        cut[start : start + int(rng.integers(25, 250))] = True
    return cut


def carry_dead_ball(match: Match, rng: np.random.Generator) -> np.ndarray:
    """Ball carried during stoppages."""
    # The open data freezes it
    ball = match.ball.copy()
    players = np.concatenate([match.home, match.away], axis=1)
    n = match.n_frames
    pace = rng.uniform(20, 90)
    for start, stop in runs(match.label == DEAD):
        if stop - start < 10:
            continue
        origin, restart = ball[max(start - 1, 0)], ball[min(stop, n - 1)]
        rest = min(int(rng.uniform(0, 75)), (stop - start) // 2)
        count = max(1, int((stop - start - rest) / pace))
        times = np.unique(rng.integers(start + 1, stop - rest, size=count))
        stops = [origin]
        for time in times:
            guide = origin + (restart - origin) * (time - start) / (stop - start)
            gap = np.nan_to_num(np.linalg.norm(players[time] - guide, axis=1), nan=1e6)
            carrier = players[time, rng.choice(np.argsort(gap)[:4])]
            stops.append(np.where(np.isnan(carrier), guide, carrier) + rng.normal(0, 1.0, 2))
        stops += [restart, restart]
        knots = np.concatenate(([start], times, [stop - rest, stop]))
        frames = np.arange(start, stop)
        path = np.stack([np.interp(frames, knots, np.array(stops)[:, axis]) for axis in (0, 1)], axis=1)
        ball[start:stop] = smooth(path.astype(np.float32), rng.uniform(1.0, 4.0))
    return ball


def degrade(match: Match, height: np.ndarray, seed: int) -> Match:
    """One random degraded copy."""
    rng = np.random.default_rng(seed)
    home, away, label = match.home, match.away, match.label
    if label is None:
        raise ValueError(f"{match.name} has no labels")
    cut = cutaways(match, rng)
    moving = carry_dead_ball(match, rng)
    height = np.where(label == DEAD, 0.0, height)
    views = []
    for _ in range(2):
        view = smooth(seen_from_camera(moving, height, rng), rng.uniform(4.0, 9.0))
        view = hide_ball(replace(match, ball=view), height, rng)
        view[height > rng.uniform(5, 25)] = np.nan
        views.append(view)
    ball, second = views
    # Alignment is off by frames
    second = np.roll(second, int(rng.integers(-3, 4)), axis=0)
    second[cut] = np.nan
    blur = rng.uniform(1.5, 4.0)
    home = hide_players(smooth(home, blur), rng)
    away = hide_players(smooth(away, blur), rng)

    lost = camera_loss(match, rng)
    ball[lost & (rng.random(match.n_frames) < 0.95)] = np.nan
    for players in (home, away):
        keep = rng.random(players.shape[1]) < rng.uniform(0.0, 0.5)
        players[np.ix_(lost, ~keep)] = np.nan
    ball, second = confirm_tracks(ball, rng), confirm_tracks(second, rng)

    # Pitch origin off by metres
    offset = rng.uniform(-2.5, 2.5, 2).astype(np.float32)
    home, away, ball, second = home + offset, away + offset, ball + offset, second + offset

    # Symmetry augmentations
    if rng.random() < 0.5:
        home, away = away, home
        label = np.where(label == DEAD, DEAD, 1 - label).astype(np.int8)
    for axis in (0, 1):
        if rng.random() < 0.5:
            sign = np.ones(2, dtype=np.float32)
            sign[axis] = -1
            home, away, ball, second = home * sign, away * sign, ball * sign, second * sign
    return replace(match, home=home, away=away, ball=ball, second=second, label=label, cutaway=cut)
