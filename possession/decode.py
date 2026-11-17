"""Frame probabilities to segments."""

import numpy as np

from possession.match import DEAD


def viterbi(log_prob: np.ndarray, penalty: float) -> np.ndarray:
    """Best path with switch cost."""
    n, k = log_prob.shape
    cost = np.full((k, k), -penalty)
    np.fill_diagonal(cost, 0.0)
    back = np.zeros((n, k), dtype=np.int8)
    score = log_prob[0].astype(np.float64)
    for t in range(1, n):
        step = score[:, None] + cost
        back[t] = step.argmax(axis=0)
        score = step.max(axis=0) + log_prob[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def drop_short(path: np.ndarray, min_spell: int, min_dead: int) -> np.ndarray:
    """Absorb too short runs."""
    out = path.copy()
    for _ in range(3):
        bounds = np.concatenate(([0], np.flatnonzero(out[1:] != out[:-1]) + 1, [len(out)]))
        changed = False
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            limit = min_dead if out[a] == DEAD else min_spell
            if b - a < limit and a > 0:
                out[a:b] = out[a - 1]
                changed = True
        if not changed:
            break
    return out


def best_paths(prob: np.ndarray, blocks: tuple[tuple[int, int], ...], penalty: float) -> np.ndarray:
    """Viterbi path of every block."""
    out = np.full(len(prob), DEAD, dtype=np.int8)
    for start, stop in blocks:
        out[start:stop] = viterbi(np.log(prob[start:stop] + 1e-6), penalty)
    return out


def tidy(path: np.ndarray, blocks: tuple[tuple[int, int], ...], min_spell: int, min_dead: int) -> np.ndarray:
    """Short runs removed per block."""
    out = path.copy()
    for start, stop in blocks:
        out[start:stop] = drop_short(path[start:stop], min_spell, min_dead)
    return out


def waiting(n_frames: int, blocks: tuple[tuple[int, int], ...], kickoffs: tuple[int, ...], slack: int) -> np.ndarray:
    """Block frames before a kickoff."""
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        # Anchors are exact within slack
        late = [frame for frame in kickoffs if start + slack < frame < stop]
        if late:
            # Anchor frame is still dead
            mask[start : late[0] + 1] = True
    return mask


def adapted_teams(prob: np.ndarray, adapted: np.ndarray) -> np.ndarray:
    """Team split from adapted networks."""
    # Stoppage probability stays as is
    live = adapted[:, :DEAD]
    split = live / np.maximum(live.sum(axis=1, keepdims=True), 1e-6)
    out = prob.copy()
    out[:, :DEAD] = split * (1 - prob[:, DEAD : DEAD + 1])
    return out


def decode(
    prob: np.ndarray,
    blocks: tuple[tuple[int, int], ...],
    penalty: float,
    min_spell: int,
    min_dead: int,
    known_dead: np.ndarray | None = None,
) -> np.ndarray:
    """Labels for every block frame."""
    if known_dead is not None:
        # Path must respect known stoppages
        prob = prob.copy()
        prob[known_dead] = np.eye(prob.shape[1], dtype=prob.dtype)[DEAD]
    labels = tidy(best_paths(prob, blocks, penalty), blocks, min_spell, min_dead)
    if known_dead is not None:
        labels[known_dead] = DEAD
    return labels
