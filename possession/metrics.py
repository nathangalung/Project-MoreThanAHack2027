"""Spell F1 as officially defined."""

import numpy as np

from possession.match import DEAD

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def spells(codes: np.ndarray, blocks: tuple[tuple[int, int], ...]) -> list[tuple[int, int, int]]:
    """Maximal team runs inside blocks."""
    out = []
    for start, stop in blocks:
        window = codes[start:stop]
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            if window[a] != DEAD:
                out.append((start + int(a), start + int(b), int(window[a])))
    return out


def spell_f1(truth: np.ndarray, guess: np.ndarray, blocks: tuple[tuple[int, int], ...]) -> float:
    """Mean F1 over IoU thresholds."""
    true_spells, guess_spells = spells(truth, blocks), spells(guess, blocks)
    if not true_spells or not guess_spells:
        return 0.0
    pairs = []
    j0 = 0
    for i, (a, b, team) in enumerate(true_spells):
        while j0 < len(guess_spells) and guess_spells[j0][1] <= a:
            j0 += 1
        j = j0
        while j < len(guess_spells) and guess_spells[j][0] < b:
            c, d, other = guess_spells[j]
            if other == team:
                pairs.append(((min(b, d) - max(a, c)) / (max(b, d) - min(a, c)), i, j))
            j += 1
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    scores = []
    for threshold in IOUS:
        used_true, used_guess = set(), set()
        for iou, i, j in pairs:
            if iou < threshold:
                break
            if i not in used_true and j not in used_guess:
                used_true.add(i)
                used_guess.add(j)
        hits = len(used_true)
        scores.append(2 * hits / (len(true_spells) + len(guess_spells)))
    return float(np.mean(scores))


def macro_f1(truth: np.ndarray, guess: np.ndarray) -> float:
    """Mean F1 over three classes."""
    scores = []
    for code in range(3):
        hits = np.sum((truth == code) & (guess == code))
        total = np.sum(truth == code) + np.sum(guess == code)
        scores.append(2 * hits / total if total else 0.0)
    return float(np.mean(scores))
