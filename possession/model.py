"""Sequence model over frame features."""

import numpy as np
import torch
from torch import nn

from possession.features import N_FEATURES, match_features
from possession.match import AWAY, DEAD, HOME, Match, mirrored

WINDOW = 8192
CONTEXT = 2048
# Label codes after team swap
SWAPPED = [AWAY, HOME, DEAD]


class PossessionNet(nn.Module):
    """Convolution stem and bidirectional GRU."""

    def __init__(self, hidden: int = 160, layers: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_FEATURES, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        stem = self.stem(features.transpose(1, 2)).transpose(1, 2)
        return self.head(self.gru(stem)[0])


@torch.no_grad()
def predict_proba(
    models: list[PossessionNet], features: np.ndarray, blocks: tuple[tuple[int, int], ...], device: str
) -> np.ndarray:
    """Mean class probabilities per frame."""
    prob = np.zeros((len(features), 3), dtype=np.float32)
    prob[:, 2] = 1.0
    for model in models:
        model.eval()
    for start, stop in blocks:
        # Overlapping windows keep context
        for at in range(start, stop, WINDOW):
            low, high = max(start, at - CONTEXT), min(stop, at + WINDOW + CONTEXT)
            batch = torch.from_numpy(features[low:high]).to(device)[None]
            mean = torch.stack([model(batch).softmax(dim=-1) for model in models]).mean(dim=0)[0]
            keep = slice(at - low, min(at + WINDOW, stop) - low)
            prob[at : min(at + WINDOW, stop)] = mean[keep].cpu().numpy()
    return prob


def view_features(match: Match) -> list[tuple[np.ndarray, bool]]:
    """Features of every mirrored view."""
    return [(match_features(view), swapped) for view, swapped in mirrored(match)]


def predict_views(
    models: list[PossessionNet], views: list[tuple[np.ndarray, bool]], blocks: tuple[tuple[int, int], ...], device: str
) -> np.ndarray:
    """Mean over mirrored views."""
    total = np.zeros((len(views[0][0]), 3), dtype=np.float32)
    for features, swapped in views:
        prob = predict_proba(models, features, blocks, device)
        total += prob[:, SWAPPED] if swapped else prob
    return total / len(views)


def predict_match(models: list[PossessionNet], match: Match, device: str) -> np.ndarray:
    """Mean over mirrored match views."""
    return predict_views(models, view_features(match), match.blocks, device)
