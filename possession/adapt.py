"""Optional adaptation, tested, not adopted."""

import argparse
import copy
import json

import numpy as np
import torch
from torch import nn

from possession.broadcast import target_match
from possession.decode import decode, waiting
from possession.match import Match
from possession.model import SWAPPED, PossessionNet, predict_views, view_features
from possession.paths import FPS, KICKOFFS, MODEL_DIR
from possession.predict import load_models
from possession.train import BATCH, CROP, NAME

# Chosen on other matches
STEPS = 800
RATE = 3e-4
CONFIDENCE = 0.9
UNUSED = -100


def pseudo_labels(
    models: list[PossessionNet], match: Match, views: list[tuple[np.ndarray, bool]], settings: dict, device: str
) -> np.ndarray:
    """Confident decoded labels, rest unused."""
    prob = predict_views(models, views, match.blocks, device)
    labels = decode(prob, match.blocks, **settings).astype(np.int64)
    sure = (prob.max(axis=1) >= CONFIDENCE) & (prob.argmax(axis=1) == labels)

    # Frames before kickoff stay out
    sure[waiting(match.n_frames, match.blocks, KICKOFFS, FPS)] = False
    return np.where(sure, labels, UNUSED)


def adapt(
    model: PossessionNet,
    views: list[tuple[np.ndarray, bool]],
    labels: np.ndarray,
    blocks: tuple[tuple[int, int], ...],
    seed: int,
    device: str,
) -> PossessionNet:
    """Train one network further."""
    student = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimiser = torch.optim.AdamW(student.parameters(), lr=RATE, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02, ignore_index=UNUSED)
    swapped_labels = np.where(labels == UNUSED, UNUSED, np.array(SWAPPED)[np.maximum(labels, 0)])
    student.train()
    for _ in range(STEPS):
        xs, ys = [], []
        for _ in range(BATCH):
            features, swapped = views[rng.integers(len(views))]
            start, stop = blocks[rng.integers(len(blocks))]
            at = int(rng.integers(start, stop - CROP))
            xs.append(features[at : at + CROP])
            ys.append((swapped_labels if swapped else labels)[at : at + CROP])
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = loss_fn(student(x).flatten(0, 1).float(), y.flatten())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimiser.step()
    return student


def adapt_all(models: list[PossessionNet], match: Match, settings: dict, device: str) -> list[PossessionNet]:
    """All networks adapted."""
    views = view_features(match)
    labels = pseudo_labels(models, match, views, settings, device)
    return [adapt(model, views, labels, match.blocks, seed, device) for seed, model in enumerate(models)]


def run(name: str = NAME) -> None:
    """Adapt and save the networks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    settings = json.loads((MODEL_DIR / f"{name}_decoder.json").read_text())
    students = adapt_all(load_models([name], device), target_match(), settings, device)
    torch.save([student.state_dict() for student in students], MODEL_DIR / f"{name}_adapted.pt")
    print(f"saved {MODEL_DIR / name}_adapted.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=NAME)
    run(parser.parse_args().name)


if __name__ == "__main__":
    main()
