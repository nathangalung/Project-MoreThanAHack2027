"""Write the submission file."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from possession.broadcast import target_match
from possession.decode import decode, waiting
from possession.match import DEAD, LABELS
from possession.metrics import macro_f1, spell_f1
from possession.model import PossessionNet, predict_match
from possession.paths import FPS, KICKOFFS, MODEL_DIR, PUBLIC_GT, RESULTS_DIR, TEAM
from possession.train import NAME

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
# CPU repeats exactly, CUDA not
DEVICE = "cpu"


def load_models(names: list[str], device: str) -> list[PossessionNet]:
    """Every network in the files."""
    models = []
    for name in names:
        for state in torch.load(MODEL_DIR / f"{name}.pt", map_location=device):
            model = PossessionNet().to(device)
            model.load_state_dict(state)
            models.append(model)
    return models


def public_scores(labels: np.ndarray) -> tuple[float, float]:
    """Scores on the public window."""
    truth = pl.read_csv(PUBLIC_GT)["label"].replace_strict(CODES, return_dtype=pl.Int8).to_numpy()
    graded = np.flatnonzero(truth != CODES["X"])
    block = ((int(graded[0]), int(graded[-1]) + 1),)
    return spell_f1(truth, labels, block), macro_f1(truth[graded], labels[graded])


def run(
    models: tuple[str, ...] = (NAME,),
    decoder: str = f"{NAME}_decoder",
    device: str = DEVICE,
    out: Path = RESULTS_DIR / f"{TEAM}.csv",
) -> tuple[np.ndarray, np.ndarray]:
    """Write the file, return labels."""
    match = target_match()
    prob = predict_match(load_models(list(models), device), match, device)
    settings = json.loads((MODEL_DIR / f"{decoder}.json").read_text())
    labels = decode(prob, match.blocks, **settings)

    # Dead ball before each kickoff
    labels[waiting(match.n_frames, match.blocks, KICKOFFS, FPS)] = DEAD

    text = np.full(match.n_frames, "X")
    for start, stop in match.blocks:
        text[start:stop] = LABELS[labels[start:stop]]
    pl.DataFrame({"frame": np.arange(match.n_frames), "label": text}).write_csv(out)
    spell, frame = public_scores(labels)
    print(f"wrote {out}")
    print(f"public window spellF1 {spell:.4f} macroF1 {frame:.4f}")
    return labels, prob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[NAME])
    parser.add_argument("--decoder", default=f"{NAME}_decoder")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE, help="cpu repeats exactly, cuda may not")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / f"{TEAM}.csv")
    args = parser.parse_args()
    run(tuple(args.models), args.decoder, args.device, args.out)


if __name__ == "__main__":
    main()
