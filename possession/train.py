"""Train on degraded open matches."""

import argparse
import itertools
import json
import multiprocessing
import signal
import time
from concurrent.futures import Executor, ProcessPoolExecutor

import numpy as np
import torch
from torch import nn

from possession.decode import best_paths, decode, tidy
from possession.degrade import degrade
from possession.dfl import MATCHES, load_dfl
from possession.features import match_features
from possession.match import Match
from possession.metrics import macro_f1, spell_f1
from possession.model import PossessionNet, predict_match, predict_proba
from possession.paths import CACHE_DIR, MODEL_DIR
from possession.skillcorner import REAL_MATCHES, load_real

# Defaults of a full run
VALID_IDS = ("J03WMX", "J03WR9")
EPOCHS = 24
STEPS = 250
SEEDS = (0, 1, 2, 3, 4)
NAME = "model"

VALID_COPIES = 3
VALID_SEED = 999
CROP = 2048
BATCH = 32
VARIANTS = 4
REFRESH = 1
CHECK = 4
WORKERS = 8
PENALTIES = (0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0)
MIN_SPELLS = (0, 15, 25, 40)
MIN_DEADS = (0, 25, 50)
Sample = tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]


def ignore_interrupt() -> None:
    """Leave interrupts to the parent."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def one_match(match_id: str, seed: int) -> Match:
    """One degraded match copy."""
    match, height = load_dfl(match_id)
    return degrade(match, height, seed)


def one_sample(match_id: str, seed: int) -> Sample:
    """Features and labels, one copy."""
    noisy = one_match(match_id, seed)
    assert noisy.label is not None
    return match_features(noisy), noisy.label, noisy.blocks


def copies(match_ids: list[str], variants: int, seed: int) -> tuple[list[str], list[int]]:
    """Ids and seeds of copies."""
    ids = [match_id for match_id in match_ids for _ in range(variants)]
    seeds = [seed * 10_000 + number * 100 + variant for number in range(len(match_ids)) for variant in range(variants)]
    return ids, seeds


def make_samples(match_ids: list[str], variants: int, seed: int, pool: Executor | None = None) -> list[Sample]:
    """Degraded feature and label arrays."""
    ids, seeds = copies(match_ids, variants, seed)
    if pool is None:
        return [one_sample(match_id, at) for match_id, at in zip(ids, seeds, strict=True)]
    return list(pool.map(one_sample, ids, seeds))


def draw_batch(samples: list[Sample], rng: np.random.Generator, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crops from random samples."""
    xs, ys = [], []
    for _ in range(BATCH):
        features, label, blocks = samples[rng.integers(len(samples))]
        start, stop = blocks[rng.integers(len(blocks))]
        at = int(rng.integers(start, stop - CROP))
        xs.append(features[at : at + CROP])
        ys.append(label[at : at + CROP])
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
    return x, y


def group_score(paths: list[np.ndarray], matches: list[Match], min_spell: int, min_dead: int) -> float:
    """Mean spellF1 of one group."""
    scores = []
    for path, match in zip(paths, matches, strict=True):
        assert match.label is not None
        scores.append(spell_f1(match.label, tidy(path, match.blocks, min_spell, min_dead), match.blocks))
    return float(np.mean(scores))


def tune_decoder(probs: list[list[np.ndarray]], groups: list[list[Match]]) -> tuple[dict, list[float]]:
    """Grid search over validation groups."""
    settings, table = [], []
    for penalty in PENALTIES:
        # One Viterbi pass per penalty
        paths = [
            [best_paths(prob, match.blocks, penalty) for prob, match in zip(part, matches, strict=True)]
            for part, matches in zip(probs, groups, strict=True)
        ]
        for min_spell, min_dead in itertools.product(MIN_SPELLS, MIN_DEADS):
            settings.append({"penalty": penalty, "min_spell": min_spell, "min_dead": min_dead})
            table.append(
                [group_score(part, matches, min_spell, min_dead) for part, matches in zip(paths, groups, strict=True)]
            )
        latest = table[-len(MIN_SPELLS) * len(MIN_DEADS) :]
        print(f"penalty {penalty:>5} best per group {np.round(np.max(latest, axis=0), 4)}", flush=True)

    # Minimax loss over the groups
    scores = np.array(table)
    regret = (scores.max(axis=0) - scores).max(axis=1)
    pick = int(regret.argmin())
    return settings[pick], scores[pick].tolist()


def fit(
    train_ids: list[str],
    valid: list[Sample],
    epochs: int,
    steps: int,
    seed: int,
    device: str,
    pool: Executor | None = None,
) -> PossessionNet:
    """Fit one network."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PossessionNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3, total_steps=epochs * steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    samples: list[Sample] = []

    for epoch in range(epochs):
        began = time.time()
        if epoch % REFRESH == 0:
            samples = make_samples(train_ids, VARIANTS, seed * 100 + epoch, pool)
        model.train()
        total = 0.0
        for _ in range(steps):
            x, y = draw_batch(samples, rng, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = loss_fn(model(x).flatten(0, 1).float(), y.flatten())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += loss.item()
        line = f"epoch {epoch + 1:>2} loss {total / steps:.4f} time {time.time() - began:.0f}s"
        if valid and (epoch % CHECK == CHECK - 1 or epoch == epochs - 1):
            frame, spell = [], []
            for features, label, blocks in valid:
                prob = predict_proba([model], features, blocks, device)
                frame.append(macro_f1(label, prob.argmax(1)))
                spell.append(spell_f1(label, decode(prob, blocks, 10.0, 15, 25), blocks))
            line += f" valid macroF1 {np.mean(frame):.4f} spellF1 {np.mean(spell):.4f}"
        print(line, flush=True)
    return model


def run(
    valid_ids: tuple[str, ...] = VALID_IDS,
    epochs: int = EPOCHS,
    steps: int = STEPS,
    seeds: tuple[int, ...] = SEEDS,
    name: str = NAME,
) -> list[float]:
    """Train, tune decoder, and save."""
    unknown = [match_id for match_id in valid_ids if match_id not in MATCHES]
    if unknown or not valid_ids:
        raise ValueError(f"held-out ids must come from {list(MATCHES)}, got {list(valid_ids)}")
    if not seeds:
        raise ValueError("at least one seed is needed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}", flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    MODEL_DIR.mkdir(exist_ok=True)

    # Workers build fresh degradations
    with ProcessPoolExecutor(WORKERS, multiprocessing.get_context("spawn"), ignore_interrupt) as pool:
        # Tune decoder on unseen matches
        valid = make_samples(list(valid_ids), VALID_COPIES, VALID_SEED, pool)
        train_ids = [match_id for match_id in MATCHES if match_id not in valid_ids]
        models = []
        for seed in seeds:
            print(f"step 1 of 2, seed {seed}, hold out {list(valid_ids)}", flush=True)
            models.append(fit(train_ids, valid, epochs, steps, seed, device, pool))
        # Real matches label possession differently
        print(f"loading {len(REAL_MATCHES)} real matches, first use downloads 1.8 GB", flush=True)
        groups = [
            list(pool.map(one_match, *copies(list(valid_ids), VALID_COPIES, VALID_SEED))),
            list(pool.map(load_real, REAL_MATCHES)),
        ]
        print("predicting validation matches, about 5 minutes", flush=True)
        probs = [[predict_match(models, match, device) for match in matches] for matches in groups]
        np.savez_compressed(CACHE_DIR / f"{name}_validation.npz", *probs[0], *probs[1])
        print("tuning decoder, about 6 minutes", flush=True)
        best, scores = tune_decoder(probs, groups)
        print(f"decoder {best} with {len(seeds)} networks averaged", flush=True)
        print(f"spellF1 open held-out {scores[0]:.4f}, real matches {scores[1]:.4f}", flush=True)
        (MODEL_DIR / f"{name}_decoder.json").write_text(json.dumps(best))
        del groups, probs

        # Refit on every open match
        models = []
        for seed in seeds:
            print(f"step 2 of 2, seed {seed}, fit on all matches", flush=True)
            models.append(fit(list(MATCHES), [], epochs, steps, seed, device, pool))
    torch.save([model.state_dict() for model in models], MODEL_DIR / f"{name}.pt")
    print(f"saved {MODEL_DIR / name}.pt")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid", nargs="+", default=list(VALID_IDS), help="held-out match ids")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--steps", type=int, default=STEPS, help="batches per epoch")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="one network per seed")
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args()
    run(tuple(args.valid), args.epochs, args.steps, tuple(args.seeds), args.name)


if __name__ == "__main__":
    main()
