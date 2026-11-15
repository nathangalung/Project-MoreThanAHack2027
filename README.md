# BYTZ Solution: Frame Level Ball Possession

Our entry for More than a Hack 2027, Phase 1. The task is to label every frame of one match as home possession, away possession, or dead ball.

The organiser documents are kept unchanged in `docs/`, and the scorer, the validator, and `games.py` in `evaluation/` are the organisers' files, unchanged. The documents in `docs/` describe the organisers' own repository layout, so the paths they mention (`starter/`, a root `games.py`, `requirements.txt`) do not exist here. This README describes our layout.

## Team

| Team Name | Members |
|---|---|
| BYTZ | Bryan Philinathaniel Hutagalung, Shazya Audrea Taufik, Yovanka Sandrina Maharaja, Tamara Mayranda Lubis |

## Approach

1. Train a sequence model on seven open Bundesliga matches that carry a possession team and a ball status for every frame (DFL open dataset, Bassek et al. 2025, CC-BY 4.0).
2. Before training, degrade the clean open tracking so it looks like the automatic tracking of the target match: smoothed positions, long stretches without the ball, lost player tracks, and no player identities.
3. Align the broadcast recording to the tactical timeline, use the broadcast ball as a second ball track, and mark the frames where the broadcast cuts away from the live wide shot.
4. Compute 50 features per frame that only need team membership, never identity.
5. Predict home, away, or dead ball for each frame with a convolution and bidirectional GRU network. Five networks trained from different seeds are averaged, each over eight mirrored views of the match.
6. Turn frame probabilities into possessions with a Viterbi decoder that charges a cost for each switch, then absorb runs that are too short.
7. Choose the decoder setting on matches that are never trained on: two held-out open matches, and 20 open matches with real automatic tracking (SkillCorner open data, MIT licence).
8. Label every graded frame before the kickoff of a half as dead ball. The kickoff frames are the anchors the organisers publish in `games.py`.

No label from the target match is used for training or tuning. The public 5 minutes are only printed as a check after the file is written.

## Results

| Measure | Where It Is Measured | Value |
|---|---|---|
| spellF1 | 20 real matches with automatic tracking, never trained on | 0.6712 |
| spellF1 | Two held-out open matches, degraded | 0.6497 |
| spellF1 | Public window of the target match | 0.6667 |
| macroF1 | Public window of the target match | 0.9259 |

The public window has 17 possessions and a noise band of about 0.33 on spellF1, so it is a check and not a measure. `BYTZ.md` has the full account.

## Why This Design

| Problem in the Task | What We Do About It |
|---|---|
| The ranking metric counts whole possessions, so flicker is expensive | Decode segments with a switch cost and a minimum length |
| The ball is missing on 41% of graded frames, mostly in long stretches | Train with the ball hidden in the same way, and give the model the gap length as a feature |
| No training or tuning on the target match is allowed | Train only on the open DFL matches. Tune the decoder on two held-out DFL matches and 20 real SkillCorner matches |
| A score on simulated data cannot show a flaw in the simulation | Validate on real automatic tracking from another provider, and compare all features of the simulation with the target tracking |
| Track slots are tracklets, not players | Use set features per team: nearest distances, counts, centroid, spread, speed |
| A dead ball is hard to see from tracking when the ball stays visible, for example after a foul | Use broadcast cutaways. Directors cut to replays and close-ups during stoppages. The open matches have no broadcast, so training simulates cutaways from their dead ball segments |

## What We Found in the Data

| Asset | Finding | Use |
|---|---|---|
| Tactical tracking | 46 slots, 20.5 players per graded frame, ball seen on 59.3% of graded frames | Main input |
| Tactical tracking | Ball gaps are long (median about 10 seconds) and the ball is often still visible during a dead ball | Dead ball needs more evidence than a missing ball |
| Tactical tracking | Positions are much smoother than the open data | Open data is blurred before training |
| Tactical tracking | 33 stretches where most tracks vanish at once | Simulated in training |
| Open DFL tracking | The ball is frozen during every dead ball, with a speed of exactly zero. In the target match it keeps moving, because a player carries it to the restart spot. A model trained on the frozen ball found only 32% of dead ball frames in the public window | Training replaces the frozen ball with one that moves between nearby players toward the restart spot. The final model finds 90% of them |
| Tactical tracking | The ball is never visible for only a few frames, rarely drops out for one frame, and moves much more smoothly than our first simulation (99th percentile of acceleration 32 m/s² against 62 to 288) | Simulation matched to these statistics. On 20 real matches, two averaged networks rose from 0.591 to 0.610 spellF1 |
| SkillCorner open data | 20 A-League matches with real broadcast tracking and a possession team per frame. The ball is seen on about 60% of frames, like the target. Possessions last 10 to 18 seconds, against 5.5 to 9.2 in the DFL labels | Validation and decoder setting only, never training |
| Smart Tagging | Phases cover about 40% of graded frames, set pieces are fixed 35 second windows, every throw-in has no team | Not used. An earlier version read the kickoff tags, but the tag of the second half starts 81 frames before the kickoff the organisers publish |
| Broadcast tracking | Its ball track matches the tactical one with a constant lag per half (4,014 and 26,920 frames), found by cross correlation, median ball distance under 1 metre | Fills 13% of tactical ball gaps, graded ball coverage rises from 59.3% to 64.4% |
| Both ball tracks | They differ by over 5 metres on 13% of shared frames, and 83% of those differences lie along the camera viewing axis. A raised ball seen from two camera heights gives exactly this | The distance between the two tracks is a feature. Training simulates two cameras using the real ball height in the open data |
| Broadcast homography | Invalid on 33% of graded frames (replays, close-ups). We first noticed in the public window that these cover 76% of dead ball frames and under 4% of live frames, and then confirmed the value of the input on held-out open matches | Cutaway flag and timing features |
| Tactical homography | The camera centre follows the ball along the pitch (correlation 0.83) but is 12 metres off on median, about what the player centroid already gives | Tried, not used |
| Tactical audio | Whistle tones are detectable, but in the public window they gave 3 hits, 3 false alarms, and 2 missed stoppages. The open matches have no audio, so this is the only check we could make, and it is the one choice that rests on the public window | Tried, not used |
| Video frames | Ball detection could gain at most 0.020 spellF1 with a perfect detector, because the ball is already visible on all but 2.5% of the frames predicted live | Measured, not built |
| Target tracking without labels | Networks that adapt to a real match with their own confident labels gain 0.014 spellF1 on real matches. Half of that comes from frames with no players, which the target tracking does not have, and on data built like the target the step changes nothing | Tested, not adopted. `BYTZ.md` has the numbers |

## Folder Layout

| Path | Whose | Content |
|---|---|---|
| `possession/` | Ours | The pipeline package: loading, features, model, training, prediction |
| `BYTZ.md` | Ours | The technical document |
| `pipeline.ipynb` | Ours | Runs the pipeline step by step and shows tables and charts |
| `tests/` | Ours | Tests for the metric, decoder, features, and degradation |
| `models/` | Ours | Trained weights and decoder settings, committed after training |
| `evaluation/` | Organisers | The scorer, the validator, match constants, the public ground truth, and the results folder |
| `docs/` | Organisers | Challenge, data, evaluation, and rules documents |
| `data/` | Organisers | The match data, not committed |
| `artifacts/` | Generated | Caches and downloaded open matches, not committed |

`possession/` is what we wrote to produce the labels. `evaluation/` is what the organisers wrote to score them. Keeping them apart shows which code decides the ranking and which code is ours.

## How to Run

Install uv, put the match data in `data/` as `docs/DATA.md` describes, then use either the notebook or the command line. Both call the same functions.

### Notebook

```bash
uv sync
uv run jupyter lab pipeline.ipynb
```

Run the cells from top to bottom. Clear the outputs before committing, because the match data may not be published.

### Command Line

```bash
uv sync
uv run pytest
uv run possession/train.py
uv run possession/predict.py
uv run evaluation/validate_submission.py evaluation/results/BYTZ.csv
uv run evaluation/run_evaluation.py
```

| Command | What It Does | Time on an RTX 4060 |
|---|---|---|
| `uv sync` | Installs the pinned dependencies from `uv.lock` | A few minutes |
| `uv run pytest` | Checks the metric against the scorer, and the decoder, features, and degradation | Seconds |
| `train.py` | Downloads the open matches on first use (about 4.3 GB), fits five networks on five matches, tunes the decoder on the two held-out matches and the 20 real matches, then refits the networks on all seven and saves `models/model.pt` and `models/model_decoder.json` | About 4 to 5 hours |
| `predict.py` | Writes `evaluation/results/BYTZ.csv` and prints the public window score | About 10 minutes |
| `validate_submission.py` | Runs the official format checks | Seconds |
| `run_evaluation.py` | Prints the official board on the public window | Seconds |

For a quick check that everything runs, use a short training first. It still downloads the open matches and spends about 11 minutes on decoder tuning:

```bash
uv run possession/train.py --epochs 4 --steps 60 --seeds 0 --name smoke
uv run possession/predict.py --models smoke --decoder smoke_decoder --out evaluation/results/smoke.csv
```

## Reproducing the Submitted File

An earlier version of the training gave byte identical weights when run twice on our machine. A different GPU or driver may not, so the trained weights are committed in `models/`, and the submitted file is regenerated from them:

```bash
uv run possession/predict.py
```

`predict.py` runs on CPU by default, which gave the same file on every run we tried. CUDA inference gave a slightly different file, so do not pass `--device cuda` for the final file.

## Code Layout

| File | Purpose |
|---|---|
| `possession/paths.py` | Paths and match constants, read from `games.py` |
| `possession/tracking.py` | Reads the provided tracking with Polars and caches it |
| `possession/broadcast.py` | Aligns the broadcast, merges its ball, and builds the target match arrays |
| `possession/dfl.py` | Downloads and parses the open DFL matches |
| `possession/skillcorner.py` | Downloads and parses the open SkillCorner matches with real tracking |
| `possession/match.py` | Common array format for every match, and its mirrored views |
| `possession/degrade.py` | Makes clean tracking look automatic |
| `possession/features.py` | Per frame features |
| `possession/model.py` | Network, windowed inference, and averaging over mirrored views |
| `possession/decode.py` | Viterbi decoder and short run removal |
| `possession/metrics.py` | spellF1 and macroF1 for tuning on open matches |
| `possession/train.py` | Training on fresh degraded copies each epoch, and decoder tuning on both validation sets |
| `possession/adapt.py` | Optional adaptation to the target match with pseudo-labels, tested and not adopted |
| `possession/predict.py` | Submission writer |
| `tests/test_metrics.py` | Compares our metric with the official scorer |
| `tests/test_pipeline.py` | Checks the decoder, features, and degradation |

## Dependencies

| Package | Why |
|---|---|
| polars | Reads the tracking text, the DFL XML, the SkillCorner JSON lines, and all CSV files in our code |
| pandas (group `evaluation`) | Needed only by the organisers' scorer, which we keep unchanged |
| numpy | Array maths for features, decoding, and metrics |
| torch | The sequence model, on GPU when one is present |
| ruff, pyright, pytest (group `dev`) | Lint, type check, tests |
| jupyterlab, matplotlib (group `notebook`) | The notebook and its charts |

## Training and Validation Data

| Dataset | Role | Source | Labels | Amount |
|---|---|---|---|---|
| DFL open dataset (Bassek et al. 2025, Scientific Data 12, 195), CC-BY 4.0 | Training, two matches held out | https://doi.org/10.6084/m9.figshare.28196177, mirrored at https://huggingface.co/datasets/pysport/idsse-data | `BallPossession` and `BallStatus` per frame, produced by the league data provider | 7 matches, about 1.0 million frames at 25 fps |
| SkillCorner open data, MIT licence, credit to SkillCorner | Validation and decoder setting only | https://github.com/SkillCorner/opendata, pinned to commit `4340d27` | Team in possession of each phase of play, produced automatically by SkillCorner | 20 matches, about 3.0 million frames after resampling from 10 to 25 fps |

## Limitations and Next Steps

1. The label style of the target ground truth is unknown to us. The DFL labels switch on brief touches and the SkillCorner labels do not, so the decoder setting is a compromise between the two.
2. The organisers call their kickoff frames playback anchors, so the dead ball rule may be off by a frame or two at the second kickoff.
3. Cutaway timing in training is simulated, not measured, because the DFL matches have no broadcast. The simulation uses wide random ranges so the model does not depend on one setting.
4. Not used: the Smart Tagging phases, and ball detection from video for frames where both trackers lost the ball.
