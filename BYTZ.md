# BYTZ Technical Document

| Item | Value |
|---|---|
| Competition | More than a Hack 2027, Phase 1: Ball Possession |
| Team Name | BYTZ |
| Members | 1. Bryan Philinathaniel Hutagalung (Lead)<br>2. Shazya Audrea Taufik<br>3. Tamara Mayranda Lubis<br>4. Yovanka Sandrina Maharaja |

## Summary

We label each frame as home possession, away possession, or dead ball with five sequence networks. They are trained on seven open matches of the German Bundesliga and 2. Bundesliga from the open dataset of the Deutsche Fussball Liga (DFL), which carries a possession label for every frame. Before training, the open tracking is degraded until it resembles the automatic tracking of the provided match. A Viterbi decoder then turns frame probabilities into whole possessions, because the ranking metric counts possessions and not frames. A last step adapts the networks to the provided match with their own confident predictions, and the adapted networks only decide the split between the two teams. No ground truth label from the provided match is used for training or tuning, and we labelled nothing by hand.

| Measure | Where It Is Measured | Value |
|---|---|---|
| spellF1 of Our Model | 20 real matches with automatic tracking (SkillCorner), never trained on | 0.6712 |
| spellF1 of Our Model | Two held-out open matches (DFL), degraded | 0.6497 |
| spellF1 of the Submitted File | Public window of the provided match | 0.6533 |
| macroF1 of the Submitted File | Public window of the provided match | 0.9263 |
| spellF1 of `baseline_smooth50`, the Published Floor | Full provided match, reported by the organisers | 0.2433 |

The first two rows come from networks trained on five of the seven open matches, with the other two held out, and the submitted networks were then trained again on all seven. Those rows are measured without the adaptation step and are biased upward, because the decoder setting was chosen on those same matches. With two networks, the team split from adapted networks added 0.0085 on the real matches and 0.0031 on the open ones. The public window has 17 possessions and a noise band of about 0.33 on spellF1, so we printed it after each run as a check that the pipeline works and did not choose the model, the decoder, or the number of networks on it. The file `DETAILS.md` in the repository holds every test, the ideas we did not adopt, and the six places where the public window still touched the work.

## Approach

### Why Supervised Learning on Other Matches

The rules forbid labels from the provided match, and the public five minutes hold 17 possessions, too few to tune on. Hand-written rules such as "the team nearest to the ball has it" fail in this data, because the tactical tracker sees the ball on only 59% of graded frames and the gaps have a median of about 10 seconds. A sequence model can still read the players during a gap, since both teams move as a unit around the ball, and it can carry the last known state across the gap. The open DFL dataset gives about one million labelled frames at the same rate of 25 frames per second. That is enough to train a network of about 0.9 million parameters, as long as the inputs look the same in both sources.

### Simulation of the Automatic Tracking

The open tracking comes from the calibrated multi-camera system of the league, and the provided tracking is automatic. We measured the differences on the provided files, never on a label, and reproduce each one during training with random strength. The ball is hidden in stretches of several seconds, ball tracks under 15 to 50 frames are removed, positions are blurred along time, and player tracks drop out. The open data freezes the ball during stoppages, so training replaces it with a ball that a player carries to the restart spot. A second ball track is simulated from another camera position, broadcast cutaways are placed around dead ball stretches, and both axes and the two teams are flipped at random. Two of the seven open matches carry three referee tracks each, which the loader leaves out.

### Features and Model

Each frame gets 50 numbers, and none of them needs a player identity. There are 15 for the ball, 2 for the agreement of the two ball tracks, 5 for broadcast cutaways, and 14 for each team, such as the three nearest distances to the ball, the centroid, and the spread. Two one-dimensional convolutions feed a two-layer bidirectional gated recurrent unit (GRU) with 160 hidden units per direction, followed by a linear layer with three outputs. The network is trained with cross entropy on random crops of 2,048 frames (82 seconds), and every epoch uses fresh degraded copies of every match. Five networks from different seeds are averaged, which raised the score on the real matches from 0.6497 for one network to 0.6712 for five. Each network also averages eight mirrored views of the match, which added 0.015 on the real matches.

### Decoder

Frame probabilities flicker, and each flicker creates a false possession. A Viterbi pass finds the label path with the highest probability under a constant cost for every switch. The two validation sets want different costs, because they label possession differently: a possession lasts 5.5 to 9.2 seconds on average in the DFL labels and 10.0 to 18.2 seconds in the SkillCorner labels. We keep the setting whose larger loss against the best score of each set is smallest, which is a switch cost of 20 with no minimum length. It scores 0.6497 on DFL and 0.6712 on SkillCorner, a loss of 0.025 and 0.015. Two rules sit outside the model: frames outside the two graded blocks are labelled `X`, and graded frames up to and including the kickoff frame of a half are dead ball.

### Adaptation to the Provided Match

The rules allow "pseudo-labels produced by an algorithm, provided no person decided any of them" on the provided match. The averaged networks label the match, frames with a probability of at least 0.9 are kept, and each network trains 800 more steps on them. We tested the step on the 20 real matches, where each match plays the part of the provided match and its labels are only used to score afterwards. A frame probability has two parts, the dead ball probability and the split of the rest between the two teams, so we scored each part separately. The rule for adopting a part was set before the run: at least 0.004 on the real matches at more than two standard errors, and no more than 0.003 lost on the open copies.

| Part Taken From the Adapted Networks | 20 Real Matches | Six Degraded Open Copies |
|---|---|---|
| Both | +0.0134 | +0.0011 |
| Team split only | +0.0085 | +0.0031 |
| Dead ball probability only | +0.0058 | −0.0038 |

Only the team split passes the rule, so the submitted file takes the dead ball probability from the networks trained on open data and the team split from the adapted networks. The dead ball part loses on every open copy, and on the provided match the full adaptation moved the dead ball share by 0.020 in one direction, six times more than on the real matches. The submitted file agrees with the file without adaptation on 98.2% of the graded frames, and the dead ball share is the same (0.452). The number of possessions falls from 245 to 231, which lies inside the range the same step produces on the real matches.

## Assets Used and Dropped

| Asset | Used | How or Why Not |
|---|---|---|
| Tactical tracking | Yes | Main input: player positions per team and the ball |
| Broadcast tracking | Yes | Second ball track, aligned with a constant lag per half found by cross correlation. It fills 13% of the tactical ball gaps |
| Broadcast homography | Yes | Frames with an invalid homography are replays and close-ups. They cover 33% of graded frames and give the cutaway flag |
| Smart Tagging | No | Phases cover about 40% of graded frames, set pieces are windows of exactly 35 seconds, and throw-ins carry no team |
| Tactical homography | No | The camera centre is 12 metres off the ball on median, which the player centroid already gives |
| Tactical audio | No | Whistle detection gave 3 hits, 3 false alarms, and 2 misses in the public window, and the open matches have no audio to train on |
| Video frames | No | Revealing every hidden ball during live play gains at most 0.020 on held-out open matches, so a ball detector was not built |
| Public ground truth | Check only | Printed after the file is written, never used for training or to tune the model or the decoder |

In a controlled test on held-out open matches, single networks without the broadcast inputs scored 0.6357 and 0.6291 for two seeds, against 0.6636 and 0.6497 with them. Ideas that we tested and did not adopt include training on the real matches as well, training for 40 epochs instead of 24, 256 hidden units instead of 160, ten networks, per-match switch costs, and simulated tracker errors. None of them gained on both validation sets. `DETAILS.md` gives the numbers for each.

## Training and Validation Data

| Dataset | Role | Source | Who Labelled | Amount |
|---|---|---|---|---|
| DFL open dataset (Bassek et al. 2025), Creative Commons Attribution (CC-BY) 4.0 | Training, and two matches held out for validation | https://doi.org/10.6084/m9.figshare.28196177 | The league data provider, as part of the official match data | 7 matches, 1,002,644 frames |
| SkillCorner open data, Massachusetts Institute of Technology (MIT) licence | Validation and decoder setting only, never training | https://github.com/SkillCorner/opendata, commit `4340d27` | SkillCorner, automatically, from broadcast video | 20 matches, about 3.0 million frames |

Each DFL frame carries a possession team and a ball status, which give our three labels directly. A SkillCorner frame is live when it lies in a phase of play, and the team is the team in possession of that phase. Its tracking is resampled from 10 to 25 frames per second. No ground truth label comes from the provided match, and the adaptation step trains only on labels that the networks themselves predict for it.

## Possession Rules

Apart from the two rules in the Decoder section, we did not write possession rules. The model learns the definition that the DFL data uses, and the decoder then removes changes that do not last. We checked that definition on the labels of one open match: the ball is more than 3 metres from every player on 26% of live frames, yet none of the 408 changes of possession happens there. 98.5% of them happen with the ball within 1.5 metres of a player.

| Situation | Label |
|---|---|
| A team controls the ball | That team |
| Pass in flight or loose ball | The team that last controlled it, until the other team gains control |
| Challenge, duel, or brief touch | The label switches only when the evidence for the other team lasts. 6% of the 231 possessions in the submitted file are shorter than two seconds |
| Ball out of play, foul, goal, or any other stoppage | Dead ball, from the whistle or the ball leaving the pitch until the restart is taken |
| Restart (throw-in, free kick, corner, goal kick, kickoff) | The taking team, from the moment the ball is played |
| Before the kickoff of a half | Dead ball |

## Limitations

1. The simulation is the weak point, because the model only knows what the simulation shows it. A final audit of nine statistics for each of the 50 features found 33 of 450 values outside the range of 21 degraded copies.
2. The label style of the provided ground truth is unknown to us. The decoder setting is a compromise between two styles and loses about 0.02 spellF1 on each.
3. Possessions under two seconds are mostly lost. A true possession of that length is matched 22% of the time on the real matches and 14% on the open ones.
4. Stoppage boundaries are off by up to three seconds in the public window.
5. The adaptation cannot be checked on the provided match. Its gain is measured on other real matches, which come from a broadcast tracker.
6. A track of the provided match sometimes carries the wrong team: the home team shows more than 11 players on 8% of frames. Simulating this did not help.
7. With more time, we would measure real broadcast cutaway timing on matches that have both a broadcast and labels.

## Reproduction

```bash
uv sync
uv run possession/predict.py
uv run evaluation/validate_submission.py evaluation/results/BYTZ.csv
```

The repository pins every dependency in `uv.lock` and contains the trained weights, the adapted weights, and the decoder setting in `models/`. `predict.py` runs on the central processing unit (CPU), takes about 20 minutes, and writes `evaluation/results/BYTZ.csv`. On our machine the same weights gave a byte-identical file on every run, including a run from a fresh clone of the repository with a new environment. To rebuild the weights, run `uv run possession/train.py` (about 4 to 5 hours on an RTX 4060 graphics card) and then `uv run possession/adapt.py`, or open `notebook.ipynb` and set `TRAIN = True`. Training may differ on another graphics card, and adaptation may differ even on the same card, which is why both sets of weights are committed. The open matches are downloaded on first use, and the provided match data must be placed in `data/` as `docs/DATA.md` describes.
