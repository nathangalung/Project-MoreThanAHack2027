# BYTZ Technical Document

More than a Hack 2027, Phase 1: Ball Possession.

| Team Name | Members |
|---|---|
| BYTZ | Bryan Philinathaniel Hutagalung, Shazya Audrea Taufik, Yovanka Sandrina Maharaja, Tamara Mayranda Lubis |

## Summary

We label each frame as home possession, away possession, or dead ball with an average of five sequence models that never saw a label from the provided match. Each model is trained on seven open Bundesliga matches that carry a possession label for every frame. The open tracking is clean, so before training we degrade it until it looks like the automatic tracking of the provided match. A Viterbi decoder then turns frame probabilities into whole possessions, because the ranking metric counts possessions and not frames. The decoder is tuned on matches that are never trained on, including 20 open matches with real automatic tracking. No label from the provided match is used for training. Its tracking, without labels, was only used to set how the open data is degraded.

| Measure | Where It Is Measured | Value |
|---|---|---|
| spellF1 of Our Model | 20 real matches with automatic tracking (SkillCorner), never trained on | 0.6637 |
| spellF1 of Our Model | Two held-out open matches (DFL), degraded | 0.6438 |
| spellF1 of the Submitted File | Public window of the provided match | 0.6667 |
| macroF1 of the Submitted File | Public window of the provided match | 0.8972 |
| spellF1 of `baseline_smooth50` | Public window of the provided match | 0.3256 |
| spellF1 of `baseline_smooth50`, the Published Floor | Full provided match, reported by the organisers | 0.2433 |

The scores on the open matches are optimistic. The decoder setting was chosen on those same matches, and the real matches also decided three design choices: the last simulation change, the averaging over mirrored views, and the number of networks. The networks behind the first two rows were trained on five open matches. The submitted networks were retrained on all seven.

The public window has only 17 possessions. The official bootstrap gives a noise band of about 0.33 on its spellF1, so it cannot separate two models. We printed it after each run as a check that the pipeline is correct. The model, the decoder, and the number of networks were chosen on the other two validation sets. The public window still touched the work in four places, and we list them so a reader can judge.

| Where | What the Public Window Did | What the Decision Rests On |
|---|---|---|
| Frozen ball in the simulation | A dead ball recall of 32% in the window showed that something was wrong | The open data freezes the ball at a speed of exactly zero, and the provided tracking does not. That is a fact about the inputs |
| Broadcast cutaways as an input | We first saw there that cutaways line up with stoppages | A controlled test on held-out open matches |
| Whistle detection from audio | It gave 3 hits, 3 false alarms, and 2 misses in the window, and we dropped it | Only the window. The open matches have no audio, so no other check was possible |
| Adaptation step | A lower score in the window made us look closer | The split of the gain by frame type on the real matches, and the test on held-out open matches |

The gains we report for single design choices lie between 0.005 and 0.02 spellF1. They are measured over 20 matches, where they are clear. On the one scored match the organisers give a noise band of about 0.04, so no single one of them would be visible there.

## Approach

### Why Supervised Learning on Other Matches

The rules forbid labels from the provided match, and the public five minutes are too short to tune on. Hand-written rules such as "the team nearest to the ball has it" fail in this data for a concrete reason: the tactical tracker sees the ball on only 59% of graded frames, and the gaps are long, with a median of about 10 seconds. A rule has nothing to work with during a gap. A sequence model can still read the players, since both teams move as a unit around the ball, and it can carry the last known state across the gap.

The open DFL dataset gives about one million labelled frames at the same frame rate of 25 per second. That is enough to train a small network, as long as the inputs look the same in both worlds.

### Making the Open Data Look Like the Provided Data

The open tracking comes from the calibrated multi-camera system of the league. The provided tracking is automatic. We measured the differences on the provided files and reproduce each one at training time with random strength, so the model cannot rely on one setting. Only the tracking itself is measured, never a label.

| Difference Found | How Training Reproduces It |
|---|---|
| Ball missing in long stretches, more often during stoppages and when the ball is high or off the pitch | Ball hidden in random stretches with those same biases |
| The ball is never visible for only a few frames, and it rarely drops out for a single frame | Ball tracks shorter than 15 to 50 frames are removed, and single frame gaps are rare |
| Positions much smoother than the open data | Gaussian blur along time that ignores missing frames, with a width of 4 to 9 frames for the ball |
| Player tracks drop out, and 33 stretches lose most tracks at once | Random track loss and simulated camera loss |
| No player identities, only a team per track | Features only use sets of players per team |
| Ball keeps moving during a stoppage, because a player carries it to the restart spot. The open data freezes it with a speed of exactly zero | Frozen ball replaced by a path that visits nearby players and ends at the restart spot |
| Two cameras disagree about the ball, mostly when it is in the air | A second ball track is simulated from another camera position using the real ball height in the open data, with a slow calibration error and a time shift of up to three frames |
| Broadcast cuts to replays and close-ups, mostly during stoppages | Cutaways simulated around dead ball stretches, plus a few during live play |
| Positions are a few metres off centre | Random shift of all positions by up to 2.5 metres |
| Either team can attack either way | Random flips of both axes and random swap of home and away |

Two findings changed the result more than any model change.

**The frozen ball.** A first model trained on the frozen ball reached 0.66 on held-out open matches but found only 32% of the dead ball frames in the public window, because it had learned that a still ball means a stoppage. The submitted model finds 90% of them. The held-out score barely moved after the fix, which shows that a held-out score on simulated data cannot detect a flaw in the simulation itself.

**Flicker and jerk.** We then compared all 50 features between the degraded open data and the provided match, including run lengths and frame-to-frame changes. The means agreed, but the timing did not.

| Statistic of the Tactical Ball Track | Provided Match | Simulation Before | Simulation After |
|---|---|---|---|
| Hidden runs per 10 minutes | 14 | 43 to 121 | 12 to 18 |
| Median hidden run in frames | 268 | 1 to 34 | 103 to 436 |
| Share of visible runs under 4 frames | 0% | 2% to 13% | 0% |
| 99th percentile of ball acceleration in m/s² | 32 | 62 to 288 | 31 to 92 |

The ranges cover several random copies of the open matches.

On the 20 real matches, the average of two networks rose from 0.591 to 0.610 spellF1 after this change, and both single networks improved.

### Features

Each frame gets 50 numbers. None of them needs a player identity.

| Group | Count | Content |
|---|---|---|
| Ball | 15 | Seen flag, time since and until it was seen, gap length, position, distance to the lines, out-of-pitch flag, speed, velocity, acceleration |
| Two Cameras | 2 | Both balls seen, distance between the two ball tracks |
| Broadcast | 5 | Cutaway flag and four clocks around cutaways |
| Each Team | 14 per team | Three nearest distances to the ball, players within 1.5 and 4 metres, speed of the nearest player, mean speed, player count, centroid, ball offset from centroid, spread, drift |

### Model

Two one-dimensional convolutions feed a two-layer bidirectional GRU with 160 hidden units per direction, followed by a linear layer with three outputs. The network has about 0.9 million parameters. It is trained with cross entropy, label smoothing of 0.02, AdamW, and a one-cycle learning rate that peaks at 0.002, on random crops of 2,048 frames (82 seconds).

A recurrent network fits the task because possession is a state that persists: the label at a frame depends on what happened seconds before and after.

| Choice | Evidence |
|---|---|
| Fresh degraded copies of every match in every epoch | With fixed copies the training loss jumped each time they were replaced, so the network was memorising them. Held-out spellF1 rose from 0.6458 to 0.6636 for one seed and from 0.6459 to 0.6497 for another |
| Average of five networks from different seeds | On the real matches the score rises with every network added: 0.6415 for one, 0.6551 for two, 0.6589 for three, 0.6618 for four, and 0.6637 for five. Five against three is +0.0047 with a paired standard error of 0.0011, better on 17 of 20 matches. The held-out open matches stay level |
| Average over eight mirrored views of the match (both flips and the team swap) | The network is trained with these as augmentations, so every view is a valid input. On the real matches it added 0.015 spellF1, with a paired standard error of 0.003 |

Every training frame of all seven matches is used. Random crops only decide which stretches share a batch.

### Decoder

Frame probabilities flicker, and each flicker creates a false possession. A Viterbi pass finds the best label path under a fixed cost for every switch, and runs shorter than a minimum length are absorbed by their left neighbour.

The two validation sets want different settings, because they label possession differently. The DFL labels switch on brief touches, and a possession lasts 5.5 to 9.2 seconds on average, depending on the match. The SkillCorner labels switch only when a team gains a possession phase, and a possession lasts 10.0 to 18.2 seconds. The challenge definition asks for controlled possession and says that a touch, a deflection, or a won aerial duel is not a change, which is closer to the second style. We do not know exactly where the provided ground truth sits between the two.

| Switch Cost | spellF1 on Held-Out DFL | spellF1 on Real SkillCorner |
|---|---|---|
| 0 | 0.6621 | 0.6510 |
| 5 | 0.6632 | 0.6534 |
| 10 | 0.6574 | 0.6557 |
| 20 | 0.6438 | 0.6672 |
| 40 | 0.6139 | 0.6773 |
| 60 | 0.5832 | 0.6840 |
| 120 | 0.5056 | 0.6600 |

Each cell is the best score over the minimum lengths, and the table leaves out some costs. The best score anywhere in the grid is 0.6639 for DFL (cost 2) and 0.6840 for SkillCorner (cost 60). We keep the setting whose larger loss against these two is smallest. That is a switch cost of 20 with no minimum length. It scores 0.6438 on DFL and 0.6637 on SkillCorner, a loss of 0.020 on each. This rule was fixed on these two sets only. We also looked at the public window score for several costs as a check, and did not use it for the choice.

### Adapting to the Provided Match, Tested and Not Adopted

The simulation can only approximate the real tracker. The rules allow one way to close part of that gap: on the provided game "learning without human labels" remains fine, including "pseudo-labels produced by an algorithm, provided no person decided any of them". We built and tested that step, and in the end left it out of the submitted file. The code is kept in `possession/adapt.py`.

1. The networks predict a match, and the decoder turns the prediction into labels.
2. A frame is kept as a pseudo-label when the networks give its label a probability of at least 0.9.
3. Each network is trained for 800 more steps on the kept frames, with the eight mirrored views as augmentation.
4. The adapted networks predict the match again.

We could test this step honestly, because each real SkillCorner match can play the part of the provided match: the networks adapt to it without its labels, and the labels are only used to score afterwards. The steps and the threshold were fixed after the first test and not tuned further.

| Test | Before | After | Change |
|---|---|---|---|
| 20 real matches, two networks trained on five open matches | 0.6565 | 0.6702 | +0.0136, standard error 0.0035 |
| Eight real matches, three networks trained on all seven open matches | 0.6597 | 0.6742 | +0.0145, standard error 0.0046 |
| Six degraded copies of the held-out open matches | 0.6398 | 0.6416 | +0.0018, standard error 0.0048 |

The step clearly helps on the real matches and changes nothing on the open ones. It is also more than extra smoothing: with the decoder rule applied to each side separately, the adapted networks score 0.007 higher on both validation sets. These tests ran with two and three networks.

We also tried three changes to the adaptation itself, on ten of the real matches, with the other ten kept aside to confirm a winner.

| Change | Result |
|---|---|
| A second round: label again with the adapted networks and adapt once more | +0.0040 on the first ten matches, but −0.0028 on the ten kept aside. Not confirmed |
| Mixing batches of degraded open data into the adaptation | 0.0057 worse than plain adaptation. It pulls the networks back toward the simulation |
| Learning the probabilities of the networks, not their decoded labels | No gain over the networks before adaptation. The confident decoded labels are what helps |

We did not adopt the step, for three reasons.

1. Half of the gain does not carry over. The SkillCorner tracker reports no players at all while the broadcast shows a replay, and half of the gain in spellF1 comes from those frames (0.007 of 0.0145). The provided tactical tracking keeps its players during replays, so that half has nothing to work on.
2. On the data built to resemble the provided match, the degraded open matches, the step changes nothing.
3. The step has a cost. It adds a stage that cannot be checked on the provided match, and it would put weights trained on the provided tracking into a public repository.

From the first two points we expect a gain of about 0.005 on the provided match, which is about as large as its own uncertainty. We judged that too little for the cost, and preferred the model that uses open data only. A team that weighs the cost differently could reasonably keep the step.

The public window was not part of this choice, but it is why we looked closer. When we printed it as the usual check, the adapted networks scored 0.600 spellF1 against 0.667 before adaptation, with macroF1 unchanged. Two possessions next to stoppages explain the difference, which is within what 17 possessions produce by chance. Looking into those frames showed the mechanism: adaptation makes the networks call a stoppage sooner when they cannot see the ball. That led us to split the gain on the real matches by frame type, which gave the first reason above.

### Rules Outside the Model

| Rule | Reason |
|---|---|
| Frames outside the two graded blocks are labelled `X` | They are not scored |
| Graded frames before the kickoff of a half are dead ball | The organisers state that the second graded block starts 2,934 frames before the kickoff and that all of those frames are dead ball. The kickoff frames are the anchors they publish in `games.py`. A block that starts within one second of its kickoff, like the first half, is left as it is |

## Assets Used and Dropped

| Asset | Used | How or Why Not |
|---|---|---|
| Tactical tracking | Yes | Main input: player positions per team and the ball |
| Broadcast tracking | Yes | Second ball track. Aligned to the tactical timeline with a constant lag per half (4,014 and 26,920 frames), found by cross correlation of the ball position. It fills 13% of the tactical ball gaps |
| Broadcast homography | Yes | Frames with an invalid homography are replays and close-ups. They cover 33% of graded frames |
| Smart Tagging | No | Phases cover about 40% of graded frames, set pieces are fixed 35 second windows, and throw-ins carry no team, so it was too coarse to help. An earlier version used the two kickoff tags, but the tag of the second half starts 81 frames before the kickoff that the organisers publish |
| Tactical homography | No | The camera centre follows the ball (correlation 0.83) but is 12 metres off on median, which the player centroid already gives |
| Tactical audio | No | Whistle detection gave 3 hits, 3 false alarms, and 2 misses in the public window. That is the only check we could make, because the open matches have no audio |
| Video frames | No | We measured what ball detection could give at best before building it. On held-out open matches, revealing every hidden ball during live play raises spellF1 by 0.020, half of them by 0.010, and with false detections in 30% of gaps the gain is gone. In the provided match both cameras miss the ball on 36% of graded frames, but only 2.5% of graded frames are predicted live with the ball hidden (measured with an earlier three-network model). The gaps lie in stoppages, where a found ball does not change the label |
| Public ground truth | Check only | Printed after the file is written. Never used for training or to tune a setting. The summary lists the four places where it still touched the work |

In a controlled test on held-out open matches, single networks without the broadcast inputs scored 0.6357 and 0.6291 for two random seeds, against 0.6636 and 0.6497 with them, so we kept them. The test ran before the last simulation change. The gain is measured on simulated cutaways and a simulated second camera, so it shows that the model can use these inputs, not how large the gain is on the provided match.

### Tried and Not Adopted

| Idea | Result |
|---|---|
| Extra outputs that predict the hidden ball position and the possession boundaries | No gain on held-out open matches |
| Training on the real SkillCorner matches as well | Raised spellF1 on six held-out real matches to 0.73. We did not adopt it, because that gain is measured on the same tracker and the same label style as the added training data, and because it would leave us no real match that the model has never seen |
| Separate switch costs for team changes and for stoppages, since the two label styles should only disagree about team changes | Tested on the saved validation predictions of an earlier three-network model. The larger loss fell from 0.025 to 0.024, which is no real gain, so the single cost stays |
| Longer training (40 epochs instead of 24) and a wider network (256 hidden units instead of 160) | On the real matches a single network scored 0.608 and 0.609, against 0.616 for the same seed with the submitted settings. No gain, so the smaller and shorter setting stays |
| Training on labels without brief touches: a possession of under two seconds between two possessions of the other team is merged into them, to teach the definition of the challenge directly | With two networks, the best score on the real matches fell from 0.679 to 0.669, and at the chosen switch cost it stayed at 0.656. On the original open labels it fell from 0.640 to 0.599. The networks only learn the smoothing that the decoder already does |
| Label shift correction: the share of each label in a match is estimated from the network outputs without labels (Saerens et al. 2002), and the probabilities are rescaled before decoding | The estimated shares were close to the true ones, but the score did not rise. At full strength it fell from 0.664 to 0.658 on the real matches and from 0.644 to 0.633 on the open matches. The networks read a long context and already adjust to the match |
| Ten networks in place of five | At the chosen switch cost the real matches moved from 0.664 to 0.667 and the open matches from 0.644 to 0.639. The gain is inside the noise and the cost is twice the training time |
| Two more training matches from the open sample data of Metrica Sports, with possession taken from the event file | Two networks trained with them were compared with the same two seeds trained without them. At the chosen switch cost the real matches moved from 0.6565 to 0.6568 and the open matches from 0.6398 to 0.6405, so the extra matches gave no measurable gain |
| Writing `X` on uncertain frames | The scorer fills each `X` with the previous label before it matches possessions, so this equals repeating the last answer, and it costs macroF1 |
| A pretrained model | We found no released weights for possession from tracking data. Published models for this kind of input (for example TranSPORTmer and Ball Radar) come as papers. Video ball detectors exist, but we could not validate one on this video in the time left |

## Training and Validation Data

| Dataset | Role | Source | Who Labelled | Amount |
|---|---|---|---|---|
| DFL open dataset (Bassek et al. 2025, Scientific Data 12, 195), CC-BY 4.0 | Training, and two matches held out for validation | https://doi.org/10.6084/m9.figshare.28196177, mirrored at https://huggingface.co/datasets/pysport/idsse-data | The league data provider, as part of the official match data. Each frame carries `BallPossession` (home or away) and `BallStatus` (alive or dead) | 7 matches, 1,002,644 frames at 25 frames per second |
| SkillCorner open data, MIT licence, credit to SkillCorner | Validation and decoder setting only, never training | https://github.com/SkillCorner/opendata, pinned to commit `4340d27` | SkillCorner, automatically. The tracking comes from computer vision on broadcast video. A frame is live when it lies in a phase of play, and the team is the team in possession of that phase | 20 matches of the Australian A-League 2024/25, about 3.0 million frames after resampling from 10 to 25 frames per second |

The SkillCorner README mentions 10 matches. Its `matches.json` lists 20, and all 20 have data.

No label comes from the provided match, and we labelled nothing by hand.

## Possession Rules

We did not write possession rules. The model learns the definition that the DFL data uses, and the decoder then removes changes that do not last. We state the learned definition here so a reader knows where the lines fall. We checked it on the labels: in one open match, the ball is more than 3 metres from every player on 26% of live frames, yet none of the 408 changes of possession happens there. 98.5% of them happen with the ball within 1.5 metres of a player.

| Situation | Label |
|---|---|
| A team controls the ball | That team |
| Pass in flight or loose ball | The team that last controlled it, until the other team gains control |
| Challenge, duel, or brief touch | The label switches only when the evidence for the other team lasts. In the submitted file one of the 256 possessions is shorter than one second, and 6% are shorter than two seconds |
| Ball out of play, foul, goal, or any other stoppage | Dead ball, from the whistle or the ball leaving the pitch until the restart is taken |
| Restart (throw-in, free kick, corner, goal kick, kickoff) | The taking team, from the moment the ball is played |
| Before the kickoff of a half | Dead ball |

## Limitations

1. The simulation is the weak point. The model only knows what the simulation shows it, and two wrong details cost more than any model change gained. Other gaps may remain.
2. The label style of the provided ground truth is unknown to us. Our decoder setting is a compromise between two styles, and it loses about 0.02 spellF1 on each compared with the best setting for that style.
3. The real validation matches come from a broadcast tracker. During replays they have no players at all, while the provided tactical tracking keeps its players, so they are close to the provided data but not the same.
4. Stoppage boundaries are off by up to three seconds in the public window, and one false stoppage of five seconds appears.
5. Short possessions are mostly lost. The public window has one of 0.8 seconds and one of 1.3 seconds, and we miss both.
6. The organisers call their kickoff frames playback anchors, so the dead ball rule may be off by a frame or two at the second kickoff.
7. The networks never see the real tracker of the provided match in training. Adapting to it with pseudo-labels helped on other real matches, but we could not show that it would help here.
8. With more time, we would measure real broadcast cutaway timing on matches that have both a broadcast and labels.

## Reproduction

The repository pins every dependency in `uv.lock` and contains the trained weights in `models/`.

```bash
uv sync
uv run possession/predict.py
uv run evaluation/validate_submission.py evaluation/results/BYTZ.csv
```

`predict.py` runs on CPU, takes about 10 minutes, and writes `evaluation/results/BYTZ.csv`. On our machine, the same weights gave a byte-identical file on every run, including a run from a fresh clone of the repository with a new environment. Prediction on CUDA gave a slightly different file, so CPU is the default.

To rebuild the weights, run `uv run possession/train.py` (about 4 to 5 hours on an RTX 4060), or open `pipeline.ipynb`, set `TRAIN = True`, and run all cells. An earlier version of the training gave byte-identical weights when we ran it twice on our machine. We did not repeat that test for the final version, and a different GPU or driver may give other weights, which is why the weights are committed.

The adaptation that we tested and did not adopt can be run with `uv run possession/adapt.py` (about 12 minutes) and `uv run possession/predict.py --models model_adapted --out evaluation/results/adapted.csv`.

The open matches are downloaded on first use: about 2.5 GB for DFL and 1.8 GB for SkillCorner. The provided match data must be placed in `data/` as `docs/DATA.md` describes, and it is not part of the repository.
