"""Check decoder, features, and degradation."""

from dataclasses import replace

import numpy as np
import polars as pl

from possession.decode import decode, drop_short, viterbi, waiting
from possession.degrade import carry_dead_ball, degrade, runs, smooth
from possession.dfl import club_ids, squads
from possession.features import FEATURE_NAMES, N_FEATURES, match_features
from possession.match import AWAY, DEAD, HOME, Match, mirrored
from possession.skillcorner import hold, upsample
from possession.train import tune_decoder


def toy_match(n: int = 3000) -> Match:
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.2, (n, 2)).astype(np.float32)
    ball = np.cumsum(steps, axis=0)
    home = ball[:, None, :] + rng.normal(0, 5, (n, 11, 2)).astype(np.float32)
    away = ball[:, None, :] + rng.normal(0, 5, (n, 11, 2)).astype(np.float32)
    label = np.repeat(np.array([HOME, DEAD, AWAY], dtype=np.int8), n // 3)
    none = np.zeros(n, dtype=bool)
    return Match("toy", home, away, ball, np.full_like(ball, np.nan), ((0, n),), none, label)


def test_dead_ball_moves_like_a_carried_one() -> None:
    match = toy_match()
    frozen = match.ball.copy()
    frozen[match.label == DEAD] = frozen[999]
    carried = carry_dead_ball(replace(match, ball=frozen), np.random.default_rng(3))
    dead = np.flatnonzero(match.label == DEAD)
    speed = np.linalg.norm(np.diff(carried[dead], axis=0), axis=1) * 25
    assert np.median(speed) > 0.2
    assert np.array_equal(carried[match.label != DEAD], frozen[match.label != DEAD])


def test_viterbi_removes_flicker() -> None:
    prob = np.full((200, 3), 0.05)
    prob[:100, HOME] = 0.9
    prob[100:, AWAY] = 0.9
    prob[50, HOME], prob[50, AWAY] = 0.05, 0.9
    path = viterbi(np.log(prob), penalty=5.0)
    assert (path[:100] == HOME).all() and (path[100:] == AWAY).all()


def test_drop_short_absorbs_brief_runs() -> None:
    path = np.array([HOME] * 50 + [AWAY] * 3 + [HOME] * 50 + [DEAD] * 5 + [HOME] * 20, dtype=np.int8)
    assert (drop_short(path, min_spell=10, min_dead=10) == HOME).all()


def test_decode_marks_outside_blocks_dead() -> None:
    prob = np.zeros((100, 3))
    prob[:, HOME] = 1.0
    labels = decode(prob, ((20, 80),), penalty=1.0, min_spell=0, min_dead=0)
    assert (labels[20:80] == HOME).all() and (labels[:20] == DEAD).all() and (labels[80:] == DEAD).all()


def test_waiting_covers_frames_before_a_late_kickoff() -> None:
    blocks = ((100, 500), (600, 900))
    mask = waiting(1000, blocks, kickoffs=(101, 700), slack=25)
    assert not mask[:600].any()
    assert mask[600:700].all() and not mask[700:].any()


def test_features_have_no_missing_values() -> None:
    match = degrade(toy_match(), np.zeros(3000, dtype=np.float32), seed=1)
    features = match_features(match)
    assert features.shape == (3000, N_FEATURES)
    assert np.isfinite(features).all()


def test_feature_names_match_their_columns() -> None:
    match = toy_match(200)
    features = match_features(match)
    assert len(set(FEATURE_NAMES)) == N_FEATURES == 50
    assert (features[:, FEATURE_NAMES.index("Ball Seen")] == 1).all()
    assert (features[:, FEATURE_NAMES.index("Cutaway")] == 0).all()
    assert np.allclose(features[:, FEATURE_NAMES.index("Home Players Seen")], 1.0)
    assert np.allclose(
        features[:, FEATURE_NAMES.index("Away Centroid Y")], np.nanmean(match.away[:, :, 1], axis=1) / 34.0
    )


def test_degrade_is_repeatable() -> None:
    height = np.zeros(3000, dtype=np.float32)
    first, second = degrade(toy_match(), height, seed=7), degrade(toy_match(), height, seed=7)
    assert np.array_equal(first.ball, second.ball, equal_nan=True)
    assert first.label is not None and second.label is not None
    assert np.array_equal(first.label, second.label)


def test_simulated_ball_has_no_brief_tracks() -> None:
    match = degrade(toy_match(), np.zeros(3000, dtype=np.float32), seed=4)
    lengths = [stop - start for start, stop in runs(~np.isnan(match.ball[:, 0]))]
    assert lengths and min(lengths) >= 15


def test_decoder_tuning_balances_both_groups() -> None:
    coarse = np.repeat(np.array([HOME, AWAY], dtype=np.int8), 500)
    blocks = ((0, 1000),)
    prob = np.full((1000, 3), 0.05)
    prob[np.arange(1000), coarse] = 0.9
    # One group counts brief touches
    prob[100:900:100] = prob[100:900:100][:, [1, 0, 2]]
    fine = prob.argmax(axis=1).astype(np.int8)
    matches = [replace(toy_match(1000), blocks=blocks, label=label) for label in (fine, coarse)]
    best, scores = tune_decoder([[prob], [prob]], [[matches[0]], [matches[1]]])
    alone = [tune_decoder([[prob]], [[match]])[1][0] for match in matches]
    assert alone == [1.0, 1.0]
    assert max(1.0 - score for score in scores) < 1.0
    assert set(best) == {"penalty", "min_spell", "min_dead"}


def test_squads_leave_out_referees() -> None:
    frames = pl.DataFrame({"team": ["club-a", "referee", "club-b", "BALL", "club-b"], "person": list("abcde")})
    assert squads(frames, "club-a", "club-b") == {"home": ["a"], "away": ["c", "e"]}


def test_club_ids_come_from_metadata() -> None:
    assert club_ids('<General HomeTeamId="club-a" GuestTeamId="club-b" />') == ("club-a", "club-b")


def test_mirrored_views_swap_teams() -> None:
    match = toy_match(100)
    views = mirrored(match)
    assert len(views) == 8
    plain, swapped = views[0], views[1]
    assert not plain[1] and swapped[1]
    assert np.array_equal(swapped[0].home, match.away)
    assert np.array_equal(views[-1][0].ball, -match.ball)


def test_upsample_keeps_missing_frames_missing() -> None:
    values = np.array([[0.0], [10.0], [np.nan], [30.0]], dtype=np.float32)
    out = upsample(values, 10)
    assert np.isclose(out[1, 0], 4.0)
    assert np.isnan(out[3:8]).all()
    assert hold(np.array([0, 1, 2, 3]), 10).tolist() == [0, 0, 1, 1, 2, 2, 2, 3, 3, 3]


def test_smooth_ignores_missing_frames() -> None:
    values = np.ones((50, 1), dtype=np.float32) * 10
    values[20:30] = np.nan
    out = smooth(values, sigma=2.0)
    assert np.isnan(out[20:30]).all()
    assert np.allclose(out[~np.isnan(out)], 10.0)
