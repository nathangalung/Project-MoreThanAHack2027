"""Check loaders on small files."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from possession import broadcast, dfl, paths, skillcorner, tracking
from possession.match import AWAY, DEAD, HOME
from possession.tracking import Tracking

META = """<Root><Teams><Team id="a"/><Team id="b"/></Teams>
<Players><Player teamId="a"/><Player teamId="b"/></Players></Root>"""


def write_tracking(folder: Path, frame_ids: tuple[int, ...] = (1, 2)) -> dict[str, Path]:
    meta, raw = folder / "meta.xml", folder / "raw.txt"
    meta.write_text(META)
    raw.write_text("".join(f"{frame}:0.1,0.2,1.0;0.3,0.4,2.0:0.5,NaN,3.0\n" for frame in frame_ids))
    return {"atd_meta": meta, "atd_raw": raw}


def test_asset_finds_exact_and_suffixed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exact, suffixed = tmp_path / "meta.xml", tmp_path / "video-001.mp4"
    exact.write_text("x")
    suffixed.write_text("x")
    listed = {"meta": exact, "video": tmp_path / "video.mp4", "audio": tmp_path / "audio.wav"}
    monkeypatch.setattr(paths.games, "asset_paths", lambda source: listed)
    assert paths.asset("meta") == exact
    assert paths.asset("video") == suffixed
    with pytest.raises(FileNotFoundError):
        paths.asset("audio")


def test_tracking_is_parsed_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = write_tracking(tmp_path)
    monkeypatch.setattr(tracking, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(tracking, "asset", lambda name, source: files[name])
    first = tracking.load_tracking("tactical")
    assert first.n_frames == 2
    assert first.team.tolist() == [HOME, AWAY]
    assert np.allclose(first.xy[0], [[0.1, 0.2], [0.3, 0.4]])
    assert np.isclose(first.ball[0, 0], 0.5) and np.isnan(first.ball[0, 1])
    files["atd_raw"].unlink()
    assert np.array_equal(tracking.load_tracking("tactical").xy, first.xy)


def test_tracking_rejects_malformed_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = write_tracking(tmp_path, frame_ids=(1, 3))
    monkeypatch.setattr(tracking, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(tracking, "asset", lambda name, source: files[name])
    with pytest.raises(ValueError, match="frame ids"):
        tracking.load_tracking()
    with pytest.raises(pl.exceptions.PolarsError):
        tracking.parse_raw(files["atd_raw"], 3)
    files["atd_meta"].write_text("<Root><Team id='a'/></Root>")
    with pytest.raises(ValueError, match="expected 2 teams"):
        tracking.slot_teams(files["atd_meta"])


def shifted_pair(n: int = 600, lag: int = 40) -> tuple[Tracking, Tracking]:
    rng = np.random.default_rng(0)
    path = 0.5 + np.cumsum(rng.normal(0, 0.01, (n + lag, 2)), axis=0).astype(np.float32)
    xy = np.full((n, 2, 2), 0.5, dtype=np.float32)
    team = np.array([HOME, AWAY], dtype=np.int8)
    return Tracking(xy, team, path[lag:]), Tracking(xy, team, path[:n])


def patch_broadcast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tactical: Tracking, other: Tracking) -> None:
    table = tmp_path / "homography.csv"
    valid = np.ones(other.n_frames, dtype=int)
    valid[:50] = 0
    pl.DataFrame({"valid": valid}).write_csv(table)
    monkeypatch.setattr(broadcast, "load_tracking", lambda source: tactical if source == "tactical" else other)
    monkeypatch.setattr(broadcast, "HALF_BLOCKS", ((1, 100, 499),))
    monkeypatch.setattr(broadcast, "MIN_SHARED", 50)
    monkeypatch.setattr(broadcast, "asset", lambda name, source: table)


def test_broadcast_lag_is_recovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tactical, other = shifted_pair()
    patch_broadcast(monkeypatch, tmp_path, tactical, other)
    assert broadcast.half_lags() == [-40]
    match = broadcast.target_match()
    assert match.blocks == ((100, 500),)
    assert match.home.shape == (600, 1, 2) and match.away.shape == (600, 1, 2)
    # Outside the half is cutaway
    assert match.cutaway[:100].all() and not match.cutaway[100:500].any()
    assert np.isnan(match.second[:100]).all()
    assert np.allclose(match.second[100:500], match.ball[100:500], atol=1e-4)


def test_broadcast_alignment_fails_on_unrelated_balls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tactical, other = shifted_pair()
    noise = np.random.default_rng(1).random(other.ball.shape).astype(np.float32)
    patch_broadcast(monkeypatch, tmp_path, tactical, Tracking(other.xy, other.team, noise))
    with pytest.raises(ValueError, match="alignment failed"):
        broadcast.half_lags()
    assert broadcast.ball_error(tactical, other, 0, 10, 0) == float("inf")


def dfl_lines(skip_frame: bool = False) -> str:
    lines = []
    for section, first in (("firstHalf", 10), ("secondHalf", 100)):
        frames = [first, first + 2] if skip_frame else [first, first + 1]
        lines.append(f'<FrameSet GameSection="{section}" TeamId="BALL" PersonId="ball">')
        for number, frame in enumerate(frames):
            lines.append(
                f'<Frame N="{frame}" X="1.0" Y="2.0" Z="0.5" BallPossession="{number + 1}" BallStatus="{number}" />'
            )
        for club, person in (("club-a", "p1"), ("club-b", "p2"), ("referee", "p3")):
            lines.append(f'<FrameSet GameSection="{section}" TeamId="{club}" PersonId="{person}">')
            lines.append(f'<Frame N="{first}" X="3.0" Y="4.0" />')
    return "\n".join(lines)


def patch_dfl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tracking_text: str) -> list[str]:
    calls = []

    def retrieve(url: str, target: Path) -> None:
        calls.append(url)
        is_meta = "matchinformation" in url
        Path(target).write_text('<General HomeTeamId="club-a" GuestTeamId="club-b" />' if is_meta else tracking_text)

    monkeypatch.setattr(dfl, "DFL_DIR", tmp_path / "dfl")
    monkeypatch.setattr(dfl.urllib.request, "urlretrieve", retrieve)
    return calls


def test_dfl_match_is_downloaded_once_and_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_dfl(monkeypatch, tmp_path, dfl_lines())
    match, height = dfl.load_dfl("J03WMX")
    assert match.blocks == ((0, 2), (2, 4))
    assert match.home.shape == (4, 1, 2) and match.away.shape == (4, 1, 2)
    assert np.allclose(match.home[0, 0], [3.0, 4.0]) and np.isnan(match.home[1, 0]).all()
    assert match.label is not None and match.label.tolist() == [DEAD, AWAY, DEAD, AWAY]
    assert np.allclose(height, 0.5) and np.isnan(match.second).all()
    assert len(calls) == 2
    dfl.load_dfl("J03WMX")
    assert len(calls) == 2


def test_dfl_rejects_missing_ball_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_dfl(monkeypatch, tmp_path, dfl_lines(skip_frame=True))
    with pytest.raises(ValueError, match="not contiguous"):
        dfl.build("J03WMX")


def real_files(match_id: str) -> dict[str, str]:
    meta = {
        "home_team": {"id": 1},
        "pitch_length": 105,
        "pitch_width": 68,
        "players": [{"id": 11, "team_id": 1}, {"id": 22, "team_id": 2}],
        "match_periods": [{"start_frame": 0, "end_frame": 2}, {"start_frame": 2, "end_frame": 4}],
    }
    rows = []
    for frame in range(4):
        rows.append(
            {
                "frame": frame,
                "ball_data": {"x": 1.0 * frame, "y": 2.0, "is_detected": frame != 1},
                "image_corners_projection": {"x_top_left": None if frame == 3 else 1.0},
                "player_data": [{"player_id": 11, "x": 5.0, "y": 6.0}, {"player_id": 22, "x": 7.0, "y": 8.0}],
            }
        )
    return {
        f"{match_id}_match.json": json.dumps(meta),
        f"{match_id}_tracking_extrapolated.jsonl": "\n".join(json.dumps(row) for row in rows),
        f"{match_id}_phases_of_play.csv": "frame_start,frame_end,team_in_possession_id\n0,1,1\n2,3,2\n",
    }


def test_real_match_is_downloaded_once_and_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files, calls = real_files("77"), []

    def retrieve(url: str, target: Path) -> None:
        calls.append(url)
        Path(target).write_text(files[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(skillcorner, "REAL_DIR", tmp_path / "real")
    monkeypatch.setattr(skillcorner.urllib.request, "urlretrieve", retrieve)
    match = skillcorner.load_real("77")
    length = 4 * skillcorner.FPS // skillcorner.SOURCE_FPS
    assert match.n_frames == length
    assert match.blocks == ((0, length // 2), (length // 2, length))
    assert np.allclose(match.home[0, 0], [5.0, 6.0]) and np.allclose(match.away[0, 0], [7.0, 8.0])
    assert match.label is not None
    assert match.label[0] == HOME and match.label[-1] == DEAD and AWAY in match.label
    # Undetected ball stays missing
    assert np.isnan(match.ball[2, 0]) and not np.isnan(match.ball[5, 0])
    assert match.cutaway[-1] and not match.cutaway[0]
    assert sum(url.startswith(skillcorner.LARGE_URL) for url in calls) == 1
    skillcorner.load_real("77")
    assert len(calls) == 3
