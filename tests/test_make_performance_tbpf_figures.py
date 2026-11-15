"""
Tests of Analysis/make_performance_tbpf_figures.py on synthetic campaigns (tests/synthetic_campaign.py): the three figures render for a campaign with
and without the post-processing steps and for a one-node stats job, as 24-bit PNG files of the size of the figure, with no text outside it; the numbers
the figures draw add up to the node-hours charged; the JSON is checked. Needs matplotlib and PIL (not in the CI environment: skipped there).
"""
import os
import sys

import pytest

pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Analysis"))
import make_performance_tbpf_figures as F  # noqa: E402

from synthetic_campaign import campaign_json  # noqa: E402


@pytest.fixture(autouse=True)
def strict(monkeypatch):
    """A text that sticks out of a figure is an error in these tests."""
    monkeypatch.setattr(F, "STRICT", True)


def test_the_three_figures_render_as_rgb_png_files_of_the_size_of_the_figure(tmp_path):
    paths = F.make_figures(campaign_json(), str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["perf_tbpf_synthetic_2009_campaign.png", "perf_tbpf_synthetic_2009_timeline.png",
                                                     "perf_tbpf_synthetic_2009_steps.png"]
    sizes = {os.path.basename(p).split("_")[-1]: Image.open(p) for p in paths}
    assert all(im.mode == "RGB" for im in sizes.values())
    # the width of a figure in inches x 100 dpi (the campaign figure is 14 inches wide, the other two 11.5 and 11.8)
    assert [sizes[k].size[0] for k in ("campaign.png", "timeline.png", "steps.png")] == [1400, 1150, 1180]
    assert all(im.size[1] > 300 for im in sizes.values())


@pytest.mark.parametrize("kwargs", [dict(post=False), dict(stats_nodes=1), dict(n_jobs=3, failed_node_hours=1.4), dict(n_jobs=1)])
def test_other_campaigns_render_without_text_outside_the_figure(tmp_path, kwargs):
    # no post-processing: two rows; one node: the stats row is as long as the wall-clock; a few months; a failed attempt in the subtitle; one month
    paths = F.make_figures(campaign_json(**kwargs), str(tmp_path), name="x")
    assert len(paths) == 3 and all(os.path.getsize(p) > 5000 for p in paths)


def test_the_figure_options_choose_the_figures_and_the_name(tmp_path):
    paths = F.make_figures(campaign_json(), str(tmp_path / "out"), name="imerg_2019", only=["steps"])
    assert [os.path.basename(p) for p in paths] == ["perf_tbpf_imerg_2019_steps.png"]


def test_the_nodes_hours_that_are_drawn_add_up_to_the_node_hours_charged():
    data = campaign_json()
    c = F.campaign_rows(data["derived"])
    drawn = c["totals"]["track_nh"] + c["totals"]["stats_nh"] + c["totals"]["post_nh"]
    # the rest is the start and shutdown of the jobs and the checks between the steps: positive and small
    assert c["totals"]["charged_nh"] == pytest.approx(data["derived"]["node_hours"]["track"] + data["derived"]["node_hours"]["stats"])
    assert 0 < c["other_nh"] < 0.1 * c["totals"]["charged_nh"] and c["totals"]["charged_nh"] == pytest.approx(drawn + c["other_nh"])
    # rows: the track steps (x the jobs), the stats job (setup, the steps up to speed), the post-processing
    assert [r[0] for r in c["stats_rows"]] == ["setup", "trackstats", "identifymcs", "matchpf", "robustmcs", "mapfeature", "speed"]
    assert [r[0] for r in c["post_rows"]] == ["rainmap", "trackpcp"]
    assert c["track"]["idfeature"][0] == pytest.approx(sum(j["steps_min"]["idfeature"] for j in data["derived"]["track_jobs"]) / 60.0)
    # four nodes: a stats step of 14.2 minutes costs 14.2 x 4 / 60 node-hours
    assert dict((r[0], r[1]) for r in c["stats_rows"])["trackstats"] == pytest.approx(14.2 * 4 / 60.0)
    # the campaign without the post-processing has no third row
    assert F.campaign_rows(campaign_json(post=False)["derived"])["post_rows"] == []


def test_the_json_is_checked_before_a_figure_is_made(tmp_path):
    data = campaign_json()
    F.check_schema(data)
    del data["derived"]["stats_job"]["setup_min"]
    data["schema"] = "other/1"
    with pytest.raises(ValueError, match=r"derived.stats_job.setup_min") as err:
        F.make_figures(data, str(tmp_path))
    assert "schema == tbpf_campaign_performance/1" in str(err.value)
    assert os.listdir(tmp_path) == []


def test_a_text_outside_the_figure_is_found():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3, 2))
    fig.text(0.5, 0.5, "inside")
    fig.text(0.95, 0.5, "this text is much too long for the little figure")
    assert F.texts_outside(fig) == ["this text is much too long for the little figure"]
    plt.close(fig)


def test_small_helpers():
    assert F.nice_ticks(10.4) == [0.0, 2.5, 5.0, 7.5, 10.0] and F.nice_ticks(0.0) == [0.0]
    assert max(F.nice_ticks(41.7)) <= 41.7
    assert F.slug("E3SM ne120 2009") == "e3sm_ne120_2009" and F.slug("IMERG V07B (2019)") == "imerg_v07b_2019"
    assert F.counts_text(campaign_json()) == "8,760 frames, 1.18 million tracks, 41,000 Tb MCS, 7,600 robust MCS" and F.counts_text({}) == ""
    assert F.text_color("#2a78d6") == "#ffffff" and F.text_color("#eda100") == "#0b0b0b"


def test_a_campaign_with_a_hung_stats_job_and_post_jobs_of_their_own_renders_and_adds_up(tmp_path):
    # the stats job (4 nodes) hung in the rain maps; a debug job (2 nodes) timed out; the two steps ran on one interactive node and again on 4 nodes: the
    # last run supplies the numbers, the 1-node run is superseded (charged, not drawn)
    data = campaign_json(incident=True)
    c = F.campaign_rows(data["derived"])
    assert [r[0] for r in c["post_rows"]] == ["rainmap", "trackpcp"] and c["post_nodes"] == [4] and c["post_scale"] == pytest.approx(4.0)
    assert c["post_source"] == {"rainmap": "post job, 4 nodes", "trackpcp": "post job, 4 nodes"} and c["drawn_post_jobs"] == ["post_tail4"]
    # node-hours of the post row: the steps of the 4-node job; the 1-node run is superseded; the hung time and the debug job are 'failed', not drawn
    assert c["totals"]["post_nh"] == pytest.approx((1.5 + 11.3) * 4 / 60.0)
    assert c["superseded_nh"] == pytest.approx((3.0 + 8.6) / 60.0)
    nh = data["derived"]["node_hours"]
    assert nh["failed"] > 10 and c["totals"]["charged_nh"] == pytest.approx(nh["track"] + nh["stats"] + nh["post"])
    assert 0 < c["other_nh"] < 0.2 * c["totals"]["charged_nh"]
    paths = F.make_figures(data, str(tmp_path), name="incident")
    assert len(paths) == 3 and all(os.path.getsize(p) > 5000 for p in paths)


def test_post_steps_that_a_post_job_ran_again_replace_those_of_the_stats_job():
    data = campaign_json(rerun=True)
    c = F.campaign_rows(data["derived"])
    stats = data["derived"]["stats_job"]
    # the stats job's own rainmap and trackpcp (5.2 and 14.8 min on 4 nodes) are superseded by the 4-node job's
    assert dict((r[0], r[2]) for r in c["post_rows"]) == {"rainmap": 1.6, "trackpcp": 20.3}
    assert c["post_source"] == {"rainmap": "post job, 4 nodes", "trackpcp": "post job, 4 nodes"}
    assert c["superseded_nh"] == pytest.approx((5.2 + 14.8) * 4 / 60.0)
    # the node-hours charged are the stats job's (all its steps) and the post job's; what is drawn plus what is superseded plus the rest is that
    assert c["totals"]["charged_nh"] == pytest.approx(data["derived"]["node_hours"]["track"] + data["derived"]["node_hours"]["stats"] + data["derived"]["node_hours"]["post"])
    assert 0 < c["other_nh"] < 0.1 * c["totals"]["charged_nh"]
    # compute time: the longest track job + the stats job without the superseded steps + the post job (queue waits are not in it)
    longest = max(j["elapsed_min"] for j in data["derived"]["track_jobs"])
    stats_min = stats["setup_min"] + sum(v for k, v in stats["steps_min"].items() if k not in ("rainmap", "trackpcp")) + stats["other_min"]
    assert c["track_longest_min"] == pytest.approx(longest) and c["stats_compute_min"] == pytest.approx(stats_min)
    assert c["post_compute_min"] == pytest.approx(24.0) and c["compute_min"] == pytest.approx(longest + stats_min + 24.0)


def test_without_post_jobs_the_compute_time_is_the_longest_track_job_plus_the_stats_job():
    data = campaign_json()
    c = F.campaign_rows(data["derived"])
    longest = max(j["elapsed_min"] for j in data["derived"]["track_jobs"])
    assert c["superseded_nh"] == 0 and c["drawn_post_jobs"] == [] and c["post_compute_min"] == 0
    assert c["compute_min"] == pytest.approx(longest + data["derived"]["stats_job"]["elapsed_min"])


@pytest.mark.parametrize("kwargs", [dict(), dict(incident=True), dict(rerun=True), dict(post=False), dict(stats_nodes=1)])
def test_the_timeline_has_no_queue_wait_the_stats_job_starts_when_the_longest_track_job_ends(kwargs):
    data = campaign_json(**kwargs)
    c = F.campaign_rows(data["derived"])
    lay = F.timeline_layout(c)
    longest = max(j["elapsed_min"] for j in data["derived"]["track_jobs"]) / 60.0
    # every track job starts at 0 and has the length of its elapsed time; the stats row starts at the end of the longest one
    assert all(row["segments"][0][1] == 0.0 for row in lay["track_rows"])
    assert [round(sum(seg[2] for seg in row["segments"]), 6) for row in lay["track_rows"]] == [round(j["elapsed_min"] / 60.0, 6) for j in data["derived"]["track_jobs"]]
    assert lay["stats_start_h"] == pytest.approx(longest) and lay["stats_segments"][0][1] == pytest.approx(longest)
    # no gap in the stats row, and its length is the compute time of the stats job and the post jobs that supply steps
    ends = [seg[1] + seg[2] for seg in lay["stats_segments"]]
    assert all(a == pytest.approx(b[1]) for a, b in zip(ends, lay["stats_segments"][1:]))
    assert lay["end_h"] - longest == pytest.approx((c["stats_compute_min"] + c["post_compute_min"]) / 60.0)
    # a cancelled stats job's hung time is not on the axis
    if kwargs.get("incident"):
        assert lay["end_h"] < longest + 2.0 and lay["post_span"] is not None


def test_a_campaign_with_post_steps_in_the_stats_job_still_says_so():
    c = F.campaign_rows(campaign_json()["derived"])
    assert c["post_nodes"] == [4] and c["post_source"] == {"rainmap": "stats job, 4 nodes", "trackpcp": "stats job, 4 nodes"} and c["post_scale"] == pytest.approx(4.0)


def test_the_figures_leave_the_matplotlib_settings_of_the_session_alone(tmp_path):
    # the look of the figures was set for the whole session: the pictures of a later test of the same session (test_viz_frames) changed
    import matplotlib.pyplot as plt
    before = dict(plt.rcParams)
    F.make_figures(campaign_json(n_jobs=2), str(tmp_path), only=["steps"])
    assert dict(plt.rcParams) == before
