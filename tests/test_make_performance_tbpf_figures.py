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
    # the stats job (4 nodes) hung in the rain maps; a debug job (2 nodes) timed out; the two steps ran on one interactive node
    data = campaign_json(incident=True)
    c = F.campaign_rows(data["derived"])
    assert [r[0] for r in c["post_rows"]] == ["rainmap", "trackpcp"] and c["post_nodes"] == [1] and c["post_scale"] == pytest.approx(1.0)
    assert c["post_source"] == {"rainmap": "1 node, interactive", "trackpcp": "1 node, interactive"}
    # node-hours of the post row: the steps of the interactive run on 1 node; the hung time and the debug job are 'failed', not drawn
    assert c["totals"]["post_nh"] == pytest.approx((3.0 + 8.6) / 60.0)
    nh = data["derived"]["node_hours"]
    assert nh["failed"] > 10 and c["totals"]["charged_nh"] == pytest.approx(nh["track"] + nh["stats"] + nh["post"])
    paths = F.make_figures(data, str(tmp_path), name="incident")
    assert len(paths) == 3 and all(os.path.getsize(p) > 5000 for p in paths)


def test_a_campaign_with_post_steps_in_the_stats_job_still_says_so():
    c = F.campaign_rows(campaign_json()["derived"])
    assert c["post_nodes"] == [4] and c["post_source"] == {"rainmap": "stats job, 4 nodes", "trackpcp": "stats job, 4 nodes"} and c["post_scale"] == pytest.approx(4.0)
