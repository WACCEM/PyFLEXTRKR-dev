"""
Tests of config/generate_tbpf_mcstracking_scripts.py: the months and their shared seams, the editing of the config text, the run flags of
the three kinds of jobs, the files of a campaign for the E3SM template (job list, dependencies, links and stitching in front of the stats
job, tokens of the slurm scripts, configs that load_config accepts), the checks of the template, and the submission (a stand-in sbatch).
Nothing is submitted.
"""
import json
import os
import re
import stat
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "config"))
import generate_tbpf_mcstracking_scripts as G  # noqa: E402

from pyflextrkr.ft_utilities import load_config  # noqa: E402

TEMPLATE = os.path.join(REPO, "config", "config_mcs_tbpf_e3sm_0p1deg_template.yml")
IMERG_TEMPLATE = os.path.join(REPO, "config", "config_mcs_tbpf_imergv7_global_template.yml")


def _campaign(tmp_path, start="2009-01", end="2009-12", template=TEMPLATE, **kwargs):
    """Write the files of a campaign for the E3SM template (or another one) and return the jobs."""
    kwargs.setdefault("input_dir", str(tmp_path / "input"))
    kwargs.setdefault("root_path", str(tmp_path / "root"))
    kwargs.setdefault("account", "m9999")
    return G.build_campaign(G.month_range(start, end), template, str(tmp_path / "out"), **kwargs)


def _read(path):
    with open(path) as f:
        return f.read()


def test_months_share_the_instant_of_their_seam_and_a_year_ends_at_the_next_january():
    months = G.month_range("2009-11", "2010-02")
    assert [m["tag"] for m in months] == ["200911", "200912", "201001", "201002"]
    assert [(m["start"], m["end"]) for m in months] == [("20091101.0000", "20091201.0000"), ("20091201.0000", "20100101.0000"),
                                                       ("20100101.0000", "20100201.0000"), ("20100201.0000", "20100301.0000")]
    assert all(a["end"] == b["start"] for a, b in zip(months, months[1:]))
    assert [m["name"] for m in months] == ["2009-11", "2009-12", "2010-01", "2010-02"]
    with pytest.raises(SystemExit):
        G.month_range("2009-05", "2009-04")


def test_the_seam_offset_moves_the_end_of_a_month_but_not_the_start_of_the_next():
    months = G.month_range("2009-12", "2010-01", seam_offset_minutes=30)
    assert [(m["start"], m["end"]) for m in months] == [("20091201.0000", "20100101.0030"), ("20100101.0000", "20100201.0030")]
    with pytest.raises(SystemExit):
        G.month_range("2009-01", "2009-02", seam_offset_minutes=-1)


def test_chunks_of_months():
    months = G.month_range("2009-01", "2010-05")
    assert [len(c) for c in G.chunks_of(months, 12)] == [12, 5]
    assert [len(c) for c in G.chunks_of(months, None)] == [17]
    assert [len(c) for c in G.chunks_of(months, 1)] == [1] * 17
    with pytest.raises(SystemExit):
        G.chunks_of(months, 0)


def test_set_key_keeps_the_comment_appends_a_missing_key_and_refuses_a_double_key():
    text = "run_a : True         # parallel\nrun_b: False\nname: x  # a name\n"
    out = G.set_key(text, "run_a", False)
    assert out.splitlines()[0] == "run_a : False         # parallel" and out.splitlines()[1:] == text.splitlines()[1:]
    assert G.set_key(text, "run_b", True).splitlines()[1] == "run_b: True"
    assert G.set_key(text, "name", "a b").splitlines()[2] == "name: 'a b'  # a name"
    assert G.set_key(text, "run_parallel", 2).splitlines()[-1] == "run_parallel: 2"
    assert G.set_key(text, "x", None).splitlines()[-1] == "x: null"
    with pytest.raises(SystemExit):
        G.set_key(text + "run_a: True\n", "run_a", False)


def test_run_flags_of_the_three_kinds_of_jobs_and_the_options_for_the_optional_steps():
    text = "run_dense_netcdf: True\nrun_rainmap: True\nrun_trackpcp: False\nrun_preprocess: True\nrun_remap_healpix: False\n"
    track, stats, month = (G.run_flags(kind, text) for kind in ("track", "stats", "month"))
    assert [s for s, v in track.items() if v] == ["idfeature", "tracksingle", "gettracks", "preprocess"]
    assert [s for s, v in stats.items() if v] == ["trackstats", "identifymcs", "matchpf", "robustmcs", "mapfeature", "speed", "dense_netcdf",
                                                  "rainmap"]
    assert [s for s, v in month.items() if v] == ["idfeature", "tracksingle", "gettracks", "trackstats", "identifymcs", "matchpf",
                                                  "robustmcs", "mapfeature", "speed", "preprocess", "dense_netcdf", "rainmap"]
    # a track job never writes the dense file, and a template without the flag adds none (the run script decides: rows mode with speed)
    assert not track["dense_netcdf"] and not G.run_flags("stats", "")["dense_netcdf"]
    # --rainmap off, --trackpcp on replace the flags of the template; a template without the flag has it off
    on = G.run_flags("stats", text, {"rainmap": False, "trackpcp": True})
    assert not on["rainmap"] and on["trackpcp"]
    assert not G.run_flags("stats", "")["rainmap"]


def test_fill_config_fills_the_placeholders_sets_the_flags_and_adds_no_flag_that_is_off_and_absent():
    text = "run_idfeature: True\nstartdate: 'STARTDATE'\nenddate: 'ENDDATE'\nclouddata_path: INPUT_DIR/\nroot_path: TRACK_DIR/\n" \
           "tracking_path_name: tracking/YEAR-MONTH\n"
    flags = {"idfeature": False, "trackpcp": False, "rainmap": True}
    out = G.fill_config(text, {"start": "20090101.0000", "end": "20090201.0000", "name": "2009-01"}, "/in/", "/out", flags, {"run_parallel": 2})
    config = yaml.safe_load(out)
    assert config["startdate"] == "20090101.0000" and config["enddate"] == "20090201.0000" and config["tracking_path_name"] == "tracking/2009-01"
    assert config["clouddata_path"] == "/in/" and config["root_path"] == "/out/" and config["run_parallel"] == 2
    assert config["run_idfeature"] is False and config["run_rainmap"] is True and "run_trackpcp" not in config


def test_a_campaign_of_a_year_has_a_track_job_per_month_and_one_stats_job_after_all_of_them(tmp_path):
    jobs = _campaign(tmp_path, stats_nodes=4)
    names = [j["name"] for j in jobs]
    assert names == [f"track2009{m:02d}" for m in range(1, 13)] + ["stats200901_200912"]
    assert [j["kind"] for j in jobs] == ["track"] * 12 + ["stats"]
    assert jobs[-1]["needs"] == names[:12] and all(j["needs"] == [] for j in jobs[:12])
    # the files of the jobs and the record of the jobs
    out = tmp_path / "out"
    assert (out / "logs").is_dir() and all(os.path.isfile(j["script"]) and os.path.isfile(j["config"]) for j in jobs)
    assert [j["name"] for j in json.load(open(out / "jobs.json"))] == names
    # a track job: the period of its month, its own folder, only the first three steps, one node
    feb = yaml.safe_load(_read(jobs[1]["config"]))
    assert (feb["startdate"], feb["enddate"], feb["tracking_path_name"]) == ("20090201.0000", "20090301.0000", "tracking/2009-02")
    assert [k for k in feb if k.startswith("run_") and feb[k] is True] == ["run_idfeature", "run_tracksingle", "run_gettracks"]
    assert feb["run_dense_netcdf"] is False
    assert feb["clouddata_path"] == str(tmp_path / "input") + "/" and feb["root_path"] == str(tmp_path / "root") + "/"
    assert feb["run_parallel"] == 1 and feb["stats_format"] == "rows" and feb["robustmcs_definition"] == "mcsmip"
    # the stats job: the whole year, the folder of links, the other steps, the rain maps on and the track accumulation off (the template)
    stats = yaml.safe_load(_read(jobs[-1]["config"]))
    assert (stats["startdate"], stats["enddate"], stats["tracking_path_name"]) == ("20090101.0000", "20100101.0000", "tracking/campaign_200901_200912")
    assert [k for k in stats if k.startswith("run_") and stats[k] is True] == [
        "run_trackstats", "run_identifymcs", "run_matchpf", "run_robustmcs", "run_mapfeature", "run_speed", "run_dense_netcdf", "run_rainmap"]
    assert stats["dense_netcdf_block_tracks"] == 4000 and stats["dense_netcdf_variables"] is None
    assert stats["run_parallel"] == 2 and stats["run_trackpcp"] is False


def test_the_stats_job_links_the_months_and_stitches_their_track_numbers_before_the_steps(tmp_path):
    jobs = _campaign(tmp_path, stats_nodes=4, mail_user="someone@example.org", track_time="01:30:00", stats_time="05:00:00")
    script = _read(jobs[-1]["script"])
    root = str(tmp_path / "root")
    link = [line for line in script.splitlines() if "link_period_folders.py" in line][0]
    stitch = [line for line in script.splitlines() if "stitch_tracknumbers.py" in line][0]
    # the months in time order, then the link folder, in front of the Dask start and the run
    assert link.split()[2] == f"{root}/tracking/campaign_200901_200912" and link.split()[3:5] == [f"{root}/tracking/2009-01/", f"{root}/tracking/2009-02/"]
    assert link.endswith(f"{root}/tracking/2009-12/ || exit 1") and len(link.split()) == 3 + 12 + 3
    # the stitched file: the year, then the 12 monthly files, then the tracking folders of the months (the seam files are compared)
    words = stitch.split()
    assert words[2] == f"{root}/stats/tracknumbers_20090101.0000_20100101.0000.nc"
    assert words[3] == f"{root}/stats/tracknumbers_20090101.0000_20090201.0000.nc" and words[14] == f"{root}/stats/tracknumbers_20091201.0000_20100101.0000.nc"
    assert words[15] == "--tracking-dirs" and words[16] == f"{root}/tracking/2009-01/" and stitch.endswith("|| exit 1")
    assert script.index("link_period_folders.py") < script.index("stitch_tracknumbers.py") < script.index("dask-scheduler") < script.index("run_mcs_tbpf_unified.py")
    # the header: account, nodes, time of the stats job, mail, the job name and its log; the workers
    assert "#SBATCH -A m9999" in script and "#SBATCH --nodes=4" in script and "#SBATCH --time=05:00:00" in script
    assert "#SBATCH --mail-user=someone@example.org" in script and "#SBATCH --output=logs/log_stats200901_200912_%j.log" in script
    assert "workers_per_node=64" in script and "--cpus-per-task=2" in script and "--memory-limit=7GB" in script
    assert "#SBATCH --time=01:30:00" in _read(jobs[0]["script"]) and "--mail-user" in _read(jobs[0]["script"])


def _strings(value):
    """All the strings in a loaded YAML value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def test_the_stitch_command_stops_the_job_when_a_boundary_has_no_shared_file(tmp_path):
    jobs = _campaign(tmp_path, stats_nodes=1)
    stitch = [line for line in _read(jobs[-1]["script"]).splitlines() if "stitch_tracknumbers.py" in line][0]
    assert stitch.endswith("--require-seams || exit 1")
    # with a seam offset the names of the monthly track numbers files carry it, and the stitched file runs to the last end
    periods = G.month_range("2009-01", "2009-03", seam_offset_minutes=30)
    jobs = G.build_campaign(periods, TEMPLATE, str(tmp_path / "out2"), str(tmp_path / "in"), str(tmp_path / "root2"), account="a")
    stitch = [line for line in _read(jobs[-1]["script"]).splitlines() if "stitch_tracknumbers.py" in line][0].split()
    assert stitch[2].endswith("tracknumbers_20090101.0000_20090401.0030.nc") and stitch[3].endswith("tracknumbers_20090101.0000_20090201.0030.nc")
    assert yaml.safe_load(_read(jobs[0]["config"]))["enddate"] == "20090201.0030"
    assert yaml.safe_load(_read(jobs[-1]["config"]))["enddate"] == "20090401.0030"


def test_the_e3sm_template_has_the_types_of_the_reference_config():
    # YAML reads 3.6e6 (no sign in the exponent) as a string: the values the code multiplies or compares must be numbers like the reference's
    template, reference = (yaml.safe_load(_read(p)) for p in (TEMPLATE, os.path.join(REPO, "config", "config_mcs_tbpf_imergv7_mcsmip.yml")))
    assert isinstance(template["pcp_convert_factor"], float) and template["pcp_convert_factor"] == 3.6e6
    different = {k: (reference[k], v) for k, v in template.items() if k in reference and isinstance(reference[k], (int, float))
                 and not isinstance(v, (int, float))}
    assert different == {}
    assert all(isinstance(template[k], bool) for k in ("olr2tb", "run_rainmap", "run_trackpcp", "verify_step_outputs", "skip_completed_steps"))


def test_the_imerg_template_is_the_reference_config_with_the_placeholders_and_the_row_tables():
    # every key that the reference config (IMERGv7, MCSMIP) has has the same value, except the paths and dates the generator fills, the folder of
    # the month and the dense trackstats file (not needed: the MCS steps read the sparse file)
    template, reference = (yaml.safe_load(_read(p)) for p in (IMERG_TEMPLATE, os.path.join(REPO, "config", "config_mcs_tbpf_imergv7_mcsmip.yml")))
    different = {k for k in set(template) & set(reference) if template[k] != reference[k]}
    assert different == {"clouddata_path", "root_path", "startdate", "enddate", "tracking_path_name", "trackstats_dense_netcdf"}
    # what the template adds: the definition (the generator refuses without it), the row tables, the checks and the optional steps
    assert template["robustmcs_definition"] == "mcsmip" and template["stats_format"] == "rows" and template["clouddata_subdir_format"] == "%Y"
    assert template["run_rainmap"] is True and template["run_trackpcp"] is False and template["trackpcp_data_path"] is None
    assert template["idclouds_minute"] == 0 and template["clouddatasource"] == "gpmirimerg" and template["olr2tb"] is False


def test_the_imerg_and_the_e3sm_templates_differ_only_in_the_data_keys():
    # so that a campaign of each differs only in the data
    imerg, e3sm = (yaml.safe_load(_read(p)) for p in (IMERG_TEMPLATE, TEMPLATE))
    assert {k for k in set(imerg) | set(e3sm) if imerg.get(k) != e3sm.get(k)} == {
        "databasename", "olr2tb", "olr_varname", "tb_varname", "pcp_varname", "pcp_convert_factor", "clouddatasource", "idclouds_hourly",
        "idclouds_minute", "idclouds_dt_thresh", "calendar"}
    # the model has no 29 February, the observations have one
    assert (e3sm["calendar"], imerg["calendar"]) == ("noleap", "standard")


def test_an_imerg_campaign_has_the_same_jobs_without_a_seam_offset_and_its_configs_load(tmp_path):
    jobs = _campaign(tmp_path, start="2019-01", end="2019-12", template=IMERG_TEMPLATE, stats_nodes=4, profile_dir=str(tmp_path / "profile"))
    assert [j["name"] for j in jobs] == [f"track2019{m:02d}" for m in range(1, 13)] + ["stats201901_201912"]
    dec = yaml.safe_load(_read(jobs[11]["config"]))
    assert (dec["startdate"], dec["enddate"], dec["databasename"], dec["clouddatasource"]) == ("20191201.0000", "20200101.0000", "merg_", "gpmirimerg")
    for job in jobs:
        assert not re.search(r"@[A-Z_]+@", _read(job["script"])), job["script"]
        config = load_config(job["config"], make_dirs=False)
        assert config["feature_type"] == "tb_pf" and config["tb_varname"] == "Tb" and config["idclouds_hourly"] == 1


def test_no_placeholder_is_left_in_any_file_and_the_configs_load(tmp_path):
    jobs = _campaign(tmp_path, stats_nodes=2)
    for job in jobs:
        # the slurm script has no @TOKEN@ left, the config no placeholder in any value (the comments of the template name them)
        assert not re.search(r"@[A-Z_]+@", _read(job["script"])), job["script"]
        values = " ".join(_strings(yaml.safe_load(_read(job["config"]))))
        assert not re.search(r"STARTDATE|ENDDATE|YEAR-MONTH|INPUT_DIR|TRACK_DIR", values), job["config"]
        # load_config reads the config the way the run script does (the output folders are not made)
        config = load_config(job["config"], make_dirs=False)
        assert config["feature_type"] == "tb_pf" and config["tracking_outpath"].startswith(str(tmp_path / "root"))
    assert not os.path.exists(tmp_path / "root")


def test_one_node_stats_job_and_a_chunk_of_one_month_use_the_single_node_template(tmp_path):
    jobs = _campaign(tmp_path, start="2009-03", end="2009-05")                       # three months, one chunk, --stats-nodes 1
    stats = _read(jobs[-1]["script"])
    assert "dask-scheduler" not in stats and "link_period_folders.py" in stats and "#SBATCH --nodes=1" in stats
    assert yaml.safe_load(_read(jobs[-1]["config"]))["run_parallel"] == 1
    # a chunk of one month is the plain job: all the steps in one job, no links
    plain = _campaign(tmp_path / "plain", start="2009-03", end="2009-04", chunk_size=1)
    assert [(j["name"], j["kind"], j["needs"]) for j in plain] == [("month200903", "month", []), ("month200904", "month", [])]
    flags = yaml.safe_load(_read(plain[0]["config"]))
    assert [k for k in flags if k.startswith("run_") and flags[k] is True] == [
        "run_idfeature", "run_tracksingle", "run_gettracks", "run_trackstats", "run_identifymcs", "run_matchpf", "run_robustmcs",
        "run_mapfeature", "run_speed", "run_dense_netcdf", "run_rainmap"]
    assert "link_period_folders" not in _read(plain[0]["script"])


def test_the_options_replace_the_optional_steps_of_the_template(tmp_path):
    jobs = _campaign(tmp_path, start="2009-01", end="2009-02", overrides={"rainmap": False, "trackpcp": True})
    stats = yaml.safe_load(_read(jobs[-1]["config"]))
    assert stats["run_rainmap"] is False and stats["run_trackpcp"] is True and stats["trackpcp_steps"] == [2]
    assert stats["trackpcp_data_path"] is None and stats["trackpcp_time_match"] == "nearest" and stats["trackpcp_class_varname"] is None


def test_the_template_is_checked_before_a_file_is_written(tmp_path):
    text = _read(TEMPLATE)
    bad = tmp_path / "bad.yml"
    cases = [(text.replace("startdate: 'STARTDATE'", "startdate: '20090101.0000'"), "needs the placeholder STARTDATE in startdate"),
             (text.replace("robustmcs_definition: mcsmip", ""), "robustmcs_definition must be one of"),
             (text.replace("mcs_min_rainvol_thresh: 20000", ""), "needs the config keys"),
             (text.replace("robustmcs_definition: mcsmip", "robustmcs_definition: saag"), "robustmcs_definition must be one of")]
    for broken, message in cases:
        bad.write_text(broken)
        with pytest.raises(SystemExit) as stop:
            G.build_campaign(G.month_range("2009-01", "2009-02"), str(bad), str(tmp_path / "out2"), "/in", "/out", account="a")
        assert message in str(stop.value)
    assert not (tmp_path / "out2").exists()
    # with the robust step off the definition is not needed
    bad.write_text(text.replace("robustmcs_definition: mcsmip", "").replace("run_robustmcs : True", "run_robustmcs : False"))
    G.check_template(bad.read_text())


def test_submit_waits_for_the_jobs_it_needs_and_a_listed_job_alone_takes_the_others_as_finished(tmp_path):
    jobs = _campaign(tmp_path, start="2009-01", end="2009-03")
    # a stand-in for sbatch: it prints a job id (a counter) and records its arguments
    record, counter = tmp_path / "calls.txt", tmp_path / "counter"
    fake = tmp_path / "fake_sbatch"
    fake.write_text(f'#!/bin/bash\necho "$@" >> {record}\nn=$(cat {counter} 2>/dev/null || echo 100); n=$((n+1)); echo $n > {counter}; echo $n\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    ids = G.submit(jobs, str(tmp_path / "out"), sbatch=str(fake))
    assert ids == {"track200901": "101", "track200902": "102", "track200903": "103", "stats200901_200903": "104"}
    calls = record.read_text().splitlines()
    assert "--dependency" not in calls[0] and calls[3].startswith("--parsable --dependency=afterok:101:102:103 --kill-on-invalid-dep=yes")
    # only a track job again and the stats job: the other track jobs are taken as finished, so the stats job waits for the one that was submitted
    record.write_text("")
    again = G.submit(jobs, str(tmp_path / "out"), names=["track200902", "stats200901_200903"], sbatch=str(fake))
    assert list(again) == ["track200902", "stats200901_200903"]
    assert record.read_text().splitlines()[1].startswith(f"--parsable --dependency=afterok:{again['track200902']} --kill-on-invalid-dep=yes")
    with pytest.raises(SystemExit, match="no such job"):
        G.submit(jobs, str(tmp_path / "out"), names=["track209901"], sbatch=str(fake))


def test_submit_records_every_job_with_its_id_and_the_ids_it_waits_for(tmp_path):
    jobs = _campaign(tmp_path, start="2009-01", end="2009-03")
    fake = tmp_path / "fake_sbatch"
    fake.write_text('#!/bin/bash\nn=$(cat %s 2>/dev/null || echo 200); n=$((n+1)); echo $n > %s; echo "$n;cluster"\n' % (tmp_path / "c", tmp_path / "c"))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    G.submit(jobs, str(tmp_path / "out"), sbatch=str(fake))
    lines = [json.loads(line) for line in _read(tmp_path / "out" / "submissions.jsonl").splitlines()]
    assert [(r["name"], r["job_id"]) for r in lines] == [("track200901", "201"), ("track200902", "202"), ("track200903", "203"), ("stats200901_200903", "204")]
    assert lines[-1]["after"] == ["track200901", "track200902", "track200903"] and lines[-1]["after_ids"] == ["201", "202", "203"]
    assert lines[0]["after_ids"] == [] and lines[0]["script"] == jobs[0]["script"] and lines[0]["config"] == jobs[0]["config"]
    # a second submission of one job appends (the first record stays: the job ids of every attempt are kept)
    G.submit(jobs, str(tmp_path / "out"), names=["track200902"], sbatch=str(fake))
    again = [json.loads(line) for line in _read(tmp_path / "out" / "submissions.jsonl").splitlines()]
    assert len(again) == 5 and again[-1]["name"] == "track200902" and again[-1]["job_id"] == "205"


def test_every_job_gets_a_ledger_folder_of_its_own_with_profile_dir(tmp_path):
    jobs = _campaign(tmp_path, start="2009-01", end="2009-03", profile_dir=str(tmp_path / "profile"))
    folders = [yaml.safe_load(_read(j["config"]))["step_profile_dir"] for j in jobs]
    assert folders == [str(tmp_path / "profile" / j["name"]) for j in jobs] and len(set(folders)) == 4
    # without the option the key of the template stays (off)
    plain = _campaign(tmp_path / "plain", start="2009-01", end="2009-03")
    assert all(yaml.safe_load(_read(j["config"])).get("step_profile_dir") is None for j in plain)


def test_pin_code_makes_the_jobs_import_the_checkout_of_the_generator(tmp_path):
    pinned = _campaign(tmp_path, start="2009-01", end="2009-03", stats_nodes=2, pin_code=True)
    line = f"export PYTHONPATH={REPO}${{PYTHONPATH:+:$PYTHONPATH}}"
    for job in pinned:
        script = _read(job["script"])
        # after the environment is activated and before anything runs (the links, the stitching, the Dask start, the run script)
        assert line in script and script.index("source activate") < script.index(line) < script.index("python ")
    plain = _campaign(tmp_path / "plain", start="2009-01", end="2009-03", stats_nodes=2)
    assert all("PYTHONPATH" not in _read(j["script"]) for j in plain)


def test_the_logs_carry_the_job_id_and_a_dead_worker_does_not_kill_the_others(tmp_path):
    jobs = _campaign(tmp_path, start="2009-01", end="2009-03", stats_nodes=2)
    assert all(f"#SBATCH --output=logs/log_{j['name']}_%j.log" in _read(j["script"]) for j in jobs)
    stats = _read(jobs[-1]["script"])
    assert "srun --kill-on-bad-exit=0 -N $SLURM_NNODES" in stats and "dask_scheduler_${SLURM_JOB_ID}.log" in stats


def test_periods_of_any_length_make_a_campaign(tmp_path):
    # the internals take a list of periods: three days, tracked as three jobs and stitched (the stitching test of the demos uses this)
    days = [{"tag": f"2025082{d}", "name": f"2025-08-2{d}", "start": f"2025082{d}.0000", "end": f"2025082{d + 1}.0000"} for d in (2, 3, 4)]
    jobs = G.build_campaign(days, TEMPLATE, str(tmp_path / "out"), str(tmp_path / "in"), str(tmp_path / "root"), account="a")
    assert [j["name"] for j in jobs] == ["track20250822", "track20250823", "track20250824", "stats20250822_20250824"]
    stitch = [line for line in _read(jobs[-1]["script"]).splitlines() if "stitch_tracknumbers.py" in line][0]
    assert "tracknumbers_20250822.0000_20250825.0000.nc" in stitch and "tracknumbers_20250823.0000_20250824.0000.nc" in stitch
