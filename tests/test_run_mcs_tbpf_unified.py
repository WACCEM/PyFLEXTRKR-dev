"""
Tests of the wiring of runscripts/run_mcs_tbpf_unified.py: the start-up checks of the config (feature type, stats_format and the robust
MCS definition, which is required and has no default), the functions of every step for the two statistics formats and the two
definitions, the optional steps, a sys.exit() without a code becoming exit code 1, the order of the steps in run_step, and main.
The steps themselves are replaced by recording functions: what the steps calculate is tested in the tests of the modules.
"""
import importlib.util
import os
import sys
import types

import pytest

pytest.importorskip("distributed")

from pyflextrkr import robustmcspf, robustmcspf_saag  # noqa: E402
from pyflextrkr.identifymcs import identifymcs_tb  # noqa: E402
from pyflextrkr.matchtbpf_driver import match_tbpf_tracks  # noqa: E402
from synthetic_tbpf import tb_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    """The run script as a module (it is a script in runscripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("run_mcs_tbpf_unified", os.path.join(REPO, "runscripts", "run_mcs_tbpf_unified.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unified = _load_script()


def _config(**extra):
    """A Tb+PF config that passes the start-up checks (the MCSMIP definition, dense statistics)."""
    return tb_config("s/", **{"feature_type": "tb_pf", "run_robustmcs": True, "robustmcs_definition": "mcsmip", **extra})


def _exit_message(config):
    with pytest.raises(SystemExit) as stop:
        unified.check_config(config)
    return str(stop.value)


def test_a_good_config_passes_and_the_run_script_is_for_tbpf_only():
    unified.check_config(_config())
    unified.check_config(_config(stats_format="rows", robustmcs_definition="feng21"))
    assert "Tb+PF tracking" in _exit_message(dict(_config(), feature_type="radar_feature"))
    assert "run_mcs_radar3d.py" in _exit_message(dict(_config(), feature_type="radar_feature"))
    assert "stats_format must be 'dense' or 'rows', not 'row'" in _exit_message(_config(stats_format="row"))


@pytest.mark.parametrize("value", [None, "saag", "MCSMIP"])
def test_the_robust_definition_is_required_and_the_message_explains_both_options(value):
    config = _config()
    config.pop("robustmcs_definition")
    if value is not None:
        config["robustmcs_definition"] = value
    message = _exit_message(config)
    # both options, the old scripts each one replaces and the keys each one needs
    assert "robustmcs_definition must be one of ('feng21', 'mcsmip')" in message
    for text in ("mcsmip", "feng21", "run_mcs_tbpf_mcsmip.py", "run_mcs_tbpf_saag.py", "run_mcs_tbpf.py", "run_mcs_tbpf_wrf.py",
                 "run_mcs_tbpf_gpm.py", "mcs_min_rainvol_thresh", "coefs_pf_area"):
        assert text in message, text


def test_the_keys_of_the_chosen_definition_are_checked():
    config = {k: v for k, v in _config().items() if k != "mcs_min_rainvol_thresh"}
    assert "needs the config keys ['mcs_min_rainvol_thresh']" in _exit_message(config)
    config = {k: v for k, v in _config(robustmcs_definition="feng21").items() if k != "coefs_pf_skew"}
    assert "needs the config keys ['coefs_pf_skew']" in _exit_message(config)


def test_a_run_without_the_robust_step_needs_no_definition_but_a_wrong_one_is_refused():
    config = _config(run_robustmcs=False)
    config.pop("robustmcs_definition")
    unified.check_config(config)
    assert "must be one of" in _exit_message(dict(config, robustmcs_definition="saag"))


def test_the_functions_of_the_steps_for_both_formats_and_both_definitions():
    names = lambda steps: [s[0] for s in steps]
    base = ["idfeature", "tracksingle", "gettracks", "trackstats", "identifymcs", "matchpf", "robustmcs", "mapfeature", "speed"]
    # dense statistics: the old modules; the robust function follows the definition
    mcsmip = unified.build_steps(_config(mcsrobust_filebase="mcs_tracks_robust_"))
    assert names(mcsmip) == base
    funcs = {name: func.__wrapped__ for name, func, _ in mcsmip}
    assert funcs["identifymcs"] is identifymcs_tb and funcs["matchpf"] is match_tbpf_tracks
    assert funcs["robustmcs"] is robustmcspf_saag.define_robust_mcs_pf
    feng21 = unified.build_steps(_config(robustmcs_definition="feng21", mcsrobust_filebase="mcs_tracks_robust_"))
    assert {n: f.__wrapped__ for n, f, _ in feng21}["robustmcs"] is robustmcspf.define_robust_mcs_pf
    # rows statistics: the row modules (any definition: robustmcs_rows reads the key itself)
    from pyflextrkr.identifymcs_tb_rows import identifymcs_tb_rows
    from pyflextrkr.mapfeature_rows import mapfeature_driver_rows
    from pyflextrkr.matchtbpf_rows import match_tbpf_tracks_rows
    from pyflextrkr.movement_speed_rows import movement_speed_rows
    from pyflextrkr.robustmcs_rows import define_robust_mcs_pf_rows
    rows = {n: (f.__wrapped__, k) for n, f, k in unified.build_steps(_config(stats_format="rows", mcsrobust_filebase="r_"))}
    assert rows["identifymcs"][0] is identifymcs_tb_rows and rows["matchpf"][0] is match_tbpf_tracks_rows
    assert rows["robustmcs"][0] is define_robust_mcs_pf_rows and rows["mapfeature"][0] is mapfeature_driver_rows
    assert rows["speed"][0] is movement_speed_rows
    # the pixel-level files are made from the robust statistics
    assert rows["mapfeature"][1] == {"trackstats_filebase": "r_"}


def test_the_optional_steps_are_in_the_list_only_when_their_flag_is_on(monkeypatch):
    # rainmap has its own module (imported only when the flag is on): a stand-in module is enough here
    fake = types.ModuleType("pyflextrkr.rainmap_driver")
    fake.rainmap_driver = lambda config: None
    monkeypatch.setitem(sys.modules, "pyflextrkr.rainmap_driver", fake)
    config = _config(mcsrobust_filebase="r_")
    assert [s[0] for s in unified.build_steps(config)][-1] == "speed"
    on = unified.build_steps(dict(config, run_rainmap=True, run_trackpcp=True, run_remap_healpix=True, run_regrid_mask=True))
    assert [s[0] for s in on][-4:] == ["rainmap", "trackpcp", "remap_healpix", "regrid_mask"]


def test_a_silent_exit_of_a_step_becomes_exit_code_1_and_other_exits_are_kept(caplog):
    def make(code):
        def step(config, extra=None):
            if code == "return":
                return ("ok", extra)
            sys.exit(code)
        return unified.fail_on_silent_exit("identifymcs", step)

    assert make("return")({}, extra=3) == ("ok", 3)                           # the return value and the arguments pass
    for silent in (None, 0):
        with caplog.at_level("CRITICAL"), pytest.raises(SystemExit) as stop:
            make(silent)({})
        assert stop.value.code == 1 and any("Step identifymcs ended the run" in r.message for r in caplog.records)
    with pytest.raises(SystemExit) as stop:
        make("No robust MCS found!")({})
    assert stop.value.code == "No robust MCS found!"                          # a message is an error already
    with pytest.raises(SystemExit) as stop:
        make(3)({})
    assert stop.value.code == 3
    assert make("return").__name__ == "step" and make("return").__wrapped__ is not None


def test_run_steps_runs_the_steps_in_order_when_their_flag_is_on_and_stops_before_mapfeature_without_the_robust_statistics(tmp_path):
    calls = []
    record = lambda name: (name, lambda config, **kwargs: calls.append((name, kwargs)), {})
    steps = [record(n) for n in ("idfeature", "tracksingle", "gettracks", "trackstats", "identifymcs", "matchpf", "robustmcs", "speed")]
    steps.insert(7, ("mapfeature", lambda config, **kwargs: calls.append(("mapfeature", kwargs)), {"trackstats_filebase": "robust_"}))
    flags = {f"run_{n}": True for n in ("idfeature", "gettracks", "matchpf", "mapfeature", "speed")}
    config = dict(tb_config(str(tmp_path) + "/"), feature_type="tb_pf", mcsrobust_filebase="robust_", verify_step_outputs=False, **flags)
    # the robust statistics are not there: mapfeature does not run, the run stops with exit code 1
    with pytest.raises(SystemExit) as stop:
        unified.run_steps(config, None, steps, {"stale": False})
    assert stop.value.code == 1 and [c[0] for c in calls] == ["idfeature", "gettracks", "matchpf"]
    # with them the steps that are on run in the order of the list, with the keyword arguments of the list
    calls.clear()
    open(unified.stats_output(config, "robust_"), "w").close()
    unified.run_steps(config, None, steps, {"stale": False})
    assert [c[0] for c in calls] == ["idfeature", "gettracks", "matchpf", "mapfeature", "speed"]
    assert dict(calls)["mapfeature"] == {"trackstats_filebase": "robust_"}


def test_main_runs_the_preprocess_before_the_client_and_closes_the_client(monkeypatch, tmp_path):
    order = []
    config = dict(tb_config(str(tmp_path) + "/"), feature_type="tb_pf", run_parallel=1, verify_step_outputs=False, run_preprocess=True,
                  run_idfeature=True, mcsrobust_filebase="r_")
    monkeypatch.setattr(unified, "load_config", lambda path: config)
    monkeypatch.setattr(unified, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(unified, "start_dask_client", lambda cfg, argv: order.append("client") or types.SimpleNamespace(
        close=lambda: order.append("closed")))
    monkeypatch.setattr(unified, "build_steps", lambda cfg: [("idfeature", lambda cfg, **kw: order.append("idfeature"), {})])
    fake = types.ModuleType("pyflextrkr.preprocess_wrf_tb_rainrate")
    fake.preprocess_wrf_tb_rainrate = lambda cfg: order.append("preprocess")
    monkeypatch.setitem(sys.modules, "pyflextrkr.preprocess_wrf_tb_rainrate", fake)
    unified.main(["run_mcs_tbpf_unified.py", "config.yml"])
    assert order == ["preprocess", "client", "idfeature", "closed"]


def test_main_stops_at_the_start_when_the_definition_is_missing(monkeypatch, tmp_path):
    config = dict(tb_config(str(tmp_path) + "/"), feature_type="tb_pf", run_robustmcs=True)
    monkeypatch.setattr(unified, "load_config", lambda path: config)
    monkeypatch.setattr(unified, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(unified, "start_dask_client", lambda cfg, argv: pytest.fail("the client must not be started before the checks"))
    with pytest.raises(SystemExit) as stop:
        unified.main(["run_mcs_tbpf_unified.py", "config.yml"])
    assert "robustmcs_definition must be one of" in str(stop.value)
