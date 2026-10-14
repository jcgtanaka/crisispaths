"""Command line: argument handling, exit codes and messages, end to end on files."""

import json

import pytest

from crisispaths import cli

STRAT = "helpers_strategies:sma_trend"
SMALL = ["--n-paths", "3", "--workers", "1", "--costs-bps", "5"]


@pytest.fixture(scope="module")
def files(market, tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    px = market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]]
    px.rename_axis("date").to_csv(d / "prices.csv")
    market["factors"].rename_axis("date").to_csv(d / "factors.csv")
    px.rename_axis("date").to_parquet(d / "prices.parquet")
    return d


def call(argv, capsys):
    code = cli.main([str(a) for a in argv])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def base(files, out, *extra):
    return ["run", "--prices", files / "prices.csv", "--factors", files / "factors.csv",
            "--strategy", STRAT, "--out", out, *extra]


def test_scenarios_lists_aliases(capsys):
    code, out, _ = call(["scenarios"], capsys)
    assert code == 0
    assert "ai_bubble_worst" in out and "us_bankruptcy_haven_v_recovery" in out
    assert "ai_bubble" in out.split("families")[-1]


def test_scenarios_family_axes(capsys):
    code, out, _ = call(["scenarios", "--family", "ai_bubble"], capsys)
    assert code == 0
    for word in ("crash_source", "analog_exact", "parametric", "default"):
        assert word in out


def test_scenarios_grid_report(capsys):
    code, out, _ = call(["scenarios", "--grid", "us_bankruptcy", "--axis", "usd=debasement,haven",
                         "--axis", "aftermath=prolonged_bear,v_recovery"], capsys)
    assert code == 0
    assert "combinations: 4" in out and "distinct scenarios: 4" in out


def test_unknown_family_exit_1(capsys):
    code, out, err = call(["scenarios", "--family", "nope"], capsys)
    assert code == 1
    assert "nope" in err and "available" in err


def test_run_requires_costs(files, tmp_path, capsys):
    argv = ["run", "--prices", files / "prices.csv", "--factors", files / "factors.csv",
            "--strategy", STRAT, "--scenario", "ai_bubble_worst", "--out", tmp_path]
    code, _, err = call(argv, capsys)
    assert code == 2
    assert "--costs-bps" in err and "not zero" in err
    assert not (tmp_path / "REPORT.md").exists()


def test_run_scenario_end_to_end(files, tmp_path, capsys):
    out = tmp_path / "o"
    code, stdout, err = call(base(files, out, "--scenario", "ai_bubble_worst", *SMALL), capsys)
    assert code == 0, err
    assert (out / "REPORT.md").exists()
    assert (out / "results_ai_bubble_worst.csv").exists()
    man = json.loads((out / "manifest_ai_bubble_worst.json").read_text())
    assert man["cost_multiplier"] == 3.0 and man["costs_bps"] == 5.0
    assert "REPORT.md" in stdout


def test_run_resume_and_no_resume(files, tmp_path, capsys):
    out = tmp_path / "o"
    argv = base(files, out, "--scenario", "ai_bubble_worst", *SMALL)
    assert call(argv, capsys)[0] == 0
    assert call([*argv], capsys)[0] == 0  # resume: nothing to do
    changed = [*argv, "--seed", "9"]
    code, _, err = call(changed, capsys)
    assert code == 1 and "seed" in err
    code, _, _ = call([*changed, "--no-resume"], capsys)
    assert code == 0


def test_run_family_sweep_with_overrides(files, tmp_path, capsys):
    out = tmp_path / "o"
    argv = base(files, out, "--family", "ai_bubble", "--axis", "complexity=minimal",
                "--axis", "crash_source=parametric", "--axis", "depth=as_2000,severe",
                "--override", "anchors.EQ.params.median=0.5", *SMALL)
    code, stdout, err = call(argv, capsys)
    assert code == 0, err
    assert len(list(out.glob("results_*.csv"))) == 2
    man = json.loads(next(out.glob("manifest_*.json")).read_text())
    assert man["overrides"] == {"anchors.EQ.params.median": 0.5}
    assert "Variant comparison" in (out / "REPORT.md").read_text()


def test_sweep_guard_exit_1(files, tmp_path, capsys):
    argv = base(files, tmp_path / "o", "--family", "ai_bubble", "--axis", "depth=as_2000,severe",
                "--axis", "aftermath=analog,prolonged_bear", "--max-scenarios", "2", *SMALL)
    code, _, err = call(argv, capsys)
    assert code == 1 and "max_scenarios" in err and "4" in err


@pytest.mark.parametrize(
    "extra, needle",
    [
        (["--axis", "novalue"], "key=value"),
        (["--override", "novalue"], "path=value"),
    ],
)
def test_malformed_axis_override_exit_2(files, tmp_path, capsys, extra, needle):
    argv = base(files, tmp_path / "o", "--family", "ai_bubble", *extra, *SMALL)
    code, _, err = call(argv, capsys)
    assert code == 2 and needle in err


def test_scenario_xor_family(files, tmp_path, capsys):
    code, _, err = call(base(files, tmp_path / "o", *SMALL), capsys)
    assert code == 2 and "--scenario" in err and "--family" in err
    code, _, err = call(base(files, tmp_path / "o", "--scenario", "ai_bubble_worst",
                             "--family", "ai_bubble", *SMALL), capsys)
    assert code == 2


def test_bad_strategy_and_prices_exit_1(files, tmp_path, capsys):
    argv = base(files, tmp_path / "o", "--scenario", "ai_bubble_worst", *SMALL)
    bad = [*argv]
    bad[bad.index("--strategy") + 1] = "helpers_strategies:missing"
    code, _, err = call(bad, capsys)
    assert code == 1 and "cannot import" in err
    bad = [*argv]
    bad[bad.index("--prices") + 1] = str(tmp_path / "nope.csv")
    code, _, err = call(bad, capsys)
    assert code == 1 and "nope.csv" in err
    bad[bad.index("--prices") + 1] = str(tmp_path / "x.txt")
    (tmp_path / "x.txt").write_text("a")
    code, _, err = call(bad, capsys)
    assert code == 1 and ".csv" in err


def test_invalid_override_path_exit_1(files, tmp_path, capsys):
    argv = base(files, tmp_path / "o", "--family", "ai_bubble", "--override", "no.such.path=1",
                *SMALL)
    code, _, err = call(argv, capsys)
    assert code == 1 and "override path" in err


def test_parquet_prices(files, tmp_path, capsys):
    argv = base(files, tmp_path / "o", "--scenario", "ai_bubble_recovery", *SMALL)
    argv[argv.index("--prices") + 1] = str(files / "prices.parquet")
    assert call(argv, capsys)[0] == 0


def test_report_subcommand(files, tmp_path, capsys):
    out = tmp_path / "o"
    assert call(base(files, out, "--scenario", "ai_bubble_worst", *SMALL), capsys)[0] == 0
    (out / "REPORT.md").unlink()
    code, stdout, _ = call(["report", out], capsys)
    assert code == 0 and (out / "REPORT.md").exists()
    code, _, err = call(["report", tmp_path / "missing"], capsys)
    assert code == 1 and "no results" in err or "not a directory" in err


def test_no_subcommand_is_usage_error(capsys):
    code, _, err = call([], capsys)
    assert code == 2


def test_calm_drift_flag(files, tmp_path, capsys):
    out = tmp_path / "o"
    argv = base(files, out, "--scenario", "ai_bubble_worst", *SMALL)
    assert call([*argv, "--calm-drift", "historical"], capsys)[0] == 0
    man = json.loads((out / "manifest_ai_bubble_worst.json").read_text())
    assert man["calm_drift"] == "historical"
    assert "calm drift" in (out / "REPORT.md").read_text().lower()
    code, _, err = call([*argv, "--calm-drift", "bogus"], capsys)
    assert code == 2 and "calm-drift" in err
    code, _, err = call([*argv], capsys)  # default zero against a historical pin
    assert code == 1 and "calibration_hash" in err
