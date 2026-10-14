"""summary.json: schema, content, and the absence of any cross-scenario aggregate."""

import json
import math

import numpy as np
import pytest

from crisispaths import cli
from crisispaths.report import BANNER
from crisispaths.runner import run_mc, run_sweep
from crisispaths.summary import SCHEMA_VERSION, load_schema, write_summary

jsonschema = pytest.importorskip("jsonschema")

STRAT = "helpers_strategies:sma_trend"
AXES = {"aftermath": ["prolonged_bear", "v_recovery"], "depth": ["as_2000", "severe"],
        "crash_source": "parametric", "complexity": "minimal"}


@pytest.fixture(scope="module")
def prices(market):
    return market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]]


@pytest.fixture(scope="module")
def sweep(prices, market, clock, tmp_path_factory):
    out = tmp_path_factory.mktemp("sw")
    tab = run_sweep(prices, STRAT, "ai_bubble", AXES, n_paths=30, seed=1, clock=clock,
                    factors=market["factors"], out_dir=out, workers=1, costs_bps=5.0,
                    cost_multiplier=3.0)
    path = write_summary(tab, out, n_boot=60)
    return out, tab, json.loads(path.read_text())


@pytest.fixture(scope="module")
def single(prices, market, clock, tmp_path_factory):
    out = tmp_path_factory.mktemp("si")
    tab = run_mc(prices, STRAT, "ai_bubble_worst", n_paths=30, seed=2, clock=clock,
                 factors=market["factors"], out_dir=out, workers=1, costs_bps=5.0)
    return out, tab, json.loads(write_summary(tab, out, n_boot=60).read_text())


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def test_schema_validates_and_is_versioned(sweep, single):
    schema = load_schema()
    for doc in (sweep[2], single[2]):
        jsonschema.validate(doc, schema)
        assert doc["schema_version"] == SCHEMA_VERSION == 1
        assert doc["banner"] == BANNER
    assert schema["properties"]["schema_version"]["const"] == 1


def test_schema_rejects_a_broken_document(single):
    bad = json.loads(json.dumps(single[2]))
    del bad["banner"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, load_schema())
    bad = json.loads(json.dumps(single[2]))
    bad["scenarios"][0]["seed"] = "zero"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, load_schema())


def test_content_blocks(single):
    doc = single[2]
    import crisispaths as cp

    assert doc["package_version"] == cp.__version__
    sc = doc["scenarios"][0]
    for k in ("key", "family", "axis_choices", "overrides", "content_hash", "seed", "calm_drift",
              "frequency", "profile_source", "judgment_parameters", "loadings",
              "path_sanity", "metrics", "n_paths"):
        assert k in sc
    assert sc["frequency"]["bars_per_year"] == 252
    assert sc["calm_drift"] == "zero"
    assert doc["caveats"] and any("what-if" in c for c in doc["caveats"])
    assert sc["judgment_parameters"] and all(j["rationale"] for j in sc["judgment_parameters"])
    ld = sc["loadings"]["assets"][0]
    assert {"asset", "n_obs", "std_errors", "shrinkage", "imputed", "unaffected"} <= set(ld)
    m = sc["metrics"]
    assert m["ruin"]["n"] == 30 and "km" not in m["recovery"]
    assert m["benchmark_comparison"] and m["censoring"]["n_censored"] >= 0
    assert any(q["metric"] == "terminal_wealth" for q in m["quantiles"])
    assert doc["variants"] is None


def test_sweep_has_variant_ranges_but_no_aggregate(sweep):
    out, tab, doc = sweep
    assert len(doc["scenarios"]) == 4 and doc["variants"] is not None
    v = doc["variants"]
    assert len(v["scenarios"]) == 4 and v["axis_ranges"]
    forbidden = ("mean", "average", "avg", "expected", "pooled", "overall", "total_return")
    for k in _keys(doc):
        assert not any(w in k.lower() for w in forbidden), k
    per = [s["metrics"]["median_terminal_wealth"] for s in doc["scenarios"]]
    pooled = [float(np.mean(per)), float(np.median(per)),
              float(tab["terminal_wealth"].mean()), float(tab["terminal_wealth"].median())]
    nums = [v for _, v in _leaves(doc) if isinstance(v, float)]
    for p in pooled:
        assert not any(math.isclose(p, n, rel_tol=1e-9) for n in nums)


def test_strict_json_and_determinism(sweep, tmp_path):
    out, tab, doc = sweep
    a = write_summary(out, tmp_path / "a", n_boot=60).read_text()
    b = write_summary(out, tmp_path / "b", n_boot=60).read_text()
    assert a == b
    json.loads(a, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_path_sanity_and_imputed_flags_in_json(single, tmp_path):
    out, _, _ = single
    src = tmp_path / "src"
    src.mkdir()
    for f in out.iterdir():
        (src / f.name).write_bytes(f.read_bytes())
    mf = next(src.glob("manifest_*.json"))
    m = json.loads(mf.read_text())
    asset = m["loadings"]["assets"][0]
    m["loadings"]["imputed"] = [asset]
    m["loadings"]["imputation"] = {asset: {"amplification": 1.25, "neighbours": [],
                                           "clipped": False, "degenerate": False}}
    mf.write_text(json.dumps(m))
    doc = json.loads(write_summary(src, tmp_path / "o", n_boot=40).read_text())
    jsonschema.validate(doc, load_schema())
    sc = doc["scenarios"][0]
    row = next(r for r in sc["loadings"]["assets"] if r["asset"] == asset)
    assert row["imputed"] and row["amplification"] == 1.25
    prow = next(r for r in sc["path_sanity"] if r["asset"] == asset)
    assert prow["imputed"]


def test_cli_run_writes_and_report_json_regenerates(market, tmp_path, capsys):
    d = tmp_path / "data"
    d.mkdir()
    px = market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]]
    px.rename_axis("date").to_csv(d / "p.csv")
    market["factors"].rename_axis("date").to_csv(d / "f.csv")
    out = tmp_path / "o"
    code = cli.main(["run", "--prices", str(d / "p.csv"), "--factors", str(d / "f.csv"),
                     "--strategy", STRAT, "--scenario", "ai_bubble_worst", "--n-paths", "3",
                     "--workers", "1", "--costs-bps", "5", "--out", str(out)])
    assert code == 0 and (out / "summary.json").exists()
    first = (out / "summary.json").read_text()
    (out / "summary.json").unlink()
    assert cli.main(["report", str(out), "--json"]) == 0
    assert (out / "summary.json").read_text() == first
    capsys.readouterr()
