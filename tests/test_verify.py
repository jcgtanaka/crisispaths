"""verify-commentary: traceability and framing heuristics (not semantic truth)."""

import json

import pytest

from crisispaths import cli
from crisispaths.report import write_report
from crisispaths.runner import run_mc
from crisispaths.summary import write_summary
from crisispaths.verify import verify_commentary

STRAT = "helpers_strategies:sma_trend"


@pytest.fixture(scope="module")
def doc(market, clock, tmp_path_factory):
    out = tmp_path_factory.mktemp("vc")
    px = market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]]
    tab = run_mc(px, STRAT, "ai_bubble_worst", n_paths=30, seed=2, clock=clock,
                 factors=market["factors"], out_dir=out, workers=1, costs_bps=5.0)
    path = write_summary(tab, out, n_boot=60)
    write_report(tab, out, n_boot=60, plots=False)
    return out, path, json.loads(path.read_text())


def _good(d):
    sc = d["scenarios"][0]
    m = sc["metrics"]
    tw = m["median_terminal_wealth"]
    p = m["ruin"]["p"]
    return (
        "Conditional on this scenario occurring, the strategy is a what-if exercise, not a "
        f"forecast. Across {sc['n_paths']} paths the median terminal wealth is {tw:.3f} of the "
        f"starting equity, and the ruin share is {100 * p:.1f}%. The seed was {sc['seed']}.\n"
    )


def test_good_commentary_passes(doc):
    r = verify_commentary(_good(doc[2]), doc[2])
    assert r.passed and not r.errors and not r.untraceable


def test_untraceable_number_is_listed_with_its_sentence(doc):
    text = _good(doc[2]) + "The median terminal wealth is 7.321 in this run.\n"
    r = verify_commentary(text, doc[2])
    assert not r.passed
    assert any(u["number"].startswith("7.321") and "7.321" in u["sentence"]
               for u in r.untraceable)


def test_displayed_precision_is_the_tolerance(doc):
    d = doc[2]
    tw = d["scenarios"][0]["metrics"]["median_terminal_wealth"]
    ok = verify_commentary(_good(d).replace(f"{tw:.3f}", f"{tw:.2f}"), d)
    assert ok.passed
    off = verify_commentary(_good(d).replace(f"{tw:.3f}", f"{tw + 0.05:.3f}"), d)
    assert not off.passed


@pytest.mark.parametrize("sentence", [
    "The expected return across the grid is attractive.",
    "On average across scenarios the strategy holds up.",
    "This is a forecast of the next bear market.",
    "The crash will happen within two years.",
    "The probability of a US default is 5%.",
    "The strategy is safe in every scenario.",
    "Results are guaranteed by the model.",
])
def test_forbidden_framings_fail(doc, sentence):
    r = verify_commentary(_good(doc[2]) + sentence + "\n", doc[2])
    assert not r.passed
    assert r.framing


def test_negated_forecast_is_allowed(doc):
    r = verify_commentary(_good(doc[2]) + "This is not a forecast.\n", doc[2])
    assert r.passed


def test_must_acknowledge_conditionality(doc):
    r = verify_commentary("The median terminal wealth is fine.\n", doc[2])
    assert not r.passed and any("conditional" in e for e in r.errors)


def test_paper_and_live_performance_claims_fail(doc):
    for s in ("The paper trading record confirms this.", "Live trading results match."):
        r = verify_commentary(_good(doc[2]) + s + "\n", doc[2])
        assert not r.passed


def test_imputed_and_warning_assets_must_be_mentioned_warn_level(doc):
    d = json.loads(json.dumps(doc[2]))
    sc = d["scenarios"][0]
    for r in sc["path_sanity"]:
        r["warning"], r["imputed"] = None, False
    sc["path_sanity"][0]["imputed"] = True
    sc["path_sanity"][1]["warning"] = "WARNING: sign mismatch"
    r = verify_commentary(_good(d), d)
    assert r.passed  # warn level only
    names = " ".join(r.warnings)
    assert sc["path_sanity"][0]["asset"] in names and sc["path_sanity"][1]["asset"] in names
    text = _good(d) + (f"{sc['path_sanity'][0]['asset']} is imputed and "
                       f"{sc['path_sanity'][1]['asset']} carries a path sanity WARNING.\n")
    assert not verify_commentary(text, d).warnings


def test_cli_exit_codes_and_help(doc, tmp_path, capsys):
    out, summary, d = doc
    good = tmp_path / "good.md"
    good.write_text(_good(d))
    bad = tmp_path / "bad.md"
    bad.write_text("The expected return is 12.5%.\n")
    assert cli.main(["verify-commentary", str(good), "--summary", str(summary)]) == 0
    assert cli.main(["verify-commentary", str(bad), "--summary", str(summary)]) == 1
    err = capsys.readouterr().out
    assert "12.5" in err and "FAIL" in err
    assert cli.main(["verify-commentary", str(tmp_path / "nope.md"), "--summary",
                     str(summary)]) == 2
    assert cli.main(["verify-commentary", "--help"]) == 0
    assert "not semantic truth" in capsys.readouterr().out
