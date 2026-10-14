"""The offline example must keep running (small n_paths)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_offline_demo_runs(tmp_path, capsys):
    demo = _load("offline_demo")
    report = demo.main(n_paths=4, out_dir=tmp_path, workers=1)
    text = report.read_text(encoding="utf-8")
    assert report.name == "REPORT.md"
    assert "Conditional on this scenario occurring" in text
    assert "## Variant comparison" in text
    assert "report:" in capsys.readouterr().out


def test_synthetic_market_shape():
    mod = _load("synthetic_market")
    prices, factors = mod.make_market()
    assert list(prices.columns) == list(mod.ASSET_BETAS)
    assert (prices > 0).all().all() and prices.index.equals(factors.index)


def test_notebook_is_valid_and_has_no_outputs():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(EXAMPLES / "notebook.ipynb", as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == "code":
            assert c.outputs == [] and c.execution_count is None


def test_module_entry_point(tmp_path):
    r = subprocess.run([sys.executable, "-m", "crisispaths", "scenarios"], capture_output=True,
                       text=True, cwd=tmp_path)
    assert r.returncode == 0 and "ai_bubble_worst" in r.stdout
    r = subprocess.run([sys.executable, "-m", "crisispaths", "run"], capture_output=True,
                       text=True, cwd=tmp_path)
    assert r.returncode == 2


@pytest.mark.network
def test_spy_example_runs(tmp_path, monkeypatch):
    """Downloads real data (Yahoo Finance, FRED); skipped by default, run with -m network."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    mod = _load("spy_sma_ai_bubble_us_bankruptcy")
    head = mod.main(n_paths=5, out_dir=tmp_path, workers=1)
    assert len(head) == 5 and (tmp_path / "ai_bubble" / "REPORT.md").exists()
