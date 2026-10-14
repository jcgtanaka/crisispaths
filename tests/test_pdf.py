"""PDF report: section order, commentary handling, determinism, missing reportlab."""

import pytest

pytest.importorskip("reportlab")
pypdf = pytest.importorskip("pypdf")

from crisispaths import cli  # noqa: E402
from crisispaths.report import BANNER, write_report  # noqa: E402
from crisispaths.runner import run_mc  # noqa: E402
from crisispaths.summary import write_summary  # noqa: E402

STRAT = "helpers_strategies:sma_trend"
HEADINGS = ["Caveats", "Scenario provenance", "Factor loadings", "Path sanity", "Results"]


@pytest.fixture(scope="module")
def outdir(market, clock, tmp_path_factory):
    out = tmp_path_factory.mktemp("pdf")
    px = market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]]
    tab = run_mc(px, STRAT, "ai_bubble_worst", n_paths=30, seed=2, clock=clock,
                 factors=market["factors"], out_dir=out, workers=1, costs_bps=5.0)
    write_report(tab, out, n_boot=60, plots=True)
    write_summary(tab, out, n_boot=60)
    return out


def _text(path):
    r = pypdf.PdfReader(str(path))
    return "\n".join(p.extract_text() for p in r.pages), len(r.pages)


def _flat(t):
    return " ".join(t.split())


def test_pdf_has_the_report_order(outdir, tmp_path):
    assert cli.main(["report", str(outdir), "--pdf", "--out", str(tmp_path)]) == 0
    pdf = tmp_path / "REPORT.pdf"
    assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF")
    text, pages = _text(pdf)
    flat = _flat(text)
    pos = [flat.index(_flat(BANNER))] + [flat.index(h) for h in HEADINGS]
    assert pos == sorted(pos) and len(set(pos)) == len(pos)
    assert pages >= 1
    assert "AI-generated commentary" not in text


def test_pdf_embeds_charts(outdir, tmp_path):
    pytest.importorskip("matplotlib")
    cli.main(["report", str(outdir), "--pdf", "--out", str(tmp_path)])
    r = pypdf.PdfReader(str(tmp_path / "REPORT.pdf"))
    n_img = sum(len(p.images) for p in r.pages)
    assert n_img >= 4


def test_pdf_bytes_are_deterministic(outdir, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cli.main(["report", str(outdir), "--pdf", "--out", str(a)])
    cli.main(["report", str(outdir), "--pdf", "--out", str(b)])
    assert (a / "REPORT.pdf").read_bytes() == (b / "REPORT.pdf").read_bytes()


def _commentary(outdir, tmp_path):
    import json

    d = json.loads((outdir / "summary.json").read_text())
    sc = d["scenarios"][0]
    p = tmp_path / "c.md"
    p.write_text(
        "Conditional on this scenario occurring, this is a what-if, not a forecast. "
        f"The median terminal wealth is {sc['metrics']['median_terminal_wealth']:.3f}.\n"
    )
    return p


def test_commentary_is_appended_last_after_verification(outdir, tmp_path):
    c = _commentary(outdir, tmp_path)
    o = tmp_path / "o"
    assert cli.main(["report", str(outdir), "--pdf", "--commentary", str(c),
                     "--out", str(o)]) == 0
    text, _ = _text(o / "REPORT.pdf")
    flat = _flat(text)
    h = "AI-generated commentary (not part of the deterministic report)"
    assert h in flat
    assert flat.index(h) > flat.index("Results") and flat.rindex(h) == flat.index(h)
    assert "what-if" in flat[flat.index(h):]
    assert "UNVERIFIED COMMENTARY" not in flat


def test_failing_commentary_blocks_pdf_and_no_verify_stamps(outdir, tmp_path, capsys):
    bad = tmp_path / "bad.md"
    bad.write_text("The expected return is 9.99%.\n")
    o = tmp_path / "o"
    assert cli.main(["report", str(outdir), "--pdf", "--commentary", str(bad),
                     "--out", str(o)]) == 1
    assert not (o / "REPORT.pdf").exists()
    assert cli.main(["report", str(outdir), "--pdf", "--commentary", str(bad), "--no-verify",
                     "--out", str(o)]) == 0
    text, _ = _text(o / "REPORT.pdf")
    assert "UNVERIFIED COMMENTARY" in _flat(text)
    capsys.readouterr()


def test_commentary_needs_pdf_and_missing_reportlab_exits_2(outdir, tmp_path, monkeypatch, capsys):
    c = _commentary(outdir, tmp_path)
    assert cli.main(["report", str(outdir), "--commentary", str(c)]) == 2
    import crisispaths.pdf as pdfmod

    def boom(*a, **k):
        raise ImportError("No module named 'reportlab'")

    monkeypatch.setattr(pdfmod, "write_pdf", boom)
    assert cli.main(["report", str(outdir), "--pdf", "--out", str(tmp_path / "x")]) == 2
    assert "crisispaths[pdf]" in capsys.readouterr().err
