"""PDF rendering of REPORT.md (optional ``[pdf]`` extra: reportlab, matplotlib).

The PDF is a rendering of REPORT.md, so its section order is identical (banner,
caveats, provenance, loadings, path sanity, results, variant comparison) and
the PNG charts written next to it are embedded. Reportlab is run in
``invariant`` mode (fixed dates and document ids) and nothing in the body is
time dependent, so the same inputs give the same bytes with the same reportlab
and matplotlib versions. Residual nondeterminism: none known; PNG bytes depend
on the matplotlib version, and fonts are the built-in Helvetica/Courier.

An optional commentary (Markdown) is appended LAST under the heading
:data:`COMMENTARY_HEADING`. If it was not verified a red ``UNVERIFIED
COMMENTARY`` stamp is printed on the section and in every page footer.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

COMMENTARY_HEADING = "AI-generated commentary (not part of the deterministic report)"
UNVERIFIED_STAMP = "UNVERIFIED COMMENTARY"


def _inline(text: str) -> str:
    s = escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', s)
    return s


def _flowables(md: str, base: Path, styles: dict, *, level_offset: int = 0) -> list:
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import HRFlowable, Image, Paragraph, Spacer, Table, TableStyle

    out: list = []
    lines = md.splitlines()
    i = 0
    width = 17.0 * cm
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip().replace("\\|", "|") for c in r.strip("|").split("|")]
                    for r in block if not re.fullmatch(r"\|[\s|:-]+\|", r)]
            if rows:
                n = max(len(r) for r in rows)
                data = [[Paragraph(_inline(c), styles["cell_b" if k == 0 else "cell"])
                         for c in r + [""] * (n - len(r))] for k, r in enumerate(rows)]
                t = Table(data, colWidths=[width / n] * n, repeatRows=1)
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                out += [t, Spacer(1, 6)]
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = min(3, len(m.group(1)) + level_offset)
            out.append(Paragraph(_inline(m.group(2)), styles[f"h{lvl}"]))
            i += 1
            continue
        if s.startswith(">"):
            out.append(Paragraph(_inline(s.lstrip("> ").strip()), styles["quote"]))
            i += 1
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
        if m:
            img = base / m.group(2)
            if img.exists():
                from reportlab.lib.utils import ImageReader

                iw, ih = ImageReader(str(img)).getSize()
                w = min(width * 0.8, iw)
                out += [Image(str(img), width=w, height=w * ih / iw), Spacer(1, 6)]
            i += 1
            continue
        if s == "---":
            out.append(HRFlowable(width="100%"))
            i += 1
            continue
        m = re.match(r"^(\s*)[-*]\s+(.*)$", ln)
        if m:
            indent = len(m.group(1))
            style = styles["bullet2" if indent >= 2 else "bullet"]
            out.append(Paragraph(_inline(m.group(2)), style, bulletText="-"))
            i += 1
            continue
        para = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(\||#|>|!\[|---|\s*[-*]\s)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(Paragraph(_inline(" ".join(para)), styles["body"]))
    return out


def _styles() -> dict:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    ss = getSampleStyleSheet()
    base = ss["BodyText"]
    small = ParagraphStyle("cell", parent=base, fontSize=6.5, leading=8, splitLongWords=1)
    return {
        "body": ParagraphStyle("body", parent=base, fontSize=9, leading=12, splitLongWords=1),
        "cell": small,
        "cell_b": ParagraphStyle("cell_b", parent=small, fontName="Helvetica-Bold"),
        "bullet": ParagraphStyle("bullet", parent=base, fontSize=9, leading=12, leftIndent=14,
                                 bulletIndent=4, splitLongWords=1),
        "bullet2": ParagraphStyle("bullet2", parent=base, fontSize=9, leading=12, leftIndent=28,
                                  bulletIndent=18, splitLongWords=1),
        "quote": ParagraphStyle("quote", parent=base, fontSize=10, leading=14, leftIndent=8,
                                borderPadding=6, borderColor=colors.black, borderWidth=1,
                                spaceAfter=10, backColor=colors.HexColor("#f6f6f6")),
        "h1": ParagraphStyle("h1", parent=ss["Title"], fontSize=18, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=ss["Heading1"], fontSize=14, spaceBefore=12),
        "h3": ParagraphStyle("h3", parent=ss["Heading2"], fontSize=10.5, spaceBefore=8),
        "stamp": ParagraphStyle("stamp", parent=base, fontSize=11, textColor=colors.red,
                                fontName="Helvetica-Bold"),
    }


def write_pdf(report_md: str | Path, out_pdf: str | Path, *, commentary: str | None = None,
              unverified: bool = False) -> Path:
    """Render ``report_md`` (and optional commentary) to ``out_pdf``; needs reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    src = Path(report_md)
    styles = _styles()
    story = _flowables(src.read_text(encoding="utf-8"), src.parent, styles)
    stamp = unverified and commentary is not None
    if commentary is not None:
        story += [PageBreak(), Paragraph(_inline(COMMENTARY_HEADING), styles["h2"])]
        if stamp:
            story.append(Paragraph(
                f"{UNVERIFIED_STAMP}: this text was not checked by verify-commentary.",
                styles["stamp"]))
        story.append(Spacer(1, 6))
        story += _flowables(commentary, src.parent, styles, level_offset=1)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(2 * cm, 1.1 * cm, "crisispaths: conditional what-ifs, not forecasts")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"page {doc.page}")
        if stamp:
            canvas.setFillColorRGB(0.8, 0, 0)
            canvas.drawCentredString(A4[0] / 2, 0.6 * cm, UNVERIFIED_STAMP)
        canvas.restoreState()

    out = Path(out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
        bottomMargin=1.8 * cm, title="crisispaths scenario report", author="crisispaths",
        subject="Conditional what-if scenario report", creator="crisispaths", invariant=True,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out
