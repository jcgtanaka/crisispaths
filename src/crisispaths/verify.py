"""Commentary verifier: traceability and framing, NOT semantic truth.

``verify_commentary`` checks, deterministically and with simple heuristics,
that a human or LLM written commentary on a crisispaths run:

a. contains no decimal number, percentage or ratio that cannot be traced to a
   numeric value of ``summary.json`` (at the precision displayed, also as a
   percent, sign ignored); small bare integers, years and structural counts
   (paths, seed, censored, bars) are allowed;
b. avoids forbidden framings (expected return, averages across scenarios,
   forecasts outside a negation, "will happen", scenario probabilities,
   "safe" or "guaranteed");
c. acknowledges conditionality somewhere, and (warning level only) mentions
   every asset flagged IMPUTED or carrying a path sanity WARNING;
d. does not claim paper or live trading performance.

A pass means the text is traceable and carefully framed. It does not mean the
text is true, relevant or complete; a human still has to read it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NUMBER_RE = re.compile(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)(\s*%|x\b)?")
CONDITIONAL_PHRASES = ("conditional on", "what-if", "what if", "scenario assumption",
                       "assuming the scenario", "if this scenario")
NEGATION_RE = re.compile(
    r"\b(not|no|nothing|none|never|nor|without|neither|cannot)\b|n't\b", re.I)
FORBIDDEN = [
    ("expected return", re.compile(r"\bexpected\s+returns?\b", re.I)),
    ("average across scenarios",
     re.compile(r"\b(on\s+average|averag\w*|mean)\s+(across|over|of)\s+(all\s+|the\s+)?"
                r"scenarios?\b", re.I)),
    ("forecast", re.compile(r"\bforecast(?:s|ed|ing)?\b", re.I)),
    ("will happen", re.compile(r"\bwill\s+(?:certainly\s+|definitely\s+)?happen\b", re.I)),
    ("scenario probability",
     re.compile(r"\b(?:probabilit\w+|chance|likelihood|odds)\s+(?:of|that)\s+(?:an?\s+|the\s+)?"
                r"(?:US\b|U\.S\.|AI\b|default|bubble|crash|crisis|recession|bankruptcy|"
                r"collapse|debasement)", re.I)),
    ("safe", re.compile(r"\bsafe(?:ly|ty)?\b", re.I)),
    ("guaranteed", re.compile(r"\bguarantee[sd]?\b", re.I)),
]
NEGATABLE = {"forecast", "safe", "guaranteed", "will happen"}
PERFORMANCE = re.compile(
    r"\b(?:paper|live)[\s-]+(?:trad\w*|performance|record|results?)\b|\btrack\s+record\b", re.I)


@dataclass
class VerifyResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    untraceable: list[dict] = field(default_factory=list)  # {"number", "sentence"}
    framing: list[dict] = field(default_factory=list)  # {"rule", "sentence"}

    def report(self) -> str:
        lines = ["verify-commentary: " + ("PASS" if self.passed else "FAIL")]
        for u in self.untraceable:
            lines.append(f"  untraceable number {u['number']!r} in: {u['sentence']}")
        for f in self.framing:
            lines.append(f"  forbidden framing [{f['rule']}] in: {f['sentence']}")
        lines += [f"  error: {e}" for e in self.errors]
        lines += [f"  warning: {w}" for w in self.warnings]
        lines.append(
            "Heuristic check of traceability and framing only; it does not establish that the "
            "text is true."
        )
        return "\n".join(lines)


def _numeric_leaves(obj, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numeric_leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numeric_leaves(v, out)


def _structural(doc: dict) -> set[float]:
    s: set[float] = set()
    for sc in doc.get("scenarios", []):
        for k in ("seed", "n_paths", "n_paths_requested"):
            s.add(float(sc[k]))
        s.add(float(sc["frequency"]["bars_per_year"]))
        s.add(float(sc["frequency"]["bars_per_day"]))
        m = sc["metrics"]
        s.update(float(m["censoring"][k]) for k in ("n_censored", "n"))
        s.add(float(m["ruin"]["n_ruined"]))
        h = m["recovery"].get("horizon_bars")
        if h is not None:
            s.add(float(h))
    return s


def _traceable(value: float, decimals: int, percent: bool, leaves: list[float]) -> bool:
    tol = 0.5 * 10.0 ** (-decimals) + 1e-9
    for v in leaves:
        for scale in ((100.0, 1.0) if percent else (1.0,)):
            if abs(abs(v) * scale - abs(value)) <= tol:
                return True
    return False


def _sentences(text: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    out = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*#>|])", text):
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", block):
            s = " ".join(s.split())
            if s:
                out.append(s)
    return out


def _clean_for_numbers(sentence: str) -> str:
    s = re.sub(r"`[^`]*`", " ", sentence)
    s = re.sub(r"\S*[A-Za-z]_[A-Za-z0-9_.\-]*\S*", " ", s)  # identifiers with underscores
    s = re.sub(r"https?://\S+", " ", s)
    return s


def _negated_before(sentence: str, start: int) -> bool:
    return bool(NEGATION_RE.search(sentence[max(0, start - 45): start]))


def verify_commentary(text: str, summary: dict) -> VerifyResult:
    """Check ``text`` against a ``summary.json`` document (see the module docstring)."""
    res = VerifyResult(passed=True)
    leaves: list[float] = []
    _numeric_leaves(summary, leaves)
    structural = _structural(summary)

    for sent in _sentences(text):
        for m in NUMBER_RE.finditer(_clean_for_numbers(sent)):
            raw, suffix = m.group(1), (m.group(2) or "").strip()
            val = float(raw)
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            percent = suffix == "%"
            if decimals == 0 and not percent and suffix != "x":
                if abs(val) < 10 or 1900 <= abs(val) <= 2100 or abs(val) in structural:
                    continue
            if abs(val) in structural and decimals == 0:
                continue
            if _traceable(val, decimals, percent, leaves):
                continue
            res.untraceable.append({"number": raw + (suffix if suffix else ""), "sentence": sent})
    if res.untraceable:
        res.errors.append(f"{len(res.untraceable)} number(s) cannot be traced to summary.json")

    for sent in _sentences(text):
        for rule, rx in FORBIDDEN:
            for m in rx.finditer(sent):
                if rule in NEGATABLE and _negated_before(sent, m.start()):
                    continue
                res.framing.append({"rule": rule, "sentence": sent})
                break
        pm = PERFORMANCE.search(sent)
        if pm and not _negated_before(sent, pm.start()):
            res.framing.append({"rule": "paper or live trading performance claim",
                                "sentence": sent})
    if res.framing:
        res.errors.append(f"{len(res.framing)} forbidden framing(s)")

    low = text.lower()
    if not any(p in low for p in CONDITIONAL_PHRASES):
        res.errors.append(
            "the commentary never acknowledges that results are conditional on the scenario "
            "(expected a phrase such as 'conditional on' or 'what-if')"
        )

    for sc in summary.get("scenarios", []):
        for r in sc.get("path_sanity", []):
            a = r["asset"]
            if r.get("imputed") and (a.lower() not in low or "imputed" not in low):
                res.warnings.append(
                    f"{sc['key']}: asset {a} is IMPUTED (stress loadings borrowed) but the "
                    "commentary does not say so"
                )
            if r.get("warning") and a.lower() not in low:
                res.warnings.append(
                    f"{sc['key']}: asset {a} has a path sanity WARNING that the commentary "
                    "does not mention"
                )
    res.passed = not res.errors
    return res
