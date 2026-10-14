"""Composable scenario families.

A *family* is package data (or a user mapping) made of a ``base`` scenario
dict and a list of *axes*. Every axis has named options; an option is a
partial patch that is merged onto the base. ``compose`` picks one option per
axis and produces a validated :class:`~crisispaths.spec.ScenarioSpec` that
records exactly what was chosen in ``spec.metadata``.

Composition rules
-----------------
1. Axes are applied in declaration order (file order for YAML families).
2. Patches are merged recursively. Conflict rule, applied at every level:
   a later axis wins; the exception is a ``rationale`` string, which
   accumulates (joined with a space, in axis order), and ``calibrated``,
   which is the logical AND (any ``false`` wins), so a replayed analog that is
   later scaled by judgment is correctly flagged as not calibrated.
3. An option may declare ``requires: {axis: [options]}``. When the final
   choices do not satisfy it the option is NOT applied. A ``ScenarioWarning``
   is emitted when the option was chosen explicitly, and the skip is always
   listed in ``spec.metadata['skipped']``. An optional ``fallback`` (a
   ``patch`` plus ``describe``) is applied instead.
4. A patch entry may carry ``when_anchor: NAME`` so it only applies when that
   anchor already exists (used to scale "every anchor that is present").
5. ``overrides`` (dotted paths) are applied last and must hit existing keys.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .spec import ScenarioSpec, SpecError, _from_dict, _to_dict

_PKG = "crisispaths.families"
_FAMILY_KEYS = {"name", "description", "description_suffix", "base", "axes"}
_AXIS_KEYS = {"default", "doc", "options"}
_OPTION_KEYS = {"summary", "describe", "requires", "patch", "patches", "fallback"}
_FALLBACK_KEYS = {"patch", "describe"}
_RESERVED_CALL_NAMES = {"family", "overrides"}
_PROTECTED = {"name", "variant", "metadata"}


class ScenarioWarning(UserWarning):
    """An axis option was requested but does not apply to the other choices."""


@dataclass(frozen=True)
class Family:
    """A validated scenario family."""

    name: str
    description: str
    description_suffix: str
    base: dict
    axes: dict


# --- helpers ------------------------------------------------------------------


def _norm(obj: Any) -> Any:
    """Normalise to plain JSON-like data (tuples to lists, dates to strings)."""
    return json.loads(json.dumps(obj, default=str))


def _merge(dst: dict, src: dict) -> None:
    for key, val in src.items():
        cur = dst.get(key)
        if isinstance(val, dict) and isinstance(cur, dict):
            _merge(cur, val)
        elif key == "rationale" and isinstance(val, str) and isinstance(cur, str):
            dst[key] = " ".join(p for p in (cur, val) if p)
        elif key == "calibrated" and isinstance(val, bool) and isinstance(cur, bool):
            dst[key] = cur and val
        else:
            dst[key] = copy.deepcopy(val)


def _as_list(value: Any, where: str) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise SpecError(f"{where}: expected a name or a list of names, got {value!r}")


# --- family loading -----------------------------------------------------------


def _normalise_option(fam: str, axis: str, name: str, opt: Any) -> dict:
    where = f"family {fam!r} axis {axis!r} option {name!r}"
    if not isinstance(opt, dict):
        raise SpecError(f"{where}: must be a mapping")
    extra = set(opt) - _OPTION_KEYS
    if extra:
        raise SpecError(f"{where}: unknown field(s) {sorted(extra)}")
    if not isinstance(opt.get("summary"), str) or not opt["summary"]:
        raise SpecError(f"{where}: a non-empty summary is required")
    if "patch" in opt and "patches" in opt:
        raise SpecError(f"{where}: use either patch or patches, not both")
    if "patches" in opt:
        entries = opt["patches"]
        if not isinstance(entries, list):
            raise SpecError(f"{where}: patches must be a list")
        patches = []
        for e in entries:
            if not isinstance(e, dict) or set(e) - {"patch", "when_anchor"}:
                raise SpecError(f"{where}: each patches entry needs patch and optional when_anchor")
            if not isinstance(e.get("patch"), dict):
                raise SpecError(f"{where}: patch must be a mapping")
            patches.append({"patch": _norm(e["patch"]), "when_anchor": e.get("when_anchor")})
    else:
        p = opt.get("patch", {})
        if not isinstance(p, dict):
            raise SpecError(f"{where}: patch must be a mapping, got {type(p).__name__}")
        patches = [{"patch": _norm(p), "when_anchor": None}]
    fb = opt.get("fallback")
    if fb is not None:
        if not isinstance(fb, dict) or set(fb) - _FALLBACK_KEYS:
            raise SpecError(f"{where}: fallback takes only patch and describe")
        if not isinstance(fb.get("patch", {}), dict):
            raise SpecError(f"{where}: fallback patch must be a mapping")
        fb = {"patch": _norm(fb.get("patch", {})), "describe": fb.get("describe", "")}
    req = opt.get("requires") or {}
    if not isinstance(req, dict):
        raise SpecError(f"{where}: requires must be a mapping of axis to options")
    return {
        "summary": opt["summary"],
        "describe": opt.get("describe", ""),
        "requires": {a: _as_list(v, f"{where} requires") for a, v in req.items()},
        "patches": patches,
        "fallback": fb,
    }


def _build_family(data: dict) -> Family:
    if not isinstance(data, dict):
        raise SpecError("family must be a mapping")
    extra = set(data) - _FAMILY_KEYS
    if extra:
        raise SpecError(f"family: unknown field(s) {sorted(extra)}")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise SpecError("family: missing required field 'name'")
    base = data.get("base") or {}
    allowed = set(ScenarioSpec.__dataclass_fields__) - _PROTECTED
    bad = set(base) - allowed
    if bad:
        raise SpecError(f"family {name!r}: base has unknown field(s) {sorted(bad)}")
    axes_in = data.get("axes") or {}
    if not isinstance(axes_in, dict):
        raise SpecError(f"family {name!r}: axes must be a mapping")
    axes: dict[str, dict] = {}
    for axis, spec in axes_in.items():
        where = f"family {name!r} axis {axis!r}"
        if axis in _RESERVED_CALL_NAMES:
            raise SpecError(f"{where}: axis name is reserved")
        if not isinstance(spec, dict) or set(spec) - _AXIS_KEYS:
            raise SpecError(f"{where}: expected a mapping with {sorted(_AXIS_KEYS)}")
        options = spec.get("options")
        if not isinstance(options, dict) or not options:
            raise SpecError(f"{where}: options must be a non-empty mapping")
        default = spec.get("default")
        if default not in options:
            raise SpecError(
                f"{where}: default {default!r} is not one of the options {list(options)}"
            )
        axes[axis] = {
            "default": default,
            "doc": spec.get("doc", ""),
            "options": {
                str(o): _normalise_option(name, axis, str(o), v) for o, v in options.items()
            },
        }
    for axis, spec in axes.items():
        for oname, opt in spec["options"].items():
            for ra, ros in opt["requires"].items():
                if ra not in axes:
                    raise SpecError(
                        f"family {name!r} axis {axis!r} option {oname!r}: requires unknown "
                        f"axis {ra!r}; axes are {list(axes)}"
                    )
                missing = [o for o in ros if o not in axes[ra]["options"]]
                if missing:
                    raise SpecError(
                        f"family {name!r} axis {axis!r} option {oname!r}: requires unknown "
                        f"option(s) {missing} of axis {ra!r}"
                    )
    fam = Family(
        name=name,
        description=data.get("description", ""),
        description_suffix=data.get("description_suffix", ""),
        base=_norm(base),
        axes=axes,
    )
    _smoke(fam)
    return fam


def _smoke(fam: Family) -> None:
    """Compose the defaults and every option one at a time, to fail early."""
    defaults = {a: s["default"] for a, s in fam.axes.items()}
    trials = [("defaults", dict(defaults))]
    for axis, spec in fam.axes.items():
        for opt in spec["options"]:
            trials.append((f"{axis}={opt}", {**defaults, axis: opt}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScenarioWarning)
        for label, choices in trials:
            try:
                _build(fam, choices, set(), None)
            except SpecError as exc:
                raise SpecError(f"family {fam.name!r} ({label}): {exc}") from exc


def load_family(source: str | Path | dict) -> Family:
    """Load and validate a family from a YAML path or a mapping."""
    if isinstance(source, Family):
        return source
    if isinstance(source, dict):
        return _build_family(copy.deepcopy(source))
    path = Path(source)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"cannot read family file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid YAML in {path}: {exc}") from exc
    return _build_family(data)


def list_families() -> list[str]:
    """Names of the packaged scenario families, sorted."""
    root = resources.files(_PKG)
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".yaml"))


@cache
def _packaged(name: str) -> Family:
    text = (resources.files(_PKG) / f"{name}.yaml").read_text(encoding="utf-8")
    return _build_family(yaml.safe_load(text))


def _as_family(family: Any) -> Family:
    if isinstance(family, Family):
        return family
    if isinstance(family, dict):
        return load_family(family)
    if isinstance(family, str) and family in list_families():
        return _packaged(family)
    raise SpecError(f"unknown family {family!r}; available: {list_families()}")


# --- composition --------------------------------------------------------------


def _resolve(fam: Family, choices: dict) -> dict:
    unknown = [a for a in choices if a not in fam.axes]
    if unknown:
        raise SpecError(f"unknown axis {unknown[0]!r}; valid axes: {list(fam.axes)}")
    out = {}
    for axis, spec in fam.axes.items():
        val = choices.get(axis, spec["default"])
        if val not in spec["options"]:
            raise SpecError(
                f"unknown option {val!r} for axis {axis!r}; valid options: "
                f"{list(spec['options'])}"
            )
        out[axis] = val
    return out


def _set_path(raw: dict, path: str, value: Any) -> None:
    segs = path.split(".")
    if segs[0] in _PROTECTED:
        raise SpecError(f"override path {path!r}: {segs[0]!r} cannot be overridden")
    node: Any = raw
    for i, seg in enumerate(segs):
        here = ".".join(segs[:i]) or "top level"
        last = i == len(segs) - 1
        if isinstance(node, dict):
            if seg not in node:
                raise SpecError(
                    f"override path {path!r}: no key {seg!r} at {here}; "
                    f"valid keys: {sorted(node)}"
                )
            if last:
                node[seg] = copy.deepcopy(value)
            else:
                node = node[seg]
        elif isinstance(node, list):
            try:
                idx = int(seg)
                node[idx]
            except (ValueError, IndexError) as exc:
                raise SpecError(
                    f"override path {path!r}: {seg!r} is not a valid index at {here}"
                ) from exc
            if last:
                node[idx] = copy.deepcopy(value)
            else:
                node = node[idx]
        else:
            raise SpecError(f"override path {path!r}: cannot descend into {here}")


def _check_documented(raw: dict, label: str) -> None:
    def need(where: str, node: Any) -> None:
        if isinstance(node, dict) and node.get("calibrated") is False:
            if not str(node.get("rationale", "")).strip():
                raise SpecError(f"{label}: {where} is a judgment (calibrated=false) but has "
                                "no rationale")

    for k, v in (raw.get("anchors") or {}).items():
        need(f"anchor {k!r}", v)
    need("aftermath", raw.get("aftermath"))
    for k, v in (raw.get("param_meta") or {}).items():
        need(f"param_meta.{k}", v)


def _variant(choices: dict, overrides: dict) -> str:
    v = "-".join(f"{a}.{o}" for a, o in choices.items())
    if overrides:
        blob = json.dumps(overrides, sort_keys=True, default=str)
        v += "-ov" + hashlib.sha256(blob.encode()).hexdigest()[:8]
    return v


def _build(fam: Family, choices: dict, explicit: set, overrides: dict | None) -> ScenarioSpec:
    defaults = _to_dict(ScenarioSpec(name="_"))
    for k in _PROTECTED:
        defaults.pop(k, None)
    raw: dict = _norm(defaults)
    _merge(raw, copy.deepcopy(fam.base))
    describes: list[str] = []
    skipped: list[str] = []
    for axis, spec in fam.axes.items():
        chosen = choices[axis]
        opt = spec["options"][chosen]
        unmet = [
            f"{ra}={choices[ra]!r} (needs {' or '.join(map(repr, ros))})"
            for ra, ros in opt["requires"].items()
            if choices[ra] not in ros
        ]
        if unmet:
            tag = f"{axis}={chosen}"
            skipped.append(tag)
            fb = opt["fallback"]
            if axis in explicit:
                warnings.warn(
                    f"option {tag} does not apply: " + "; ".join(unmet)
                    + ("; a fallback was applied instead" if fb else "; it was skipped"),
                    ScenarioWarning,
                    stacklevel=4,
                )
            if fb:
                _merge(raw, copy.deepcopy(fb["patch"]))
                if fb["describe"]:
                    describes.append(fb["describe"])
            continue
        for entry in opt["patches"]:
            anchor = entry["when_anchor"]
            if anchor is not None and anchor not in raw.get("anchors", {}):
                continue
            _merge(raw, copy.deepcopy(entry["patch"]))
        if opt["describe"]:
            describes.append(opt["describe"])
    parts = [raw.get("description", ""), *describes, fam.description_suffix]
    raw["description"] = " ".join(p.strip() for p in parts if p and p.strip())
    ov = _norm(overrides or {})
    for path, value in ov.items():
        _set_path(raw, path, value)
    label = f"family {fam.name!r}"
    _check_documented(raw, label)
    raw["name"] = fam.name
    raw["variant"] = _variant(choices, ov)
    raw["metadata"] = {
        "family": fam.name,
        "axis_choices": dict(choices),
        "overrides": ov,
        "skipped": skipped,
    }
    try:
        return _from_dict(raw)
    except SpecError as exc:
        raise SpecError(f"{label} choices {dict(choices)}: {exc}") from exc


def compose(family: Any, *, overrides: dict | None = None, **axis_choices: str) -> ScenarioSpec:
    """Compose one scenario from a family by choosing one option per axis.

    Axes not given take their documented default (see :func:`describe_axes`).
    ``overrides`` is a dict of dotted paths applied last, for example
    ``{"anchors.EQ.params.median": 0.5, "crash_length": "6M"}``.
    """
    fam = _as_family(family)
    for axis, val in axis_choices.items():
        if not isinstance(val, str):
            raise SpecError(
                f"axis {axis!r}: compose takes one option name (str), got {val!r}; "
                "use enumerate_scenarios to sweep"
            )
    resolved = _resolve(fam, axis_choices)
    return _build(fam, resolved, set(axis_choices), overrides)


def _sweep(fam: Family, axes: dict) -> dict[str, list[str]]:
    unknown = [a for a in axes if a not in fam.axes]
    if unknown:
        raise SpecError(f"unknown axis {unknown[0]!r}; valid axes: {list(fam.axes)}")
    out: dict[str, list[str]] = {}
    for axis, spec in fam.axes.items():
        val = axes.get(axis, spec["default"])
        vals = list(val) if isinstance(val, (list, tuple)) else [val]
        if not vals:
            raise SpecError(f"axis {axis!r}: empty list of options")
        for v in vals:
            if v not in spec["options"]:
                raise SpecError(
                    f"unknown option {v!r} for axis {axis!r}; valid options: "
                    f"{list(spec['options'])}"
                )
        out[axis] = list(dict.fromkeys(vals))
    return out


def grid_size(family: Any, **axes: Any) -> int:
    """Number of scenarios in the cartesian grid (lists swept, scalars fixed)."""
    sweep = _sweep(_as_family(family), axes)
    n = 1
    for vals in sweep.values():
        n *= len(vals)
    return n


@dataclass(frozen=True)
class GridReport:
    """Summary of a sweep: how many combinations, how many distinct scenarios.

    ``groups`` has one entry per distinct scenario (by content hash, in
    enumeration order); each entry lists the axis choices that produce it, the
    kept (first) combination first.
    """

    n_total: int
    n_unique: int
    groups: tuple[tuple[dict, ...], ...]


def _iter_all(
    fam: Family, sweep: dict, explicit: set, overrides: dict | None
) -> Iterator[tuple[dict, ScenarioSpec]]:
    for combo in itertools.product(*sweep.values()):
        choices = dict(zip(sweep, combo, strict=True))
        yield choices, _build(fam, choices, explicit, overrides)


def _group(pairs: Iterator[tuple[dict, ScenarioSpec]]) -> list[list[tuple[dict, ScenarioSpec]]]:
    groups: dict[str, list[tuple[dict, ScenarioSpec]]] = {}
    for choices, spec in pairs:
        groups.setdefault(spec.content_hash, []).append((choices, spec))
    return list(groups.values())


def enumerate_scenarios(
    family: Any, *, overrides: dict | None = None, dedupe: bool = True, **axes: Any
) -> Iterator[tuple[dict, ScenarioSpec]]:
    """Iterate ``(choices, spec)`` over the full cartesian grid.

    An axis given as a list is swept, a scalar is fixed, an omitted axis takes
    its default. Order is axis declaration order, last axis fastest.

    With ``dedupe=True`` (default) combinations whose ``content_hash`` is equal
    collapse to the first one; the choices of the merged combinations are
    recorded in ``spec.metadata["merged_choices"]`` of the kept spec. Use
    ``dedupe=False`` to keep every combination.
    """
    fam = _as_family(family)
    sweep = _sweep(fam, axes)
    explicit = set(axes)
    if not dedupe:
        return _iter_all(fam, sweep, explicit, overrides)

    def gen() -> Iterator[tuple[dict, ScenarioSpec]]:
        for group in _group(_iter_all(fam, sweep, explicit, overrides)):
            choices, spec = group[0]
            if len(group) > 1:
                spec.metadata["merged_choices"] = [dict(c) for c, _ in group[1:]]
            yield choices, spec

    return gen()


def grid_report(family: Any, *, overrides: dict | None = None, **axes: Any) -> GridReport:
    """Count total and distinct scenarios of a sweep and list the merge groups."""
    fam = _as_family(family)
    sweep = _sweep(fam, axes)
    groups = _group(_iter_all(fam, sweep, set(axes), overrides))
    return GridReport(
        n_total=sum(len(g) for g in groups),
        n_unique=len(groups),
        groups=tuple(tuple(dict(c) for c, _ in g) for g in groups),
    )


def describe_axes(family: Any) -> dict:
    """Structured description of a family: axis order, defaults, option summaries."""
    fam = _as_family(family)
    return {
        "family": fam.name,
        "description": fam.description,
        "axis_order": list(fam.axes),
        "axes": {
            axis: {
                "default": spec["default"],
                "doc": spec["doc"],
                "options": {
                    o: {
                        "summary": opt["summary"],
                        "requires": {a: list(v) for a, v in opt["requires"].items()},
                        "is_default": o == spec["default"],
                    }
                    for o, opt in spec["options"].items()
                },
            }
            for axis, spec in fam.axes.items()
        },
    }
