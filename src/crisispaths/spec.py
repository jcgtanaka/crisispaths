"""Declarative scenario specification: frozen dataclasses, YAML load and dump.

Every numeric parameter group may carry ``calibrated`` and ``rationale``
metadata so reports can flag which numbers are judgment calls rather than
estimates from data.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .clock import parse_duration

CUM_DISTS = ("fixed", "uniform", "lognormal_loss", "analog")
AFTERMATH_MODES = ("analog", "prolonged_bear", "v_recovery", "stagnation")
LABELS = ("main_case", "COUNTERFACTUAL")
ANALOG_MODES = ("exact", "block_bootstrap")
_REQUIRED_PARAMS = {
    "fixed": ("value",),
    "uniform": ("low", "high"),
    "lognormal_loss": ("median", "sigma", "cap"),
    "analog": (),
}
_DURATION_PARAMS = ("liquidation_length",)
_META_KEYS = (
    "calibration_windows",
    "analog_window",
    "crash_length",
    "horizon",
    "onset",
    "vol_multiplier",
    "factor_df",
    "copula_df",
)


class SpecError(ValueError):
    """Raised when a scenario specification is invalid."""


def _date(value: Any, where: str) -> str:
    if isinstance(value, (_dt.date, _dt.datetime)):
        value = value.isoformat()[:10]
    try:
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise SpecError(f"{where}: invalid date {value!r}") from exc


def _window(value: Any, where: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SpecError(f"{where}: a window must be a (start, end) pair, got {value!r}")
    a, b = _date(value[0], where), _date(value[1], where)
    if not pd.Timestamp(a) < pd.Timestamp(b):
        raise SpecError(f"{where}: window start {a} must be before end {b}")
    return (a, b)


def _duration(value: Any, where: str) -> str:
    try:
        parse_duration(value)
    except ValueError as exc:
        raise SpecError(f"{where}: {exc}") from exc
    return value


def _pair(value: Any, where: str, *, positive: bool = False, ordered: bool = False):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value, value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SpecError(f"{where} must be a number or a pair of numbers, got {value!r}")
    try:
        a, b = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{where} must be numeric, got {value!r}") from exc
    if positive and (a <= 0 or b <= 0):
        raise SpecError(f"{where} must be positive, got {value!r}")
    if ordered and a > b:
        raise SpecError(f"{where} must be ordered (low, high), got {value!r}")
    return (a, b)


def _meta(calibrated: Any, rationale: Any, where: str) -> tuple[bool | None, str]:
    if calibrated is not None and not isinstance(calibrated, bool):
        raise SpecError(f"{where}: calibrated must be true or false, got {calibrated!r}")
    if not isinstance(rationale, str):
        raise SpecError(f"{where}: rationale must be a string")
    return calibrated, rationale


def _no_unknown(data: dict, allowed: set[str], where: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise SpecError(f"{where}: unknown field(s) {sorted(extra)}")


@dataclass(frozen=True)
class FactorShock:
    """Cumulative shock on one factor over the crash window.

    ``params`` depend on ``cum_dist``: ``fixed`` needs ``value``; ``uniform``
    needs ``low`` and ``high``; ``lognormal_loss`` needs ``median``, ``sigma``
    and ``cap``; ``analog`` needs none. Extra keys (for example
    ``liquidation_length`` or ``unit``) are preserved for later milestones.
    """

    cum_dist: str
    params: dict = field(default_factory=dict)
    aftermath_drift_scale: float = 0.0
    calibrated: bool | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.cum_dist not in CUM_DISTS:
            raise SpecError(f"unknown cum_dist {self.cum_dist!r}, expected one of {CUM_DISTS}")
        if not isinstance(self.params, dict):
            raise SpecError("params must be a mapping")
        for key in _REQUIRED_PARAMS[self.cum_dist]:
            if key not in self.params:
                raise SpecError(f"cum_dist {self.cum_dist!r} requires param {key!r}")
        if self.cum_dist == "uniform" and self.params["low"] > self.params["high"]:
            raise SpecError("uniform requires low <= high")
        for key in _DURATION_PARAMS:
            if key in self.params:
                _duration(self.params[key], f"params.{key}")
        _meta(self.calibrated, self.rationale, "FactorShock")


@dataclass(frozen=True)
class AftermathSpec:
    """What happens after the crash window."""

    mode: str = "analog"
    bear_annual_drift: float = -0.20
    recovery_fraction: tuple[float, float] = (0.5, 0.8)
    recovery_period: str = "9M"
    vol_half_life: str = "6W"
    calibrated: bool | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.mode not in AFTERMATH_MODES:
            raise SpecError(
                f"unknown aftermath mode {self.mode!r}, expected one of {AFTERMATH_MODES}"
            )
        lo, hi = _pair(self.recovery_fraction, "recovery_fraction", ordered=True)
        if not (0 < lo and hi <= 1.0):
            raise SpecError("recovery_fraction must lie within (0, 1]")
        object.__setattr__(self, "recovery_fraction", (lo, hi))
        _duration(self.recovery_period, "recovery_period")
        _duration(self.vol_half_life, "vol_half_life")
        _meta(self.calibrated, self.rationale, "aftermath")


@dataclass(frozen=True)
class ScenarioSpec:
    """A complete crisis scenario."""

    name: str
    variant: str = "base"
    label: str = "main_case"
    calibration_windows: tuple = ()
    crash_length: str = "3M"
    horizon: str = "2Y"
    onset: Any = "0D"
    anchors: dict = field(default_factory=dict)
    analog_window: tuple | None = None
    analog_mode: str = "exact"
    vol_multiplier: tuple[float, float] = (3.0, 4.0)
    factor_df: float = 3.0
    copula_df: tuple[float, float] = (10.0, 4.0)
    aftermath: AftermathSpec = field(default_factory=AftermathSpec)
    description: str = ""
    param_meta: dict = field(default_factory=dict)
    conditional_factors: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SpecError("name must be a non-empty string")
        if self.label not in LABELS:
            raise SpecError(f"unknown label {self.label!r}, expected one of {LABELS}")
        if self.analog_mode not in ANALOG_MODES:
            raise SpecError(
                f"unknown analog_mode {self.analog_mode!r}, expected one of {ANALOG_MODES}"
            )
        wins = tuple(_window(w, "calibration_windows") for w in self.calibration_windows)
        object.__setattr__(self, "calibration_windows", wins)
        if self.analog_window is not None:
            object.__setattr__(self, "analog_window", _window(self.analog_window, "analog_window"))
        _duration(self.crash_length, "crash_length")
        _duration(self.horizon, "horizon")
        if isinstance(self.onset, str):
            _duration(self.onset, "onset")
        else:
            if not isinstance(self.onset, (list, tuple)) or len(self.onset) != 2:
                raise SpecError("onset must be a duration or a (low, high) pair of durations")
            on = (_duration(self.onset[0], "onset"), _duration(self.onset[1], "onset"))
            if parse_duration(on[0]) > parse_duration(on[1]):
                raise SpecError("onset must be ordered (low, high)")
            object.__setattr__(self, "onset", on)
        object.__setattr__(
            self,
            "vol_multiplier",
            _pair(self.vol_multiplier, "vol_multiplier", positive=True, ordered=True),
        )
        if not isinstance(self.factor_df, (int, float)) or not self.factor_df > 2:
            raise SpecError(f"factor_df must be a number > 2, got {self.factor_df}")
        object.__setattr__(self, "factor_df", float(self.factor_df))
        cdf = _pair(self.copula_df, "copula_df", positive=True)
        if min(cdf) <= 2:
            raise SpecError(f"copula_df must be > 2, got {self.copula_df}")
        object.__setattr__(self, "copula_df", cdf)
        for k, v in self.anchors.items():
            if not isinstance(v, FactorShock):
                raise SpecError(f"anchor {k!r} must be a FactorShock")
        if not isinstance(self.metadata, dict):
            raise SpecError("metadata must be a mapping")
        cf = self.conditional_factors
        if isinstance(cf, str) or not all(isinstance(f, str) and f for f in cf):
            raise SpecError("conditional_factors must be a list of factor names")
        object.__setattr__(self, "conditional_factors", tuple(cf))
        clash = set(cf) & set(self.anchors)
        if clash:
            raise SpecError(
                f"conditional_factors {sorted(clash)} are also anchored; a factor is either "
                "anchored or generated conditionally"
            )
        bad = set(self.param_meta) - set(_META_KEYS)
        if bad:
            raise SpecError(f"param_meta: unknown key(s) {sorted(bad)}")
        for k, m in self.param_meta.items():
            _no_unknown(m, {"calibrated", "rationale"}, f"param_meta.{k}")
            _meta(m.get("calibrated"), m.get("rationale", ""), f"param_meta.{k}")

    @property
    def key(self) -> str:
        """Unique identifier, also the packaged file stem."""
        return f"{self.name}_{self.variant}"

    @property
    def content_hash(self) -> str:
        """SHA-256 of the scenario content, excluding name, variant and metadata.

        Two specs with the same numbers, windows and description hash equal even
        when they were built through different routes (alias versus compose).
        """
        d = _to_dict(self)
        for k in ("name", "variant", "metadata"):
            d.pop(k, None)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- (de)serialisation -------------------------------------------------------


def _shock_from_dict(name: str, d: Any) -> FactorShock:
    if not isinstance(d, dict):
        raise SpecError(f"anchor {name!r} must be a mapping")
    _no_unknown(
        d, {"cum_dist", "params", "aftermath_drift_scale", "calibrated", "rationale"},
        f"anchor {name!r}",
    )
    if "cum_dist" not in d:
        raise SpecError(f"anchor {name!r}: missing cum_dist")
    try:
        return FactorShock(
            cum_dist=d["cum_dist"],
            params=dict(d.get("params") or {}),
            aftermath_drift_scale=float(d.get("aftermath_drift_scale", 0.0)),
            calibrated=d.get("calibrated"),
            rationale=d.get("rationale", ""),
        )
    except SpecError as exc:
        raise SpecError(f"anchor {name!r}: {exc}") from exc


def _aftermath_from_dict(d: Any) -> AftermathSpec:
    if not isinstance(d, dict):
        raise SpecError("aftermath must be a mapping")
    allowed = {f for f in AftermathSpec.__dataclass_fields__}
    _no_unknown(d, allowed, "aftermath")
    kw = dict(d)
    if "recovery_fraction" in kw:
        kw["recovery_fraction"] = tuple(kw["recovery_fraction"])
    return AftermathSpec(**kw)


def _from_dict(data: dict) -> ScenarioSpec:
    allowed = set(ScenarioSpec.__dataclass_fields__)
    _no_unknown(data, allowed, "scenario")
    if "name" not in data:
        raise SpecError("scenario: missing required field 'name'")
    kw = dict(data)
    kw["anchors"] = {k: _shock_from_dict(k, v) for k, v in (data.get("anchors") or {}).items()}
    if "aftermath" in kw:
        kw["aftermath"] = _aftermath_from_dict(kw["aftermath"])
    kw["calibration_windows"] = tuple(tuple(w) if isinstance(w, list) else w
                                      for w in data.get("calibration_windows") or ())
    for key in ("analog_window", "onset", "vol_multiplier", "copula_df", "conditional_factors"):
        if isinstance(kw.get(key), list):
            kw[key] = tuple(kw[key])
    kw["param_meta"] = dict(data.get("param_meta") or {})
    kw["metadata"] = dict(data.get("metadata") or {})
    return ScenarioSpec(**kw)


def load_spec(source: str | Path | dict) -> ScenarioSpec:
    """Load a scenario from a YAML file path or an already-parsed mapping."""
    if isinstance(source, dict):
        return _from_dict(source)
    path = Path(source)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"cannot read scenario file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{path}: top level must be a mapping")
    return _from_dict(data)


def _shock_to_dict(s: FactorShock) -> dict:
    out: dict[str, Any] = {"cum_dist": s.cum_dist, "params": dict(s.params)}
    if s.aftermath_drift_scale:
        out["aftermath_drift_scale"] = s.aftermath_drift_scale
    if s.calibrated is not None:
        out["calibrated"] = s.calibrated
    if s.rationale:
        out["rationale"] = s.rationale
    return out


def _to_dict(spec: ScenarioSpec) -> dict:
    a = spec.aftermath
    aft: dict[str, Any] = {
        "mode": a.mode,
        "bear_annual_drift": a.bear_annual_drift,
        "recovery_fraction": list(a.recovery_fraction),
        "recovery_period": a.recovery_period,
        "vol_half_life": a.vol_half_life,
    }
    if a.calibrated is not None:
        aft["calibrated"] = a.calibrated
    if a.rationale:
        aft["rationale"] = a.rationale
    out: dict[str, Any] = {
        "name": spec.name,
        "variant": spec.variant,
        "label": spec.label,
        "description": spec.description,
        "calibration_windows": [list(w) for w in spec.calibration_windows],
        "crash_length": spec.crash_length,
        "horizon": spec.horizon,
        "onset": spec.onset if isinstance(spec.onset, str) else list(spec.onset),
        "anchors": {k: _shock_to_dict(v) for k, v in spec.anchors.items()},
        "analog_window": None if spec.analog_window is None else list(spec.analog_window),
        "analog_mode": spec.analog_mode,
        "vol_multiplier": list(spec.vol_multiplier),
        "factor_df": spec.factor_df,
        "copula_df": list(spec.copula_df),
        "aftermath": aft,
    }
    if spec.param_meta:
        out["param_meta"] = {k: dict(v) for k, v in spec.param_meta.items()}
    if spec.conditional_factors:
        out["conditional_factors"] = list(spec.conditional_factors)
    if spec.metadata:
        out["metadata"] = json.loads(json.dumps(spec.metadata, default=list))
    return out


def dump_spec(spec: ScenarioSpec, path: str | Path | None = None) -> str:
    """Serialise a scenario to YAML. Writes to ``path`` when given."""
    text = yaml.safe_dump(_to_dict(spec), sort_keys=False, default_flow_style=False,
                          allow_unicode=False, width=100)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
