"""Named scenarios: thin aliases defined as compositions of a family."""

from __future__ import annotations

import dataclasses

from .compose import compose
from .spec import ScenarioSpec

# alias -> (family, name, variant, axis choices). The key of the resulting spec
# is "<name>_<variant>", which equals the alias.
_US_FULL = {"complexity": "full", "depth": "base", "crash_speed": "base"}
_AI_STD = {"complexity": "standard", "crash_source": "analog_exact", "tech_spread": "as_2000",
           "depth": "as_2000"}


def _build_aliases() -> dict[str, tuple[str, dict]]:
    out: dict[str, tuple[str, dict]] = {
        "ai_bubble_worst": ("ai_bubble", {**_AI_STD, "aftermath": "analog"}),
        "ai_bubble_recovery": ("ai_bubble", {**_AI_STD, "aftermath": "v_recovery"}),
    }
    for usd in ("debasement", "haven"):
        for aftermath in ("prolonged_bear", "v_recovery"):
            for suffix, rc in (("", "shock"), ("_history_consistent", "history_consistent")):
                out[f"us_bankruptcy_{usd}_{aftermath}{suffix}"] = (
                    "us_bankruptcy",
                    {**_US_FULL, "usd": usd, "aftermath": aftermath, "rates_credit": rc},
                )
    return out


ALIASES: dict[str, tuple[str, dict]] = _build_aliases()


def list_scenarios() -> list[str]:
    """Names of the named scenario aliases, sorted."""
    return sorted(ALIASES)


def load_scenario(name: str) -> ScenarioSpec:
    """Load a named scenario, for example ``'ai_bubble_worst'``.

    Each name is an alias for ``compose(family, **choices)``; its spec carries
    the same provenance metadata as the composed scenario.
    """
    if name not in ALIASES:
        raise KeyError(f"unknown scenario {name!r}; available: {list_scenarios()}")
    family, choices = ALIASES[name]
    spec = compose(family, **choices)
    family_name = spec.name
    variant = name[len(family_name) + 1:]
    return dataclasses.replace(spec, name=family_name, variant=variant)
