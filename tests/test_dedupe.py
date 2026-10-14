import warnings

import pytest

from crisispaths import ScenarioWarning, describe_axes, enumerate_scenarios, grid_report

COUNTS = [("us_bankruptcy", 486, 297), ("ai_bubble", 144, 144)]


def full_axes(family):
    return {a: list(v["options"]) for a, v in describe_axes(family)["axes"].items()}


def run(family, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScenarioWarning)
        return list(enumerate_scenarios(family, **full_axes(family), **kw))


@pytest.mark.parametrize("family, total, unique", COUNTS)
def test_grid_report_numbers(family, total, unique):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScenarioWarning)
        rep = grid_report(family, **full_axes(family))
    assert rep.n_total == total
    assert rep.n_unique == unique
    assert sum(len(g) for g in rep.groups) == total
    assert len(rep.groups) == unique
    assert all(isinstance(c, dict) for g in rep.groups for c in g)


@pytest.mark.parametrize("family, total, unique", COUNTS)
def test_enumerate_dedupes_by_default(family, total, unique):
    out = run(family)
    assert len(out) == unique
    assert len({s.content_hash for _, s in out}) == unique
    assert len(run(family, dedupe=False)) == total


def test_merged_choices_recorded_on_kept_spec_and_first_is_kept():
    out = run("us_bankruptcy")
    merged = [(c, s) for c, s in out if s.metadata.get("merged_choices")]
    assert merged
    raw = run("us_bankruptcy", dedupe=False)
    by_hash = {}
    for c, s in raw:
        by_hash.setdefault(s.content_hash, []).append(c)
    for c, s in merged:
        group = by_hash[s.content_hash]
        assert group[0] == c  # first in declaration order is kept
        assert s.metadata["merged_choices"] == group[1:]
    # unmerged specs carry no merged_choices key
    singles = [s for _, s in out if len(by_hash[s.content_hash]) == 1]
    assert singles and all("merged_choices" not in s.metadata for s in singles)
