# Contributing

## Setup

```bash
pip install -e ".[dev,data]"   # Python 3.12 or newer
ruff check .
pytest
```

Tests are offline by default (`-m 'not network'`). Write the failing test
first, then the code. Keep user-facing text in English and do not commit data
downloaded from a vendor.

## Add a scenario family

A family is one YAML file in `src/crisispaths/families/`; the file stem is the
family name and it is picked up by `list_families()` automatically.

```yaml
name: my_family
description: One sentence.
description_suffix: ""        # appended after the chosen options
base:                         # a ScenarioSpec without name/variant/metadata
  calibration_windows: [["2008-09-01", "2009-03-09"]]
  crash_length: 3M
  horizon: 2Y
  anchors:
    EQ:
      cum_dist: lognormal_loss
      params: {median: 0.4, sigma: 0.276, cap: 0.95}
      calibrated: false
      rationale: Why this number, and that it is assumed, not estimated.
axes:
  depth:
    default: base
    doc: How deep the crash is.
    options:
      base: {summary: median loss 40%}
      severe:
        summary: median loss 55%
        patch:
          anchors: {EQ: {params: {median: 0.55}}}
```

Rules to respect (they are enforced at load time and by `_smoke`):

- Every option needs a non-empty `summary`; an option patch is merged
  recursively onto the base, later axes win, `rationale` accumulates and
  `calibrated` is the logical AND.
- Anything with `calibrated: false` must carry a `rationale`.
- `requires: {axis: [options]}` skips an option when the final choices do not
  meet it; give a `fallback` if the base needs a replacement.
- `patches: [{when_anchor: NAME, patch: ...}]` applies a patch only when that
  anchor exists.
- Axis names must not be `family` or `overrides`.

For a private family use `cp.load_family("my_family.yaml")` and pass the
`Family` to `compose`, `enumerate_scenarios`, `run_sweep`.

## Add an axis or an option to an existing family

1. Add the axis under `axes:` in the family file. Axis order is application
   order, so put it after the axes it must override.
2. Update the axis tables in `README.md` and the grid size in the family
   description if you quote one.
3. Add or extend a test in `tests/test_compose.py` (the defaults and every
   single option are composed once at load time, so a broken patch fails
   immediately).
4. If a named alias should exist, add it to `src/crisispaths/library.py`.
5. Note the change in `CHANGELOG.md`.

## Add a factor or a report section

Factors live in `src/crisispaths/factors.py` (`FACTOR_REGISTRY`); report
sections in `src/crisispaths/report.py`. The report order and the "no figure
pooled across scenarios" rule are tested in `tests/test_report.py`; keep them.

## Style

Python 3.10+, line length 100, `ruff` rules E, F, I, B, UP. No absolute paths
and no private references in repository files.
