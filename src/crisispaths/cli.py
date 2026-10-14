"""Command line interface (argparse only).

Commands
--------
``crisispaths scenarios [--family F [--grid]] ...``
    List scenario aliases and families, describe a family's axes, or print the
    grid report of a sweep.
``crisispaths run --prices FILE --strategy module:fn (--scenario NAME | --family F ...)``
    Run the Monte Carlo (resumable, parallel) and write ``REPORT.md``.
``crisispaths report DIR``
    Rebuild ``REPORT.md`` from an output directory.

Exit codes: 0 success; 1 runtime error (bad data, unknown scenario, pin
mismatch, sweep guard, ...); 2 usage error (argparse); 130 interrupted (partial
results are kept and the same command resumes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPT = 0, 1, 2, 130
DEFAULT_COST_MULTIPLIER = 3.0


class CliError(Exception):
    """A user-facing failure (exit code 1 unless ``code`` says otherwise)."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


def _kv(text: str, what: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key.strip() or not value.strip():
        raise argparse.ArgumentTypeError(
            f"{what} must look like {what_example(what)}, got {text!r}"
        )
    return key.strip(), value.strip()


def what_example(what: str) -> str:
    return "key=value1,value2" if what == "--axis" else "path=value"


def _axis(text: str) -> tuple[str, str]:
    return _kv(text, "--axis")


def _override(text: str) -> tuple[str, str]:
    return _kv(text, "--override")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crisispaths",
        description="Backtest a strategy under simulated crisis scenarios "
        "(conditional what-ifs, not forecasts).",
    )
    sub = p.add_subparsers(
        dest="command", required=True,
        metavar="{scenarios,run,report,verify-commentary}",
    )

    sc = sub.add_parser("scenarios", help="list scenarios, describe a family, print a grid report")
    sc.add_argument("--family", metavar="F", help="describe the axes of family F")
    sc.add_argument("--grid", metavar="F", help="print the grid report of family F "
                    "(use --axis key=v1,v2 to choose what is swept)")
    sc.add_argument("--axis", action="append", type=_axis, default=[], metavar="KEY=V1,V2",
                    help="axis choices for --grid (repeatable)")

    r = sub.add_parser("run", help="run a scenario or a sweep and write REPORT.md")
    r.add_argument("--prices", required=True, metavar="FILE",
                   help="csv or parquet of price levels; the first column is the date index")
    r.add_argument("--strategy", required=True, metavar="MODULE:FN",
                   help="importable strategy, prices -> weights (searched in the current "
                   "directory too)")
    r.add_argument("--scenario", metavar="NAME", help="a scenario alias or a family (defaults)")
    r.add_argument("--family", metavar="F", help="compose from family F with --axis/--override")
    r.add_argument("--axis", action="append", type=_axis, default=[], metavar="KEY=V1,V2",
                   help="axis choice(s); several values sweep the axis (repeatable)")
    r.add_argument("--override", action="append", type=_override, default=[],
                   metavar="PATH=VALUE", help="dotted-path override of the composed spec "
                   "(repeatable; the value is parsed as YAML)")
    r.add_argument("--factors", metavar="FILE",
                   help="csv or parquet of factor returns; omitted: download proxies "
                   "(needs the [data] extra and network)")
    r.add_argument("--calm-drift", choices=("zero", "historical"), default="zero",
                   help="calm-period bootstrap trend: zero removes each asset's calm mean "
                   "(default), historical keeps it")
    r.add_argument("--n-paths", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--costs-bps", type=float, default=None,
                   help="cost per unit of turnover in bps (REQUIRED: crisis costs are not zero)")
    r.add_argument("--cost-multiplier", type=float, default=DEFAULT_COST_MULTIPLIER,
                   help="crisis cost multiplier on --costs-bps (default %(default)s)")
    r.add_argument("--workers", type=int, default=None,
                   help="worker processes (default min(cpu-1, n-paths); 1 = in-process)")
    r.add_argument("--max-scenarios", type=int, default=64,
                   help="refuse sweeps with more distinct scenarios (default %(default)s)")
    r.add_argument("--out", required=True, metavar="DIR", help="output directory")
    r.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="resume finished paths found in --out (default); --no-resume restarts")
    r.set_defaults(_parser=r)

    rp = sub.add_parser("report", help="rebuild REPORT.md from an output directory")
    rp.add_argument("dir", metavar="DIR")
    rp.add_argument("--out", metavar="DIR", help="where to write (default: DIR)")
    rp.add_argument("--json", action="store_true", help="also (re)write summary.json")
    rp.add_argument("--pdf", action="store_true",
                    help="also write REPORT.pdf (needs the [pdf] extra)")
    rp.add_argument("--commentary", metavar="FILE.md",
                    help="append this commentary to the PDF after verify-commentary passes")
    rp.add_argument("--no-verify", action="store_true",
                    help="skip verify-commentary and stamp the commentary UNVERIFIED")
    rp.set_defaults(_parser=rp)

    vc = sub.add_parser(
        "verify-commentary",
        help="check a commentary's number traceability and framing against summary.json",
        description=(
            "Heuristic, deterministic check of a commentary: every decimal/percentage must "
            "trace to a value in summary.json, forbidden framings (expected return, forecast, "
            "scenario probabilities, ...) are rejected, conditionality must be acknowledged. "
            "It checks traceability and framing, not semantic truth. Exit 0 pass, 1 fail, "
            "2 usage."
        ),
    )
    vc.add_argument("commentary", metavar="COMMENTARY.md")
    vc.add_argument("--summary", required=True, metavar="summary.json")
    return p


# --- helpers -----------------------------------------------------------------------------------


def _load_frame(path: str, what: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in (".csv", ".parquet", ".pq"):
        raise CliError(f"{what} file {p.name!r} must be a .csv or .parquet file")
    if not p.exists():
        raise CliError(f"{what} file not found: {path}")
    try:
        if suffix == ".csv":
            df = pd.read_csv(p, index_col=0, parse_dates=True)
        else:
            df = pd.read_parquet(p)
    except ImportError as exc:
        raise CliError(f"cannot read {p.name}: {exc}") from exc
    if not isinstance(df.index, pd.DatetimeIndex):
        first = df.columns[0] if len(df.columns) else None
        try:
            df = df.set_index(pd.to_datetime(df[first])).drop(columns=[first])
        except Exception as exc:
            raise CliError(
                f"{what} file {p.name}: the first column must be a date index"
            ) from exc
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _split_axes(pairs) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise CliError(f"--axis {key} given twice; list the options as {key}=a,b")
        vals = [v.strip() for v in value.split(",") if v.strip()]
        out[key] = vals if len(vals) > 1 else vals[0]
    return out


def _parse_overrides(pairs) -> dict:
    out: dict = {}
    for path, value in pairs:
        try:
            out[path] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise CliError(f"--override {path}: cannot parse value {value!r}: {exc}") from exc
    return out


# --- commands ----------------------------------------------------------------------------------


def _cmd_scenarios(args) -> int:
    from .compose import describe_axes, grid_report, list_families
    from .library import list_scenarios

    if args.grid:
        axes = _split_axes(args.axis)
        rep = grid_report(args.grid, **axes)
        print(f"family: {args.grid}")
        print(f"combinations: {rep.n_total}")
        print(f"distinct scenarios: {rep.n_unique}")
        for g in rep.groups:
            if len(g) > 1:
                print(f"  merged (identical content): {g[0]} <- {list(g[1:])}")
        return EXIT_OK
    if args.family:
        d = describe_axes(args.family)
        print(f"family: {d['family']}")
        if d["description"]:
            print(d["description"].strip())
        for axis in d["axis_order"]:
            a = d["axes"][axis]
            print(f"\naxis {axis} (default: {a['default']})")
            if a["doc"]:
                print(f"  {' '.join(a['doc'].split())}")
            for name, o in a["options"].items():
                tag = " [default]" if o["is_default"] else ""
                req = f" (requires {o['requires']})" if o["requires"] else ""
                print(f"  - {name}{tag}: {o['summary']}{req}")
        return EXIT_OK
    names = list_scenarios()
    print(f"scenario aliases ({len(names)}):")
    for n in names:
        print(f"  {n}")
    fams = list_families()
    print(f"families ({len(fams)}): {', '.join(fams)}")
    print("describe one with: crisispaths scenarios --family NAME")
    return EXIT_OK


def _cmd_run(args) -> int:
    from .compose import enumerate_scenarios
    from .report import write_report
    from .runner import run_mc, run_sweep

    if bool(args.scenario) == bool(args.family):
        args._parser.error("give exactly one of --scenario NAME or --family F")
    if args.costs_bps is None:
        args._parser.error(
            "--costs-bps is required: crisis costs are not zero "
            "(pass --costs-bps 0 to run without costs on purpose)"
        )
    if args.scenario and (args.axis or args.override):
        args._parser.error("--axis and --override need --family, not --scenario")
    if args.n_paths < 1:
        args._parser.error("--n-paths must be >= 1")

    prices = _load_frame(args.prices, "prices")
    factors = _load_frame(args.factors, "factors") if args.factors else None
    source = None
    if factors is None:
        from .data import YFinanceFredSource

        source = YFinanceFredSource()
    sys.path.insert(0, os.getcwd())
    out = Path(args.out)
    common = dict(
        n_paths=args.n_paths, seed=args.seed, out_dir=out, workers=args.workers,
        costs_bps=args.costs_bps, cost_multiplier=args.cost_multiplier, resume=args.resume,
        factors=factors, source=source, calm_drift=args.calm_drift,
    )
    if sys.stderr.isatty():
        def progress(key, done, total):
            print(f"\r{key}: {done}/{total}", end="", file=sys.stderr, flush=True)
            if done == total:
                print(file=sys.stderr)
        common["progress"] = progress

    if args.scenario:
        table = run_mc(prices, args.strategy, args.scenario, **common)
    else:
        axes = _split_axes(args.axis)
        overrides = _parse_overrides(args.override)
        swept = any(isinstance(v, list) for v in axes.values())
        if swept:
            table = run_sweep(prices, args.strategy, args.family, axes, overrides=overrides,
                              max_scenarios=args.max_scenarios, **common)
        else:
            specs = [s for _, s in enumerate_scenarios(args.family, overrides=overrides, **axes)]
            table = run_mc(prices, args.strategy, specs, **common)
    path = write_report(table, out)
    from .summary import write_summary

    write_summary(table, out)
    for key, g in table.groupby("scenario", sort=False):
        print(f"{key}: {len(g)} paths, median terminal wealth "
              f"{g['terminal_wealth'].median():.3f}, P(ruin) {g['ruined'].mean():.1%}")
    print(f"report: {path}")
    return EXIT_OK


def _cmd_report(args) -> int:
    from .report import write_report

    src = Path(args.dir)
    if args.commentary and not args.pdf:
        args._parser.error("--commentary needs --pdf")
    out = Path(args.out) if args.out else src
    path = write_report(src, out)
    print(f"report: {path}")
    from .summary import build_summary, write_summary

    if args.json or args.pdf:
        sp = write_summary(src, out)
        if args.json:
            print(f"summary: {sp}")
    if args.pdf:
        text = None
        unverified = False
        if args.commentary:
            cf = Path(args.commentary)
            if not cf.exists():
                raise CliError(f"commentary file not found: {args.commentary}", EXIT_USAGE)
            text = cf.read_text(encoding="utf-8")
            if args.no_verify:
                unverified = True
            else:
                from .verify import verify_commentary

                res = verify_commentary(text, build_summary(src))
                if not res.passed:
                    print(res.report())
                    print("crisispaths: error: commentary failed verification; no PDF written "
                          "(fix it, or pass --no-verify to stamp it UNVERIFIED)",
                          file=sys.stderr)
                    return EXIT_ERROR
        try:
            from .pdf import write_pdf

            pdf = write_pdf(path, out / "REPORT.pdf", commentary=text, unverified=unverified)
        except ImportError as exc:
            raise CliError(
                f"PDF output needs the [pdf] extra: pip install 'crisispaths[pdf]' ({exc})",
                EXIT_USAGE,
            ) from exc
        print(f"pdf: {pdf}")
    return EXIT_OK


def _load_json(path: str, what: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise CliError(f"{what} not found: {path}", EXIT_USAGE)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"{what} is not valid JSON: {exc}", EXIT_USAGE) from exc


def _cmd_verify(args) -> int:
    from .verify import verify_commentary

    cp_ = Path(args.commentary)
    if not cp_.exists():
        raise CliError(f"commentary file not found: {args.commentary}", EXIT_USAGE)
    res = verify_commentary(cp_.read_text(encoding="utf-8"),
                            _load_json(args.summary, "summary"))
    print(res.report())
    return EXIT_OK if res.passed else EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    handler = {"scenarios": _cmd_scenarios, "run": _cmd_run, "report": _cmd_report,
               "verify-commentary": _cmd_verify}[args.command]
    try:
        return handler(args)
    except SystemExit as exc:  # parser.error inside a command
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    except KeyboardInterrupt:
        print("crisispaths: interrupted; finished paths are kept, run the same command to "
              "resume", file=sys.stderr)
        return EXIT_INTERRUPT
    except (CliError, ValueError, KeyError, OSError, ImportError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"crisispaths: error: {msg}", file=sys.stderr)
        return exc.code if isinstance(exc, CliError) else EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
