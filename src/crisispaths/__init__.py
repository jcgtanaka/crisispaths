"""crisispaths: backtest any strategy under simulated crisis scenarios."""

from .backtest import BacktestResult, backtest, calibration_hash
from .calibrate import Calibration, calibrate
from .clock import Clock
from .compose import (
    Family,
    GridReport,
    ScenarioWarning,
    compose,
    describe_axes,
    enumerate_scenarios,
    grid_report,
    grid_size,
    list_families,
    load_family,
)
from .disaggregate import (
    IntradayProfile,
    aggregate_log_returns,
    disaggregate,
    estimate_intraday_profile,
)
from .generate import GeneratedPath, generate_path, generate_paths
from .harness import (
    Strategy,
    StrategyResult,
    buy_and_hold_equal_weight,
    run_strategy,
)
from .library import list_scenarios, load_scenario
from .lookahead import LookaheadError, LookaheadReport, check_lookahead
from .metrics import (
    Summary,
    bootstrap_ci,
    kaplan_meier,
    path_metrics,
    relative_metrics,
    summarize,
)
from .report import write_report
from .runner import PinMismatchError, run_mc, run_sweep
from .spec import AftermathSpec, FactorShock, ScenarioSpec, SpecError, dump_spec, load_spec

__version__ = "0.1.0"

__all__ = [
    "AftermathSpec",
    "BacktestResult",
    "Calibration",
    "Clock",
    "FactorShock",
    "Family",
    "GeneratedPath",
    "GridReport",
    "IntradayProfile",
    "LookaheadError",
    "LookaheadReport",
    "PinMismatchError",
    "ScenarioSpec",
    "ScenarioWarning",
    "SpecError",
    "Strategy",
    "StrategyResult",
    "Summary",
    "aggregate_log_returns",
    "backtest",
    "bootstrap_ci",
    "buy_and_hold_equal_weight",
    "calibrate",
    "calibration_hash",
    "check_lookahead",
    "compose",
    "describe_axes",
    "disaggregate",
    "dump_spec",
    "estimate_intraday_profile",
    "enumerate_scenarios",
    "generate_path",
    "generate_paths",
    "grid_report",
    "grid_size",
    "kaplan_meier",
    "list_families",
    "list_scenarios",
    "load_family",
    "load_scenario",
    "load_spec",
    "path_metrics",
    "relative_metrics",
    "run_mc",
    "run_strategy",
    "run_sweep",
    "summarize",
    "write_report",
]
