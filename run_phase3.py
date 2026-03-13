#!/usr/bin/env python3
"""
Phase 3 Runner - BTC Futures Curve Trading (Part 2)

Entry point for the BTC futures curve trading pipeline.
Invokes the master orchestrator in strategies/futures_curve/phase3run.py.

Usage:
    python run_phase3.py                     # Full pipeline (default)
    python run_phase3.py --mode full         # Full pipeline with reports
    python run_phase3.py --mode full_backtest # Comprehensive backtest
    python run_phase3.py --mode walk_forward # Walk-forward optimization
    python run_phase3.py --mode verify       # Module verification

Modules:
    Core types, term structure analysis, funding rate analysis,
    4 strategy modules (calendar, cross-venue, synthetic, roll),
    backtest engine, walk-forward optimizer, orchestrators,
    computation acceleration, and report generation.

Part 2 compliance:
    6 venues (Binance, Deribit, CME, Hyperliquid, dYdX, GMX)
    4 strategies with walk-forward validation (18m train / 6m test)
    60+ performance metrics, crisis event analysis
"""

import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# ANSI formatting helpers
# ---------------------------------------------------------------------------

class _Fmt:
    """ANSI escape codes for terminal formatting."""
    RESET   = '\033[0m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    BLUE    = '\033[94m'
    CYAN    = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE   = '\033[97m'
    GRAY    = '\033[90m'

    # Progress bar characters
    FILL  = '\u2588'  # Full block
    LIGHT = '\u2591'  # Light shade
    MID   = '\u2593'  # Dark shade

    @staticmethod
    def enabled() -> bool:
        return sys.stdout.isatty()

    @classmethod
    def c(cls, text: str, *codes: str) -> str:
        if not cls.enabled():
            return text
        prefix = ''.join(codes)
        return f"{prefix}{text}{cls.RESET}"


def _hr(char: str = '\u2500', width: int = 72) -> str:
    """Horizontal rule."""
    return char * width


def _box_line(text: str, width: int = 72) -> str:
    """Line inside a box."""
    inner = width - 4
    return f"\u2502 {text:<{inner}} \u2502"


def _box_top(width: int = 72) -> str:
    return '\u250C' + '\u2500' * (width - 2) + '\u2510'


def _box_bottom(width: int = 72) -> str:
    return '\u2514' + '\u2500' * (width - 2) + '\u2518'


def _box_sep(width: int = 72) -> str:
    return '\u251C' + '\u2500' * (width - 2) + '\u2524'


# ---------------------------------------------------------------------------
# Animated spinner for long operations
# ---------------------------------------------------------------------------

class Spinner:
    """Animated terminal spinner for indicating activity."""

    FRAMES = ['\u280B', '\u2819', '\u2839', '\u2838',
              '\u283C', '\u2834', '\u2826', '\u2827',
              '\u2807', '\u280F']

    def __init__(self, message: str = "Processing"):
        self.message = message
        self._running = False
        self._thread = None
        self._start_time = None

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, final_message: str = None):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        elapsed = time.time() - self._start_time if self._start_time else 0
        msg = final_message or self.message
        sys.stdout.write(f"\r{_Fmt.c('[OK]', _Fmt.GREEN, _Fmt.BOLD)} {msg} "
                         f"{_Fmt.c(f'({_format_elapsed(elapsed)})', _Fmt.DIM)}\n")
        sys.stdout.flush()

    def fail(self, error_msg: str = "Failed"):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        sys.stdout.write(f"\r{_Fmt.c('[!!]', _Fmt.RED, _Fmt.BOLD)} {error_msg}\n")
        sys.stdout.flush()

    def _spin(self):
        idx = 0
        while self._running:
            elapsed = time.time() - self._start_time
            frame = self.FRAMES[idx % len(self.FRAMES)]
            line = (f"\r{_Fmt.c(frame, _Fmt.CYAN)} {self.message} "
                    f"{_Fmt.c(f'[{_format_elapsed(elapsed)}]', _Fmt.DIM)}")
            sys.stdout.write(line)
            sys.stdout.flush()
            idx += 1
            time.sleep(0.10)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ---------------------------------------------------------------------------
# Banner and summary display
# ---------------------------------------------------------------------------

def _print_banner(mode: str, capital: float, output_dir: str):
    """Print a styled startup banner."""
    W = 72
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print(_Fmt.c(_box_top(W), _Fmt.BLUE))
    print(_Fmt.c(_box_line("", W), _Fmt.BLUE))
    print(_Fmt.c(_box_line(
        _Fmt.c("BTC Futures Curve Trading", _Fmt.BOLD, _Fmt.WHITE)
        + "  " + _Fmt.c("Part 2", _Fmt.DIM), W), _Fmt.BLUE))
    print(_Fmt.c(_box_line("", W), _Fmt.BLUE))
    print(_Fmt.c(_box_sep(W), _Fmt.BLUE))
    print(_Fmt.c(_box_line(f"Mode        {_Fmt.c(mode.upper(), _Fmt.BOLD)}", W), _Fmt.BLUE))
    print(_Fmt.c(_box_line(f"Capital     {_Fmt.c(f'${capital:,.0f}', _Fmt.GREEN)}", W), _Fmt.BLUE))
    print(_Fmt.c(_box_line(f"Output      {output_dir}", W), _Fmt.BLUE))
    print(_Fmt.c(_box_line(f"Started     {now}", W), _Fmt.BLUE))
    print(_Fmt.c(_box_sep(W), _Fmt.BLUE))

    venues = "Binance  CME  Deribit  Hyperliquid  dYdX  GMX"
    strategies = "Calendar  Cross-Venue  Synthetic  Roll"
    print(_Fmt.c(_box_line(f"Venues      {_Fmt.c(venues, _Fmt.DIM)}", W), _Fmt.BLUE))
    print(_Fmt.c(_box_line(f"Strategies  {_Fmt.c(strategies, _Fmt.DIM)}", W), _Fmt.BLUE))
    print(_Fmt.c(_box_line("", W), _Fmt.BLUE))
    print(_Fmt.c(_box_bottom(W), _Fmt.BLUE))
    print()


def _print_results(results, elapsed_total: float):
    """Print formatted results summary."""
    W = 72

    print()
    print(_Fmt.c(_hr('\u2550', W), _Fmt.GREEN))
    print(_Fmt.c("  EXECUTION COMPLETE", _Fmt.BOLD, _Fmt.GREEN))
    print(_Fmt.c(_hr('\u2550', W), _Fmt.GREEN))
    print()

    summary = results.summary if hasattr(results, 'summary') else {}
    perf = summary.get('performance', summary)

    # Key metrics
    metrics = [
        ("Total Return",    perf.get('total_return_pct', 'N/A'),    '%'),
        ("Sharpe Ratio",    perf.get('sharpe_ratio', 'N/A'),        ''),
        ("Max Drawdown",    perf.get('max_drawdown_pct', 'N/A'),    '%'),
        ("BTC Correlation", perf.get('btc_correlation', 'N/A'),     ''),
        ("Win Rate",        perf.get('win_rate_pct', 'N/A'),        '%'),
        ("Total Trades",    perf.get('total_trades', 'N/A'),        ''),
    ]

    print(f"  {'Metric':<20} {'Value':>12}")
    print(f"  {_hr('-', 34)}")
    for label, value, unit in metrics:
        if isinstance(value, float):
            val_str = f"{value:.2f}{unit}"
        else:
            val_str = str(value)
        color = _Fmt.GREEN if label == "Total Return" else _Fmt.WHITE
        print(f"  {label:<20} {_Fmt.c(val_str, color, _Fmt.BOLD):>12}")

    print()
    print(f"  Elapsed: {_Fmt.c(_format_elapsed(elapsed_total), _Fmt.CYAN)}")
    print(_Fmt.c(_hr('\u2550', W), _Fmt.GREEN))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for Phase 3."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Phase 3: BTC Futures Curve Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_phase3.py                        # Full pipeline (default)
  python run_phase3.py --mode full            # Full pipeline with reports
  python run_phase3.py --mode full_backtest   # Comprehensive backtest
  python run_phase3.py --mode walk_forward    # Walk-forward optimization
  python run_phase3.py --mode verify          # Verify all modules
  python run_phase3.py --capital 5000000      # $5M initial capital
        """
    )
    parser.add_argument('--mode', '-m', default='full',
                        choices=['verify', 'full_backtest', 'walk_forward',
                                 'analysis_only', 'full'],
                        help='Execution mode (default: full)')
    parser.add_argument('--capital', '-c', type=float, default=1_000_000,
                        help='Initial capital in USD (default: 1,000,000)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (ISO format: 2022-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (ISO format: 2024-01-01)')
    parser.add_argument('--output-dir', '-o', type=str, default='output/phase3',
                        help='Output directory (default: output/phase3)')

    args = parser.parse_args()

    _print_banner(args.mode, args.capital, args.output_dir)

    # Import orchestrator
    sp = Spinner("Loading orchestrator modules")
    sp.start()
    try:
        from strategies.futures_curve.phase3run import run_phase3, Phase3Mode
        sp.stop("Orchestrator modules loaded")
    except ImportError as e:
        sp.fail(f"Import failed: {e}")
        print("\n  Ensure you are running from the project root directory.\n")
        return 1

    # Parse dates
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None

    # Execute pipeline
    t0 = time.time()
    try:
        results = run_phase3(
            mode=args.mode,
            initial_capital=args.capital,
            start_date=start_date,
            end_date=end_date,
            output_dir=args.output_dir
        )

        elapsed = time.time() - t0
        _print_results(results, elapsed)

        # Save summary
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / 'execution_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results.summary, f, indent=2, default=str)
        print(f"  Summary saved to: {_Fmt.c(str(summary_file), _Fmt.CYAN)}")
        print()
        return 0

    except Exception as e:
        elapsed = time.time() - t0
        print()
        print(_Fmt.c(_hr('\u2550', 72), _Fmt.RED))
        print(_Fmt.c("  EXECUTION FAILED", _Fmt.BOLD, _Fmt.RED))
        print(_Fmt.c(_hr('\u2550', 72), _Fmt.RED))
        print(f"\n  Error: {e}")
        print(f"  Elapsed: {_format_elapsed(elapsed)}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
