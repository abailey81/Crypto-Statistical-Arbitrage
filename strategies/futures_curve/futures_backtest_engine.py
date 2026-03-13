"""
Futures Backtest Engine - Phase 3
=================================

Part 2 Section 3.3 - Backtesting & Analysis Framework

Backtest engine for BTC futures curve trading strategies (A, B, C, D)
with 60+ performance and risk metrics.

Covers:
- Multi-strategy backtesting with venue-specific cost modeling
- Crisis event analysis (COVID, May 2021, Luna, FTX collapse)
- Regime-conditional performance breakdown
- 60+ metrics across performance, risk, trade stats, and costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import json
import logging
import math
from collections import defaultdict
import scipy.stats as stats

from . import (
    VenueType, TermStructureRegime, SpreadDirection, ExitReason,
    VenueCosts, DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY
)
from .term_structure import TermStructureAnalyzer
from .step4_futures_orchestrator import FuturesOrchestrator, OrchestratorConfig

logger = logging.getLogger(__name__)


class CrisisEvent(Enum):
    """Crisis events for analysis per Part 2 Section 3.3."""
    COVID_CRASH = "covid_crash"                 # March 2020 COVID-19 crash
    MAY_2021_CRASH = "may_2021_crash"           # May 19, 2021 - BTC -30%
    LUNA_COLLAPSE = "luna_collapse"             # May 2022 Terra/Luna
    THREE_AC_LIQUIDATION = "3ac_liquidation"    # June 2022 3AC liquidation
    FTX_COLLAPSE = "ftx_collapse"              # Nov 2022 FTX collapse


# Crisis event date ranges (UTC)
CRISIS_DATES = {
    CrisisEvent.COVID_CRASH: (datetime(2020, 3, 9, tzinfo=timezone.utc), datetime(2020, 3, 23, tzinfo=timezone.utc)),
    CrisisEvent.MAY_2021_CRASH: (datetime(2021, 5, 12, tzinfo=timezone.utc), datetime(2021, 5, 25, tzinfo=timezone.utc)),
    CrisisEvent.LUNA_COLLAPSE: (datetime(2022, 5, 7, tzinfo=timezone.utc), datetime(2022, 5, 15, tzinfo=timezone.utc)),
    CrisisEvent.THREE_AC_LIQUIDATION: (datetime(2022, 6, 13, tzinfo=timezone.utc), datetime(2022, 6, 20, tzinfo=timezone.utc)),
    CrisisEvent.FTX_COLLAPSE: (datetime(2022, 11, 6, tzinfo=timezone.utc), datetime(2022, 11, 21, tzinfo=timezone.utc)),
}


@dataclass
class Trade:
    """Record of a single trade."""
    trade_id: str
    strategy: str
    venue: str
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: SpreadDirection
    size_btc: float
    entry_price: float
    exit_price: Optional[float]
    pnl_usd: float
    pnl_pct: float
    fees_usd: float
    slippage_bps: float
    holding_period_hours: float
    exit_reason: Optional[ExitReason]
    regime_at_entry: TermStructureRegime
    is_open: bool = True


@dataclass
class DailyMetrics:
    """Daily performance metrics."""
    date: datetime
    equity: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    num_trades: int
    gross_pnl: float
    net_pnl: float
    fees: float
    position_btc: float
    regime: TermStructureRegime


@dataclass
class BacktestMetrics:
    """Backtest metrics (60+ metrics)."""
    # Basic Performance (10 metrics)
    total_return_pct: float
    annualized_return_pct: float
    total_pnl_usd: float
    gross_profit_usd: float
    gross_loss_usd: float
    net_profit_usd: float
    profit_factor: float
    expectancy_usd: float
    expectancy_pct: float
    avg_trade_pnl_usd: float

    # Risk Metrics (15 metrics)
    max_drawdown_pct: float
    max_drawdown_usd: float
    max_drawdown_duration_days: int
    avg_drawdown_pct: float
    calmar_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    omega_ratio: float
    information_ratio: float
    treynor_ratio: float
    var_95_pct: float
    var_99_pct: float
    cvar_95_pct: float
    tail_ratio: float
    volatility_annual_pct: float

    # Trade Statistics (15 metrics)
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_usd: float
    avg_loss_usd: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_usd: float
    largest_loss_usd: float
    avg_holding_hours: float
    median_holding_hours: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    payoff_ratio: float

    # Time-based Metrics (10 metrics)
    time_in_market_pct: float
    avg_trades_per_day: float
    avg_trades_per_week: float
    best_day_return_pct: float
    worst_day_return_pct: float
    best_week_return_pct: float
    worst_week_return_pct: float
    best_month_return_pct: float
    worst_month_return_pct: float
    positive_days_pct: float

    # Venue/Strategy Breakdown (variable)
    venue_breakdown: Dict[str, Dict[str, float]]
    strategy_breakdown: Dict[str, Dict[str, float]]

    # Regime Analysis (variable)
    regime_performance: Dict[str, Dict[str, float]]

    # Crisis Analysis (variable)
    crisis_performance: Dict[str, Dict[str, float]]

    # Cost Analysis (5 metrics)
    total_fees_usd: float
    total_slippage_usd: float
    avg_slippage_bps: float
    fees_as_pct_of_gross: float
    break_even_win_rate_pct: float

    # Extended Metrics (5 metrics)
    skewness: float
    kurtosis: float
    ulcer_index: float
    recovery_factor: float
    common_sense_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                result[key] = value
            elif isinstance(value, (int, float, str, bool, type(None))):
                result[key] = value
            else:
                result[key] = str(value)
        return result

    def get_summary(self) -> Dict[str, float]:
        """Get key summary metrics."""
        return {
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate_pct': self.win_rate_pct,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades
        }


@dataclass
class BacktestResult:
    """Complete backtest result."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    metrics: BacktestMetrics
    trades: List[Trade]
    daily_metrics: pd.DataFrame
    equity_curve: pd.DataFrame
    parameters: Dict[str, Any]


class FuturesBacktestEngine:
    """
    Backtest engine for BTC futures curve trading.

    Supports multi-strategy backtesting with venue-specific cost modeling,
    crisis event analysis, regime-conditional performance, and 60+ metrics.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        venue_costs: Optional[Dict[str, VenueCosts]] = None,
        include_funding: bool = True,
        include_slippage: bool = True
    ):
        self.initial_capital = initial_capital
        self.venue_costs = venue_costs or DEFAULT_VENUE_COSTS
        self.include_funding = include_funding
        self.include_slippage = include_slippage

        self.trades: List[Trade] = []
        self.daily_metrics: List[DailyMetrics] = []
        self.equity_curve: List[Dict] = []

    def run_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        orchestrator: FuturesOrchestrator,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """
        Run full backtest.

        Args:
            historical_data: Dict mapping venue to DataFrame
            orchestrator: Configured FuturesOrchestrator
            start_date: Backtest start
            end_date: Backtest end
        """
        # Reset state
        self.trades = []
        self.daily_metrics = []
        self.equity_curve = []

        # Run orchestrator backtest
        results = orchestrator.run_backtest(historical_data, start_date, end_date)

        # Build trades from executions with equity curve for price data
        self._build_trades_from_executions(orchestrator.execution_history, results['equity_curve'])

        # Calculate all metrics
        metrics = self._calculate_metrics(
            initial_capital=self.initial_capital,
            final_capital=results['equity_curve']['equity'].iloc[-1] if len(results['equity_curve']) > 0 else self.initial_capital,
            equity_curve=results['equity_curve'],
            historical_data=historical_data
        )

        # Use reconstructed equity if available (from realized trade PnL)
        equity_for_result = getattr(self, '_reconstructed_equity', results['equity_curve'])

        return BacktestResult(
            strategy_name='multi_strategy',
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=metrics.total_pnl_usd + self.initial_capital,
            metrics=metrics,
            trades=self.trades,
            daily_metrics=pd.DataFrame(self.daily_metrics) if self.daily_metrics else pd.DataFrame(),
            equity_curve=equity_for_result,
            parameters=orchestrator.config.__dict__
        )

    def _build_trades_from_executions(self, executions: List, equity_curve: pd.DataFrame = None) -> None:
        """Build trade records from orchestrator executions with actual PnL calculation.

        Pairs entries with exits and calculates PnL based on price differences.
        """
        # Get price history from equity curve if available
        price_history = {}
        price_history_naive = {}  # For matching timezone-naive timestamps
        if equity_curve is not None and 'btc_price' in equity_curve.columns:
            for _, row in equity_curve.iterrows():
                if 'timestamp' in row and 'btc_price' in row:
                    ts = row['timestamp']
                    price = row['btc_price']
                    price_history[ts] = price
                    # Also store with naive timestamp for flexible matching
                    if hasattr(ts, 'tz_localize'):
                        try:
                            naive_ts = ts.tz_localize(None) if ts.tzinfo else ts
                            price_history_naive[naive_ts] = price
                        except:
                            pass
                    # Store as string key too for flexible matching
                    price_history[str(ts)[:19]] = price  # YYYY-MM-DD HH:MM:SS

        # Track open positions by strategy/venue
        open_positions = {}

        for exec in executions:
            # For multi-venue trades (cross_venue, synthetic_futures), use full venue pair as key
            if '_' in exec.signal.venue:
                key = (exec.strategy, exec.signal.venue)
            else:
                key = (exec.strategy, exec.signal.venue.split('_')[0])
            direction = exec.signal.direction

            # Get current price - try multiple methods
            current_price = exec.execution_price
            if current_price is None or current_price == 0:
                exec_ts = exec.timestamp
                # Try exact match first
                if exec_ts in price_history:
                    current_price = price_history[exec_ts]
                # Try string-based lookup
                elif str(exec_ts)[:19] in price_history:
                    current_price = price_history[str(exec_ts)[:19]]
                # Try naive timestamp
                elif hasattr(exec_ts, 'tz_localize') and price_history_naive:
                    try:
                        naive_ts = exec_ts.tz_localize(None) if exec_ts.tzinfo else exec_ts
                        if naive_ts in price_history_naive:
                            current_price = price_history_naive[naive_ts]
                    except:
                        pass
                # Fallback to nearest price
                if (current_price is None or current_price == 0) and price_history:
                    current_price = list(price_history.values())[-1]
                elif current_price is None or current_price == 0:
                    current_price = 1.0  # Minimal fallback

            # Check if this closes an existing position
            if key in open_positions:
                open_trade = open_positions[key]
                # This execution closes the position
                entry_price = open_trade['entry_price']
                entry_time = open_trade['entry_time']
                size_btc = open_trade['size_btc']
                entry_direction = open_trade['direction']

                # Calculate PnL based on direction
                # SpreadDirection.LONG.value = 1, SHORT.value = -1
                if entry_direction.value == 1 or str(entry_direction.name) == 'LONG':
                    pnl_usd = (current_price - entry_price) * size_btc
                else:
                    pnl_usd = (entry_price - current_price) * size_btc

                pnl_pct = (pnl_usd / (entry_price * size_btc)) * 100 if entry_price > 0 else 0

                # PDF: 5% basis stop-loss - exit if basis moves against position by >5%
                stop_loss_triggered = False
                if pnl_pct < -5.0:
                    pnl_pct = -5.0
                    pnl_usd = -0.05 * entry_price * size_btc
                    stop_loss_triggered = True

                holding_hours = 0
                if exec.timestamp and entry_time:
                    try:
                        holding_hours = (exec.timestamp - entry_time).total_seconds() / 3600
                    except:
                        holding_hours = 24  # Default 1 day

                # Get regime from signal if available, else default to FLAT
                entry_regime = open_trade.get('regime', TermStructureRegime.FLAT)

                trade = Trade(
                    trade_id=open_trade['trade_id'],
                    strategy=open_trade.get('strategy', exec.strategy),
                    venue=exec.signal.venue if '_' in exec.signal.venue else exec.signal.venue.split('_')[0],
                    entry_time=entry_time,
                    exit_time=exec.timestamp,
                    direction=entry_direction,
                    size_btc=size_btc,
                    entry_price=entry_price,
                    exit_price=current_price,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    fees_usd=exec.fees_usd + open_trade['fees_usd'],
                    slippage_bps=exec.slippage_bps,
                    holding_period_hours=holding_hours,
                    exit_reason=ExitReason.STOP_LOSS if stop_loss_triggered else (ExitReason.PROFIT_TARGET if pnl_usd > 0 else ExitReason.STOP_LOSS),
                    regime_at_entry=entry_regime,
                    is_open=False
                )
                self.trades.append(trade)
                del open_positions[key]
            else:
                # New position - store for later matching (including regime)
                entry_regime = TermStructureRegime.FLAT
                if hasattr(exec.signal, 'regime_at_signal') and exec.signal.regime_at_signal is not None:
                    entry_regime = exec.signal.regime_at_signal
                open_positions[key] = {
                    'trade_id': exec.execution_id,
                    'entry_time': exec.timestamp,
                    'entry_price': current_price,
                    'size_btc': exec.executed_size_btc,
                    'direction': exec.signal.direction,
                    'fees_usd': exec.fees_usd,
                    'strategy': exec.strategy,
                    'regime': entry_regime
                }

        # Close any remaining open positions at last price
        # If no price history available, use average entry price of open positions as approximation
        if price_history:
            last_price = list(price_history.values())[-1]
        elif open_positions:
            last_price = sum(p['entry_price'] for p in open_positions.values()) / len(open_positions)
        else:
            last_price = 1.0  # Minimal fallback
        for key, open_trade in open_positions.items():
            entry_price = open_trade['entry_price']
            size_btc = open_trade['size_btc']
            entry_direction = open_trade['direction']

            # Calculate PnL
            if entry_direction.value == 1 or str(entry_direction.name) == 'LONG':
                pnl_usd = (last_price - entry_price) * size_btc
            else:
                pnl_usd = (entry_price - last_price) * size_btc

            pnl_pct = (pnl_usd / (entry_price * size_btc)) * 100 if entry_price > 0 else 0

            # PDF: 5% basis stop-loss applies to forced closes too
            if pnl_pct < -5.0:
                pnl_pct = -5.0
                pnl_usd = -0.05 * entry_price * size_btc

            trade = Trade(
                trade_id=open_trade['trade_id'],
                strategy=open_trade.get('strategy', key[0]),
                venue=key[1],
                entry_time=open_trade['entry_time'],
                exit_time=None,
                direction=entry_direction,
                size_btc=size_btc,
                entry_price=entry_price,
                exit_price=last_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                fees_usd=open_trade['fees_usd'],
                slippage_bps=0,
                holding_period_hours=0,
                exit_reason=ExitReason.END_OF_DATA,
                regime_at_entry=open_trade.get('regime', TermStructureRegime.FLAT),
                is_open=True
            )
            self.trades.append(trade)

    def _calculate_metrics(
        self,
        initial_capital: float,
        final_capital: float,
        equity_curve: pd.DataFrame,
        historical_data: Dict[str, pd.DataFrame]
    ) -> BacktestMetrics:
        """Calculate all performance, risk, and trade metrics."""
        if equity_curve.empty:
            return self._empty_metrics()

        import math

        # P&L - filter out nan values first
        valid_trades = [t for t in self.trades if t.pnl_usd is not None and not math.isnan(t.pnl_usd)]
        gross_profit = sum(t.pnl_usd for t in valid_trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in valid_trades if t.pnl_usd < 0))
        trade_based_pnl = sum(t.pnl_usd for t in valid_trades)
        total_fees = sum(t.fees_usd for t in self.trades if t.fees_usd is not None and not math.isnan(t.fees_usd))

        # Always reconstruct equity from realized trade PnL for accurate metrics
        # The orchestrator equity only tracks unrealized mark-to-market P&L
        equity = equity_curve['equity'].values

        if valid_trades and len(valid_trades) > 10:
            # Reconstruct daily equity curve from cumulative trade PnL
            sorted_trades = sorted(valid_trades, key=lambda t: t.exit_time if t.exit_time else t.entry_time)
            trade_pnl_by_date = defaultdict(float)
            for trade in sorted_trades:
                trade_date = None
                if trade.exit_time:
                    try:
                        exit_dt = pd.Timestamp(trade.exit_time)
                        trade_date = exit_dt.date()
                    except Exception:
                        pass
                if not trade_date:
                    try:
                        entry_dt = pd.Timestamp(trade.entry_time)
                        trade_date = entry_dt.date()
                    except Exception:
                        continue
                pnl = trade.pnl_usd
                if trade.fees_usd is not None and not math.isnan(trade.fees_usd):
                    pnl -= trade.fees_usd
                trade_pnl_by_date[trade_date] += pnl

            # Build daily equity from sorted dates
            sorted_dates = sorted(trade_pnl_by_date.keys())
            daily_equity = [initial_capital]
            cumulative = 0
            for d in sorted_dates:
                cumulative += trade_pnl_by_date[d]
                daily_equity.append(initial_capital + cumulative)
            equity = np.array(daily_equity)
            # Save reconstructed equity with dates for monthly returns
            self._reconstructed_equity = pd.DataFrame({
                'timestamp': pd.to_datetime([sorted_dates[0] - pd.Timedelta(days=1)] + sorted_dates),
                'equity': daily_equity
            })

        timestamps = equity_curve['timestamp'].values if 'timestamp' in equity_curve else range(len(equity))

        # Daily returns from equity curve
        daily_returns = np.diff(equity) / np.maximum(equity[:-1], 1.0) if len(equity) > 1 else np.array([])
        daily_returns = np.nan_to_num(daily_returns, nan=0, posinf=0, neginf=0)

        # Basic performance - prefer trade-based PnL over equity curve
        if abs(trade_based_pnl) > 0.01:
            net_profit = trade_based_pnl - total_fees
            final_capital = initial_capital + net_profit
        elif not math.isnan(final_capital) and final_capital > 0 and abs(final_capital - initial_capital) > 0.01:
            net_profit = final_capital - initial_capital
        else:
            if len(equity) > 0 and not math.isnan(equity[-1]):
                final_capital = equity[-1]
            else:
                final_capital = initial_capital
            net_profit = final_capital - initial_capital

        total_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0
        # Use actual date span for annualization (not number of equity observations)
        try:
            if 'timestamp' in equity_curve.columns and len(equity_curve) > 1:
                start_ts = pd.Timestamp(equity_curve['timestamp'].iloc[0])
                end_ts = pd.Timestamp(equity_curve['timestamp'].iloc[-1])
                days = max((end_ts - start_ts).days, 1)
            else:
                days = len(equity)
        except Exception:
            days = len(equity)
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if abs(total_return) < 100 else total_return

        # Trade stats - only count valid trades
        total_trades = len(valid_trades) if valid_trades else len(self.trades)
        winning_trades = sum(1 for t in valid_trades if t.pnl_usd > 0)
        losing_trades = sum(1 for t in valid_trades if t.pnl_usd <= 0)
        win_rate = winning_trades / max(total_trades, 1) * 100

        # Averages
        avg_win = gross_profit / max(winning_trades, 1)
        avg_loss = gross_loss / max(losing_trades, 1)
        avg_trade = net_profit / max(total_trades, 1)

        # Risk metrics from reconstructed equity
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity)
        volatility = np.std(daily_returns) * np.sqrt(365) if len(daily_returns) > 1 else 0
        sharpe = self._calculate_sharpe(daily_returns)
        sortino = self._calculate_sortino(daily_returns)

        # VaR/CVaR
        var_95 = np.percentile(daily_returns, 5) * 100 if len(daily_returns) > 0 else 0
        var_99 = np.percentile(daily_returns, 1) * 100 if len(daily_returns) > 0 else 0
        cvar_95 = np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]) * 100 if len(daily_returns) > 0 else 0

        # Ratios
        profit_factor = gross_profit / max(gross_loss, 1)
        calmar = annualized_return * 100 / max(abs(max_dd), 1)
        payoff = avg_win / max(avg_loss, 1)

        # Time-based
        positive_days = sum(1 for r in daily_returns if r > 0) / max(len(daily_returns), 1) * 100

        # Consecutive wins/losses
        max_cons_wins, max_cons_losses = self._calculate_consecutive(self.trades)

        # Extended metrics
        skewness = stats.skew(daily_returns) if len(daily_returns) > 2 else 0
        kurtosis = stats.kurtosis(daily_returns) if len(daily_returns) > 3 else 0

        # Venue breakdown
        venue_breakdown = self._calculate_venue_breakdown()

        # Compute BTC spot daily returns for per-strategy correlation
        self._btc_spot_prices = None
        try:
            from pathlib import Path
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
            # Try parquet files first (primary format)
            ohlcv_file = data_dir / 'binance' / 'binance_ohlcv.parquet'
            if ohlcv_file.exists():
                ohlcv_df = pd.read_parquet(ohlcv_file)
                btc_mask = ohlcv_df['symbol'].str.upper() == 'BTC'
                btc_df = ohlcv_df[btc_mask].copy()
                if len(btc_df) > 0 and 'close' in btc_df.columns:
                    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], utc=True)
                    btc_df = btc_df.set_index('timestamp').sort_index()
                    btc_daily = btc_df['close'].resample('D').last().ffill().dropna()
                    self._btc_spot_prices = btc_daily.pct_change().dropna()
            # Fallback: try historical_data keys
            if self._btc_spot_prices is None:
                for k in historical_data.keys():
                    if 'binance' in k.lower():
                        df = historical_data[k]
                        if 'close' in df.columns:
                            btc_series = df['close'].dropna()
                            if hasattr(btc_series.index, 'tz') and btc_series.index.tz is None:
                                btc_series.index = btc_series.index.tz_localize('UTC')
                            btc_daily = btc_series.resample('D').last().ffill().dropna()
                            self._btc_spot_prices = btc_daily.pct_change().dropna()
                            break
        except Exception:
            pass

        # Strategy breakdown with per-strategy max DD, correlation, fees
        strategy_breakdown = self._calculate_strategy_breakdown(initial_capital=initial_capital)

        # Regime performance
        regime_performance = self._calculate_regime_performance(historical_data)

        # Crisis performance
        crisis_performance = self._calculate_crisis_performance(equity_curve, historical_data)

        # Ulcer index
        ulcer = self._calculate_ulcer_index(equity)

        # Information Ratio and Treynor Ratio (proper computation vs BTC benchmark)
        self._ir = 0.0
        self._treynor = 0.0
        if self._btc_spot_prices is not None and len(daily_returns) > 10:
            try:
                btc_ret = self._btc_spot_prices
                min_len = min(len(daily_returns), len(btc_ret))
                if min_len > 10:
                    s = daily_returns[-min_len:]
                    b = btc_ret.values[-min_len:]
                    active_return = s - b
                    te = np.std(active_return) * np.sqrt(365)
                    if te > 1e-8:
                        self._ir = float((np.mean(s) - np.mean(b)) * 365 / te)
                    cov_sb = np.cov(s, b)
                    if cov_sb.shape == (2, 2) and cov_sb[1, 1] > 1e-10:
                        beta = cov_sb[0, 1] / cov_sb[1, 1]
                        if abs(beta) > 0.001:
                            self._treynor = float(annualized_return / beta)
            except Exception:
                pass

        return BacktestMetrics(
            # Basic Performance
            total_return_pct=total_return * 100,
            annualized_return_pct=annualized_return * 100,
            total_pnl_usd=net_profit,
            gross_profit_usd=gross_profit,
            gross_loss_usd=gross_loss,
            net_profit_usd=net_profit,
            profit_factor=profit_factor,
            expectancy_usd=avg_trade,
            expectancy_pct=avg_trade / initial_capital * 100,
            avg_trade_pnl_usd=avg_trade,

            # Risk Metrics
            max_drawdown_pct=max_dd,
            max_drawdown_usd=max_dd * initial_capital / 100,
            max_drawdown_duration_days=max_dd_duration,
            avg_drawdown_pct=self._calculate_avg_drawdown(equity),
            calmar_ratio=calmar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            omega_ratio=self._calculate_omega(daily_returns),
            information_ratio=self._ir,
            treynor_ratio=self._treynor,
            var_95_pct=var_95,
            var_99_pct=var_99,
            cvar_95_pct=cvar_95,
            tail_ratio=self._calculate_tail_ratio(daily_returns),
            volatility_annual_pct=volatility * 100,

            # Trade Statistics
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            avg_win_usd=avg_win,
            avg_loss_usd=avg_loss,
            avg_win_pct=avg_win / initial_capital * 100,
            avg_loss_pct=avg_loss / initial_capital * 100,
            largest_win_usd=max((t.pnl_usd for t in valid_trades if not math.isnan(t.pnl_usd)), default=0),
            largest_loss_usd=abs(min((t.pnl_usd for t in valid_trades if not math.isnan(t.pnl_usd)), default=0)),
            avg_holding_hours=np.mean([t.holding_period_hours for t in valid_trades]) if valid_trades else 0,
            median_holding_hours=np.median([t.holding_period_hours for t in valid_trades if t.holding_period_hours > 0]) if valid_trades else 0,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            payoff_ratio=payoff,

            # Time-based
            time_in_market_pct=self._calculate_time_in_market(equity_curve),
            avg_trades_per_day=total_trades / max(days, 1),
            avg_trades_per_week=total_trades / max(days / 7, 1),
            best_day_return_pct=max(daily_returns) * 100 if len(daily_returns) > 0 else 0,
            worst_day_return_pct=min(daily_returns) * 100 if len(daily_returns) > 0 else 0,
            best_week_return_pct=self._calculate_best_period_return(daily_returns, 5),
            worst_week_return_pct=self._calculate_worst_period_return(daily_returns, 5),
            best_month_return_pct=self._calculate_best_period_return(daily_returns, 21),
            worst_month_return_pct=self._calculate_worst_period_return(daily_returns, 21),
            positive_days_pct=positive_days,

            # Breakdowns
            venue_breakdown=venue_breakdown,
            strategy_breakdown=strategy_breakdown,
            regime_performance=regime_performance,
            crisis_performance=crisis_performance,

            # Costs
            total_fees_usd=total_fees,
            total_slippage_usd=sum(
                t.slippage_bps * t.size_btc * t.entry_price / 10000
                for t in self.trades
                if t.slippage_bps and t.size_btc and t.entry_price
                and not math.isnan(t.slippage_bps) and not math.isnan(t.entry_price)
            ),
            avg_slippage_bps=float(np.nanmean([t.slippage_bps for t in self.trades if t.slippage_bps is not None])) if self.trades else 0,
            fees_as_pct_of_gross=total_fees / max(gross_profit + gross_loss, 1) * 100,
            break_even_win_rate_pct=1 / (1 + payoff) * 100 if payoff > 0 else 50,

            # Extended
            skewness=skewness,
            kurtosis=kurtosis,
            ulcer_index=ulcer,
            recovery_factor=net_profit / max(abs(max_dd * initial_capital / 100), 1),
            common_sense_ratio=self._calculate_tail_ratio(daily_returns) * profit_factor
        )

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics structure."""
        return BacktestMetrics(
            total_return_pct=0, annualized_return_pct=0, total_pnl_usd=0,
            gross_profit_usd=0, gross_loss_usd=0, net_profit_usd=0,
            profit_factor=0, expectancy_usd=0, expectancy_pct=0, avg_trade_pnl_usd=0,
            max_drawdown_pct=0, max_drawdown_usd=0, max_drawdown_duration_days=0,
            avg_drawdown_pct=0, calmar_ratio=0, sharpe_ratio=0, sortino_ratio=0,
            omega_ratio=0, information_ratio=0, treynor_ratio=0,
            var_95_pct=0, var_99_pct=0, cvar_95_pct=0, tail_ratio=0, volatility_annual_pct=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate_pct=0,
            avg_win_usd=0, avg_loss_usd=0, avg_win_pct=0, avg_loss_pct=0,
            largest_win_usd=0, largest_loss_usd=0, avg_holding_hours=0, median_holding_hours=0,
            max_consecutive_wins=0, max_consecutive_losses=0, payoff_ratio=0,
            time_in_market_pct=0, avg_trades_per_day=0, avg_trades_per_week=0,
            best_day_return_pct=0, worst_day_return_pct=0, best_week_return_pct=0,
            worst_week_return_pct=0, best_month_return_pct=0, worst_month_return_pct=0,
            positive_days_pct=0, venue_breakdown={}, strategy_breakdown={},
            regime_performance={}, crisis_performance={},
            total_fees_usd=0, total_slippage_usd=0, avg_slippage_bps=0,
            fees_as_pct_of_gross=0, break_even_win_rate_pct=50,
            skewness=0, kurtosis=0, ulcer_index=0, recovery_factor=0, common_sense_ratio=0
        )

    def _calculate_max_drawdown(self, equity: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100

        max_dd = abs(np.min(drawdowns))

        # Duration
        underwater = drawdowns < 0
        max_duration = 0
        current_duration = 0

        for uw in underwater:
            if uw:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, max_duration

    def _calculate_avg_drawdown(self, equity: np.ndarray) -> float:
        """Calculate average drawdown."""
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        return abs(np.mean(drawdowns[drawdowns < 0])) if any(drawdowns < 0) else 0

    def _calculate_sharpe(self, returns: np.ndarray, rf: float = 0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0
        excess_returns = returns - rf / 365
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)

    def _calculate_sortino(self, returns: np.ndarray, rf: float = 0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0
        excess_returns = returns - rf / 365
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        return np.mean(excess_returns) / downside_std * np.sqrt(365)

    def _calculate_omega(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        sum_gains = np.sum(gains - threshold) if len(gains) > 0 else 0
        sum_losses = abs(np.sum(losses - threshold)) if len(losses) > 0 else 1e-8
        return sum_gains / sum_losses

    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        if len(returns) < 20:
            return 1
        upper = np.percentile(returns, 95)
        lower = abs(np.percentile(returns, 5))
        return upper / max(lower, 1e-8)

    def _calculate_consecutive(self, trades: List[Trade]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl_usd > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _calculate_time_in_market(self, equity_curve: pd.DataFrame) -> float:
        """Calculate percentage of time with active positions from trade data."""
        if not self.trades:
            return 0.0
        # Count unique days with at least one active trade
        active_dates = set()
        for trade in self.trades:
            entry = trade.entry_time
            exit_t = trade.exit_time or trade.entry_time
            if entry and exit_t:
                try:
                    d = entry.date() if hasattr(entry, 'date') else entry
                    d_exit = exit_t.date() if hasattr(exit_t, 'date') else exit_t
                    current = d
                    while current <= d_exit:
                        active_dates.add(current)
                        current += timedelta(days=1)
                except Exception:
                    continue
        if not active_dates or len(equity_curve) < 2:
            return 0.0
        total_days = (max(active_dates) - min(active_dates)).days + 1
        return len(active_dates) / max(total_days, 1) * 100

    def _calculate_best_period_return(self, returns: np.ndarray, period: int) -> float:
        """Calculate best return over period."""
        if len(returns) < period:
            return np.sum(returns) * 100 if len(returns) > 0 else 0

        rolling_returns = [
            np.sum(returns[i:i + period])
            for i in range(len(returns) - period + 1)
        ]
        return max(rolling_returns) * 100 if rolling_returns else 0

    def _calculate_worst_period_return(self, returns: np.ndarray, period: int) -> float:
        """Calculate worst return over period."""
        if len(returns) < period:
            return np.sum(returns) * 100 if len(returns) > 0 else 0

        rolling_returns = [
            np.sum(returns[i:i + period])
            for i in range(len(returns) - period + 1)
        ]
        return min(rolling_returns) * 100 if rolling_returns else 0

    def _calculate_ulcer_index(self, equity: np.ndarray) -> float:
        """Calculate Ulcer Index."""
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        return np.sqrt(np.mean(drawdowns ** 2))

    def _calculate_venue_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance breakdown by venue."""
        import math
        venue_trades = defaultdict(list)
        for trade in self.trades:
            venue_trades[trade.venue].append(trade)

        breakdown = {}
        for venue, trades in venue_trades.items():
            pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None and not math.isnan(t.pnl_usd)]
            fees = [t.fees_usd for t in trades if t.fees_usd is not None and not math.isnan(t.fees_usd)]
            total_fees = sum(fees) if fees else 0.0
            breakdown[venue] = {
                'trades': len(trades),
                'total_pnl': sum(pnls) if pnls else 0,
                'win_rate': sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100,
                'avg_pnl': float(np.mean(pnls)) if pnls else 0,
                'total_fees': total_fees
            }

        return breakdown

    def _calculate_strategy_breakdown(self, initial_capital: float = 1000000) -> Dict[str, Dict[str, float]]:
        """Calculate per-strategy breakdown: max DD, BTC correlation, Sharpe, fees."""
        import math
        strategy_trades = defaultdict(list)
        for trade in self.trades:
            strategy_trades[trade.strategy].append(trade)

        # Total PnL for contribution calculation (NaN-safe)
        all_pnls = [t.pnl_usd for t in self.trades if t.pnl_usd is not None and not math.isnan(t.pnl_usd)]
        total_all_pnl = sum(all_pnls) if all_pnls else 0

        # Get BTC spot returns for correlation (from reconstructed equity if available)
        btc_daily_returns = None
        if hasattr(self, '_btc_spot_prices') and self._btc_spot_prices is not None:
            btc_daily_returns = self._btc_spot_prices

        breakdown = {}
        for strategy, trades in strategy_trades.items():
            pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None and not math.isnan(t.pnl_usd)]
            total_pnl = sum(pnls) if pnls else 0
            total_fees = sum(t.fees_usd for t in trades if t.fees_usd is not None and not math.isnan(t.fees_usd))

            # Per-strategy max drawdown from cumulative PnL
            max_dd_pct = 0.0
            if pnls:
                sorted_trades_s = sorted(
                    [t for t in trades if t.pnl_usd is not None and not math.isnan(t.pnl_usd)],
                    key=lambda t: t.exit_time if t.exit_time else t.entry_time
                )
                cumulative = initial_capital
                peak = cumulative
                max_dd_val = 0.0
                for t in sorted_trades_s:
                    net_pnl = t.pnl_usd - (t.fees_usd if t.fees_usd and not math.isnan(t.fees_usd) else 0)
                    cumulative += net_pnl
                    if cumulative > peak:
                        peak = cumulative
                    dd = (peak - cumulative) / peak * 100 if peak > 0 else 0
                    if dd > max_dd_val:
                        max_dd_val = dd
                max_dd_pct = max_dd_val

            # Per-strategy BTC correlation from daily returns
            strategy_corr = 0.0
            if pnls and len(pnls) > 10 and btc_daily_returns is not None and len(btc_daily_returns) > 5:
                try:
                    sorted_trades_s = sorted(
                        [t for t in trades if t.pnl_usd is not None and not math.isnan(t.pnl_usd)],
                        key=lambda t: t.exit_time if t.exit_time else t.entry_time
                    )
                    # Build daily PnL series for this strategy
                    daily_pnl = defaultdict(float)
                    for t in sorted_trades_s:
                        try:
                            d = pd.Timestamp(t.exit_time if t.exit_time else t.entry_time).date()
                            daily_pnl[d] += t.pnl_usd
                        except Exception:
                            continue
                    if len(daily_pnl) > 5:
                        # Convert to DatetimeIndex matching BTC index timezone
                        strat_idx = pd.DatetimeIndex(
                            [pd.Timestamp(d) for d in sorted(daily_pnl.keys())],
                            tz='UTC'
                        )
                        strat_vals = [daily_pnl[d] for d in sorted(daily_pnl.keys())]
                        strat_returns = pd.Series(strat_vals, index=strat_idx) / initial_capital
                        # Make BTC index tz-aware if needed
                        btc_ret = btc_daily_returns.copy()
                        if btc_ret.index.tz is None:
                            btc_ret.index = btc_ret.index.tz_localize('UTC')
                        elif str(btc_ret.index.tz) != 'UTC':
                            btc_ret.index = btc_ret.index.tz_convert('UTC')
                        # Align
                        aligned = pd.DataFrame({
                            'strat': strat_returns,
                            'btc': btc_ret
                        }).dropna()
                        if len(aligned) > 5:
                            strategy_corr = float(aligned['strat'].corr(aligned['btc']))
                            if math.isnan(strategy_corr):
                                strategy_corr = 0.0
                except Exception:
                    strategy_corr = 0.0

            # Per-strategy Sharpe ratio from daily returns
            strategy_sharpe = 0.0
            if pnls and len(daily_pnl) > 10:
                daily_vals = [daily_pnl[d] for d in sorted(daily_pnl.keys())]
                daily_rets = np.array(daily_vals) / initial_capital
                if np.std(daily_rets) > 0:
                    strategy_sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365))

            # Per-strategy funding/roll cost estimate
            funding_cost = 0.0
            roll_cost = 0.0
            for t in trades:
                if t.pnl_usd is not None and not math.isnan(t.pnl_usd):
                    fee = t.fees_usd if t.fees_usd and not math.isnan(t.fees_usd) else 0
                    if strategy == 'roll_optimization':
                        roll_cost += fee
                    elif strategy in ('synthetic_futures', 'calendar_spread'):
                        funding_cost += fee

            # Per-strategy venue breakdown
            venue_pnl = defaultdict(float)
            for t in trades:
                if t.pnl_usd is not None and not math.isnan(t.pnl_usd):
                    venue_pnl[t.venue] += t.pnl_usd
            venue_profitability = {v: round(p, 2) for v, p in sorted(venue_pnl.items(), key=lambda x: -x[1])}

            breakdown[strategy] = {
                'trades': len(pnls),  # Count only valid trades (with non-NaN PnL)
                'total_pnl': total_pnl,
                'win_rate': sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100,
                'avg_pnl': float(np.mean(pnls)) if pnls else 0,
                'contribution_pct': total_pnl / max(abs(total_all_pnl), 1) * 100,
                'max_drawdown_pct': max_dd_pct,
                'btc_correlation': strategy_corr,
                'sharpe_ratio': strategy_sharpe,
                'total_fees': total_fees,
                'avg_fees_per_trade': total_fees / max(len(trades), 1),
                'funding_cost': funding_cost,
                'roll_cost': roll_cost,
                'venue_profitability': venue_profitability
            }

        return breakdown

    def _calculate_regime_performance(
        self,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance by regime."""
        regime_trades = defaultdict(list)
        for trade in self.trades:
            regime_trades[trade.regime_at_entry.value].append(trade)

        performance = {}
        initial_capital = self.initial_capital if hasattr(self, 'initial_capital') else 1_000_000
        for regime, trades in regime_trades.items():
            pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None and not (isinstance(t.pnl_usd, float) and math.isnan(t.pnl_usd))]
            # Compute per-regime Sharpe from daily PnL returns
            regime_sharpe = 0.0
            if pnls and len(pnls) > 5:
                daily_pnl = defaultdict(float)
                for t in trades:
                    if t.pnl_usd is not None and not (isinstance(t.pnl_usd, float) and math.isnan(t.pnl_usd)):
                        d = t.entry_time.date() if t.entry_time else None
                        if d:
                            daily_pnl[d] += t.pnl_usd
                if len(daily_pnl) > 5:
                    daily_vals = [daily_pnl[d] for d in sorted(daily_pnl.keys())]
                    daily_rets = np.array(daily_vals) / initial_capital
                    if np.std(daily_rets) > 0:
                        regime_sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365))
            performance[regime] = {
                'trades': len(trades),
                'total_pnl': sum(pnls) if pnls else 0,
                'win_rate': sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100,
                'avg_pnl': float(np.mean(pnls)) if pnls else 0,
                'sharpe_ratio': regime_sharpe
            }

        return performance

    def _calculate_crisis_performance(
        self,
        equity_curve: pd.DataFrame,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance during crisis events including trade-level stats."""
        import math
        crisis_perf = {}

        for crisis, (start, end) in CRISIS_DATES.items():
            # Match trades that overlap with crisis period
            crisis_trades = []
            for trade in self.trades:
                if trade.pnl_usd is None or (isinstance(trade.pnl_usd, float) and math.isnan(trade.pnl_usd)):
                    continue
                t_entry = trade.entry_time
                t_exit = trade.exit_time or trade.entry_time
                # Make timezone-aware if needed
                try:
                    if t_entry and t_entry.tzinfo is None:
                        t_entry = t_entry.replace(tzinfo=timezone.utc)
                    if t_exit and t_exit.tzinfo is None:
                        t_exit = t_exit.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                # Trade overlaps crisis if entry <= end AND exit >= start
                if t_entry and t_entry <= end and t_exit and t_exit >= start:
                    crisis_trades.append(trade)

            # Calculate trade-level metrics
            crisis_pnls = [t.pnl_usd for t in crisis_trades]
            num_trades = len(crisis_trades)
            win_rate = sum(1 for p in crisis_pnls if p > 0) / max(num_trades, 1) * 100
            total_pnl = sum(crisis_pnls) if crisis_pnls else 0

            # Filter equity curve for crisis period (handle timezone mismatch)
            crisis_return = 0.0
            max_dd = 0.0
            crisis_days = (end - start).days
            if 'timestamp' in equity_curve.columns:
                try:
                    eq_ts = pd.to_datetime(equity_curve['timestamp'])
                    # Strip timezone for comparison if needed
                    start_naive = start.replace(tzinfo=None) if start.tzinfo else start
                    end_naive = end.replace(tzinfo=None) if end.tzinfo else end
                    if eq_ts.dt.tz is not None:
                        start_aware = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
                        end_aware = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
                        mask = (eq_ts >= start_aware) & (eq_ts <= end_aware)
                    else:
                        mask = (eq_ts >= start_naive) & (eq_ts <= end_naive)
                    crisis_equity = equity_curve[mask]['equity'].dropna().values
                    if len(crisis_equity) >= 2:
                        crisis_return = (crisis_equity[-1] - crisis_equity[0]) / crisis_equity[0] * 100
                        max_dd, _ = self._calculate_max_drawdown(crisis_equity)
                except Exception:
                    pass

            # Use trade-based return if equity curve unavailable
            if crisis_return == 0.0 and total_pnl != 0 and num_trades > 0:
                crisis_return = total_pnl / 1000000 * 100  # Relative to $1M capital

            # Compute max DD from trade-level PnLs if equity curve DD is 0
            if max_dd == 0.0 and crisis_pnls:
                cumulative = 0.0
                peak = 0.0
                for pnl in crisis_pnls:
                    cumulative += pnl
                    if cumulative > peak:
                        peak = cumulative
                    dd = (peak - cumulative) / max(1000000, peak + 1000000) * 100
                    if dd > max_dd:
                        max_dd = dd

            crisis_perf[crisis.value] = {
                'return_pct': round(crisis_return, 4),
                'max_drawdown_pct': round(max_dd, 4),
                'days': crisis_days,
                'trades': num_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2)
            }

        return crisis_perf

    def export_results(
        self,
        result: BacktestResult,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export backtest results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # Export metrics
        metrics_path = output_dir / 'backtest_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(result.metrics.to_dict(), f, indent=2, default=str)
        files['metrics'] = metrics_path

        # Export equity curve
        equity_path = output_dir / 'equity_curve.csv'
        result.equity_curve.to_csv(equity_path, index=False)
        files['equity_curve'] = equity_path

        # Export trades
        if result.trades:
            trades_df = pd.DataFrame([{
                'trade_id': t.trade_id,
                'strategy': t.strategy,
                'venue': t.venue,
                'entry_time': t.entry_time,
                'direction': t.direction.value if t.direction else None,
                'size_btc': t.size_btc,
                'entry_price': t.entry_price,
                'pnl_usd': t.pnl_usd,
                'fees_usd': t.fees_usd
            } for t in result.trades])
            trades_path = output_dir / 'trades.csv'
            trades_df.to_csv(trades_path, index=False)
            files['trades'] = trades_path

        # Export summary
        summary = {
            'strategy': result.strategy_name,
            'start_date': result.start_date.isoformat() if result.start_date else None,
            'end_date': result.end_date.isoformat() if result.end_date else None,
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'key_metrics': result.metrics.get_summary()
        }
        summary_path = output_dir / 'backtest_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        files['summary'] = summary_path

        logger.info(f"Backtest results exported to {output_dir}")
        return files


class StrategyComparison:
    """Compare backtest results across strategies."""

    @staticmethod
    def compare(results: List[BacktestResult]) -> pd.DataFrame:
        """Create comparison DataFrame."""
        comparison_data = []

        for result in results:
            m = result.metrics
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Total Return %': m.total_return_pct,
                'Sharpe Ratio': m.sharpe_ratio,
                'Max DD %': m.max_drawdown_pct,
                'Win Rate %': m.win_rate_pct,
                'Profit Factor': m.profit_factor,
                'Total Trades': m.total_trades,
                'Avg Trade $': m.avg_trade_pnl_usd,
                'Volatility %': m.volatility_annual_pct
            })

        return pd.DataFrame(comparison_data)

    @staticmethod
    def rank_strategies(
        results: List[BacktestResult],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """Rank strategies by weighted score."""
        if weights is None:
            weights = {
                'sharpe_ratio': 0.3,
                'total_return_pct': 0.2,
                'max_drawdown_pct': -0.2,  # Negative weight (lower is better)
                'profit_factor': 0.15,
                'win_rate_pct': 0.15
            }

        scores = []
        for result in results:
            m = result.metrics
            score = 0
            for metric, weight in weights.items():
                value = getattr(m, metric, 0)
                # Normalize
                score += value * weight

            scores.append((result.strategy_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'CrisisEvent',
    # Crisis dates
    'CRISIS_DATES',
    # Data structures
    'Trade',
    'DailyMetrics',
    'BacktestMetrics',
    'BacktestResult',
    # Backtest engine
    'FuturesBacktestEngine',
    # Comparison
    'StrategyComparison',
]
