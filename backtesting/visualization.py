"""
Visualization Module for BTC Futures Curve Statistical Arbitrage Backtesting
=============================================================================

Generates publication-quality charts for backtest analysis, performance attribution,
and risk reporting. Covers equity curves, 3D term structure surfaces, venue heatmaps,
regime-conditioned performance, rolling metrics, trade scatter analysis, funding rate
comparisons, basis spread overlays, correlation clustering, drawdown decomposition,
monthly return calendars, risk dashboards, capacity degradation, and roll cost breakdowns.

All plots follow the project specification reporting requirements for
Section 2.4 deliverables.

Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
logger = logging.getLogger(__name__)

# =============================================================================
# COLOR PALETTES
# =============================================================================

# Venue classification colors
VENUE_COLORS: Dict[str, str] = {
    'CEX': '#2196F3',
    'Hybrid': '#4CAF50',
    'DEX': '#FF9800',
}

# Individual venue colors derived from classification palette
VENUE_SPECIFIC_COLORS: Dict[str, str] = {
    'Binance': '#1565C0',
    'CME': '#1976D2',
    'Deribit': '#2196F3',
    'Hyperliquid': '#388E3C',
    'dYdX': '#E65100',
    'GMX': '#FF9800',
}

# Strategy colors
STRATEGY_COLORS: Dict[str, str] = {
    'Calendar': '#1976D2',
    'Cross-Venue': '#388E3C',
    'Synthetic': '#F57C00',
    'Roll': '#7B1FA2',
}

# Market regime colors
REGIME_COLORS: Dict[str, str] = {
    'Bull': '#4CAF50',
    'Bear': '#F44336',
    'Sideways': '#FFC107',
    'Crisis': '#9C27B0',
    'High Vol': '#FF5722',
}

# Crisis event definitions for annotation overlays
CRISIS_EVENTS: Dict[str, Dict[str, str]] = {
    'UST/Luna Collapse': {'start': '2022-05-07', 'end': '2022-05-13'},
    'FTX Bankruptcy': {'start': '2022-11-06', 'end': '2022-11-14'},
    'USDC Depeg': {'start': '2023-03-10', 'end': '2023-03-15'},
    'SEC Lawsuits': {'start': '2023-06-05', 'end': '2023-06-12'},
    'Aug 2024 Unwind': {'start': '2024-08-05', 'end': '2024-08-07'},
}

# Consistent figure styling
FIGURE_DPI = 150
FIGURE_FACECOLOR = '#FAFAFA'
GRID_ALPHA = 0.3
GRID_COLOR = '#CCCCCC'


def _apply_style(ax: plt.Axes, title: str = '', xlabel: str = '', ylabel: str = '') -> None:
    """Apply consistent axis styling across all plots."""
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, alpha=GRID_ALPHA, color=GRID_COLOR, linestyle='-', linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#888888')


def _format_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def _format_currency(value: float) -> str:
    """Format a dollar value with comma separators."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.2f}"


def _compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute the drawdown series from an equity curve.

    Returns a non-positive series where 0 indicates a new high-water mark.
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return drawdown


def _compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 90,
    annualization: float = np.sqrt(365)
) -> pd.Series:
    """Compute rolling Sharpe ratio using a specified window in calendar days."""
    rolling_mean = returns.rolling(window=window, min_periods=max(30, window // 3)).mean()
    rolling_std = returns.rolling(window=window, min_periods=max(30, window // 3)).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (rolling_mean / rolling_std) * annualization


# =============================================================================
# MAIN VISUALIZATION CLASS
# =============================================================================

class BacktestVisualization:
    """
    Visualization engine for BTC futures curve statistical arbitrage backtests.

    Accepts equity curves, trade records, term structure data, and regime
    classifications to produce a complete set of analytical charts. All plots
    are designed for PDF embedding at 150 DPI with consistent styling.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio equity indexed by datetime.
    trades : pd.DataFrame
        Trade records with columns: entry_time, exit_time, strategy, venue,
        venue_type, net_pnl, gross_pnl, fees, slippage, gas_cost, mev_cost,
        holding_period_hours, entry_price, exit_price, quantity, regime,
        exit_reason.
    benchmark_curves : dict of pd.Series, optional
        Named benchmark equity curves for overlay comparison.
    term_structure : pd.DataFrame, optional
        Term structure data with columns: date, maturity_days, basis_annualized.
    regime_labels : pd.Series, optional
        Market regime classification indexed by datetime.
    funding_rates : pd.DataFrame, optional
        Funding rate data with venue columns indexed by datetime.
    initial_capital : float
        Starting capital for the backtest (default 1,000,000).
    output_dir : str or Path
        Directory for saving generated figures.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        benchmark_curves: Optional[Dict[str, pd.Series]] = None,
        term_structure: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None,
        funding_rates: Optional[pd.DataFrame] = None,
        initial_capital: float = 1_000_000,
        output_dir: Union[str, Path] = './output/charts',
    ):
        self.equity_curve = equity_curve.copy()
        self.trades = trades.copy()
        self.benchmark_curves = benchmark_curves or {}
        self.term_structure = term_structure.copy() if term_structure is not None else None
        self.regime_labels = regime_labels.copy() if regime_labels is not None else None
        self.funding_rates = funding_rates.copy() if funding_rates is not None else None
        self.initial_capital = initial_capital
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Derived series
        self.returns = self.equity_curve.pct_change().dropna()
        self.drawdown = _compute_drawdown_series(self.equity_curve)

        # Precompute trade DataFrame columns if missing
        if 'holding_period_days' not in self.trades.columns and 'holding_period_hours' in self.trades.columns:
            self.trades['holding_period_days'] = self.trades['holding_period_hours'] / 24.0

        logger.info(
            "BacktestVisualization initialized: %d equity points, %d trades, %d benchmarks",
            len(self.equity_curve), len(self.trades), len(self.benchmark_curves),
        )

    # -----------------------------------------------------------------
    # 1. Equity Curve with Drawdown Shading
    # -----------------------------------------------------------------
    def plot_equity_curve(
        self,
        show_drawdown: bool = True,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot equity curve with optional drawdown shading and benchmark overlays.

        Parameters
        ----------
        show_drawdown : bool
            Shade drawdown periods beneath the equity curve.
        log_scale : bool
            Use logarithmic y-axis for equity.
        figsize : tuple
            Figure dimensions (width, height) in inches.
        save : bool
            Persist figure to output directory.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(
            2 if show_drawdown else 1, 1, figsize=figsize,
            gridspec_kw={'height_ratios': [3, 1]} if show_drawdown else None,
            sharex=True,
        )
        fig.set_facecolor(FIGURE_FACECOLOR)

        ax_eq = axes[0] if show_drawdown else axes

        # Strategy equity
        ax_eq.plot(
            self.equity_curve.index, self.equity_curve.values,
            color='#1565C0', linewidth=1.5, label='Strategy', zorder=3,
        )

        # Benchmark overlays
        benchmark_styles = [
            ('--', '#757575', 1.0), ('-.', '#9E9E9E', 0.9),
            (':', '#BDBDBD', 0.9), ('--', '#616161', 0.8),
        ]
        for idx, (name, bm_curve) in enumerate(self.benchmark_curves.items()):
            style = benchmark_styles[idx % len(benchmark_styles)]
            # Normalize benchmark to same starting equity
            normalized = bm_curve / bm_curve.iloc[0] * self.initial_capital
            ax_eq.plot(
                normalized.index, normalized.values,
                linestyle=style[0], color=style[1], linewidth=style[2],
                label=name, alpha=0.8, zorder=2,
            )

        # Crisis event shading
        for event_name, dates in CRISIS_EVENTS.items():
            start = pd.Timestamp(dates['start'])
            end = pd.Timestamp(dates['end'])
            if start >= self.equity_curve.index.min() and start <= self.equity_curve.index.max():
                ax_eq.axvspan(start, end, alpha=0.08, color='#F44336', zorder=1)
                ypos = ax_eq.get_ylim()[1] * 0.98
                ax_eq.annotate(
                    event_name, xy=(start, ypos), fontsize=6.5,
                    color='#C62828', rotation=90, va='top', ha='right',
                )

        if log_scale:
            ax_eq.set_yscale('log')
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_currency(x)))
        ax_eq.legend(loc='upper left', fontsize=9, framealpha=0.9)
        _apply_style(ax_eq, title='Portfolio Equity Curve', ylabel='Portfolio Value (USD)')

        # Drawdown panel
        if show_drawdown:
            ax_dd = axes[1]
            ax_dd.fill_between(
                self.drawdown.index, self.drawdown.values, 0,
                color='#EF5350', alpha=0.4, zorder=2,
            )
            ax_dd.plot(
                self.drawdown.index, self.drawdown.values,
                color='#C62828', linewidth=0.8, zorder=3,
            )
            ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
            _apply_style(ax_dd, ylabel='Drawdown')
            ax_dd.set_ylim(top=0)

        fig.autofmt_xdate()
        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'equity_curve.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved equity_curve.png")
        return fig

    # -----------------------------------------------------------------
    # 2. 3D Term Structure Surface
    # -----------------------------------------------------------------
    def plot_3d_term_structure(
        self,
        figsize: Tuple[float, float] = (14, 10),
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot a 3D surface of BTC futures term structure (basis vs maturity vs date).

        Requires self.term_structure DataFrame with columns:
        date, maturity_days, basis_annualized.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.term_structure is None or self.term_structure.empty:
            logger.warning("No term structure data available; returning empty figure.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Term structure data not available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#999999')
            return fig

        ts = self.term_structure.copy()
        ts['date'] = pd.to_datetime(ts['date'])

        # Create a pivot for the surface mesh
        pivot = ts.pivot_table(
            values='basis_annualized', index='date', columns='maturity_days', aggfunc='mean',
        )
        pivot = pivot.interpolate(axis=1, limit_direction='both').ffill().bfill()

        dates_num = mdates.date2num(pivot.index.to_pydatetime())
        maturities = pivot.columns.values.astype(float)
        X, Y = np.meshgrid(maturities, dates_num)
        Z = pivot.values

        fig = plt.figure(figsize=figsize, facecolor=FIGURE_FACECOLOR)
        ax = fig.add_subplot(111, projection='3d')

        # Surface with colormap
        norm = mcolors.TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=max(Z.max(), 0.01))
        surf = ax.plot_surface(
            Y, X, Z, cmap='RdYlGn', norm=norm,
            alpha=0.85, rstride=max(1, len(dates_num) // 50),
            cstride=max(1, len(maturities) // 20), edgecolor='none', antialiased=True,
        )

        ax.set_xlabel('Date', fontsize=10, labelpad=12)
        ax.set_ylabel('Maturity (days)', fontsize=10, labelpad=12)
        ax.set_zlabel('Annualized Basis (%)', fontsize=10, labelpad=10)
        ax.set_title('BTC Futures Term Structure Surface', fontsize=13, fontweight='bold', pad=20)

        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        for label in ax.xaxis.get_ticklabels():
            label.set_fontsize(7)
            label.set_rotation(30)

        ax.view_init(elev=25, azim=-55)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Annualized Basis (%)', pad=0.1)
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / 'term_structure_3d.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved term_structure_3d.png")
        return fig

    # -----------------------------------------------------------------
    # 3. 3D PnL Surface (Regimes x Strategies)
    # -----------------------------------------------------------------
    def plot_3d_pnl_surface(
        self,
        figsize: Tuple[float, float] = (14, 10),
        save: bool = True,
    ) -> plt.Figure:
        """
        3D bar/surface plot of PnL across market regimes and strategies.

        X-axis: strategy, Y-axis: regime, Z-axis: cumulative net PnL.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if 'regime' not in self.trades.columns or 'strategy' not in self.trades.columns:
            logger.warning("Missing regime/strategy columns; returning empty figure.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Regime/strategy data not available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#999999')
            return fig

        pivot = self.trades.pivot_table(
            values='net_pnl', index='regime', columns='strategy', aggfunc='sum', fill_value=0,
        )

        strategies = list(pivot.columns)
        regimes = list(pivot.index)
        x_pos = np.arange(len(strategies))
        y_pos = np.arange(len(regimes))
        xpos, ypos = np.meshgrid(x_pos, y_pos)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        dz = pivot.values.flatten()

        fig = plt.figure(figsize=figsize, facecolor=FIGURE_FACECOLOR)
        ax = fig.add_subplot(111, projection='3d')

        # Color bars by regime
        colors = []
        for yi in range(len(regimes)):
            regime_name = regimes[yi]
            color = REGIME_COLORS.get(regime_name, '#607D8B')
            for _ in range(len(strategies)):
                colors.append(color)

        dx = dy = 0.6
        ax.bar3d(
            xpos, ypos, zpos, dx, dy, dz,
            color=colors, alpha=0.85, edgecolor='#444444', linewidth=0.3,
        )

        ax.set_xticks(x_pos + dx / 2)
        ax.set_xticklabels(strategies, fontsize=8, rotation=15)
        ax.set_yticks(y_pos + dy / 2)
        ax.set_yticklabels(regimes, fontsize=8)
        ax.set_zlabel('Cumulative Net PnL (USD)', fontsize=10)
        ax.set_title('PnL by Strategy and Market Regime', fontsize=13, fontweight='bold', pad=20)
        ax.view_init(elev=25, azim=-45)

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'pnl_surface_3d.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved pnl_surface_3d.png")
        return fig

    # -----------------------------------------------------------------
    # 4. Venue Profitability Heatmap
    # -----------------------------------------------------------------
    def plot_venue_heatmap(
        self,
        figsize: Tuple[float, float] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Heatmap of venue profitability across strategies and metrics.

        Rows: venues, Columns: strategies. Cell values show net PnL, Sharpe,
        win rate, and cost drag in a multi-metric tiled heatmap.

        Returns
        -------
        matplotlib.figure.Figure
        """
        metrics_list = []
        for (venue, strategy), grp in self.trades.groupby(['venue', 'strategy']):
            daily_pnl = grp.set_index('exit_time')['net_pnl'].resample('D').sum().dropna()
            n_days = max(len(daily_pnl), 1)
            total_pnl = grp['net_pnl'].sum()
            mean_daily = daily_pnl.mean() if len(daily_pnl) > 0 else 0
            std_daily = daily_pnl.std() if len(daily_pnl) > 1 else np.nan
            sharpe = (mean_daily / std_daily * np.sqrt(365)) if std_daily and std_daily > 0 else 0
            win_rate = (grp['net_pnl'] > 0).mean() * 100
            total_cost = grp[['fees', 'slippage', 'gas_cost', 'mev_cost']].sum().sum()
            gross = grp['gross_pnl'].sum()
            cost_drag = (total_cost / gross * 100) if gross > 0 else 0

            metrics_list.append({
                'venue': venue, 'strategy': strategy,
                'net_pnl': total_pnl, 'sharpe': sharpe,
                'win_rate': win_rate, 'cost_drag_pct': cost_drag,
            })

        if not metrics_list:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No trade data for venue heatmap', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#999999')
            return fig

        df_metrics = pd.DataFrame(metrics_list)
        metric_names = ['net_pnl', 'sharpe', 'win_rate', 'cost_drag_pct']
        metric_labels = ['Net PnL (USD)', 'Sharpe Ratio', 'Win Rate (%)', 'Cost Drag (%)']
        cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn', 'RdYlGn_r']

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.set_facecolor(FIGURE_FACECOLOR)
        fig.suptitle('Venue Profitability Heatmap', fontsize=14, fontweight='bold', y=1.02)

        for ax, metric, label, cmap in zip(axes.flat, metric_names, metric_labels, cmaps):
            pivot = df_metrics.pivot_table(values=metric, index='venue', columns='strategy', aggfunc='mean')
            fmt = ',.0f' if metric == 'net_pnl' else '.2f'
            sns.heatmap(
                pivot, ax=ax, annot=True, fmt=fmt, cmap=cmap,
                linewidths=0.5, linecolor='#EEEEEE', cbar_kws={'shrink': 0.8},
            )
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', labelsize=8)

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'venue_heatmap.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved venue_heatmap.png")
        return fig

    # -----------------------------------------------------------------
    # 5. Regime Performance Bar Charts
    # -----------------------------------------------------------------
    def plot_regime_performance(
        self,
        figsize: Tuple[float, float] = (14, 10),
        save: bool = True,
    ) -> plt.Figure:
        """
        Grouped bar charts of strategy performance segmented by market regime.

        Panels: Total Return, Sharpe Ratio, Win Rate, Max Drawdown contribution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if 'regime' not in self.trades.columns:
            logger.warning("No regime labels in trade data.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Regime data not available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#999999')
            return fig

        regimes = sorted(self.trades['regime'].unique())
        strategies = sorted(self.trades['strategy'].unique())
        n_strategies = len(strategies)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.set_facecolor(FIGURE_FACECOLOR)
        fig.suptitle('Strategy Performance by Market Regime', fontsize=14, fontweight='bold', y=1.02)

        bar_width = 0.8 / max(n_strategies, 1)
        x = np.arange(len(regimes))

        # Panel 1: Total Net PnL
        ax = axes[0, 0]
        for i, strat in enumerate(strategies):
            vals = []
            for regime in regimes:
                mask = (self.trades['strategy'] == strat) & (self.trades['regime'] == regime)
                vals.append(self.trades.loc[mask, 'net_pnl'].sum())
            color = STRATEGY_COLORS.get(strat, f'C{i}')
            ax.bar(x + i * bar_width, vals, bar_width, label=strat, color=color, edgecolor='white', linewidth=0.3)
        ax.set_xticks(x + bar_width * (n_strategies - 1) / 2)
        ax.set_xticklabels(regimes, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _format_currency(v)))
        ax.legend(fontsize=8, ncol=2)
        _apply_style(ax, title='Net PnL by Regime', ylabel='Net PnL (USD)')

        # Panel 2: Sharpe Ratio per regime-strategy
        ax = axes[0, 1]
        for i, strat in enumerate(strategies):
            vals = []
            for regime in regimes:
                mask = (self.trades['strategy'] == strat) & (self.trades['regime'] == regime)
                subset = self.trades.loc[mask]
                if len(subset) > 1 and 'exit_time' in subset.columns:
                    daily_pnl = subset.set_index('exit_time')['net_pnl'].resample('D').sum().dropna()
                    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
                        vals.append(daily_pnl.mean() / daily_pnl.std() * np.sqrt(365))
                    else:
                        vals.append(0)
                else:
                    vals.append(0)
            color = STRATEGY_COLORS.get(strat, f'C{i}')
            ax.bar(x + i * bar_width, vals, bar_width, label=strat, color=color, edgecolor='white', linewidth=0.3)
        ax.set_xticks(x + bar_width * (n_strategies - 1) / 2)
        ax.set_xticklabels(regimes, fontsize=9)
        ax.axhline(y=0, color='#888888', linewidth=0.5)
        _apply_style(ax, title='Sharpe Ratio by Regime', ylabel='Sharpe Ratio')

        # Panel 3: Win Rate
        ax = axes[1, 0]
        for i, strat in enumerate(strategies):
            vals = []
            for regime in regimes:
                mask = (self.trades['strategy'] == strat) & (self.trades['regime'] == regime)
                subset = self.trades.loc[mask]
                win_rate = (subset['net_pnl'] > 0).mean() * 100 if len(subset) > 0 else 0
                vals.append(win_rate)
            color = STRATEGY_COLORS.get(strat, f'C{i}')
            ax.bar(x + i * bar_width, vals, bar_width, label=strat, color=color, edgecolor='white', linewidth=0.3)
        ax.set_xticks(x + bar_width * (n_strategies - 1) / 2)
        ax.set_xticklabels(regimes, fontsize=9)
        ax.axhline(y=50, color='#888888', linewidth=0.5, linestyle='--')
        ax.set_ylim(0, 100)
        _apply_style(ax, title='Win Rate by Regime', ylabel='Win Rate (%)')

        # Panel 4: Trade count
        ax = axes[1, 1]
        for i, strat in enumerate(strategies):
            vals = []
            for regime in regimes:
                mask = (self.trades['strategy'] == strat) & (self.trades['regime'] == regime)
                vals.append(mask.sum())
            color = STRATEGY_COLORS.get(strat, f'C{i}')
            ax.bar(x + i * bar_width, vals, bar_width, label=strat, color=color, edgecolor='white', linewidth=0.3)
        ax.set_xticks(x + bar_width * (n_strategies - 1) / 2)
        ax.set_xticklabels(regimes, fontsize=9)
        _apply_style(ax, title='Trade Count by Regime', ylabel='Number of Trades')

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'regime_performance.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved regime_performance.png")
        return fig

    # -----------------------------------------------------------------
    # 6. Rolling Metrics (Sharpe, Correlation, Drawdown)
    # -----------------------------------------------------------------
    def plot_rolling_metrics(
        self,
        window: int = 90,
        figsize: Tuple[float, float] = (14, 12),
        save: bool = True,
    ) -> plt.Figure:
        """
        Subplots of rolling Sharpe, rolling BTC correlation, and rolling drawdown.

        Parameters
        ----------
        window : int
            Rolling window in calendar days.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.set_facecolor(FIGURE_FACECOLOR)
        fig.suptitle(f'Rolling Metrics ({window}-Day Window)', fontsize=14, fontweight='bold', y=1.01)

        # Rolling Sharpe
        ax = axes[0]
        rolling_sharpe = _compute_rolling_sharpe(self.returns, window=window)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='#1565C0', linewidth=1.2)
        ax.axhline(y=0, color='#888888', linewidth=0.5)
        ax.axhline(y=2.0, color='#4CAF50', linewidth=0.5, linestyle='--', alpha=0.6)
        ax.fill_between(
            rolling_sharpe.index, rolling_sharpe.values, 0,
            where=rolling_sharpe.values > 0, alpha=0.15, color='#4CAF50',
        )
        ax.fill_between(
            rolling_sharpe.index, rolling_sharpe.values, 0,
            where=rolling_sharpe.values < 0, alpha=0.15, color='#F44336',
        )
        _apply_style(ax, title=f'Rolling {window}d Sharpe Ratio', ylabel='Sharpe Ratio')

        # Rolling BTC correlation
        ax = axes[1]
        btc_key = None
        for key in self.benchmark_curves:
            if 'btc' in key.lower() or 'buy' in key.lower():
                btc_key = key
                break

        if btc_key is not None:
            btc_returns = self.benchmark_curves[btc_key].pct_change().dropna()
            aligned = pd.DataFrame({
                'strategy': self.returns,
                'btc': btc_returns,
            }).dropna()
            rolling_corr = aligned['strategy'].rolling(window=window, min_periods=30).corr(aligned['btc'])
            ax.plot(rolling_corr.index, rolling_corr.values, color='#F57C00', linewidth=1.2)
            ax.axhline(y=0, color='#888888', linewidth=0.5)
            ax.axhline(y=0.3, color='#F44336', linewidth=0.5, linestyle='--', alpha=0.6,
                        label='Target ceiling (0.3)')
            ax.axhline(y=-0.3, color='#F44336', linewidth=0.5, linestyle='--', alpha=0.6)
            ax.fill_between(
                rolling_corr.index, rolling_corr.values, 0, alpha=0.1, color='#F57C00',
            )
            ax.set_ylim(-1, 1)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'BTC benchmark not available for correlation',
                    transform=ax.transAxes, ha='center', va='center', fontsize=11, color='#999999')
        _apply_style(ax, title=f'Rolling {window}d BTC Correlation', ylabel='Correlation')

        # Rolling drawdown
        ax = axes[2]
        rolling_dd = self.drawdown.rolling(window=window, min_periods=1).min()
        ax.fill_between(
            rolling_dd.index, rolling_dd.values, 0,
            color='#EF5350', alpha=0.4,
        )
        ax.plot(rolling_dd.index, rolling_dd.values, color='#C62828', linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.set_ylim(top=0)
        _apply_style(ax, title=f'Rolling {window}d Maximum Drawdown', ylabel='Max Drawdown')

        fig.autofmt_xdate()
        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'rolling_metrics.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved rolling_metrics.png")
        return fig

    # -----------------------------------------------------------------
    # 7. Trade Scatter (PnL vs Holding Period by Venue)
    # -----------------------------------------------------------------
    def plot_trade_scatter(
        self,
        figsize: Tuple[float, float] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Scatter of individual trades: net PnL vs holding period, colored by venue.

        Marker size scales with trade notional. A horizontal line at PnL=0 and
        marginal histograms on each axis provide distributional context.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=figsize, facecolor=FIGURE_FACECOLOR)
        gs = gridspec.GridSpec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )

        ax_main = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

        hold_col = 'holding_period_days' if 'holding_period_days' in self.trades.columns else 'holding_period_hours'
        hold_label = 'Holding Period (days)' if hold_col == 'holding_period_days' else 'Holding Period (hours)'

        venues = self.trades['venue'].unique() if 'venue' in self.trades.columns else ['Unknown']
        for venue in venues:
            mask = self.trades['venue'] == venue if 'venue' in self.trades.columns else pd.Series(True, index=self.trades.index)
            subset = self.trades.loc[mask]
            color = VENUE_SPECIFIC_COLORS.get(venue, VENUE_COLORS.get(
                subset['venue_type'].iloc[0] if 'venue_type' in subset.columns and len(subset) > 0 else 'CEX', '#607D8B'))

            notional = (subset['entry_price'].abs() * subset['quantity'].abs()) if 'entry_price' in subset.columns else pd.Series(100, index=subset.index)
            sizes = np.clip(notional / notional.quantile(0.95) * 40 if len(notional) > 0 else 10, 5, 80)

            ax_main.scatter(
                subset[hold_col], subset['net_pnl'],
                s=sizes, c=color, alpha=0.5, edgecolors='white', linewidth=0.3,
                label=venue, zorder=3,
            )

        ax_main.axhline(y=0, color='#888888', linewidth=0.5, zorder=2)
        ax_main.legend(fontsize=8, loc='upper right', framealpha=0.9)
        _apply_style(ax_main, xlabel=hold_label, ylabel='Net PnL (USD)')

        # Marginal histograms
        ax_histx.hist(
            self.trades[hold_col].dropna(), bins=50, color='#90CAF9', edgecolor='white', linewidth=0.3,
        )
        ax_histx.tick_params(labelbottom=False)
        ax_histx.set_title('Trade Scatter: PnL vs Holding Period', fontsize=12, fontweight='bold')

        ax_histy.hist(
            self.trades['net_pnl'].dropna(), bins=50, orientation='horizontal',
            color='#90CAF9', edgecolor='white', linewidth=0.3,
        )
        ax_histy.tick_params(labelleft=False)

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'trade_scatter.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved trade_scatter.png")
        return fig

    # -----------------------------------------------------------------
    # 8. Funding Rate Comparison
    # -----------------------------------------------------------------
    def plot_funding_rate_comparison(
        self,
        figsize: Tuple[float, float] = (14, 7),
        save: bool = True,
    ) -> plt.Figure:
        """
        Multi-venue funding rate time series overlay.

        Requires self.funding_rates DataFrame with venue name columns and a
        datetime index.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.funding_rates is None or self.funding_rates.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Funding rate data not available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#999999')
            return fig

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig.set_facecolor(FIGURE_FACECOLOR)

        ax_rates = axes[0]
        ax_spread = axes[1]

        for venue_col in self.funding_rates.columns:
            color = VENUE_SPECIFIC_COLORS.get(venue_col, '#607D8B')
            series = self.funding_rates[venue_col].dropna()
            # 24h rolling mean to smooth 8h funding rate noise
            smoothed = series.rolling(window=3, min_periods=1).mean()
            ax_rates.plot(smoothed.index, smoothed.values * 100, label=venue_col,
                          color=color, linewidth=1.0, alpha=0.85)

        ax_rates.axhline(y=0, color='#888888', linewidth=0.5)
        ax_rates.legend(fontsize=8, ncol=3, loc='upper right')
        ax_rates.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f%%'))
        _apply_style(ax_rates, title='Perpetual Funding Rates by Venue', ylabel='Funding Rate (%)')

        # Spread between max and min venue funding
        if self.funding_rates.shape[1] >= 2:
            spread = (self.funding_rates.max(axis=1) - self.funding_rates.min(axis=1)) * 100
            spread_smooth = spread.rolling(window=3, min_periods=1).mean()
            ax_spread.fill_between(spread_smooth.index, spread_smooth.values, 0,
                                   color='#AB47BC', alpha=0.3)
            ax_spread.plot(spread_smooth.index, spread_smooth.values, color='#7B1FA2', linewidth=0.8)
        _apply_style(ax_spread, ylabel='Max-Min Spread (%)')

        fig.autofmt_xdate()
        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'funding_rate_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved funding_rate_comparison.png")
        return fig

    # -----------------------------------------------------------------
    # 9. Basis Spread with Entry/Exit Signals
    # -----------------------------------------------------------------
    def plot_basis_spread_analysis(
        self,
        basis_series: Optional[pd.Series] = None,
        figsize: Tuple[float, float] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Basis spread time series with trade entry/exit signal markers overlaid.

        Parameters
        ----------
        basis_series : pd.Series, optional
            Annualized basis spread series. If not provided, derived from
            term_structure at the shortest maturity.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if basis_series is None and self.term_structure is not None:
            ts = self.term_structure.copy()
            ts['date'] = pd.to_datetime(ts['date'])
            # Use the nearest-expiry basis as the reference spread
            shortest_mat = ts.groupby('date').apply(
                lambda g: g.loc[g['maturity_days'].idxmin(), 'basis_annualized']
            )
            basis_series = shortest_mat.sort_index()

        if basis_series is None or basis_series.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Basis spread data not available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#999999')
            return fig

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig.set_facecolor(FIGURE_FACECOLOR)

        ax_basis = axes[0]
        ax_basis.plot(basis_series.index, basis_series.values * 100, color='#1565C0',
                      linewidth=1.0, label='Annualized Basis', zorder=2)

        # Rolling mean and bands (Bollinger-style on basis)
        window = 30
        basis_ma = basis_series.rolling(window=window, min_periods=10).mean()
        basis_std = basis_series.rolling(window=window, min_periods=10).std()
        upper = (basis_ma + 2 * basis_std) * 100
        lower = (basis_ma - 2 * basis_std) * 100
        ax_basis.plot(basis_ma.index, basis_ma.values * 100, color='#FFA726', linewidth=0.8,
                      linestyle='--', label=f'{window}d MA', zorder=2)
        ax_basis.fill_between(basis_ma.index, lower, upper, color='#FFA726', alpha=0.1, zorder=1)

        # Overlay entry/exit markers from calendar spread trades
        calendar_trades = self.trades[self.trades['strategy'].str.contains('Calendar', case=False, na=False)]
        if len(calendar_trades) > 0:
            for _, trade in calendar_trades.iterrows():
                entry_t = pd.Timestamp(trade['entry_time'])
                exit_t = pd.Timestamp(trade['exit_time'])
                if entry_t in basis_series.index or len(basis_series) > 0:
                    nearest_entry = basis_series.index[basis_series.index.searchsorted(entry_t, side='right') - 1] \
                        if entry_t <= basis_series.index.max() else None
                    nearest_exit = basis_series.index[basis_series.index.searchsorted(exit_t, side='right') - 1] \
                        if exit_t <= basis_series.index.max() else None
                    if nearest_entry is not None:
                        ax_basis.scatter(nearest_entry, basis_series.loc[nearest_entry] * 100,
                                         marker='^', color='#4CAF50', s=30, zorder=4, alpha=0.7)
                    if nearest_exit is not None:
                        ax_basis.scatter(nearest_exit, basis_series.loc[nearest_exit] * 100,
                                         marker='v', color='#F44336', s=30, zorder=4, alpha=0.7)

        ax_basis.axhline(y=0, color='#888888', linewidth=0.5)
        ax_basis.legend(fontsize=8)
        _apply_style(ax_basis, title='BTC Futures Basis Spread Analysis', ylabel='Annualized Basis (%)')

        # Lower panel: z-score of basis
        z_score = (basis_series - basis_ma) / basis_std.replace(0, np.nan)
        ax_z = axes[1]
        ax_z.bar(z_score.index, z_score.values, width=1.0, color=np.where(z_score.values > 0, '#4CAF50', '#F44336'),
                 alpha=0.6, linewidth=0)
        ax_z.axhline(y=0, color='#888888', linewidth=0.5)
        ax_z.axhline(y=2, color='#F44336', linewidth=0.5, linestyle='--', alpha=0.5)
        ax_z.axhline(y=-2, color='#4CAF50', linewidth=0.5, linestyle='--', alpha=0.5)
        _apply_style(ax_z, ylabel='Basis Z-Score')

        fig.autofmt_xdate()
        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'basis_spread_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved basis_spread_analysis.png")
        return fig

    # -----------------------------------------------------------------
    # 10. Strategy Correlation Matrix with Hierarchical Clustering
    # -----------------------------------------------------------------
    def plot_correlation_matrix(
        self,
        figsize: Tuple[float, float] = (10, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Heatmap of daily return correlations across strategies with hierarchical
        clustering dendrogram ordering.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if 'strategy' not in self.trades.columns:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Strategy labels not available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#999999')
            return fig

        # Build daily PnL matrix per strategy
        strategy_daily = {}
        for strategy, grp in self.trades.groupby('strategy'):
            daily_pnl = grp.set_index('exit_time')['net_pnl'].resample('D').sum()
            strategy_daily[strategy] = daily_pnl

        pnl_matrix = pd.DataFrame(strategy_daily).fillna(0)
        if pnl_matrix.shape[1] < 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Insufficient strategies for correlation analysis',
                    transform=ax.transAxes, ha='center', va='center', fontsize=14, color='#999999')
            return fig

        corr = pnl_matrix.corr()

        # Hierarchical clustering for reordering
        from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
        from scipy.spatial.distance import squareform

        dist_matrix = np.clip(1 - corr.values, 0, 2)
        np.fill_diagonal(dist_matrix, 0)
        # Ensure symmetry
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method='ward')
        order = leaves_list(Z)
        corr_ordered = corr.iloc[order, order]

        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(FIGURE_FACECOLOR)

        mask = np.triu(np.ones_like(corr_ordered, dtype=bool), k=1)
        sns.heatmap(
            corr_ordered, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, linecolor='#EEEEEE',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            ax=ax,
        )
        _apply_style(ax, title='Strategy Daily PnL Correlation Matrix (Clustered)')
        ax.tick_params(axis='both', labelsize=9)

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'correlation_matrix.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved correlation_matrix.png")
        return fig

    # -----------------------------------------------------------------
    # 11. Drawdown Analysis (Underwater Chart)
    # -----------------------------------------------------------------
    def plot_drawdown_analysis(
        self,
        top_n: int = 5,
        figsize: Tuple[float, float] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Underwater chart with the top N drawdown periods highlighted and
        crisis event annotations.

        Parameters
        ----------
        top_n : int
            Number of largest drawdowns to highlight.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
        fig.set_facecolor(FIGURE_FACECOLOR)

        # Top panel: equity curve for context
        ax_eq = axes[0]
        ax_eq.plot(self.equity_curve.index, self.equity_curve.values, color='#1565C0', linewidth=1.0)
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_currency(x)))
        _apply_style(ax_eq, title='Equity Curve & Drawdown Analysis', ylabel='Portfolio Value')

        # Bottom panel: underwater chart
        ax_dd = axes[1]
        ax_dd.fill_between(
            self.drawdown.index, self.drawdown.values * 100, 0,
            color='#EF5350', alpha=0.35, zorder=2,
        )
        ax_dd.plot(self.drawdown.index, self.drawdown.values * 100, color='#C62828', linewidth=0.7, zorder=3)

        # Identify top N drawdown periods
        dd_series = self.drawdown.copy()
        dd_periods = []
        temp = dd_series.copy()
        for _ in range(top_n):
            if temp.min() >= 0:
                break
            trough_idx = temp.idxmin()
            trough_val = temp.loc[trough_idx]

            # Walk backward to find start (last zero-crossing before trough)
            before_trough = temp.loc[:trough_idx]
            zero_crossings = before_trough[before_trough >= 0]
            start_idx = zero_crossings.index[-1] if len(zero_crossings) > 0 else before_trough.index[0]

            # Walk forward to find recovery (first zero after trough)
            after_trough = temp.loc[trough_idx:]
            recovery_points = after_trough[after_trough >= 0]
            end_idx = recovery_points.index[0] if len(recovery_points) > 0 else after_trough.index[-1]

            dd_periods.append({
                'start': start_idx, 'trough': trough_idx, 'end': end_idx,
                'depth': trough_val,
                'duration_days': (end_idx - start_idx).days,
            })

            # Mask this period to find next
            temp.loc[start_idx:end_idx] = 0

        # Highlight top drawdown periods
        highlight_colors = ['#D32F2F', '#E64A19', '#F57C00', '#FFA000', '#FBC02D']
        for i, dd_info in enumerate(dd_periods):
            color = highlight_colors[i % len(highlight_colors)]
            ax_dd.axvspan(dd_info['start'], dd_info['end'], alpha=0.12, color=color, zorder=1)
            ax_dd.annotate(
                f"#{i + 1}: {dd_info['depth'] * 100:.2f}% ({dd_info['duration_days']}d)",
                xy=(dd_info['trough'], dd_info['depth'] * 100),
                xytext=(0, -15), textcoords='offset points',
                fontsize=7, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
            )

        # Crisis annotations
        for event_name, dates in CRISIS_EVENTS.items():
            start = pd.Timestamp(dates['start'])
            if start >= self.drawdown.index.min() and start <= self.drawdown.index.max():
                ax_dd.axvline(x=start, color='#9C27B0', linewidth=0.6, linestyle=':', alpha=0.7)
                ax_dd.annotate(
                    event_name, xy=(start, ax_dd.get_ylim()[0] * 0.95),
                    fontsize=6, color='#7B1FA2', rotation=90, va='bottom',
                )

        ax_dd.set_ylim(top=0.5)
        ax_dd.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))
        _apply_style(ax_dd, ylabel='Drawdown (%)')

        fig.autofmt_xdate()
        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'drawdown_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved drawdown_analysis.png")
        return fig

    # -----------------------------------------------------------------
    # 12. Monthly Returns Heatmap
    # -----------------------------------------------------------------
    def plot_monthly_returns_heatmap(
        self,
        figsize: Tuple[float, float] = (14, 6),
        save: bool = True,
    ) -> plt.Figure:
        """
        Calendar heatmap of monthly portfolio returns (rows = years, cols = months).

        Returns
        -------
        matplotlib.figure.Figure
        """
        monthly_returns = self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = monthly_returns.index.to_period('M')

        # Build year x month pivot
        years = sorted(monthly_returns.index.year.unique())
        months = range(1, 13)
        data = pd.DataFrame(index=years, columns=months, dtype=float)
        for period, ret in monthly_returns.items():
            data.loc[period.year, period.month] = ret * 100

        data = data.astype(float)

        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(FIGURE_FACECOLOR)

        # Diverging colormap centered at zero
        vmax = max(abs(data.min().min()), abs(data.max().max()), 1)
        sns.heatmap(
            data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            vmin=-vmax, vmax=vmax, linewidths=1, linecolor='white',
            cbar_kws={'label': 'Return (%)', 'shrink': 0.8},
            ax=ax, annot_kws={'fontsize': 9},
        )

        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels, fontsize=9)
        ax.set_yticklabels([str(y) for y in years], fontsize=9, rotation=0)

        # Annotate annual totals in a right-side column effect
        annual_returns = self.returns.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100
        for i, yr in enumerate(years):
            yr_mask = annual_returns.index.year == yr
            if yr_mask.any():
                ann_ret = annual_returns[yr_mask].values[0]
                ax.text(
                    12.6, i + 0.5, f'{ann_ret:+.1f}%',
                    fontsize=9, fontweight='bold', va='center',
                    color='#4CAF50' if ann_ret > 0 else '#F44336',
                )

        _apply_style(ax, title='Monthly Returns Heatmap (%)')
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel('Year', fontsize=10)

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'monthly_returns_heatmap.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved monthly_returns_heatmap.png")
        return fig

    # -----------------------------------------------------------------
    # 13. Risk Metrics Dashboard (VaR, CVaR, Tail Risk)
    # -----------------------------------------------------------------
    def plot_risk_metrics_dashboard(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save: bool = True,
    ) -> plt.Figure:
        """
        Multi-panel risk dashboard: return distribution, VaR/CVaR, QQ plot,
        rolling volatility, tail ratio, and leverage utilization.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=figsize, facecolor=FIGURE_FACECOLOR)
        fig.suptitle('Risk Metrics Dashboard', fontsize=15, fontweight='bold', y=1.01)
        gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

        returns = self.returns.dropna()

        # Panel 1: Return distribution with VaR/CVaR lines
        ax1 = fig.add_subplot(gs[0, 0])
        n_bins = min(100, max(30, len(returns) // 10))
        ax1.hist(returns * 100, bins=n_bins, color='#90CAF9', edgecolor='white',
                 linewidth=0.3, density=True, alpha=0.8, zorder=2)

        # Kernel density overlay
        from scipy.stats import gaussian_kde
        if len(returns) > 10:
            kde = gaussian_kde(returns * 100)
            x_range = np.linspace(returns.min() * 100 - 1, returns.max() * 100 + 1, 300)
            ax1.plot(x_range, kde(x_range), color='#1565C0', linewidth=1.5, zorder=3)

        # VaR and CVaR markers
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        ax1.axvline(var_95, color='#FF9800', linewidth=1.2, linestyle='--', label=f'VaR 95%: {var_95:.3f}%')
        ax1.axvline(var_99, color='#F44336', linewidth=1.2, linestyle='--', label=f'VaR 99%: {var_99:.3f}%')
        ax1.axvline(cvar_95, color='#9C27B0', linewidth=1.2, linestyle=':', label=f'CVaR 95%: {cvar_95:.3f}%')
        ax1.legend(fontsize=7, loc='upper right')
        _apply_style(ax1, title='Daily Return Distribution', xlabel='Return (%)', ylabel='Density')

        # Panel 2: QQ plot (normal reference)
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats as sp_stats
        sorted_returns = np.sort(returns.values)
        theoretical_quantiles = sp_stats.norm.ppf(
            np.linspace(0.001, 0.999, len(sorted_returns))
        )
        ax2.scatter(theoretical_quantiles, sorted_returns * 100, s=4, color='#1565C0', alpha=0.5, zorder=3)
        # 45-degree reference line
        q_min, q_max = theoretical_quantiles.min(), theoretical_quantiles.max()
        ref_line = np.array([q_min, q_max])
        mean_r, std_r = returns.mean() * 100, returns.std() * 100
        ax2.plot(ref_line, ref_line * std_r + mean_r, color='#F44336', linewidth=1.0,
                 linestyle='--', label='Normal reference', zorder=2)
        ax2.legend(fontsize=8)
        _apply_style(ax2, title='QQ Plot vs Normal', xlabel='Theoretical Quantiles', ylabel='Sample Quantiles (%)')

        # Panel 3: Rolling volatility (30d and 90d)
        ax3 = fig.add_subplot(gs[1, 0])
        vol_30 = returns.rolling(30, min_periods=10).std() * np.sqrt(365) * 100
        vol_90 = returns.rolling(90, min_periods=30).std() * np.sqrt(365) * 100
        ax3.plot(vol_30.index, vol_30.values, color='#FF9800', linewidth=1.0, label='30d Rolling Vol', alpha=0.8)
        ax3.plot(vol_90.index, vol_90.values, color='#1565C0', linewidth=1.2, label='90d Rolling Vol')
        ax3.fill_between(vol_30.index, vol_30.values, 0, color='#FFE0B2', alpha=0.3)
        ax3.legend(fontsize=8)
        _apply_style(ax3, title='Rolling Annualized Volatility', ylabel='Volatility (%)')

        # Panel 4: Rolling VaR (95%)
        ax4 = fig.add_subplot(gs[1, 1])
        rolling_var = returns.rolling(90, min_periods=30).quantile(0.05) * 100
        rolling_cvar = returns.rolling(90, min_periods=30).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else x.min(),
            raw=False,
        ) * 100
        ax4.plot(rolling_var.index, rolling_var.values, color='#FF9800', linewidth=1.0, label='90d Rolling VaR 95%')
        ax4.plot(rolling_cvar.index, rolling_cvar.values, color='#F44336', linewidth=1.0, label='90d Rolling CVaR 95%')
        ax4.fill_between(rolling_cvar.index, rolling_cvar.values, rolling_var.values,
                         color='#FFCDD2', alpha=0.4)
        ax4.axhline(y=0, color='#888888', linewidth=0.5)
        ax4.legend(fontsize=8)
        _apply_style(ax4, title='Rolling Value-at-Risk (95%)', ylabel='VaR / CVaR (%)')

        # Panel 5: Tail ratio over time (gain/loss at 95th/5th percentile)
        ax5 = fig.add_subplot(gs[2, 0])
        tail_ratio = returns.rolling(90, min_periods=30).apply(
            lambda x: abs(np.percentile(x, 95) / np.percentile(x, 5)) if np.percentile(x, 5) != 0 else np.nan,
            raw=True,
        )
        ax5.plot(tail_ratio.index, tail_ratio.values, color='#7B1FA2', linewidth=1.0)
        ax5.axhline(y=1.0, color='#888888', linewidth=0.5, linestyle='--', label='Symmetric tails')
        ax5.fill_between(tail_ratio.index, tail_ratio.values, 1.0,
                         where=tail_ratio.values > 1.0, alpha=0.15, color='#4CAF50')
        ax5.fill_between(tail_ratio.index, tail_ratio.values, 1.0,
                         where=tail_ratio.values < 1.0, alpha=0.15, color='#F44336')
        ax5.legend(fontsize=8)
        _apply_style(ax5, title='Rolling Tail Ratio (90d)', ylabel='|P95| / |P5|')

        # Panel 6: Summary statistics table
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')

        skew = returns.skew()
        kurt = returns.kurtosis()
        jarque_bera = len(returns) / 6 * (skew ** 2 + (kurt ** 2) / 4)
        max_daily_loss = returns.min() * 100
        max_daily_gain = returns.max() * 100

        stats_data = [
            ['Annualized Vol', f'{returns.std() * np.sqrt(365) * 100:.2f}%'],
            ['Daily VaR (95%)', f'{var_95:.4f}%'],
            ['Daily VaR (99%)', f'{var_99:.4f}%'],
            ['Daily CVaR (95%)', f'{cvar_95:.4f}%'],
            ['Skewness', f'{skew:.3f}'],
            ['Excess Kurtosis', f'{kurt:.3f}'],
            ['Jarque-Bera Stat', f'{jarque_bera:.1f}'],
            ['Max Daily Loss', f'{max_daily_loss:.4f}%'],
            ['Max Daily Gain', f'{max_daily_gain:.4f}%'],
            ['Positive Days', f'{(returns > 0).mean() * 100:.1f}%'],
        ]

        table = ax6.table(
            cellText=stats_data, colLabels=['Metric', 'Value'],
            cellLoc='left', loc='center', colColours=['#E3F2FD', '#E3F2FD'],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#BBDEFB')
            if row == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#BBDEFB')
        ax6.set_title('Risk Summary', fontsize=11, fontweight='bold', pad=15)

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'risk_dashboard.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved risk_dashboard.png")
        return fig

    # -----------------------------------------------------------------
    # 14. Capacity Analysis (Sharpe Degradation vs AUM)
    # -----------------------------------------------------------------
    def plot_capacity_analysis(
        self,
        capacity_data: Optional[Dict[str, pd.DataFrame]] = None,
        figsize: Tuple[float, float] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Sharpe ratio degradation curve as a function of deployed capital per venue.

        Uses a square-root market impact model: Sharpe degrades proportionally to
        sqrt(AUM / venue_daily_volume). If capacity_data is not supplied, generates
        a theoretical curve from trade-level volume participation rates.

        Parameters
        ----------
        capacity_data : dict of pd.DataFrame, optional
            Per-venue DataFrames with columns: aum, sharpe_estimate.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.set_facecolor(FIGURE_FACECOLOR)
        fig.suptitle('Capacity Analysis: Sharpe Degradation by Venue', fontsize=14, fontweight='bold', y=1.02)

        if capacity_data is not None:
            ax = axes[0]
            for venue, df in capacity_data.items():
                color = VENUE_SPECIFIC_COLORS.get(venue, '#607D8B')
                ax.plot(df['aum'] / 1e6, df['sharpe_estimate'], color=color,
                        linewidth=1.5, marker='o', markersize=4, label=venue)
            ax.axhline(y=2.0, color='#4CAF50', linewidth=0.5, linestyle='--', label='Target Sharpe (2.0)')
            ax.legend(fontsize=8)
            _apply_style(ax, title='Sharpe vs AUM by Venue', xlabel='AUM ($M)', ylabel='Estimated Sharpe')
        else:
            # Theoretical model using sqrt market impact
            ax = axes[0]
            # Estimate base Sharpe from backtest
            base_sharpe = self.returns.mean() / self.returns.std() * np.sqrt(365) if self.returns.std() > 0 else 0
            venue_capacities = {
                'Binance': 500e6, 'CME': 200e6, 'Deribit': 150e6,
                'Hyperliquid': 50e6, 'dYdX': 30e6, 'GMX': 20e6,
            }
            aum_range = np.linspace(0.1e6, 50e6, 200)
            for venue, capacity in venue_capacities.items():
                color = VENUE_SPECIFIC_COLORS.get(venue, '#607D8B')
                # Impact model: Sharpe * (1 - sqrt(AUM / capacity))
                degraded = base_sharpe * np.maximum(1 - np.sqrt(aum_range / capacity), 0)
                ax.plot(aum_range / 1e6, degraded, color=color, linewidth=1.5, label=venue)

            ax.axhline(y=2.0, color='#4CAF50', linewidth=0.5, linestyle='--', alpha=0.7, label='Sharpe = 2.0')
            ax.axhline(y=0, color='#888888', linewidth=0.5)
            ax.legend(fontsize=7, ncol=2)
            _apply_style(ax, title='Theoretical Sharpe Degradation', xlabel='AUM ($M)', ylabel='Sharpe Ratio')

        # Right panel: venue allocation pie chart
        ax_pie = axes[1]
        venue_pnl = self.trades.groupby('venue')['net_pnl'].sum()
        venue_pnl = venue_pnl[venue_pnl > 0]
        if len(venue_pnl) > 0:
            colors_pie = [VENUE_SPECIFIC_COLORS.get(v, '#607D8B') for v in venue_pnl.index]
            wedges, texts, autotexts = ax_pie.pie(
                venue_pnl.values, labels=venue_pnl.index, colors=colors_pie,
                autopct='%1.1f%%', startangle=90, pctdistance=0.75,
                textprops={'fontsize': 9},
            )
            for autotext in autotexts:
                autotext.set_fontsize(8)
            ax_pie.set_title('Venue PnL Contribution', fontsize=11, fontweight='bold')
        else:
            ax_pie.text(0.5, 0.5, 'No positive venue PnL', transform=ax_pie.transAxes,
                        ha='center', va='center', fontsize=12, color='#999999')

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'capacity_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved capacity_analysis.png")
        return fig

    # -----------------------------------------------------------------
    # 15. Roll Cost vs Funding Cost Comparison
    # -----------------------------------------------------------------
    def plot_roll_cost_comparison(
        self,
        figsize: Tuple[float, float] = (14, 8),
        save: bool = True,
    ) -> plt.Figure:
        """
        Stacked bar comparison of roll costs vs funding costs across venues.

        Roll costs are computed from Calendar/Roll strategy trades.
        Funding costs are computed from perpetual-based strategy trades.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.set_facecolor(FIGURE_FACECOLOR)
        fig.suptitle('Roll Costs vs Funding Costs by Venue', fontsize=14, fontweight='bold', y=1.02)

        # Classify trades into roll-based and funding-based
        roll_strategies = ['Calendar', 'Roll']
        funding_strategies = ['Cross-Venue', 'Synthetic']

        venues = sorted(self.trades['venue'].unique()) if 'venue' in self.trades.columns else []
        roll_costs_by_venue = {}
        funding_costs_by_venue = {}

        for venue in venues:
            vmask = self.trades['venue'] == venue
            # Roll costs: total execution costs for roll/calendar trades
            roll_mask = vmask & self.trades['strategy'].isin(roll_strategies)
            roll_trades = self.trades.loc[roll_mask]
            roll_cost = roll_trades[['fees', 'slippage']].sum().sum() if len(roll_trades) > 0 else 0
            roll_costs_by_venue[venue] = roll_cost

            # Funding costs: gas + mev + fees for perp-based trades
            fund_mask = vmask & self.trades['strategy'].isin(funding_strategies)
            fund_trades = self.trades.loc[fund_mask]
            funding_cost = fund_trades[['fees', 'slippage', 'gas_cost', 'mev_cost']].sum().sum() if len(fund_trades) > 0 else 0
            funding_costs_by_venue[venue] = funding_cost

        # Left panel: stacked bar chart
        ax = axes[0]
        if venues:
            x = np.arange(len(venues))
            roll_vals = [roll_costs_by_venue.get(v, 0) for v in venues]
            fund_vals = [funding_costs_by_venue.get(v, 0) for v in venues]
            bar_colors_roll = [VENUE_SPECIFIC_COLORS.get(v, '#607D8B') for v in venues]

            ax.bar(x, roll_vals, 0.35, label='Roll Costs', color='#1976D2', edgecolor='white', linewidth=0.3)
            ax.bar(x + 0.35, fund_vals, 0.35, label='Funding Costs', color='#F57C00', edgecolor='white', linewidth=0.3)
            ax.set_xticks(x + 0.175)
            ax.set_xticklabels(venues, fontsize=9, rotation=30)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _format_currency(v)))
            ax.legend(fontsize=9)
        _apply_style(ax, title='Absolute Cost Comparison', ylabel='Total Cost (USD)')

        # Right panel: cost as % of gross PnL
        ax2 = axes[1]
        if venues:
            roll_pct = []
            fund_pct = []
            for venue in venues:
                vmask = self.trades['venue'] == venue
                gross = self.trades.loc[vmask, 'gross_pnl'].sum()
                gross = max(gross, 1)  # Avoid division by zero
                roll_pct.append(roll_costs_by_venue.get(venue, 0) / gross * 100)
                fund_pct.append(funding_costs_by_venue.get(venue, 0) / gross * 100)

            ax2.bar(x, roll_pct, 0.35, label='Roll Cost Drag', color='#1976D2', edgecolor='white', linewidth=0.3)
            ax2.bar(x + 0.35, fund_pct, 0.35, label='Funding Cost Drag', color='#F57C00', edgecolor='white', linewidth=0.3)
            ax2.set_xticks(x + 0.175)
            ax2.set_xticklabels(venues, fontsize=9, rotation=30)
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))
            ax2.legend(fontsize=9)
        _apply_style(ax2, title='Cost as % of Gross PnL', ylabel='Cost Drag (%)')

        fig.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'roll_cost_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved roll_cost_comparison.png")
        return fig

    # -----------------------------------------------------------------
    # 16. Generate Full Report
    # -----------------------------------------------------------------
    def generate_full_report(
        self,
        basis_series: Optional[pd.Series] = None,
        capacity_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate all plots and persist to the output directory.

        Returns a dictionary mapping plot names to Figure objects.

        Parameters
        ----------
        basis_series : pd.Series, optional
            Passed through to plot_basis_spread_analysis.
        capacity_data : dict, optional
            Passed through to plot_capacity_analysis.

        Returns
        -------
        dict of str to matplotlib.figure.Figure
        """
        figures: Dict[str, plt.Figure] = {}
        plot_methods = [
            ('equity_curve', lambda: self.plot_equity_curve()),
            ('term_structure_3d', lambda: self.plot_3d_term_structure()),
            ('pnl_surface_3d', lambda: self.plot_3d_pnl_surface()),
            ('venue_heatmap', lambda: self.plot_venue_heatmap()),
            ('regime_performance', lambda: self.plot_regime_performance()),
            ('rolling_metrics', lambda: self.plot_rolling_metrics()),
            ('trade_scatter', lambda: self.plot_trade_scatter()),
            ('funding_rate_comparison', lambda: self.plot_funding_rate_comparison()),
            ('basis_spread_analysis', lambda: self.plot_basis_spread_analysis(basis_series=basis_series)),
            ('correlation_matrix', lambda: self.plot_correlation_matrix()),
            ('drawdown_analysis', lambda: self.plot_drawdown_analysis()),
            ('monthly_returns_heatmap', lambda: self.plot_monthly_returns_heatmap()),
            ('risk_dashboard', lambda: self.plot_risk_metrics_dashboard()),
            ('capacity_analysis', lambda: self.plot_capacity_analysis(capacity_data=capacity_data)),
            ('roll_cost_comparison', lambda: self.plot_roll_cost_comparison()),
        ]

        for name, plot_fn in plot_methods:
            try:
                logger.info("Generating %s...", name)
                fig = plot_fn()
                figures[name] = fig
                plt.close(fig)
            except Exception as e:
                logger.error("Failed to generate %s: %s", name, str(e))

        logger.info("Full report generated: %d/%d plots succeeded.", len(figures), len(plot_methods))
        return figures


# =============================================================================
# PERFORMANCE REPORT CLASS (PDF-READY)
# =============================================================================

class PerformanceReport:
    """
    PDF-ready performance report combining all visualizations with summary tables.

    Designed to produce a structured multi-page report aligned with
    Project Specification deliverables.

    Parameters
    ----------
    visualization : BacktestVisualization
        Initialized visualization engine with all data loaded.
    report_title : str
        Title for the report header.
    """

    def __init__(
        self,
        visualization: BacktestVisualization,
        report_title: str = 'BTC Futures Curve StatArb - Backtest Report',
    ):
        self.viz = visualization
        self.report_title = report_title
        self.figures: Dict[str, plt.Figure] = {}

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute the full set of summary statistics for the report header page.

        Returns
        -------
        dict
            Nested dictionary of performance, risk, trading, and cost metrics.
        """
        returns = self.viz.returns
        equity = self.viz.equity_curve
        trades = self.viz.trades

        # Performance metrics
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
        n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.01)
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        annualized_vol = returns.std() * np.sqrt(365)
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else np.nan
        sortino = (returns.mean() * 365 / (downside_vol if downside_vol and downside_vol > 0 else np.nan))

        dd = _compute_drawdown_series(equity)
        max_dd = dd.min()
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else np.nan

        # Trade statistics
        n_trades = len(trades)
        n_winners = (trades['net_pnl'] > 0).sum() if n_trades > 0 else 0
        win_rate = n_winners / n_trades if n_trades > 0 else 0
        gross_profit = trades.loc[trades['net_pnl'] > 0, 'net_pnl'].sum() if n_trades > 0 else 0
        gross_loss = abs(trades.loc[trades['net_pnl'] < 0, 'net_pnl'].sum()) if n_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        avg_holding = trades['holding_period_hours'].mean() / 24 if 'holding_period_hours' in trades.columns and n_trades > 0 else 0

        # Cost metrics
        total_fees = trades['fees'].sum() if 'fees' in trades.columns else 0
        total_slippage = trades['slippage'].sum() if 'slippage' in trades.columns else 0
        total_gas = trades['gas_cost'].sum() if 'gas_cost' in trades.columns else 0
        total_mev = trades['mev_cost'].sum() if 'mev_cost' in trades.columns else 0
        total_costs = total_fees + total_slippage + total_gas + total_mev
        gross_pnl_total = trades['gross_pnl'].sum() if 'gross_pnl' in trades.columns else 0
        cost_drag = total_costs / gross_pnl_total if gross_pnl_total > 0 else 0

        # Risk metrics
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()

        # BTC correlation
        btc_corr = np.nan
        for key in self.viz.benchmark_curves:
            if 'btc' in key.lower() or 'buy' in key.lower():
                btc_ret = self.viz.benchmark_curves[key].pct_change().dropna()
                aligned = pd.DataFrame({'strat': returns, 'btc': btc_ret}).dropna()
                if len(aligned) > 10:
                    btc_corr = aligned['strat'].corr(aligned['btc'])
                break

        return {
            'performance': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_dd,
                'btc_correlation': btc_corr,
            },
            'trading': {
                'total_trades': n_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_holding_days': avg_holding,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
            },
            'costs': {
                'total_fees': total_fees,
                'total_slippage': total_slippage,
                'total_gas': total_gas,
                'total_mev': total_mev,
                'total_costs': total_costs,
                'cost_drag_pct': cost_drag * 100,
            },
            'risk': {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
            },
        }

    def compute_per_strategy_breakdown(self) -> pd.DataFrame:
        """
        Compute per-strategy performance summary for the report table.

        Returns
        -------
        pd.DataFrame
            One row per strategy with metrics: total_return, sharpe, max_dd,
            btc_correlation, win_rate, avg_roll_cost, avg_funding_cost,
            trade_count, profit_factor.
        """
        trades = self.viz.trades
        results = []

        for strategy, grp in trades.groupby('strategy'):
            daily_pnl = grp.set_index('exit_time')['net_pnl'].resample('D').sum().dropna()
            total_pnl = grp['net_pnl'].sum()
            n_trades = len(grp)
            win_rate = (grp['net_pnl'] > 0).mean() if n_trades > 0 else 0

            # Sharpe (annualized with sqrt(365))
            sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0

            # Max drawdown from cumulative PnL
            cumulative = daily_pnl.cumsum() + self.viz.initial_capital
            dd = _compute_drawdown_series(cumulative)
            max_dd = dd.min() if len(dd) > 0 else 0

            # Profit factor
            gross_win = grp.loc[grp['net_pnl'] > 0, 'net_pnl'].sum()
            gross_loss_val = abs(grp.loc[grp['net_pnl'] < 0, 'net_pnl'].sum())
            pf = gross_win / gross_loss_val if gross_loss_val > 0 else np.inf

            # BTC correlation
            btc_corr = np.nan
            for key in self.viz.benchmark_curves:
                if 'btc' in key.lower() or 'buy' in key.lower():
                    btc_ret = self.viz.benchmark_curves[key].pct_change().dropna()
                    aligned = pd.DataFrame({'strat': daily_pnl, 'btc': btc_ret}).dropna()
                    if len(aligned) > 10:
                        btc_corr = aligned['strat'].corr(aligned['btc'])
                    break

            # Cost breakdown
            avg_roll_cost = grp[['fees', 'slippage']].sum().sum() / max(n_trades, 1)
            avg_funding_cost = grp[['gas_cost', 'mev_cost']].sum().sum() / max(n_trades, 1)

            results.append({
                'strategy': strategy,
                'total_pnl': total_pnl,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'btc_correlation': btc_corr,
                'win_rate': win_rate,
                'avg_roll_cost': avg_roll_cost,
                'avg_funding_cost': avg_funding_cost,
                'trade_count': n_trades,
                'profit_factor': pf,
            })

        return pd.DataFrame(results).set_index('strategy')

    def plot_summary_page(
        self,
        figsize: Tuple[float, float] = (16, 20),
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate the executive summary page with key metrics table,
        mini equity curve, and per-strategy breakdown.

        Returns
        -------
        matplotlib.figure.Figure
        """
        stats = self.compute_summary_statistics()
        strategy_df = self.compute_per_strategy_breakdown()

        fig = plt.figure(figsize=figsize, facecolor='#FFFFFF')
        gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3, height_ratios=[0.8, 2, 2, 2.5])

        # Title bar
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.text(0.5, 0.7, self.report_title, fontsize=18, fontweight='bold',
                      ha='center', va='center', color='#1565C0')
        ax_title.text(0.5, 0.25,
                      f"Period: {self.viz.equity_curve.index[0].strftime('%Y-%m-%d')} to "
                      f"{self.viz.equity_curve.index[-1].strftime('%Y-%m-%d')} | "
                      f"Initial Capital: {_format_currency(self.viz.initial_capital)}",
                      fontsize=11, ha='center', va='center', color='#616161')

        # Key metrics table
        ax_metrics = fig.add_subplot(gs[1, 0])
        ax_metrics.axis('off')
        perf = stats['performance']
        risk = stats['risk']
        trading = stats['trading']

        metric_rows = [
            ['Total Return', _format_pct(perf['total_return'])],
            ['Annualized Return', _format_pct(perf['annualized_return'])],
            ['Sharpe Ratio', f"{perf['sharpe_ratio']:.2f}"],
            ['Sortino Ratio', f"{perf['sortino_ratio']:.2f}" if not np.isnan(perf['sortino_ratio']) else 'N/A'],
            ['Max Drawdown', _format_pct(perf['max_drawdown'])],
            ['Calmar Ratio', f"{perf['calmar_ratio']:.2f}" if not np.isnan(perf['calmar_ratio']) else 'N/A'],
            ['BTC Correlation', f"{perf['btc_correlation']:.3f}" if not np.isnan(perf['btc_correlation']) else 'N/A'],
            ['Win Rate', _format_pct(trading['win_rate'])],
            ['Profit Factor', f"{trading['profit_factor']:.2f}"],
            ['Total Trades', f"{trading['total_trades']:,}"],
        ]

        table = ax_metrics.table(
            cellText=metric_rows, colLabels=['Metric', 'Value'],
            cellLoc='left', loc='center', colColours=['#E3F2FD', '#E3F2FD'],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#BBDEFB')
            if row == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#BBDEFB')
        ax_metrics.set_title('Key Performance Metrics', fontsize=12, fontweight='bold', pad=10)

        # Cost breakdown table
        ax_costs = fig.add_subplot(gs[1, 1])
        ax_costs.axis('off')
        costs = stats['costs']
        cost_rows = [
            ['Trading Fees', _format_currency(costs['total_fees'])],
            ['Slippage', _format_currency(costs['total_slippage'])],
            ['Gas Costs', _format_currency(costs['total_gas'])],
            ['MEV Costs', _format_currency(costs['total_mev'])],
            ['Total Costs', _format_currency(costs['total_costs'])],
            ['Cost Drag', f"{costs['cost_drag_pct']:.2f}%"],
        ]
        table2 = ax_costs.table(
            cellText=cost_rows, colLabels=['Cost Type', 'Amount'],
            cellLoc='left', loc='center', colColours=['#FFF3E0', '#FFF3E0'],
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 1.5)
        for (row, col), cell in table2.get_celld().items():
            cell.set_edgecolor('#FFE0B2')
            if row == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#FFE0B2')
        ax_costs.set_title('Cost Breakdown', fontsize=12, fontweight='bold', pad=10)

        # Mini equity curve
        ax_eq = fig.add_subplot(gs[2, :])
        ax_eq.plot(self.viz.equity_curve.index, self.viz.equity_curve.values,
                   color='#1565C0', linewidth=1.2)
        for name, bm in self.viz.benchmark_curves.items():
            normalized = bm / bm.iloc[0] * self.viz.initial_capital
            ax_eq.plot(normalized.index, normalized.values, linewidth=0.8, alpha=0.6,
                       linestyle='--', label=name)
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_currency(x)))
        ax_eq.legend(fontsize=8, ncol=3)
        _apply_style(ax_eq, title='Equity Curve with Benchmarks', ylabel='Portfolio Value (USD)')

        # Per-strategy breakdown table
        ax_strat = fig.add_subplot(gs[3, :])
        ax_strat.axis('off')

        if len(strategy_df) > 0:
            display_df = strategy_df.copy()
            display_df['total_pnl'] = display_df['total_pnl'].apply(_format_currency)
            display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f'{x:.2f}')
            display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f'{x * 100:.2f}%')
            display_df['btc_correlation'] = display_df['btc_correlation'].apply(
                lambda x: f'{x:.3f}' if not np.isnan(x) else 'N/A')
            display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f'{x * 100:.1f}%')
            display_df['avg_roll_cost'] = display_df['avg_roll_cost'].apply(lambda x: f'${x:.2f}')
            display_df['avg_funding_cost'] = display_df['avg_funding_cost'].apply(lambda x: f'${x:.2f}')
            display_df['trade_count'] = display_df['trade_count'].apply(lambda x: f'{int(x):,}')
            display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f'{x:.2f}')

            col_labels = ['Total PnL', 'Sharpe', 'Max DD', 'BTC Corr', 'Win Rate',
                          'Avg Roll Cost', 'Avg Fund Cost', 'Trades', 'Profit Factor']

            table3 = ax_strat.table(
                cellText=display_df.values,
                rowLabels=display_df.index,
                colLabels=col_labels,
                cellLoc='center', loc='center',
                colColours=['#E8F5E9'] * len(col_labels),
            )
            table3.auto_set_font_size(False)
            table3.set_fontsize(8)
            table3.scale(1, 1.4)
            for (row, col), cell in table3.get_celld().items():
                cell.set_edgecolor('#C8E6C9')
                if row == 0:
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('#C8E6C9')
                if col == -1:
                    cell.set_text_props(fontweight='bold')

        ax_strat.set_title('Per-Strategy Breakdown', fontsize=12, fontweight='bold', pad=10)

        fig.tight_layout()
        if save:
            fig.savefig(self.viz.output_dir / 'summary_page.png', dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info("Saved summary_page.png")

        return fig

    def generate_full_pdf_report(
        self,
        basis_series: Optional[pd.Series] = None,
        capacity_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate the complete report: summary page plus all analytical plots.

        Returns
        -------
        dict of str to matplotlib.figure.Figure
        """
        all_figures: Dict[str, plt.Figure] = {}

        # Page 1: Summary
        try:
            all_figures['summary'] = self.plot_summary_page()
        except Exception as e:
            logger.error("Failed to generate summary page: %s", str(e))

        # All analytical charts
        analytical = self.viz.generate_full_report(
            basis_series=basis_series, capacity_data=capacity_data,
        )
        all_figures.update(analytical)

        total = len(all_figures)
        logger.info("Complete PDF report generated: %d figures saved to %s", total, self.viz.output_dir)
        return all_figures
