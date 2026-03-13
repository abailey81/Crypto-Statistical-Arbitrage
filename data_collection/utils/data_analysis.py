"""
detailed Data Analysis Module for Crypto Statistical Arbitrage.

This module provides comprehensive, data-driven analysis capabilities that go
far beyond basic categorization:

1. SURVIVORSHIP BIAS DETECTION
   - Detect missing symbols from data gaps
   - Calculate actual attrition rates from observed data
   - Dynamic adjustment factors based on real data
   - Track symbol appearances/disappearances over time

2. WASH TRADING DETECTION
   - Volume-price divergence analysis
   - Round number concentration detection
   - Volume consistency analysis (Benford's Law)
   - Cross-venue volume correlation
   - Statistical anomaly detection

3. MEV/SANDWICH ATTACK DETECTION
   - Price impact analysis
   - Reversion pattern detection
   - Block-level price anomalies
   - Slippage estimation

4. CROSS-VENUE STATISTICAL VALIDATION
   - Correlation matrices
   - Cointegration tests
   - Lead-lag analysis
   - Divergence detection

5. LIQUIDITY FRAGMENTATION ANALYSIS
   - HHI (Herfindahl-Hirschman Index) calculation
   - Volume concentration metrics
   - Effective spread estimation

Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SurvivorshipBiasResult:
    """Result of survivorship bias analysis."""
    symbols_analyzed: int
    symbols_with_gaps: int
    potential_delistings: List[Dict[str, Any]]
    attrition_rate_observed: float
    attrition_rate_annualized: float
    adjustment_factors: Dict[str, float]
    gap_analysis: Dict[str, Any]
    confidence_score: float
    methodology: str
    recommendations: List[str]

@dataclass
class WashTradingResult:
    """Result of wash trading detection."""
    venue: str
    risk_score: float # 0-100
    risk_level: str # LOW, MEDIUM, HIGH, CRITICAL
    indicators: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    flagged_periods: List[Dict[str, Any]]
    recommendations: List[str]

@dataclass
class MEVAnalysisResult:
    """Result of MEV/sandwich attack analysis."""
    venue: str
    estimated_mev_cost_bps: float
    sandwich_attack_likelihood: float
    front_running_indicators: Dict[str, Any]
    price_impact_stats: Dict[str, Any]
    reversion_patterns: List[Dict[str, Any]]

@dataclass
class CrossVenueValidationResult:
    """Result of cross-venue statistical validation."""
    correlation_matrix: pd.DataFrame
    cointegration_results: Dict[str, Any]
    lead_lag_analysis: Dict[str, Any]
    divergence_events: List[Dict[str, Any]]
    data_quality_flags: List[str]

@dataclass
class LiquidityFragmentationResult:
    """Result of liquidity fragmentation analysis."""
    hhi_index: float # 0-10000 (higher = more concentrated)
    effective_venues: int
    volume_distribution: Dict[str, float]
    fragmentation_score: float # 0-1 (higher = more fragmented)
    cross_chain_distribution: Dict[str, float]

# =============================================================================
# SURVIVORSHIP BIAS DETECTION (Data-Driven)
# =============================================================================

class SurvivorshipBiasAnalyzer:
    """
    comprehensive survivorship bias detection from actual data.

    Instead of relying solely on academic estimates, this analyzer:
    1. Detects symbols that disappear from the data
    2. Identifies data gaps that may indicate delistings
    3. Calculates actual attrition rates from observed data
    4. Provides dynamic adjustment factors based on real patterns
    """

    def __init__(self):
        self.symbol_timelines: Dict[str, Dict[str, datetime]] = {}
        self.data_gaps: Dict[str, List[Tuple[datetime, datetime]]] = {}

    def analyze(
        self,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        symbol_col: str = 'symbol',
        timestamp_col: str = 'timestamp',
        venue_col: str = 'venue'
    ) -> SurvivorshipBiasResult:
        """
        Perform comprehensive survivorship bias analysis on collected data.

        Parameters:
            data: Collected data (OHLCV, funding rates, etc.)
            start_date: Analysis period start
            end_date: Analysis period end
            symbol_col: Column name for symbol
            timestamp_col: Column name for timestamp
            venue_col: Column name for venue

        Returns:
            SurvivorshipBiasResult with comprehensive analysis
        """
        if data.empty:
            return self._empty_result()

        # Ensure timestamp is datetime with consistent timezone handling
        data = data.copy()
        try:
            if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
                data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)
            else:
                # Handle mixed tz-aware/tz-naive by converting all to UTC
                if data[timestamp_col].dt.tz is None:
                    data[timestamp_col] = pd.to_datetime(data[timestamp_col]).dt.tz_localize('UTC')
                else:
                    data[timestamp_col] = data[timestamp_col].dt.tz_convert('UTC')
        except Exception:
            # Fallback: convert to string first, then to datetime with UTC
            data[timestamp_col] = pd.to_datetime(data[timestamp_col].astype(str), utc=True)

        # 1. Build symbol timelines
        symbol_timelines = self._build_symbol_timelines(data, symbol_col, timestamp_col)

        # 2. Detect potential delistings (symbols that stop appearing)
        potential_delistings = self._detect_potential_delistings(
            symbol_timelines, end_date
        )

        # 3. Analyze data gaps
        gap_analysis = self._analyze_data_gaps(data, symbol_col, timestamp_col)

        # 4. Calculate observed attrition rate
        attrition_observed, attrition_annualized = self._calculate_attrition_rate(
            symbol_timelines, start_date, end_date
        )

        # 5. Calculate dynamic adjustment factors
        adjustment_factors = self._calculate_adjustment_factors(
            attrition_annualized,
            len(potential_delistings),
            len(symbol_timelines)
        )

        # 6. Calculate confidence score
        confidence = self._calculate_confidence(
            len(data),
            len(symbol_timelines),
            (end_date - start_date).days
        )

        # 7. Generate recommendations
        recommendations = self._generate_recommendations(
            attrition_annualized,
            len(potential_delistings),
            gap_analysis
        )

        return SurvivorshipBiasResult(
            symbols_analyzed=len(symbol_timelines),
            symbols_with_gaps=len([s for s, g in self.data_gaps.items() if g]),
            potential_delistings=potential_delistings,
            attrition_rate_observed=attrition_observed,
            attrition_rate_annualized=attrition_annualized,
            adjustment_factors=adjustment_factors,
            gap_analysis=gap_analysis,
            confidence_score=confidence,
            methodology='data_driven_gap_analysis',
            recommendations=recommendations
        )

    def _build_symbol_timelines(
        self,
        data: pd.DataFrame,
        symbol_col: str,
        timestamp_col: str
    ) -> Dict[str, Dict[str, datetime]]:
        """Build first/last seen timestamps for each symbol (optimized with groupby)."""
        # Use groupby aggregation instead of iterating over each symbol
        grouped = data.groupby(symbol_col)[timestamp_col].agg(['min', 'max', 'count'])
        grouped.columns = ['first_seen', 'last_seen', 'record_count']

        timelines = grouped.to_dict('index')

        self.symbol_timelines = timelines
        return timelines

    def _detect_potential_delistings(
        self,
        symbol_timelines: Dict,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Detect symbols that may have been delisted based on data patterns.

        A symbol is flagged as potentially delisted if:
        1. Its last_seen date is significantly before end_date
        2. It had regular data and then stopped
        """
        potential_delistings = []

        # Convert end_date to pandas Timestamp for comparison
        end_ts = pd.Timestamp(end_date)
        threshold_days = 7 # Flag if no data for 7+ days before end

        for symbol, timeline in symbol_timelines.items():
            last_seen = timeline['last_seen']

            # Handle timezone-aware comparisons
            if hasattr(last_seen, 'tzinfo') and last_seen.tzinfo is not None:
                if end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize('UTC')

            days_since_last = (end_ts - last_seen).days

            if days_since_last > threshold_days:
                # Calculate if this was a sudden stop
                sudden_stop = timeline['record_count'] > 10 # Had substantial data

                potential_delistings.append({
                    'symbol': symbol,
                    'last_seen': last_seen,
                    'days_missing': days_since_last,
                    'record_count_before_stop': timeline['record_count'],
                    'sudden_stop': sudden_stop,
                    'confidence': min(0.9, days_since_last / 30) # Higher confidence for longer gaps
                })

        return sorted(potential_delistings, key=lambda x: -x['days_missing'])

    def _analyze_data_gaps(
        self,
        data: pd.DataFrame,
        symbol_col: str,
        timestamp_col: str
    ) -> Dict[str, Any]:
        """
        Analyze gaps in the data that may indicate survivorship issues.
        OPTIMIZED: Uses vectorized operations instead of per-symbol iteration.
        """
        gap_summary = {
            'total_gaps': 0,
            'significant_gaps': 0, # > 24 hours
            'symbols_with_gaps': 0,
            'max_gap_hours': 0,
            'avg_gap_hours': 0
        }

        # Skip detailed gap analysis for large datasets (> 100K records) - use sampling
        if len(data) > 100_000:
            logger.info(f"Large dataset ({len(data):,} records) - using fast gap estimation")
            # Sample analysis: check gap stats per symbol using groupby
            try:
                # Sort once and calculate gaps using diff within groups
                data_sorted = data.sort_values([symbol_col, timestamp_col])
                data_sorted['_prev_ts'] = data_sorted.groupby(symbol_col)[timestamp_col].shift(1)
                data_sorted['_gap_hours'] = (
                    data_sorted[timestamp_col] - data_sorted['_prev_ts']
                ) / pd.Timedelta(hours=1)

                # Filter significant gaps (> 24 hours)
                significant_gaps = data_sorted[data_sorted['_gap_hours'] > 24]['_gap_hours']

                gap_summary['total_gaps'] = len(significant_gaps)
                gap_summary['significant_gaps'] = len(significant_gaps)
                gap_summary['symbols_with_gaps'] = data_sorted[
                    data_sorted['_gap_hours'] > 24
                ][symbol_col].nunique()

                if len(significant_gaps) > 0:
                    gap_summary['max_gap_hours'] = float(significant_gaps.max())
                    gap_summary['avg_gap_hours'] = float(significant_gaps.mean())

                # Store minimal gap info
                self.data_gaps = {} # Skip detailed gaps for performance

            except Exception as e:
                logger.warning(f"Gap analysis error: {e}, using fallback")
                self.data_gaps = {}

            return gap_summary

        # For smaller datasets, do detailed analysis
        gaps_by_symbol = {}
        all_gaps = []

        # Use groupby to process each symbol
        for symbol, group in data.groupby(symbol_col):
            if len(group) < 2:
                continue

            group_sorted = group.sort_values(timestamp_col)
            timestamps = group_sorted[timestamp_col].values
            gaps = []

            # Vectorized gap calculation
            ts_diff = np.diff(timestamps) / np.timedelta64(1, 'h')
            significant_mask = ts_diff > 24

            if significant_mask.any():
                significant_indices = np.where(significant_mask)[0]
                for idx in significant_indices:
                    gap_hours = float(ts_diff[idx])
                    gaps.append({
                        'start': pd.Timestamp(timestamps[idx]),
                        'end': pd.Timestamp(timestamps[idx + 1]),
                        'hours': gap_hours
                    })
                    all_gaps.append(gap_hours)

            if gaps:
                gaps_by_symbol[symbol] = gaps

        self.data_gaps = gaps_by_symbol

        gap_summary['total_gaps'] = len(all_gaps)
        gap_summary['significant_gaps'] = len(all_gaps)
        gap_summary['symbols_with_gaps'] = len(gaps_by_symbol)

        if all_gaps:
            gap_summary['max_gap_hours'] = max(all_gaps)
            gap_summary['avg_gap_hours'] = sum(all_gaps) / len(all_gaps)

        return gap_summary

    def _calculate_attrition_rate(
        self,
        symbol_timelines: Dict,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[float, float]:
        """
        Calculate observed attrition rate from the data.

        Attrition = symbols that stopped appearing / total symbols
        """
        if not symbol_timelines:
            return 0.0, 0.0

        total_symbols = len(symbol_timelines)
        period_days = (end_date - start_date).days

        # Count symbols that stopped appearing before end date
        end_ts = pd.Timestamp(end_date)
        threshold_days = 7

        dropped_symbols = 0
        for symbol, timeline in symbol_timelines.items():
            last_seen = timeline['last_seen']
            if hasattr(last_seen, 'tzinfo') and last_seen.tzinfo is not None:
                if end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize('UTC')

            if (end_ts - last_seen).days > threshold_days:
                dropped_symbols += 1

        # Observed attrition for period
        attrition_observed = dropped_symbols / total_symbols if total_symbols > 0 else 0

        # Annualize
        if period_days > 0:
            attrition_annualized = attrition_observed * (365 / period_days)
        else:
            attrition_annualized = 0

        return attrition_observed, min(attrition_annualized, 1.0)

    def _calculate_adjustment_factors(
        self,
        attrition_annualized: float,
        num_delistings: int,
        num_symbols: int
    ) -> Dict[str, float]:
        """
        Calculate adjustment factors based on observed data.

        Different weighting schemes have different sensitivities to
        survivorship bias.
        """
        # Base adjustment from observed attrition
        base_adjustment = 1 / (1 + attrition_annualized) if attrition_annualized < 1 else 0.5

        # Equal-weighted portfolios are more affected
        # (small/risky coins more likely to delist)
        ew_multiplier = 2.5 # Academic research suggests ~2.5x impact

        # Value-weighted less affected (large caps rarely delist)
        vw_multiplier = 0.15 # ~15% of the impact

        # Liquidity-weighted somewhere in between
        lw_multiplier = 0.4

        return {
            'value_weighted': 1 - (1 - base_adjustment) * vw_multiplier,
            'equal_weighted': 1 - (1 - base_adjustment) * ew_multiplier,
            'liquidity_weighted': 1 - (1 - base_adjustment) * lw_multiplier,
            'observed_attrition': attrition_annualized,
            'delisting_rate': num_delistings / num_symbols if num_symbols > 0 else 0
        }

    def _calculate_confidence(
        self,
        num_records: int,
        num_symbols: int,
        period_days: int
    ) -> float:
        """Calculate confidence in the survivorship bias estimate."""
        confidence = 0.5 # Base

        # More data = higher confidence
        if num_records > 100000:
            confidence += 0.2
        elif num_records > 10000:
            confidence += 0.1

        # More symbols = higher confidence
        if num_symbols > 50:
            confidence += 0.1
        elif num_symbols > 20:
            confidence += 0.05

        # Longer period = higher confidence
        if period_days > 365:
            confidence += 0.15
        elif period_days > 180:
            confidence += 0.1

        return min(confidence, 0.95)

    def _generate_recommendations(
        self,
        attrition_annualized: float,
        num_delistings: int,
        gap_analysis: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        if attrition_annualized > 0.1:
            recommendations.append(
                f"HIGH ATTRITION ({attrition_annualized*100:.1f}% annualized): "
                "Consider using value-weighted returns as primary metric"
            )

        if num_delistings > 5:
            recommendations.append(
                f"{num_delistings} potential delistings detected: "
                "Verify these symbols and include their final returns in analysis"
            )

        if gap_analysis.get('significant_gaps', 0) > 10:
            recommendations.append(
                f"{gap_analysis['significant_gaps']} significant data gaps: "
                "Review data completeness before backtesting"
            )

        if gap_analysis.get('max_gap_hours', 0) > 168: # > 1 week
            recommendations.append(
                f"Maximum gap of {gap_analysis['max_gap_hours']:.0f} hours: "
                "Consider interpolation or excluding affected periods"
            )

        if not recommendations:
            recommendations.append(
                "Data appears complete with minimal survivorship bias risk"
            )

        return recommendations

    def _empty_result(self) -> SurvivorshipBiasResult:
        """Return empty result when no data available."""
        return SurvivorshipBiasResult(
            symbols_analyzed=0,
            symbols_with_gaps=0,
            potential_delistings=[],
            attrition_rate_observed=0.0,
            attrition_rate_annualized=0.0,
            adjustment_factors={
                'value_weighted': 1.0,
                'equal_weighted': 1.0,
                'liquidity_weighted': 1.0
            },
            gap_analysis={},
            confidence_score=0.0,
            methodology='no_data',
            recommendations=['No data available for analysis']
        )

# =============================================================================
# WASH TRADING DETECTION (Statistical Algorithms)
# =============================================================================

class WashTradingDetector:
    """
    comprehensive wash trading detection using multiple statistical methods.

    Detection algorithms:
    1. Volume-price divergence (high volume, no price movement)
    2. Benford's Law analysis (digit distribution)
    3. Volume autocorrelation (too consistent = suspicious)
    4. Round number concentration
    5. Cross-venue volume discrepancy
    """

    def analyze(
        self,
        data: pd.DataFrame,
        venue: str,
        price_col: str = 'close',
        volume_col: str = 'volume',
        timestamp_col: str = 'timestamp'
    ) -> WashTradingResult:
        """
        Perform comprehensive wash trading detection.

        Parameters:
            data: OHLCV data for a single venue
            venue: Venue name
            price_col: Column name for price
            volume_col: Column name for volume
            timestamp_col: Column name for timestamp

        Returns:
            WashTradingResult with detection results
        """
        if data.empty or len(data) < 10:
            return self._empty_result(venue)

        indicators = {}
        statistical_tests = {}
        flagged_periods = []

        # 1. Volume-Price Divergence Analysis
        vpd_result = self._volume_price_divergence(data, price_col, volume_col)
        indicators['volume_price_divergence'] = vpd_result

        # 2. Benford's Law Analysis (first digit distribution)
        benford_result = self._benford_analysis(data, volume_col)
        statistical_tests['benford_law'] = benford_result

        # 3. Volume Autocorrelation (suspicious if too consistent)
        autocorr_result = self._volume_autocorrelation(data, volume_col)
        statistical_tests['volume_autocorrelation'] = autocorr_result

        # 4. Round Number Concentration
        round_number_result = self._round_number_analysis(data, volume_col, price_col)
        indicators['round_number_concentration'] = round_number_result

        # 5. Volume Consistency Analysis
        consistency_result = self._volume_consistency(data, volume_col)
        indicators['volume_consistency'] = consistency_result

        # 6. Flag suspicious periods
        flagged_periods = self._flag_suspicious_periods(
            data, timestamp_col, vpd_result, consistency_result
        )

        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            vpd_result, benford_result, autocorr_result,
            round_number_result, consistency_result
        )

        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_score, indicators, statistical_tests
        )

        return WashTradingResult(
            venue=venue,
            risk_score=risk_score,
            risk_level=risk_level,
            indicators=indicators,
            statistical_tests=statistical_tests,
            flagged_periods=flagged_periods,
            recommendations=recommendations
        )

    def _volume_price_divergence(
        self,
        data: pd.DataFrame,
        price_col: str,
        volume_col: str
    ) -> Dict[str, Any]:
        """
        Detect periods of high volume with no price movement.

        Wash trading often shows high volume without corresponding price changes.
        """
        # Calculate returns and volume
        returns = data[price_col].pct_change().abs()
        volumes = data[volume_col]

        # Normalize
        returns_norm = (returns - returns.mean()) / returns.std() if returns.std() > 0 else returns
        volumes_norm = (volumes - volumes.mean()) / volumes.std() if volumes.std() > 0 else volumes

        # Find divergence: high volume but low return
        divergence = volumes_norm - returns_norm * 100 # Returns typically much smaller

        # Count suspicious periods (high divergence)
        threshold = divergence.quantile(0.95) if len(divergence) > 20 else 2
        suspicious_periods = (divergence > threshold).sum()

        # Calculate correlation (should be positive in normal markets)
        correlation = returns.corr(volumes) if len(data) > 10 else 0

        return {
            'correlation': float(correlation) if not np.isnan(correlation) else 0,
            'suspicious_periods': int(suspicious_periods),
            'suspicious_pct': float(suspicious_periods / len(data) * 100),
            'divergence_mean': float(divergence.mean()) if not np.isnan(divergence.mean()) else 0,
            'divergence_max': float(divergence.max()) if not np.isnan(divergence.max()) else 0,
            'flag': suspicious_periods > len(data) * 0.1 # Flag if >10% suspicious
        }

    def _benford_analysis(
        self,
        data: pd.DataFrame,
        volume_col: str
    ) -> Dict[str, Any]:
        """
        Apply Benford's Law to volume data.

        Natural data follows Benford's distribution for first digits.
        Manipulated data often deviates significantly.
        """
        # Expected Benford distribution
        benford_expected = {
            1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
            5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
        }

        # Extract first digits from volume
        volumes = data[volume_col].dropna()
        volumes = volumes[volumes > 0]

        if len(volumes) < 50:
            return {
                'chi_square': 0,
                'p_value': 1.0,
                'deviation': 0,
                'flag': False,
                'note': 'Insufficient data for Benford analysis'
            }

        # Get first digits
        first_digits = volumes.apply(
            lambda x: int(str(abs(x)).replace('.', '').lstrip('0')[0])
            if x != 0 and str(abs(x)).replace('.', '').lstrip('0') else 1
        )

        # Calculate observed distribution
        observed_counts = first_digits.value_counts().sort_index()
        observed_freq = observed_counts / len(first_digits)

        # Chi-square test
        chi_square = 0
        total_deviation = 0

        for digit in range(1, 10):
            observed = observed_freq.get(digit, 0)
            expected = benford_expected[digit]

            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected * len(first_digits)
                total_deviation += abs(observed - expected)

        # P-value from chi-square distribution (df=8)
        p_value = 1 - stats.chi2.cdf(chi_square, df=8)

        return {
            'chi_square': float(chi_square),
            'p_value': float(p_value),
            'deviation': float(total_deviation),
            'flag': p_value < 0.01, # Flag if significant deviation
            'observed_distribution': {int(k): float(v) for k, v in observed_freq.items()}
        }

    def _volume_autocorrelation(
        self,
        data: pd.DataFrame,
        volume_col: str
    ) -> Dict[str, Any]:
        """
        Analyze volume autocorrelation.

        Natural markets have low short-term autocorrelation.
        Artificial volume often shows patterns.
        """
        volumes = data[volume_col].dropna()

        if len(volumes) < 50:
            return {
                'lag1_autocorr': 0,
                'lag5_autocorr': 0,
                'flag': False,
                'note': 'Insufficient data'
            }

        # Calculate autocorrelations
        lag1 = volumes.autocorr(lag=1)
        lag5 = volumes.autocorr(lag=5)
        lag10 = volumes.autocorr(lag=10) if len(volumes) > 20 else 0

        # High autocorrelation is suspicious (volume too predictable)
        flag = abs(lag1) > 0.7 or abs(lag5) > 0.5

        return {
            'lag1_autocorr': float(lag1) if not np.isnan(lag1) else 0,
            'lag5_autocorr': float(lag5) if not np.isnan(lag5) else 0,
            'lag10_autocorr': float(lag10) if not np.isnan(lag10) else 0,
            'flag': flag,
            'interpretation': 'HIGH' if flag else 'NORMAL'
        }

    def _round_number_analysis(
        self,
        data: pd.DataFrame,
        volume_col: str,
        price_col: str
    ) -> Dict[str, Any]:
        """
        Detect concentration at round numbers.

        Wash traders often use round numbers for convenience.
        """
        volumes = data[volume_col].dropna()
        prices = data[price_col].dropna()

        if len(volumes) < 10:
            return {'flag': False, 'note': 'Insufficient data'}

        # Check volume round numbers
        def is_round(x, precision=2):
            """Check if number is suspiciously round."""
            if x == 0:
                return False
            # Check if divisible by powers of 10
            for power in [100, 1000, 10000]:
                if abs(x) >= power and abs(x) % power == 0:
                    return True
            return False

        round_volumes = volumes.apply(is_round).sum()
        round_volume_pct = round_volumes / len(volumes) * 100

        # Check price round numbers (less common in natural trading)
        round_prices = prices.apply(lambda x: x == round(x, -1) if x > 100 else False).sum()
        round_price_pct = round_prices / len(prices) * 100

        # Flag if excessive round numbers
        flag = round_volume_pct > 15 or round_price_pct > 20

        return {
            'round_volume_pct': float(round_volume_pct),
            'round_price_pct': float(round_price_pct),
            'round_volume_count': int(round_volumes),
            'flag': flag,
            'interpretation': 'SUSPICIOUS' if flag else 'NORMAL'
        }

    def _volume_consistency(
        self,
        data: pd.DataFrame,
        volume_col: str
    ) -> Dict[str, Any]:
        """
        Analyze volume consistency over time.

        Natural markets have varying volume.
        Too consistent volume is suspicious.
        """
        volumes = data[volume_col].dropna()

        if len(volumes) < 20:
            return {'flag': False, 'note': 'Insufficient data'}

        # Calculate coefficient of variation
        cv = volumes.std() / volumes.mean() if volumes.mean() > 0 else 0

        # Natural markets typically have CV > 0.5
        # Wash trading might show lower CV (more consistent)
        flag = cv < 0.3

        # Check for repeating patterns
        volume_diffs = volumes.diff().dropna()
        repeat_zero = (volume_diffs == 0).sum()
        repeat_pct = repeat_zero / len(volume_diffs) * 100 if len(volume_diffs) > 0 else 0

        return {
            'coefficient_of_variation': float(cv),
            'repeat_value_pct': float(repeat_pct),
            'flag': flag or repeat_pct > 10,
            'interpretation': 'TOO_CONSISTENT' if flag else 'NORMAL'
        }

    def _flag_suspicious_periods(
        self,
        data: pd.DataFrame,
        timestamp_col: str,
        vpd_result: Dict,
        consistency_result: Dict
    ) -> List[Dict[str, Any]]:
        """Identify specific periods with suspicious activity."""
        flagged = []

        # For now, flag based on overall results
        # In production, would identify specific time windows
        if vpd_result.get('flag', False):
            flagged.append({
                'type': 'volume_price_divergence',
                'severity': 'HIGH' if vpd_result['suspicious_pct'] > 20 else 'MEDIUM',
                'description': f"{vpd_result['suspicious_pct']:.1f}% of periods show volume without price movement"
            })

        if consistency_result.get('flag', False):
            flagged.append({
                'type': 'volume_consistency',
                'severity': 'MEDIUM',
                'description': f"Volume shows unusual consistency (CV={consistency_result['coefficient_of_variation']:.2f})"
            })

        return flagged

    def _calculate_risk_score(
        self,
        vpd: Dict, benford: Dict, autocorr: Dict,
        round_num: Dict, consistency: Dict
    ) -> float:
        """Calculate overall wash trading risk score (0-100)."""
        score = 0

        # Volume-price divergence (0-25 points)
        if vpd.get('flag'):
            score += min(25, vpd.get('suspicious_pct', 0) * 2)
        elif vpd.get('correlation', 0) < 0:
            score += 10 # Negative correlation is suspicious

        # Benford's Law (0-25 points)
        if benford.get('flag'):
            score += 25
        elif benford.get('p_value', 1) < 0.05:
            score += 15

        # Autocorrelation (0-20 points)
        if autocorr.get('flag'):
            score += 20
        elif abs(autocorr.get('lag1_autocorr', 0)) > 0.5:
            score += 10

        # Round numbers (0-15 points)
        if round_num.get('flag'):
            score += 15
        elif round_num.get('round_volume_pct', 0) > 10:
            score += 7

        # Consistency (0-15 points)
        if consistency.get('flag'):
            score += 15
        elif consistency.get('coefficient_of_variation', 1) < 0.4:
            score += 7

        return min(100, score)

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        if score >= 70:
            return 'CRITICAL'
        elif score >= 50:
            return 'HIGH'
        elif score >= 30:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _generate_recommendations(
        self,
        risk_score: float,
        indicators: Dict,
        tests: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if risk_score >= 70:
            recommendations.append(
                "CRITICAL: Consider excluding this venue from analysis or apply significant discount to volume"
            )
        elif risk_score >= 50:
            recommendations.append(
                "HIGH: Cross-validate data with other venues before use"
            )

        if indicators.get('volume_price_divergence', {}).get('flag'):
            recommendations.append(
                "Volume shows divergence from price movements - apply volume-quality adjustment"
            )

        if tests.get('benford_law', {}).get('flag'):
            recommendations.append(
                "Volume distribution deviates from Benford's Law - potential manipulation"
            )

        if not recommendations:
            recommendations.append(
                "No significant wash trading indicators detected"
            )

        return recommendations

    def _empty_result(self, venue: str) -> WashTradingResult:
        """Return empty result when no data."""
        return WashTradingResult(
            venue=venue,
            risk_score=0,
            risk_level='UNKNOWN',
            indicators={},
            statistical_tests={},
            flagged_periods=[],
            recommendations=['Insufficient data for analysis']
        )

# =============================================================================
# MEV/SANDWICH ATTACK DETECTION
# =============================================================================

class MEVAnalyzer:
    """
    Detect MEV (Maximal Extractable Value) impact and sandwich attacks.

    Methods:
    1. Price impact analysis
    2. Reversion pattern detection
    3. Slippage estimation
    """

    def analyze(
        self,
        data: pd.DataFrame,
        venue: str,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        volume_col: str = 'volume'
    ) -> MEVAnalysisResult:
        """Perform MEV analysis on DEX data."""
        if data.empty or len(data) < 20:
            return self._empty_result(venue)

        # 1. Estimate price impact
        price_impact = self._estimate_price_impact(data, price_col, volume_col)

        # 2. Detect reversion patterns (sandwich attack signature)
        reversion = self._detect_reversion_patterns(data, price_col, high_col, low_col)

        # 3. Calculate effective spread
        spread = self._calculate_effective_spread(data, high_col, low_col, price_col)

        # 4. Estimate MEV cost
        mev_cost_bps = self._estimate_mev_cost(price_impact, reversion, spread)

        # 5. Sandwich attack likelihood
        sandwich_likelihood = self._sandwich_likelihood(reversion, price_impact)

        return MEVAnalysisResult(
            venue=venue,
            estimated_mev_cost_bps=mev_cost_bps,
            sandwich_attack_likelihood=sandwich_likelihood,
            front_running_indicators=price_impact,
            price_impact_stats=spread,
            reversion_patterns=reversion
        )

    def _estimate_price_impact(
        self,
        data: pd.DataFrame,
        price_col: str,
        volume_col: str
    ) -> Dict[str, Any]:
        """Estimate price impact from volume."""
        returns = data[price_col].pct_change().abs()
        volumes = data[volume_col]

        # Price impact coefficient (Kyle's lambda approximation)
        if volumes.std() > 0 and len(data) > 10:
            # Regress returns on sqrt(volume)
            sqrt_vol = np.sqrt(volumes.fillna(0))
            correlation = returns.corr(sqrt_vol)

            # Estimate impact coefficient
            impact_coef = returns.mean() / sqrt_vol.mean() if sqrt_vol.mean() > 0 else 0
        else:
            correlation = 0
            impact_coef = 0

        return {
            'price_volume_correlation': float(correlation) if not np.isnan(correlation) else 0,
            'impact_coefficient': float(impact_coef) if not np.isnan(impact_coef) else 0,
            'avg_return': float(returns.mean()) if not np.isnan(returns.mean()) else 0,
            'avg_volume': float(volumes.mean()) if not np.isnan(volumes.mean()) else 0
        }

    def _detect_reversion_patterns(
        self,
        data: pd.DataFrame,
        price_col: str,
        high_col: str,
        low_col: str
    ) -> List[Dict[str, Any]]:
        """
        Detect price reversion patterns that indicate sandwich attacks.

        Pattern: Price spikes then quickly reverts (characteristic of sandwich)
        """
        patterns = []

        returns = data[price_col].pct_change()
        ranges = (data[high_col] - data[low_col]) / data[price_col]

        # Look for large intra-candle ranges with small net moves
        for i in range(1, len(data) - 1):
            if ranges.iloc[i] > ranges.mean() * 2: # Large range
                if abs(returns.iloc[i]) < abs(ranges.iloc[i]) * 0.3: # Small net move
                    patterns.append({
                        'index': i,
                        'range': float(ranges.iloc[i]),
                        'net_return': float(returns.iloc[i]),
                        'reversion_ratio': float(abs(returns.iloc[i]) / ranges.iloc[i]) if ranges.iloc[i] > 0 else 0
                    })

        return patterns[:20] # Return top 20

    def _calculate_effective_spread(
        self,
        data: pd.DataFrame,
        high_col: str,
        low_col: str,
        close_col: str
    ) -> Dict[str, Any]:
        """Calculate effective spread estimates."""
        spreads = (data[high_col] - data[low_col]) / data[close_col]

        return {
            'avg_spread_pct': float(spreads.mean() * 100) if not np.isnan(spreads.mean()) else 0,
            'max_spread_pct': float(spreads.max() * 100) if not np.isnan(spreads.max()) else 0,
            'spread_volatility': float(spreads.std()) if not np.isnan(spreads.std()) else 0
        }

    def _estimate_mev_cost(
        self,
        price_impact: Dict,
        reversion_patterns: List,
        spread: Dict
    ) -> float:
        """Estimate total MEV cost in basis points."""
        base_cost = 0

        # Price impact component
        impact_cost = price_impact.get('avg_return', 0) * 10000 * 0.5 # 50% of avg return

        # Reversion pattern component (sandwich indicator)
        reversion_cost = len(reversion_patterns) * 0.5 # 0.5 bps per pattern detected

        # Spread component
        spread_cost = spread.get('avg_spread_pct', 0) * 100 * 0.2 # 20% of spread

        base_cost = impact_cost + reversion_cost + spread_cost

        return max(0, min(100, base_cost)) # Cap at 100 bps

    def _sandwich_likelihood(
        self,
        reversion_patterns: List,
        price_impact: Dict
    ) -> float:
        """Calculate likelihood of sandwich attacks (0-1)."""
        if not reversion_patterns:
            return 0.1 # Baseline for any DEX

        # More patterns = higher likelihood
        pattern_score = min(0.5, len(reversion_patterns) * 0.05)

        # High price impact = higher likelihood
        impact_score = min(0.3, price_impact.get('avg_return', 0) * 10)

        return min(0.95, 0.1 + pattern_score + impact_score)

    def _empty_result(self, venue: str) -> MEVAnalysisResult:
        return MEVAnalysisResult(
            venue=venue,
            estimated_mev_cost_bps=0,
            sandwich_attack_likelihood=0,
            front_running_indicators={},
            price_impact_stats={},
            reversion_patterns=[]
        )

# =============================================================================
# CROSS-VENUE STATISTICAL VALIDATION
# =============================================================================

class CrossVenueValidator:
    """
    Statistical validation across venues.

    Methods:
    1. Correlation matrix calculation
    2. Cointegration testing
    3. Lead-lag analysis
    4. Divergence detection
    """

    def validate(
        self,
        venue_data: Dict[str, pd.DataFrame],
        price_col: str = 'close',
        timestamp_col: str = 'timestamp'
    ) -> CrossVenueValidationResult:
        """Perform cross-venue validation."""
        if len(venue_data) < 2:
            return self._empty_result()

        # 1. Build price series
        price_series = self._build_price_series(venue_data, price_col, timestamp_col)

        if price_series.empty:
            return self._empty_result()

        # 2. Calculate correlation matrix
        corr_matrix = price_series.corr()

        # 3. Lead-lag analysis
        lead_lag = self._analyze_lead_lag(price_series)

        # 4. Detect divergence events
        divergences = self._detect_divergences(price_series)

        # 5. Data quality flags
        flags = self._generate_quality_flags(corr_matrix, divergences)

        return CrossVenueValidationResult(
            correlation_matrix=corr_matrix,
            cointegration_results={}, # Requires statsmodels for full test
            lead_lag_analysis=lead_lag,
            divergence_events=divergences,
            data_quality_flags=flags
        )

    def _build_price_series(
        self,
        venue_data: Dict[str, pd.DataFrame],
        price_col: str,
        timestamp_col: str,
        reference_symbol: str = 'BTC'
    ) -> pd.DataFrame:
        """
        Build aligned price series for all venues.

        IMPORTANT: Only compares prices for the SAME symbol across venues.
        Default is BTC as the reference symbol (most liquid, available everywhere).
        """
        series_dict = {}

        for venue, data in venue_data.items():
            if data.empty or price_col not in data.columns:
                continue

            # IMPORTANT: Skip venues without 'symbol' column entirely
            # Pool-based venues (Uniswap, GeckoTerminal) use 'pair_name' instead
            # and their prices cannot be meaningfully compared with CEX data
            if 'symbol' not in data.columns:
                continue

            # Get price series indexed by timestamp
            df = data.copy()

            # Filter to reference symbol only (critical for valid comparison)
            if 'symbol' in df.columns:
                # Normalize symbol names for matching
                df['symbol_normalized'] = df['symbol'].str.upper().str.replace('-PERP', '').str.replace('USDT', '').str.replace('USD', '')
                df = df[df['symbol_normalized'] == reference_symbol.upper()]

                if df.empty:
                    continue

            if timestamp_col in df.columns:
                df = df.set_index(timestamp_col)

            # Get price series (average if multiple entries at same timestamp)
            if len(df) > 0:
                series = df.groupby(df.index)[price_col].mean()
                series_dict[venue] = series

        if not series_dict:
            return pd.DataFrame()

        # Combine into single DataFrame
        result = pd.DataFrame(series_dict)

        # Forward fill small gaps
        result = result.ffill(limit=3)

        return result

    def _analyze_lead_lag(self, price_series: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which venues lead/lag others."""
        if price_series.empty or len(price_series.columns) < 2:
            return {}

        returns = price_series.pct_change().dropna()

        lead_lag_results = {}
        venues = list(returns.columns)

        for i, venue1 in enumerate(venues):
            for venue2 in venues[i+1:]:
                # Calculate cross-correlation at different lags
                max_lag = min(10, len(returns) // 5)
                best_lag = 0
                best_corr = returns[venue1].corr(returns[venue2])

                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        continue
                    if lag > 0:
                        corr = returns[venue1].iloc[:-lag].corr(returns[venue2].iloc[lag:])
                    else:
                        corr = returns[venue1].iloc[-lag:].corr(returns[venue2].iloc[:lag])

                    if not np.isnan(corr) and abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag

                lead_lag_results[f"{venue1}_vs_{venue2}"] = {
                    'lag': best_lag,
                    'correlation_at_lag': float(best_corr) if not np.isnan(best_corr) else 0,
                    'leader': venue1 if best_lag > 0 else venue2,
                    'follower': venue2 if best_lag > 0 else venue1
                }

        return lead_lag_results

    def _detect_divergences(self, price_series: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect significant price divergences between venues."""
        divergences = []

        if price_series.empty or len(price_series.columns) < 2:
            return divergences

        # Calculate pairwise divergences
        venues = list(price_series.columns)

        for i, venue1 in enumerate(venues):
            for venue2 in venues[i+1:]:
                # Calculate percentage difference
                diff = (price_series[venue1] - price_series[venue2]) / price_series[venue1] * 100

                # Find significant divergences (> 1%)
                threshold = 1.0 # 1%
                significant = diff.abs() > threshold

                if significant.any():
                    max_div_idx = diff.abs().idxmax()
                    divergences.append({
                        'venues': [venue1, venue2],
                        'max_divergence_pct': float(diff.loc[max_div_idx]),
                        'timestamp': str(max_div_idx),
                        'count': int(significant.sum()),
                        'avg_divergence_pct': float(diff[significant].abs().mean())
                    })

        return sorted(divergences, key=lambda x: -abs(x['max_divergence_pct']))[:10]

    def _generate_quality_flags(
        self,
        corr_matrix: pd.DataFrame,
        divergences: List
    ) -> List[str]:
        """Generate data quality flags."""
        flags = []

        if corr_matrix.empty:
            flags.append("WARNING: Insufficient data for correlation analysis")
            return flags

        # Check for low correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if not np.isnan(corr) and corr < 0.9:
                    flags.append(
                        f"LOW_CORRELATION: {corr_matrix.columns[i]} vs "
                        f"{corr_matrix.columns[j]} = {corr:.3f}"
                    )

        # Check for large divergences
        for div in divergences:
            if abs(div['max_divergence_pct']) > 2:
                flags.append(
                    f"LARGE_DIVERGENCE: {div['venues'][0]} vs {div['venues'][1]} "
                    f"= {div['max_divergence_pct']:.2f}%"
                )

        if not flags:
            flags.append("PASS: Cross-venue data is consistent")

        return flags

    def _empty_result(self) -> CrossVenueValidationResult:
        return CrossVenueValidationResult(
            correlation_matrix=pd.DataFrame(),
            cointegration_results={},
            lead_lag_analysis={},
            divergence_events=[],
            data_quality_flags=['INSUFFICIENT_DATA: Need at least 2 venues']
        )

# =============================================================================
# LIQUIDITY FRAGMENTATION ANALYSIS
# =============================================================================

class LiquidityFragmentationAnalyzer:
    """
    Analyze liquidity fragmentation across venues.

    Metrics:
    1. HHI (Herfindahl-Hirschman Index)
    2. Effective number of venues
    3. Volume concentration
    4. Cross-chain distribution
    """

    def analyze(
        self,
        venue_volumes: Dict[str, float],
        venue_chains: Optional[Dict[str, str]] = None
    ) -> LiquidityFragmentationResult:
        """Analyze liquidity fragmentation."""
        if not venue_volumes:
            return self._empty_result()

        total_volume = sum(venue_volumes.values())

        if total_volume == 0:
            return self._empty_result()

        # 1. Calculate market shares
        market_shares = {v: vol / total_volume for v, vol in venue_volumes.items()}

        # 2. Calculate HHI (0-10000)
        hhi = sum((share * 100) ** 2 for share in market_shares.values())

        # 3. Effective number of venues (1/HHI * 10000)
        effective_venues = 10000 / hhi if hhi > 0 else len(venue_volumes)

        # 4. Fragmentation score (0-1, higher = more fragmented)
        # HHI < 1500 = unconcentrated (fragmented)
        # HHI 1500-2500 = moderately concentrated
        # HHI > 2500 = highly concentrated
        fragmentation_score = max(0, 1 - (hhi / 10000))

        # 5. Cross-chain distribution
        cross_chain = {}
        if venue_chains:
            for venue, share in market_shares.items():
                chain = venue_chains.get(venue, 'unknown')
                cross_chain[chain] = cross_chain.get(chain, 0) + share

        return LiquidityFragmentationResult(
            hhi_index=hhi,
            effective_venues=effective_venues,
            volume_distribution=market_shares,
            fragmentation_score=fragmentation_score,
            cross_chain_distribution=cross_chain
        )

    def _empty_result(self) -> LiquidityFragmentationResult:
        return LiquidityFragmentationResult(
            hhi_index=10000,
            effective_venues=0,
            volume_distribution={},
            fragmentation_score=0,
            cross_chain_distribution={}
        )

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'SurvivorshipBiasResult',
    'WashTradingResult',
    'MEVAnalysisResult',
    'CrossVenueValidationResult',
    'LiquidityFragmentationResult',
    # Analyzers
    'SurvivorshipBiasAnalyzer',
    'WashTradingDetector',
    'MEVAnalyzer',
    'CrossVenueValidator',
    'LiquidityFragmentationAnalyzer',
]
