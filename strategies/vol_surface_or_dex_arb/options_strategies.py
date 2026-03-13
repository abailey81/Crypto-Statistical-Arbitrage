"""
Options Volatility Surface Arbitrage Strategies
Implements strategies to exploit mispricings in the BTC/ETH options volatility surface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm
from scipy.optimize import brentq


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionQuote:
    """Represents an option quote."""
    underlying: str
    strike: float
    expiry: pd.Timestamp
    option_type: OptionType
    bid: float
    ask: float
    mark_price: float
    implied_vol: float
    delta: float
    gamma: float
    vega: float
    theta: float
    open_interest: float
    venue: str


class BlackScholes:
    """Black-Scholes option pricing utilities."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 for Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 for Black-Scholes formula."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price."""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price."""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def implied_vol(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        precision: float = 1e-6
    ) -> float:
        """Calculate implied volatility from option price."""
        if T <= 0:
            return 0

        def objective(sigma):
            if option_type == OptionType.CALL:
                return BlackScholes.call_price(S, K, T, r, sigma) - price
            else:
                return BlackScholes.put_price(S, K, T, r, sigma) - price

        try:
            return brentq(objective, 0.001, 5.0, xtol=precision)
        except ValueError:
            return np.nan

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        """Calculate option delta."""
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma."""
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega."""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change


class VolatilitySurfaceArbitrage:
    """
    Volatility surface arbitrage strategy.

    Exploits mispricings in the implied volatility surface:
    1. Calendar spreads: Different IVs for same strike, different expiries
    2. Butterfly spreads: Non-smooth IV smile
    3. Put-call parity violations
    4. Cross-venue IV discrepancies (Deribit vs DEX options)
    """

    def __init__(
        self,
        min_iv_spread: float = 0.05,  # 5% IV difference to trade
        max_position_delta: float = 0.1,
        max_position_vega: float = 1000,
        transaction_cost: float = 0.001
    ):
        self.min_iv_spread = min_iv_spread
        self.max_position_delta = max_position_delta
        self.max_position_vega = max_position_vega
        self.transaction_cost = transaction_cost
        self.positions: List[Dict] = []
        self.trade_history: List[Dict] = []

    def build_volatility_surface(
        self,
        options: List[OptionQuote],
        spot_price: float
    ) -> pd.DataFrame:
        """
        Build volatility surface from option quotes.

        Returns DataFrame with moneyness, time to expiry, and IV.
        """
        data = []
        for opt in options:
            tte = (opt.expiry - pd.Timestamp.now()).days / 365
            if tte <= 0:
                continue

            moneyness = np.log(opt.strike / spot_price)

            data.append({
                "strike": opt.strike,
                "expiry": opt.expiry,
                "tte": tte,
                "moneyness": moneyness,
                "option_type": opt.option_type.value,
                "implied_vol": opt.implied_vol,
                "bid": opt.bid,
                "ask": opt.ask,
                "mark_price": opt.mark_price,
                "delta": opt.delta,
                "gamma": opt.gamma,
                "vega": opt.vega,
                "open_interest": opt.open_interest,
                "venue": opt.venue
            })

        return pd.DataFrame(data)

    def detect_calendar_spread_opportunities(
        self,
        surface: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect calendar spread arbitrage opportunities.

        Look for same strike with significantly different IVs across expiries.
        """
        opportunities = []

        for strike in surface["strike"].unique():
            strike_data = surface[surface["strike"] == strike].sort_values("tte")

            if len(strike_data) < 2:
                continue

            for i in range(len(strike_data) - 1):
                near = strike_data.iloc[i]
                far = strike_data.iloc[i + 1]

                iv_spread = far["implied_vol"] - near["implied_vol"]

                # Normally, longer-dated options have higher IV (term structure)
                # Opportunity exists when this relationship is inverted significantly
                if abs(iv_spread) >= self.min_iv_spread:
                    opportunities.append({
                        "type": "calendar_spread",
                        "strike": strike,
                        "near_expiry": near["expiry"],
                        "far_expiry": far["expiry"],
                        "near_iv": near["implied_vol"],
                        "far_iv": far["implied_vol"],
                        "iv_spread": iv_spread,
                        "trade": "sell_near_buy_far" if iv_spread < 0 else "buy_near_sell_far"
                    })

        return opportunities

    def detect_butterfly_opportunities(
        self,
        surface: pd.DataFrame,
        spot_price: float
    ) -> List[Dict]:
        """
        Detect butterfly spread opportunities.

        Look for non-convex IV smiles (arbitrage-free surface should be convex).
        """
        opportunities = []

        for expiry in surface["expiry"].unique():
            expiry_data = surface[
                (surface["expiry"] == expiry) &
                (surface["option_type"] == "call")
            ].sort_values("strike")

            if len(expiry_data) < 3:
                continue

            strikes = expiry_data["strike"].values
            ivs = expiry_data["implied_vol"].values

            # Check convexity at each interior point
            for i in range(1, len(strikes) - 1):
                k_low, k_mid, k_high = strikes[i-1], strikes[i], strikes[i+1]
                iv_low, iv_mid, iv_high = ivs[i-1], ivs[i], ivs[i+1]

                # Linear interpolation
                weight = (k_mid - k_low) / (k_high - k_low)
                iv_interpolated = iv_low + weight * (iv_high - iv_low)

                # If actual IV is significantly below interpolated, butterfly opportunity
                iv_diff = iv_interpolated - iv_mid

                if iv_diff >= self.min_iv_spread:
                    opportunities.append({
                        "type": "butterfly",
                        "expiry": expiry,
                        "strikes": (k_low, k_mid, k_high),
                        "ivs": (iv_low, iv_mid, iv_high),
                        "iv_mispricing": iv_diff,
                        "trade": "sell_wings_buy_body"  # Sell low/high strikes, buy middle
                    })

        return opportunities

    def detect_put_call_parity_violations(
        self,
        surface: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.05
    ) -> List[Dict]:
        """
        Detect put-call parity violations.

        Put-Call Parity: C - P = S - K*exp(-rT)
        """
        opportunities = []

        for expiry in surface["expiry"].unique():
            for strike in surface["strike"].unique():
                calls = surface[
                    (surface["expiry"] == expiry) &
                    (surface["strike"] == strike) &
                    (surface["option_type"] == "call")
                ]
                puts = surface[
                    (surface["expiry"] == expiry) &
                    (surface["strike"] == strike) &
                    (surface["option_type"] == "put")
                ]

                if len(calls) == 0 or len(puts) == 0:
                    continue

                call = calls.iloc[0]
                put = puts.iloc[0]

                tte = call["tte"]
                if tte <= 0:
                    continue

                # Theoretical relationship
                forward = spot_price - strike * np.exp(-risk_free_rate * tte)
                actual_spread = call["mark_price"] - put["mark_price"]

                parity_violation = actual_spread - forward
                parity_violation_pct = abs(parity_violation) / spot_price

                if parity_violation_pct >= 0.001:  # 0.1% violation
                    opportunities.append({
                        "type": "put_call_parity",
                        "expiry": expiry,
                        "strike": strike,
                        "call_price": call["mark_price"],
                        "put_price": put["mark_price"],
                        "theoretical_spread": forward,
                        "actual_spread": actual_spread,
                        "violation": parity_violation,
                        "violation_pct": parity_violation_pct,
                        "trade": "buy_call_sell_put" if parity_violation < 0 else "sell_call_buy_put"
                    })

        return opportunities

    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate aggregate portfolio Greeks."""
        total_delta = sum(p.get("delta", 0) * p.get("size", 0) for p in self.positions)
        total_gamma = sum(p.get("gamma", 0) * p.get("size", 0) for p in self.positions)
        total_vega = sum(p.get("vega", 0) * p.get("size", 0) for p in self.positions)
        total_theta = sum(p.get("theta", 0) * p.get("size", 0) for p in self.positions)

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega,
            "theta": total_theta
        }

    def analyze_surface(
        self,
        options: List[OptionQuote],
        spot_price: float
    ) -> Dict[str, List[Dict]]:
        """
        Comprehensive analysis of volatility surface for opportunities.

        Returns dict with all detected opportunities by type.
        """
        surface = self.build_volatility_surface(options, spot_price)

        return {
            "calendar_spreads": self.detect_calendar_spread_opportunities(surface),
            "butterflies": self.detect_butterfly_opportunities(surface, spot_price),
            "put_call_parity": self.detect_put_call_parity_violations(surface, spot_price),
            "surface_summary": {
                "num_options": len(surface),
                "expiries": surface["expiry"].nunique(),
                "strikes": surface["strike"].nunique(),
                "avg_iv": surface["implied_vol"].mean(),
                "iv_range": (surface["implied_vol"].min(), surface["implied_vol"].max())
            }
        }
