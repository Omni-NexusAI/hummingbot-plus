from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class StabilityMetrics:
    is_stable: bool
    current_range: Tuple[float, float]
    volatility: float
    trend_strength: float
    volume_profile: float
    confidence: float

class MarketStabilityDetector:
    """
    Analyzes market conditions to detect stable trading ranges and market state transitions.
    """
    
    def __init__(self, 
                 stability_threshold: float = 0.02,
                 volatility_threshold: float = 0.05,
                 range_samples: int = 20,
                 min_range_candles: int = 12):
        self.stability_threshold = stability_threshold
        self.volatility_threshold = volatility_threshold
        self.range_samples = range_samples
        self.min_range_candles = min_range_candles
        self.last_range: Tuple[float, float] = (0, 0)
        self.range_start_time = None
        
    def detect_price_range(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Detects the current price range using recent price action.
        """
        recent_highs = df['high'].tail(self.range_samples)
        recent_lows = df['low'].tail(self.range_samples)
        
        # Use percentile instead of min/max to avoid outliers
        range_high = np.percentile(recent_highs, 90)
        range_low = np.percentile(recent_lows, 10)
        
        return (range_low, range_high)
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> float:
        """
        Analyzes volume distribution within the current range.
        Returns a score between 0-1 indicating volume stability.
        """
        recent_volume = df['volume'].tail(self.range_samples)
        volume_std = recent_volume.std()
        volume_mean = recent_volume.mean()
        
        if volume_mean == 0:
            return 0
            
        volume_cv = volume_std / volume_mean  # Coefficient of variation
        volume_score = 1 / (1 + volume_cv)  # Normalize to 0-1
        
        return volume_score
        
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculates the strength of the current trend.
        Returns a value between 0-1 where 0 is ranging and 1 is strongly trending.
        """
        prices = df['close'].tail(self.range_samples)
        
        # Calculate linear regression
        x = np.arange(len(prices))
        y = prices.values
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope to 0-1 range
        max_slope = prices.mean() * 0.01  # 1% per period as max slope
        trend_strength = min(abs(slope) / max_slope, 1)
        
        return trend_strength
        
    def analyze_stability(self, 
                         df: pd.DataFrame, 
                         btc_df: pd.DataFrame = None) -> StabilityMetrics:
        """
        Comprehensive market stability analysis.
        """
        if len(df) < self.range_samples:
            return StabilityMetrics(
                is_stable=False,
                current_range=(0, 0),
                volatility=1.0,
                trend_strength=0,
                volume_profile=0,
                confidence=0
            )
            
        # Calculate core metrics
        current_range = self.detect_price_range(df)
        range_size = (current_range[1] - current_range[0]) / current_range[0]
        
        recent_prices = df['close'].tail(self.range_samples)
        volatility = recent_prices.pct_change().std()
        
        trend_strength = self.calculate_trend_strength(df)
        volume_profile = self.calculate_volume_profile(df)
        
        # Consider BTC volatility if available
        btc_volatility = 0
        if btc_df is not None and len(btc_df) >= self.range_samples:
            btc_prices = btc_df['close'].tail(self.range_samples)
            btc_volatility = btc_prices.pct_change().std()
        
        # Determine stability
        is_stable = (
            range_size <= self.stability_threshold and
            volatility <= self.volatility_threshold and
            trend_strength <= 0.3 and  # Not strongly trending
            volume_profile >= 0.7 and  # Consistent volume
            btc_volatility <= self.volatility_threshold
        )
        
        # Calculate confidence score
        confidence = (
            (1 - range_size/self.stability_threshold) * 0.3 +
            (1 - volatility/self.volatility_threshold) * 0.3 +
            volume_profile * 0.2 +
            (1 - trend_strength) * 0.2
        )
        confidence = max(min(confidence, 1), 0)  # Normalize to 0-1
        
        return StabilityMetrics(
            is_stable=is_stable,
            current_range=current_range,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            confidence=confidence
        )
        
    def should_trade(self, 
                    df: pd.DataFrame, 
                    btc_df: pd.DataFrame = None,
                    min_confidence: float = 0.7) -> bool:
        """
        Determines if market conditions are suitable for trading.
        """
        metrics = self.analyze_stability(df, btc_df)
        
        if not metrics.is_stable:
            return False
            
        if metrics.confidence < min_confidence:
            return False
            
        # Check if range has been stable for minimum required candles
        if self.last_range != metrics.current_range:
            self.last_range = metrics.current_range
            self.range_start_time = df.index[-1]
            return False
            
        if self.range_start_time is None:
            return False
            
        candles_in_range = len(df[df.index >= self.range_start_time])
        return candles_in_range >= self.min_range_candles