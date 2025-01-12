from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import pandas_ta as ta

from hummingbot.client.settings import AllConnectorSettings
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase
from hummingbot.core.data_type.common import OrderType, PriceType
from hummingbot.client.config.config_validators import validate_bool, validate_decimal
from hummingbot.client.config.config_var import ConfigVar

class PositionState(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    HOLDING = "holding"
    EXITING = "exiting"

@dataclass
class StabilityMetrics:
    is_stable: bool
    current_range: Tuple[float, float]
    volatility: float
    trend_strength: float
    volume_profile: float
    confidence: float

class EnhancedAdaptiveStrategy(DirectionalStrategyBase):
    """
    Enhanced Adaptive Strategy that combines market stability detection, position management,
    and range-based trading in a single strategy.
    
    Key Features:
    - Detects and trades within stable price ranges
    - Dynamic position sizing based on market conditions
    - Advanced risk management with trailing stops
    - Multiple timeframe analysis
    - Configurable through Hummingbot CLI
    """
    
    default_exchange = "coinbase_advanced_trade"
    default_trading_pair = "BTC-USD"
    markets: Dict[str, Set[str]] = {}
    
    @classmethod
    def get_markets(cls) -> Dict[str, Set[str]]:
        return {
            cls._config_map.get("exchange").value:
            {cls._config_map.get("trading_pair").value}
        }

    @property
    def config_maps(self) -> Dict:
        return {
            "exchange": ConfigVar(
                key="exchange",
                prompt="Enter the exchange name >>> ",
                type_str="str",
                validator=lambda v: v in AllConnectorSettings.get_exchange_names(),
                default=self.default_exchange),
            "trading_pair": ConfigVar(
                key="trading_pair",
                prompt="Enter the trading pair >>> ",
                type_str="str",
                default=self.default_trading_pair),
            "order_amount": ConfigVar(
                key="order_amount",
                prompt="Enter the base order amount >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, min_value=0),
                default=Decimal("0.01")),
            "max_position_size": ConfigVar(
                key="max_position_size",
                prompt="Enter maximum position size >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, min_value=0),
                default=Decimal("0.05")),
            "stop_loss_pct": ConfigVar(
                key="stop_loss_pct",
                prompt="Enter stop loss percentage >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, 0, 100),
                default=Decimal("1.58")),
            "stability_threshold": ConfigVar(
                key="stability_threshold",
                prompt="Enter price range stability threshold (%) >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, 0, 100),
                default=Decimal("2.0")),
            "min_range_size": ConfigVar(
                key="min_range_size",
                prompt="Enter minimum range size (%) >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, 0, 100),
                default=Decimal("1.0")),
            "enable_trailing_stop": ConfigVar(
                key="enable_trailing_stop",
                prompt="Do you want to enable trailing stop? (Yes/No) >>> ",
                type_str="bool",
                validator=lambda v: validate_bool(v),
                default=True),
        }

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        
        # Strategy state
        self.current_range: Optional[Tuple[Decimal, Decimal]] = None
        self.range_start_time = None
        self.min_range_candles = 12
        self.position_state = PositionState.WAITING
        self.current_position = Decimal("0")
        self.entry_prices: List[Decimal] = []
        self.trailing_stop_price = None
        
        # Initialize candles for multiple timeframes
        self.candles = [
            CandlesFactory.get_candle(CandlesConfig(
                connector=self.exchange,
                trading_pair=self.trading_pair,
                interval=interval,
                max_records=1000
            )) for interval in ["1m", "5m", "15m"]
        ]

    def detect_price_range(self, df: pd.DataFrame) -> Tuple[Decimal, Decimal]:
        """Detects the current price range using recent price action."""
        range_samples = 20
        recent_highs = df['high'].tail(range_samples)
        recent_lows = df['low'].tail(range_samples)
        
        range_high = Decimal(str(np.percentile(recent_highs, 90)))
        range_low = Decimal(str(np.percentile(recent_lows, 10)))
        
        return (range_low, range_high)

    def is_range_stable(self, df: pd.DataFrame) -> bool:
        """Determines if the current price range is stable enough for trading."""
        if self.current_range is None:
            return False
            
        stability_threshold = self._config_map.get("stability_threshold").value / Decimal("100")
        range_size = (self.current_range[1] - self.current_range[0]) / self.current_range[0]
        
        if range_size < self._config_map.get("min_range_size").value / Decimal("100"):
            return False
            
        recent_prices = df['close'].tail(20)
        volatility = Decimal(str(recent_prices.pct_change().std()))
        
        return range_size <= stability_threshold and volatility <= stability_threshold

    def should_pause_trading(self, df: pd.DataFrame) -> bool:
        """Determines if trading should be paused due to market conditions."""
        if not self.current_range:
            return True
            
        current_price = Decimal(str(df['close'].iloc[-1]))
        
        # Check for large price movements outside the range
        range_size = self.current_range[1] - self.current_range[0]
        price_deviation = abs(current_price - sum(self.current_range) / Decimal("2"))
        
        if price_deviation > range_size * Decimal("0.5"):
            return True
            
        # Check for increased volatility
        recent_volatility = Decimal(str(df['close'].pct_change().tail(20).std()))
        if recent_volatility > self._config_map.get("stability_threshold").value / Decimal("100"):
            return True
            
        return False

    def calculate_position_size(self, current_price: Decimal) -> Decimal:
        """Calculates appropriate position size based on market conditions."""
        if not self.is_range_stable(self.candles[1].candles_df):
            return Decimal("0")
            
        base_size = self._config_map.get("order_amount").value
        max_size = self._config_map.get("max_position_size").value
        
        # Scale size based on position within range
        if self.current_range:
            range_position = (current_price - self.current_range[0]) / (self.current_range[1] - self.current_range[0])
            if range_position <= Decimal("0.3"):  # Near bottom of range
                size_multiplier = Decimal("1.2")
            elif range_position >= Decimal("0.7"):  # Near top of range
                size_multiplier = Decimal("0.8")
            else:
                size_multiplier = Decimal("1.0")
                
            position_size = base_size * size_multiplier
            return min(position_size, max_size - self.current_position)
            
        return Decimal("0")

    def update_trailing_stop(self, current_price: Decimal):
        """Updates the trailing stop price based on current market price."""
        if not self._config_map.get("enable_trailing_stop").value:
            return
            
        if self.current_position > 0:
            if self.trailing_stop_price is None:
                # Initialize trailing stop
                stop_distance = current_price * self._config_map.get("stop_loss_pct").value / Decimal("100")
                self.trailing_stop_price = current_price - stop_distance
            else:
                # Update trailing stop if price moves higher
                stop_distance = current_price * self._config_map.get("stop_loss_pct").value / Decimal("100")
                new_stop = current_price - stop_distance
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop

    def should_exit_position(self, current_price: Decimal) -> bool:
        """Determines if current position should be exited."""
        if self.current_position == 0:
            return False
            
        # Check trailing stop
        if self.trailing_stop_price and current_price < self.trailing_stop_price:
            return True
            
        # Check if price moved significantly outside the range
        if self.current_range:
            if current_price < self.current_range[0] * Decimal("0.98") or \
               current_price > self.current_range[1] * Decimal("1.02"):
                return True
                
        return False

    def get_signal(self) -> int:
        """
        Main signal generation method implementing the range-based trading logic.
        Returns: 1 (buy), -1 (sell), or 0 (hold)
        """
        if not all(candle.ready for candle in self.candles):
            return 0
            
        df_5m = self.candles[1].candles_df
        current_price = Decimal(str(df_5m['close'].iloc[-1]))
        
        # Update trailing stop
        self.update_trailing_stop(current_price)
        
        # Check for position exit
        if self.should_exit_position(current_price):
            if self.current_position > 0:
                self.position_state = PositionState.EXITING
                return -1
                
        # Detect and validate price range
        new_range = self.detect_price_range(df_5m)
        if self.current_range != new_range:
            self.current_range = new_range
            self.range_start_time = df_5m.index[-1]
            return 0
            
        # Check if we should pause trading
        if self.should_pause_trading(df_5m):
            self.position_state = PositionState.HOLDING
            return 0
            
        # Only trade if range is stable
        if not self.is_range_stable(df_5m):
            return 0
            
        # Calculate technical indicators
        df_5m.ta.rsi(length=14, append=True)
        df_5m.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        rsi = Decimal(str(df_5m['RSI_14'].iloc[-1]))
        macd = Decimal(str(df_5m['MACD_12_26_9'].iloc[-1]))
        macd_signal = Decimal(str(df_5m['MACDs_12_26_9'].iloc[-1]))
        
        # Generate trading signals
        if self.position_state != PositionState.ACTIVE:
            if current_price <= self.current_range[0] * Decimal("1.02") and \
               rsi < Decimal("30") and macd > macd_signal:
                position_size = self.calculate_position_size(current_price)
                if position_size > 0:
                    self.position_state = PositionState.ACTIVE
                    return 1
                    
        elif current_price >= self.current_range[1] * Decimal("0.98") and \
             rsi > Decimal("70") and macd < macd_signal:
            self.position_state = PositionState.EXITING
            return -1
            
        return 0

    def market_data_extra_info(self) -> List[str]:
        """Provides detailed market information for strategy monitoring."""
        lines = []
        
        if self.current_range:
            lines.extend([
                f"\nCurrent Range: ${self.current_range[0]:.2f} - ${self.current_range[1]:.2f}",
                f"Range Size: {((self.current_range[1] - self.current_range[0]) / self.current_range[0] * 100):.2f}%"
            ])
            
        if self.candles[1].ready:
            df = self.candles[1].candles_df
            current_price = Decimal(str(df['close'].iloc[-1]))
            
            lines.extend([
                f"Current Price: ${current_price:.2f}",
                f"Position State: {self.position_state.value}",
                f"Current Position: {self.current_position}",
                f"Average Entry: ${sum(self.entry_prices) / len(self.entry_prices):.2f}" if self.entry_prices else "No position"
            ])
            
            if self.trailing_stop_price:
                lines.append(f"Trailing Stop: ${self.trailing_stop_price:.2f}")
                
        return lines