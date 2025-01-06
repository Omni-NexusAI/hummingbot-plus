from decimal import Decimal
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pandas_ta as ta

from .market_stability_detector import MarketStabilityDetector, StabilityMetrics
from .position_manager import PositionManager, PositionState

from hummingbot.client.settings import AllConnectorSettings
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase
from hummingbot.core.data_type.common import OrderType, PriceType
from hummingbot.client.config.config_validators import validate_bool, validate_decimal
from hummingbot.client.config.config_var import ConfigVar


class AdvancedMarketAdaptiveStrategy(DirectionalStrategyBase):
    """
    Advanced Market Adaptive Strategy that dynamically adjusts to market conditions.
    
    Key Features:
    - Market state detection (stable, volatile, trending)
    - BTC correlation tracking
    - Dynamic position management
    - Configurable through Hummingbot CLI
    """
    
    # Default config
    default_exchange = "coinbase_advanced_trade"
    default_trading_pair = "ADA-USD"
    default_ref_pair = "BTC-USD"
    
    # Strategy configs that can be changed through CLI
    markets: Dict[str, Set[str]] = {}
    
    @classmethod
    def get_markets(cls) -> Dict[str, Set[str]]:
        return {
            cls._config_map.get("exchange").value:
            {cls._config_map.get("trading_pair").value, cls._config_map.get("ref_pair").value}
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
            "ref_pair": ConfigVar(
                key="ref_pair",
                prompt="Enter the reference pair (default: BTC-USD) >>> ",
                type_str="str",
                default=self.default_ref_pair),
            "order_amount": ConfigVar(
                key="order_amount",
                prompt="Enter the order amount in quote currency >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, min_value=0),
                default=Decimal("100")),
            "max_position_size": ConfigVar(
                key="max_position_size",
                prompt="Enter maximum position size in quote currency >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, min_value=0),
                default=Decimal("1000")),
            "stop_loss_pct": ConfigVar(
                key="stop_loss_pct",
                prompt="Enter stop loss percentage >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, 0, 100),
                default=Decimal("1.58")),
            "enable_trailing_stop": ConfigVar(
                key="enable_trailing_stop",
                prompt="Do you want to enable trailing stop? (Yes/No) >>> ",
                type_str="bool",
                validator=lambda v: validate_bool(v),
                default=True),
        }

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        
        # Initialize stability detector and position manager
        self.stability_detector = MarketStabilityDetector(
            stability_threshold=0.02,
            volatility_threshold=0.05,
            range_samples=20,
            min_range_candles=12
        )
        
        self.position_manager = PositionManager(
            max_position_size=self._config_map.get("max_position_size").value,
            stop_loss_pct=self._config_map.get("stop_loss_pct").value / Decimal("100"),
            take_profit_pct=Decimal("0.15"),  # 15% initial take profit
            max_drawdown_pct=Decimal("5.0"),
            position_step_size=Decimal("0.1")
        )
        
        # Initialize candles for trading pair and BTC reference
        self.candles = [
            CandlesFactory.get_candle(CandlesConfig(
                connector=self.exchange,
                trading_pair=self.trading_pair,
                interval=interval,
                max_records=1000
            )) for interval in ["1m", "5m", "15m"]
        ]
        
        # BTC reference candles
        self.btc_candles = [
            CandlesFactory.get_candle(CandlesConfig(
                connector=self.exchange,
                trading_pair=self.ref_pair,
                interval=interval,
                max_records=1000
            )) for interval in ["1m", "5m", "15m"]
        ]

    def _calculate_market_state(self) -> str:
        """
        Determines current market state based on price action and volatility.
        Returns: "stable", "volatile", or "trending"
        """
        df = self.candles[1].candles_df  # Using 5m candles for state detection
        
        # Calculate price range and volatility
        recent_prices = df['close'].tail(self.range_samples)
        price_range = (recent_prices.max() - recent_prices.min()) / recent_prices.mean()
        volatility = recent_prices.pct_change().std()
        
        # Check BTC correlation
        btc_prices = self.btc_candles[1].candles_df['close'].tail(self.range_samples)
        btc_volatility = btc_prices.pct_change().std()
        
        if btc_volatility > self.volatility_threshold:
            return "volatile"  # BTC is volatile, consider market unstable
        
        if price_range <= self.stability_threshold and volatility <= self.volatility_threshold:
            return "stable"
        elif volatility > self.volatility_threshold:
            return "volatile"
        else:
            return "trending"

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical analysis with volatility metrics"""
        df = df.copy()
        
        # Core indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        
        # Volatility indicators
        df.ta.atr(length=14, append=True)
        df['volatility'] = df['close'].pct_change().rolling(window=14).std()
        
        # Volume analysis
        df.ta.vwap(append=True)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        return df

    def _analyze_signals(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> int:
        """
        Enhanced signal generation with market state awareness
        """
        market_state = self._calculate_market_state()
        self.market_state = market_state
        
        if market_state == "volatile":
            return 0  # No new trades during high volatility
            
        # Get latest indicator values
        rsi_5m = df_5m['RSI_14'].iloc[-1]
        macd_5m = df_5m['MACD_12_26_9'].iloc[-1]
        macd_signal_5m = df_5m['MACDs_12_26_9'].iloc[-1]
        bb_lower = df_5m['BBL_20_2.0'].iloc[-1]
        bb_upper = df_5m['BBU_20_2.0'].iloc[-1]
        current_price = df_5m['close'].iloc[-1]
        
        # Volume confirmation
        volume_trend = df_5m['volume'].iloc[-1] > df_5m['volume_sma'].iloc[-1]
        
        # BTC trend check
        btc_trend = self.btc_candles[1].candles_df['close'].pct_change(5).iloc[-1]
        
        if market_state == "stable":
            # More conservative signals during stable periods
            if (rsi_5m < 30 and current_price <= bb_lower and 
                macd_5m > macd_signal_5m and volume_trend and btc_trend > 0):
                return 1
            elif (rsi_5m > 70 and current_price >= bb_upper and 
                  macd_5m < macd_signal_5m and volume_trend and btc_trend < 0):
                return -1
        elif market_state == "trending":
            # Trend-following signals
            if (rsi_5m < 40 and macd_5m > macd_signal_5m and 
                volume_trend and btc_trend > 0):
                return 1
            elif (rsi_5m > 60 and macd_5m < macd_signal_5m and 
                  volume_trend and btc_trend < 0):
                return -1
                
        return 0

    def get_signal(self) -> int:
        """
        Main signal generation method with market state awareness and position management
        """
        if not all(candle.ready for candle in self.candles + self.btc_candles):
            return 0
            
        # Get current price and process dataframes
        current_price = Decimal(str(self.candles[1].candles_df['close'].iloc[-1]))
        df_1m = self._process_dataframe(self.candles[0].candles_df)
        df_5m = self._process_dataframe(self.candles[1].candles_df)
        df_15m = self._process_dataframe(self.candles[2].candles_df)
        
        # Check market stability
        stability_metrics = self.stability_detector.analyze_stability(
            df_5m, 
            self.btc_candles[1].candles_df
        )
        
        # Update position manager
        should_reduce, reduce_size = self.position_manager.should_reduce_position(
            current_price,
            {
                "volatility": stability_metrics.volatility,
                "is_stable": stability_metrics.is_stable,
                "confidence": stability_metrics.confidence
            }
        )
        
        if should_reduce:
            self.position_manager.update_position(
                -reduce_size,
                current_price,
                {"action": "reduce"}
            )
            return -1 if reduce_size == self.position_manager.current_position else 0
            
        # Only look for new signals if market is stable
        if not self.stability_detector.should_trade(df_5m, self.btc_candles[1].candles_df):
            return 0
            
        # Get trading signal
        signal = self._analyze_signals(df_1m, df_5m, df_15m)
        
        if signal != 0:
            # Calculate position size
            position_size = self.position_manager.calculate_position_size(
                current_price,
                {
                    "confidence": stability_metrics.confidence,
                    "volatility": stability_metrics.volatility,
                    "is_stable": stability_metrics.is_stable
                }
            )
            
            if position_size > 0:
                self.position_manager.update_position(
                    position_size if signal == 1 else -position_size,
                    current_price,
                    {"action": "new_position"}
                )
                return signal
                
        return 0

    def market_data_extra_info(self) -> List[str]:
        """Enhanced market data display with stability and position information"""
        lines = []
        
        # Get stability metrics
        df_5m = self._process_dataframe(self.candles[1].candles_df)
        stability_metrics = self.stability_detector.analyze_stability(
            df_5m,
            self.btc_candles[1].candles_df
        )
        
        # Add market state information
        lines.extend([
            "\n=== Market State Information ===",
            f"Stability: {'STABLE' if stability_metrics.is_stable else 'UNSTABLE'}",
            f"Current Range: ${stability_metrics.current_range[0]:.4f} - ${stability_metrics.current_range[1]:.4f}",
            f"Volatility: {stability_metrics.volatility:.4f}",
            f"Trend Strength: {stability_metrics.trend_strength:.4f}",
            f"Volume Profile: {stability_metrics.volume_profile:.4f}",
            f"Confidence: {stability_metrics.confidence:.4f}",
            f"BTC Correlation: {self._calculate_btc_correlation():.4f}\n"
        ])
        
        # Add position information
        position_metrics = self.position_manager.get_position_metrics()
        lines.extend([
            "=== Position Information ===",
            f"Current Position: {position_metrics['current_position']}",
            f"Position State: {position_metrics['position_state']}",
            f"Average Entry: ${position_metrics['average_entry']:.4f}",
            f"Last Update: {position_metrics['last_update']}\n"
        ])
        
        # Add candles information for both trading pair and BTC
        for i, interval in enumerate(["1m", "5m", "15m"]):
            # Trading pair candles
            df = self._process_dataframe(self.candles[i].candles_df)
            lines.extend([f"\n=== {self.trading_pair} Candles | Interval: {interval} ==="])
            columns = ["timestamp", "open", "close", "volume", "RSI_14", 
                      "MACD_12_26_9", "BBL_20_2.0", "BBU_20_2.0"]
            lines.extend(self.candles_formatted_list(df, columns))
            
            # BTC reference candles
            btc_df = self._process_dataframe(self.btc_candles[i].candles_df)
            lines.extend([f"\n=== {self.ref_pair} Reference | Interval: {interval} ==="])
            lines.extend(self.candles_formatted_list(btc_df, columns))
            
        return lines

    def _calculate_btc_correlation(self) -> float:
        """Calculate correlation with BTC price movement"""
        df = self.candles[1].candles_df['close'].pct_change()
        btc_df = self.btc_candles[1].candles_df['close'].pct_change()
        return df.corr(btc_df)