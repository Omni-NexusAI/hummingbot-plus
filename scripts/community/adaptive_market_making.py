from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
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

class AdaptiveMarketMakingStrategy(DirectionalStrategyBase):
    """
    Adaptive Market Making Strategy

    This strategy integrates multi-pair trading (using BTC as a market reference),
    dynamic technical analysis (RSI, MACD, Bollinger Bands, Volume Profile), and
    adaptive risk management (dynamic position sizing, trailing stops, max drawdown).
    
    It is built to be fully configurable via Hummingbot’s CLI using the 'create' and 'config' commands.
    """
    
    default_exchange = "coinbase_advanced_trade"
    default_trading_pair = "ETH-USD"  # Default trading pair for non-BTC assets
    markets: Dict[str, Set[str]] = {}

    @classmethod
    def get_markets(cls) -> Dict[str, Set[str]]:
        return {
            cls._config_map.get("exchange").value: {
                cls._config_map.get("trading_pair").value,
                cls._config_map.get("btc_reference_pair").value
            }
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
                prompt="Enter the trading pair (non-BTC) >>> ",
                type_str="str",
                default="ETH-USD"),
            "btc_reference_pair": ConfigVar(
                key="btc_reference_pair",
                prompt="Enter the BTC reference pair >>> ",
                type_str="str",
                default="BTC-USD"),
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
                prompt="Enter trailing stop loss percentage >>> ",
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
            "dynamic_granularity": ConfigVar(
                key="dynamic_granularity",
                prompt="Enter dynamic granularity factor >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, min_value=0),
                default=Decimal("1.0")),
            "max_drawdown_pct": ConfigVar(
                key="max_drawdown_pct",
                prompt="Enter maximum drawdown percentage >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, 0, 100),
                default=Decimal("10.0")),
            "risk_tolerance": ConfigVar(
                key="risk_tolerance",
                prompt="Enter your risk tolerance as a fraction of account equity (e.g., 0.02 for 2%) >>> ",
                type_str="decimal",
                validator=lambda v: validate_decimal(v, min_value=Decimal("0")),
                default=Decimal("0.02")),
            "enable_trailing_stop": ConfigVar(
                key="enable_trailing_stop",
                prompt="Enable trailing stop? (Yes/No) >>> ",
                type_str="bool",
                validator=lambda v: validate_bool(v),
                default=True),
        }

    def __init__(self, connectors: Dict[str, any]):
        super().__init__(connectors)
        # Strategy state initialization
        self.current_range: Optional[Tuple[Decimal, Decimal]] = None
        self.range_start_time = None
        self.position_state = PositionState.WAITING
        self.current_position = Decimal("0")
        self.entry_prices: List[Decimal] = []
        self.trailing_stop_price = None

        # Initialize candle feeds for multiple timeframes for primary trading pair
        self.candles = [
            CandlesFactory.get_candle(CandlesConfig(
                connector=self.exchange,
                trading_pair=self.trading_pair,
                interval=interval,
                max_records=1000
            )) for interval in ["1m", "5m", "15m"]
        ]
        # Initialize BTC reference candles for overall market volatility analysis
        self.btc_candles = CandlesFactory.get_candle(CandlesConfig(
            connector=self.exchange,
            trading_pair=self._config_map.get("btc_reference_pair").value,
            interval="5m",
            max_records=1000
        ))

    def calculate_account_equity(self, trading_pair: str) -> Decimal:
        base, quote = trading_pair.split("-")
        connector = self.connectors[self._config_map.get("exchange").value]
        equity = connector.get_balance(quote)
        self.logger().debug(f"Account equity for {quote}: {equity}")
        return Decimal(str(equity))

    def detect_price_range(self, df: pd.DataFrame) -> Tuple[Decimal, Decimal]:
        range_samples = 20
        recent_highs = df['high'].tail(range_samples)
        recent_lows = df['low'].tail(range_samples)
        range_high = Decimal(str(np.percentile(recent_highs, 90)))
        range_low = Decimal(str(np.percentile(recent_lows, 10)))
        return (range_low, range_high)

    def is_range_stable(self, df: pd.DataFrame) -> bool:
        if self.current_range is None:
            return False
        stability_threshold = self._config_map.get("stability_threshold").value / Decimal("100")
        range_size = (self.current_range[1] - self.current_range[0]) / self.current_range[0]
        if range_size < self._config_map.get("min_range_size").value / Decimal("100"):
            return False
        recent_prices = df['close'].tail(20)
        volatility = Decimal(str(recent_prices.pct_change().std()))
        return range_size <= stability_threshold and volatility <= stability_threshold

    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Decimal:
        prices = df['close']
        volumes = df['volume']
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
        median_bin = np.argmax(hist)
        profile_strength = Decimal(str(hist[median_bin] / volumes.sum()))
        return profile_strength

    def calculate_bollinger_bands(self, df: pd.DataFrame, length: int = 20, std: int = 2) -> Tuple[Decimal, Decimal]:
        df['BBL'], df['BBM'], df['BBU'] = ta.bbands(df['close'], length=length, std=std)
        lower_band = Decimal(str(df['BBL'].iloc[-1]))
        upper_band = Decimal(str(df['BBU'].iloc[-1]))
        return lower_band, upper_band

    def calculate_btc_volatility(self) -> Decimal:
        df = self.btc_candles.candles_df
        recent_returns = df['close'].pct_change().tail(20)
        volatility = Decimal(str(recent_returns.std()))
        return volatility

    def calculate_position_size(self, current_price: Decimal, df: pd.DataFrame) -> Decimal:
        if not self.is_range_stable(df):
            return Decimal("0")
        base_size = self._config_map.get("order_amount").value
        max_size = self._config_map.get("max_position_size").value

        # Dynamic sizing based on position within range
        if self.current_range:
            range_position = (current_price - self.current_range[0]) / (self.current_range[1] - self.current_range[0])
            if range_position <= Decimal("0.3"):
                size_multiplier = Decimal("1.2")
            elif range_position >= Decimal("0.7"):
                size_multiplier = Decimal("0.8")
            else:
                size_multiplier = Decimal("1.0")
        else:
            size_multiplier = Decimal("1.0")

        local_volatility = Decimal(str(df['close'].pct_change().tail(20).std()))
        btc_volatility = self.calculate_btc_volatility()
        volatility_factor = Decimal("1.0")
        threshold = Decimal("0.02") * self._config_map.get("dynamic_granularity").value
        if local_volatility > threshold or btc_volatility > threshold:
            volatility_factor = Decimal("0.8")

        # Risk-based adjustment
        account_equity = self.calculate_account_equity(self._config_map.get("trading_pair").value)
        risk_tolerance = self._config_map.get("risk_tolerance").value
        risk_based_size = account_equity * risk_tolerance

        position_size = base_size * size_multiplier * volatility_factor
        final_size = min(position_size, risk_based_size, max_size - self.current_position)
        self.logger().debug(f"Calculated position size: {final_size} (Base: {base_size}, Multiplier: {size_multiplier}, Volatility Factor: {volatility_factor}, Risk-Based: {risk_based_size})")
        return final_size

    def update_trailing_stop(self, current_price: Decimal):
        if not self._config_map.get("enable_trailing_stop").value:
            return
        df = self.candles[1].candles_df
        recent_volatility = Decimal(str(df['close'].pct_change().tail(20).std()))
        dynamic_stop_pct = self._config_map.get("stop_loss_pct").value / Decimal("100")
        adjusted_stop = dynamic_stop_pct + recent_volatility * self._config_map.get("dynamic_granularity").value

        if self.current_position > 0:
            if self.trailing_stop_price is None:
                self.trailing_stop_price = current_price - current_price * adjusted_stop
                self.logger().info(f"Initializing trailing stop at: {self.trailing_stop_price}")
            else:
                new_stop = current_price - current_price * adjusted_stop
                if new_stop > self.trailing_stop_price:
                    self.logger().info(f"Updating trailing stop from {self.trailing_stop_price} to {new_stop}")
                    self.trailing_stop_price = new_stop

    def should_exit_position(self, current_price: Decimal) -> bool:
        if self.current_position == 0:
            return False
        if self.trailing_stop_price and current_price < self.trailing_stop_price:
            return True
        if self.current_range:
            if current_price < self.current_range[0] * Decimal("0.98") or \
               current_price > self.current_range[1] * Decimal("1.02"):
                return True
        df = self.candles[1].candles_df
        recent_peak = max(df['close'].tail(20))
        max_drawdown = (Decimal(str(recent_peak)) - current_price) / Decimal(str(recent_peak))
        if max_drawdown > self._config_map.get("max_drawdown_pct").value / Decimal("100"):
            return True
        return False

    def check_market_extremes(self) -> Tuple[bool, bool]:
        df = self.btc_candles.candles_df
        recent_prices = df['close'].tail(20)
        current_btc_price = Decimal(str(recent_prices.iloc[-1]))
        recent_max = Decimal(str(recent_prices.max()))
        recent_min = Decimal(str(recent_prices.min()))
        drop_pct = (recent_max - current_btc_price) / recent_max
        rally_pct = (current_btc_price - recent_min) / recent_min
        extreme_drop = drop_pct > Decimal("0.15")
        extreme_rally = rally_pct > Decimal("0.15")
        return extreme_drop, extreme_rally

    def get_signal(self) -> int:
        if not all(candle.ready for candle in self.candles) or not self.btc_candles.ready:
            return 0
        df_5m = self.candles[1].candles_df
        try:
            current_price = Decimal(str(df_5m['close'].iloc[-1]))
            self.update_trailing_stop(current_price)
            extreme_drop, extreme_rally = self.check_market_extremes()
            if extreme_drop:
                if self.current_position > 0:
                    self.position_state = PositionState.EXITING
                    return -1
                else:
                    return 0
            if extreme_rally and self.position_state != PositionState.ACTIVE:
                return 0
            if self.should_exit_position(current_price):
                if self.current_position > 0:
                    self.position_state = PositionState.EXITING
                    return -1

            new_range = self.detect_price_range(df_5m)
            if self.current_range != new_range:
                self.current_range = new_range
                self.range_start_time = df_5m.index[-1]
                return 0
            if not self.is_range_stable(df_5m):
                self.position_state = PositionState.HOLDING
                return 0

            # Calculate technical indicators
            df_5m.ta.rsi(length=14, append=True)
            df_5m.ta.macd(fast=12, slow=26, signal=9, append=True)
            lower_band, upper_band = self.calculate_bollinger_bands(df_5m)
            volume_profile = self.calculate_volume_profile(df_5m)

            rsi = Decimal(str(df_5m['RSI_14'].iloc[-1]))
            macd = Decimal(str(df_5m['MACD_12_26_9'].iloc[-1]))
            macd_signal = Decimal(str(df_5m['MACDs_12_26_9'].iloc[-1]))

            if self.position_state != PositionState.ACTIVE:
                if (current_price <= self.current_range[0] * Decimal("1.02") and
                    rsi < Decimal("30") and
                    macd > macd_signal and
                    current_price <= lower_band):
                    position_size = self.calculate_position_size(current_price, df_5m)
                    if position_size > 0:
                        self.position_state = PositionState.ACTIVE
                        return 1
            else:
                if (current_price >= self.current_range[1] * Decimal("0.98") and
                    rsi > Decimal("70") and
                    macd < macd_signal and
                    current_price >= upper_band):
                    self.position_state = PositionState.EXITING
                    return -1
        except Exception as e:
            self.logger().error(f"Error in signal generation: {e}")
        return 0

    def market_data_extra_info(self) -> List[str]:
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
                f"Average Entry: ${sum(self.entry_prices)/len(self.entry_prices):.2f}" if self.entry_prices else "No position"
            ])
            if self.trailing_stop_price:
                lines.append(f"Trailing Stop: ${self.trailing_stop_price:.2f}")
        btc_vol = self.calculate_btc_volatility() * Decimal("100")
        lines.append(f"BTC Volatility (approx.): {btc_vol:.2f}%")
        return lines
