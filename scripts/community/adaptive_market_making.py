import os
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import pandas as pd
import datetime

# Import necessary Hummingbot components
from pydantic import Field

from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase
from hummingbot.core.data_type.common import OrderType, TradeType, PositionAction, PositionMode
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.client.settings import AllConnectorSettings
from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData


class AdaptiveMMConfig(BaseClientModel):
    """
    Configuration for Adaptive Market Making script. Enables interactive wizard
    via `create --script-config` and startup via `start --script ... --conf ...`.
    """
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    exchange: str = Field(
        default="kraken_paper_trade",
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Exchange where the bot will trade"),
    )
    trading_pair: str = Field(
        default="ADA-USD",
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Trading pair to trade"),
    )
    candles_interval: str = Field(
        default="5m",
        client_data=ClientFieldData(prompt_on_new=False, prompt=lambda mi: "Candles interval (e.g. 1m/5m/15m)"),
    )
    max_candle_records: int = Field(
        default=500,
        client_data=ClientFieldData(prompt_on_new=False, prompt=lambda mi: "Max candle records to fetch"),
    )
    order_amount_usd: Decimal = Field(
        default=Decimal("10"),
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order notional size in quote currency (USD)"),
    )
    leverage: int = Field(
        default=1,
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Leverage"),
    )
    position_mode: str = Field(
        default="HEDGE",
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Position mode (HEDGE/ONEWAY)"),
    )
    stop_loss: float = Field(
        default=0.015,
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Stop loss (e.g. 0.015 = 1.5%)"),
    )
    take_profit: float = Field(
        default=0.03,
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Take profit (e.g. 0.03 = 3%)"),
    )
    time_limit: int = Field(
        default=300,
        client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Time limit for a position in seconds"),
    )

class PositionState(Enum):
    WAITING = "waiting"
    ACTIVE = "active" 
    HOLDING = "holding"
    EXITING = "exiting"

class AdaptiveMarketMakingStrategy(DirectionalStrategyBase):
    """
    Adaptive Market Making Strategy
    
    A flexible market making strategy that adapts to market conditions
    using technical indicators and price action analysis.
    """
    
    # Strategy configuration
    directional_strategy_name = "adaptive_market_making"
    default_exchange = "kraken_paper_trade"  # Use paper trade by default
    default_trading_pair = "ADA-USD"
    trading_pair = default_trading_pair
    exchange = default_exchange
    candles = []
    btc_candles = None
    markets = {default_exchange: {default_trading_pair}}
    candles_interval: str = "5m"
    
    # Maximum candle records for Kraken to avoid errors
    # Kraken only supports a maximum of 720 records
    max_candle_records = 500
    
    # Strategy settings
    max_executors = 1
    position_mode = PositionMode.HEDGE
    leverage = 1
    stop_loss = 0.015
    take_profit = 0.03
    time_limit = 300
    open_order_type = OrderType.MARKET
    take_profit_order_type = OrderType.MARKET
    stop_loss_order_type = OrderType.MARKET
    time_limit_order_type = OrderType.MARKET
    trailing_stop_activation_delta = 0.01
    trailing_stop_trailing_delta = 0.005
    cooldown_after_execution = 60
    
    # Additional strategy settings
    risk_factor = 0.02  # 2% risk per trade
    max_position_size = Decimal("0.05") 
    min_volatility = Decimal("0.005")  # 0.5% minimum volatility
    max_drawdown = Decimal("0.10")  # 10% maximum drawdown
    
    # Market state variables
    current_range = None
    range_start_time = None
    entry_prices = []
    _trade_history = []

    @classmethod
    def init_markets(cls, config: AdaptiveMMConfig):
        """Called by the start command to declare which connectors/pairs are required."""
        cls.markets = {config.exchange: {config.trading_pair}}
        cls.exchange = config.exchange
        cls.trading_pair = config.trading_pair
        cls.candles_interval = config.candles_interval
        cls.max_candle_records = int(config.max_candle_records)

    @classmethod
    def get_markets(cls) -> Dict[str, Set[str]]:
        """Define the markets this strategy will use"""
        # Use paper trade version if available, otherwise use regular exchange
        if cls.exchange in AllConnectorSettings.get_exchange_names():
            return {
                cls.exchange: {
                    cls.trading_pair
                }
            }
        elif cls.exchange.replace("_paper_trade", "") in AllConnectorSettings.get_exchange_names():
            # Try with base exchange name if paper trade version isn't found
            base_exchange = cls.exchange.replace("_paper_trade", "")
            return {
                base_exchange: {
                    cls.trading_pair
                }
            }
        else:
            # Return default as fallback
            return {
                "paper_trade": {
                    cls.trading_pair
                }
            }
    
    def initialize_candles(self):
        """Initialize candles with proper record limit to avoid Kraken API errors"""
        try:
            # Initialize with a safe number of records (well below Kraken's 720 limit)
            candle_exchange = self.exchange.replace("_paper_trade", "")
            self.candles = [
                CandlesFactory.get_candle(CandlesConfig(
                    connector=candle_exchange,
                    trading_pair=self.trading_pair,
                    interval=self.candles_interval,
                    max_records=self.max_candle_records
                ))
            ]
            
            # Start the candles
            for candle in self.candles:
                candle.start()
                
            # Also initialize BTC reference candles (for market correlation)
            try:
                btc_pair = "BTC-USD"
                self.btc_candles = CandlesFactory.get_candle(CandlesConfig(
                    connector=candle_exchange,
                    trading_pair=btc_pair,
                    interval=self.candles_interval,
                    max_records=self.max_candle_records
                ))
                self.btc_candles.start()
                logging.getLogger(__name__).info(f"Successfully initialized BTC reference candles on {self.exchange}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not initialize BTC reference candles: {e}")
                self.btc_candles = None
                
            logging.getLogger(__name__).info(f"Successfully initialized candles for {self.trading_pair} on {self.exchange}")
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error initializing candles: {e}")

    def __init__(self, connectors: Dict[str, any], config: Optional[AdaptiveMMConfig] = None):
        """Initialize the strategy with proper setup"""
        try:
            # Initialize essential properties
            self.current_position = Decimal("0")
            self.position_state = PositionState.WAITING
            self.entry_price = Decimal("0")
            self.trailing_stop_price = None
            self._trade_history = []
            self.entry_prices = []
            self.config = config

            # Apply provided config values
            if config is not None:
                self.exchange = config.exchange
                self.trading_pair = config.trading_pair
                self.candles_interval = config.candles_interval
                self.max_candle_records = int(config.max_candle_records)
                self.stop_loss = float(config.stop_loss)
                self.take_profit = float(config.take_profit)
                self.time_limit = int(config.time_limit)
                self.leverage = int(config.leverage)
                try:
                    self.position_mode = (
                        PositionMode[config.position_mode.upper()]
                        if isinstance(config.position_mode, str)
                        else config.position_mode
                    )
                except Exception:
                    self.position_mode = PositionMode.HEDGE
                try:
                    self.order_amount_usd = Decimal(str(config.order_amount_usd))
                except Exception:
                    pass
            
            # Get available connector names for better error messages
            available_connectors = list(connectors.keys())
            logging.getLogger(__name__).info(f"Available connectors: {available_connectors}")
            
            # Handle paper trade for exchange name
            original_exchange = self.exchange
            self.paper_trade = False
            
            # If using paper trade, adjust exchange name
            if "_paper_trade" in self.exchange:
                base_exchange = self.exchange.replace("_paper_trade", "")
                self.paper_trade = True
                
                # Check if base exchange is available
                if base_exchange in available_connectors:
                    self.exchange = base_exchange
                    logging.getLogger(__name__).info(f"Using {base_exchange} (paper trade mode)")
                # If "paper_trade" is available as a connector, try to use it
                elif "paper_trade" in available_connectors:
                    self.exchange = "paper_trade"
                    logging.getLogger(__name__).info(f"Using paper_trade connector")
            
            # Check if exchange is available in connectors
            if self.exchange not in connectors:
                logging.getLogger(__name__).warning(f"Exchange {original_exchange} not found in available connectors. Available connectors: {available_connectors}")
                
                # Try to use another available connector
                if available_connectors:
                    self.exchange = available_connectors[0]
                    logging.getLogger(__name__).info(f"Using available exchange: {self.exchange}")
            
            # Initialize parent class
            super().__init__(connectors)
            logging.getLogger(__name__).info("Parent class initialized successfully")
            
            # Initialize candles after parent class initialization
            self.initialize_candles()
            
            # Print startup confirmation
            logging.getLogger(__name__).info(f"Adaptive Market Making Strategy initialized on {self.exchange} for {self.trading_pair}")
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error initializing strategy: {e}")
    
    def detect_price_range(self) -> Tuple[Decimal, Decimal]:
        """Detect price range based on recent data"""
        try:
            if not self.candles or len(self.candles) == 0 or not self.candles[0].ready:
                return (Decimal("0"), Decimal("0"))
                
            df = self.candles[0].candles_df
            if len(df) < 20:
                return (Decimal("0"), Decimal("0"))
                
            recent_highs = df['high'].tail(20)
            recent_lows = df['low'].tail(20)
            
            range_high = Decimal(str(recent_highs.max()))
            range_low = Decimal(str(recent_lows.min()))
            
            return (range_low, range_high)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error detecting price range: {e}")
            return (Decimal("0"), Decimal("0"))
    
    def calculate_volatility(self) -> Decimal:
        """Calculate recent price volatility"""
        try:
            if not self.candles or len(self.candles) == 0 or not self.candles[0].ready:
                return Decimal("0")
                
            df = self.candles[0].candles_df
            if len(df) < 20:
                return Decimal("0")
                
            # Calculate volatility as average of high-low ranges
            highs = df['high'].tail(20)
            lows = df['low'].tail(20)
            closes = df['close'].tail(20)
            
            avg_range = (highs - lows).mean() / closes.mean()
            return Decimal(str(avg_range))
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating volatility: {e}")
            return Decimal("0")
    
    def calculate_market_trend(self) -> int:
        """Determine market trend (1 = up, -1 = down, 0 = sideways)"""
        try:
            if not self.candles or len(self.candles) == 0 or not self.candles[0].ready:
                return 0
                
            df = self.candles[0].candles_df
            if len(df) < 20:
                return 0
            
            # Simple trend calculation based on short and long moving averages
            short_ma = df['close'].tail(7).mean()
            long_ma = df['close'].tail(21).mean()
            
            if short_ma > long_ma * Decimal("1.02"):  # 2% above
                return 1
            elif short_ma < long_ma * Decimal("0.98"):  # 2% below
                return -1
            else:
                return 0
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating market trend: {e}")
            return 0

    def get_signal(self) -> int:
        """Generate basic trading signal"""
        try:
            # Don't trade without data
            if not self.candles or len(self.candles) == 0 or not self.candles[0].ready:
                return 0
                
            # Get current price
            df = self.candles[0].candles_df
            if len(df) < 10:
                return 0
                
            current_price = Decimal(str(df['close'].iloc[-1]))
            if current_price <= Decimal("0"):
                return 0
            
            # Update market context
            if self.current_range is None or self.current_range[0] <= Decimal("0"):
                self.current_range = self.detect_price_range()
                if self.current_range[0] > Decimal("0"):
                    self.range_start_time = df.index[-1]
            
            # Update trailing stop
            if self.position_state == PositionState.ACTIVE and self.current_position > Decimal("0"):
                stop_level = current_price - (current_price * self.stop_loss)
                if self.trailing_stop_price is None or stop_level > self.trailing_stop_price:
                    self.trailing_stop_price = stop_level
            
            # Check for exit conditions
            if self.position_state == PositionState.ACTIVE and self.current_position > Decimal("0"):
                # Exit if trailing stop hit
                if self.trailing_stop_price and current_price < self.trailing_stop_price:
                    self.position_state = PositionState.EXITING
                    return -1
                
                # Exit if take profit reached
                if self.entry_price > Decimal("0"):
                    profit_level = self.entry_price * (1 + self.take_profit)
                    if current_price >= profit_level:
                        self.position_state = PositionState.EXITING
                        return -1
            
            # Calculate simple moving averages
            short_ma = df['close'].tail(7).mean()
            long_ma = df['close'].tail(21).mean()
            
            # Basic strategy logic
            if self.position_state != PositionState.ACTIVE:
                # Buy when short MA crosses above long MA
                if short_ma > long_ma:
                    self.position_state = PositionState.ACTIVE
                    return 1
            else:
                # Sell when short MA crosses below long MA
                if short_ma < long_ma:
                    self.position_state = PositionState.EXITING
                    return -1
                    
            return 0
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in get_signal: {e}")
            return 0

    def did_fill_order(self, event):
        """Process order filled events"""
        try:
            super().did_fill_order(event)
            
            price = event.price
            amount = event.amount
            trade_type = "BUY" if event.trade_type == TradeType.BUY else "SELL"
            
            # Update position
            if trade_type == "BUY":
                self.current_position += Decimal(str(amount))
                self.entry_price = Decimal(str(price))
                self.entry_prices.append(self.entry_price)
                self.position_state = PositionState.ACTIVE
            else:
                self.current_position -= Decimal(str(amount))
                if self.current_position <= Decimal("0"):
                    self.position_state = PositionState.WAITING
                    self.entry_price = Decimal("0")
                    self.entry_prices = []
                    self.trailing_stop_price = None
            
            # Record trade in history
            self._trade_history.append({
                "timestamp": datetime.datetime.now(),
                "side": trade_type,
                "price": price,
                "amount": amount,
                "value": price * amount
            })
                    
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in did_fill_order: {e}")
            
    def market_data_extra_info(self) -> List[str]:
        """Display market data info"""
        lines = []
        try:
            # Add market context information
            market_trend = self.calculate_market_trend()
            trend_str = "Bullish" if market_trend > 0 else "Bearish" if market_trend < 0 else "Sideways"
            
            lines.append(f"Market Trend: {trend_str}")
            
            volatility = self.calculate_volatility()
            if volatility > Decimal("0"):
                lines.append(f"Volatility: {volatility * 100:.2f}%")
            
            if self.current_range and self.current_range[0] > Decimal("0"):
                range_size = ((self.current_range[1] - self.current_range[0]) / self.current_range[0]) * 100
                lines.extend([
                    f"Price Range: ${self.current_range[0]:.4f} - ${self.current_range[1]:.4f}",
                    f"Range Size: {range_size:.2f}%"
                ])
            
            # Add current price and position information
            if self.candles and len(self.candles) > 0 and self.candles[0].ready:
                df = self.candles[0].candles_df
                if len(df) > 0:
                    current_price = Decimal(str(df['close'].iloc[-1]))
                    lines.extend([
                        f"Current Price: ${current_price:.4f}",
                        f"Position State: {self.position_state.value}",
                        f"Current Position: {self.current_position}"
                    ])
                    
                    if self.entry_prices:
                        avg_entry = sum(self.entry_prices) / len(self.entry_prices)
                        lines.append(f"Average Entry: ${avg_entry:.4f}")
                        
                        if current_price > Decimal("0") and avg_entry > Decimal("0"):
                            pnl_pct = ((current_price / avg_entry) - 1) * 100
                            lines.append(f"Unrealized P&L: {pnl_pct:.2f}%")
                    
                    if self.trailing_stop_price and self.trailing_stop_price > Decimal("0"):
                        lines.append(f"Trailing Stop: ${self.trailing_stop_price:.4f}")
            
            # Add BTC correlation info if available
            if self.btc_candles and self.btc_candles.ready and self.candles and self.candles[0].ready:
                btc_df = self.btc_candles.candles_df
                pair_df = self.candles[0].candles_df
                
                if len(btc_df) > 20 and len(pair_df) > 20:
                    btc_returns = btc_df['close'].pct_change().tail(20)
                    pair_returns = pair_df['close'].pct_change().tail(20)
                    if len(btc_returns) == len(pair_returns):
                        correlation = btc_returns.corr(pair_returns)
                        lines.append(f"BTC Correlation: {correlation:.2f}")
                        
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in market_data_extra_info: {e}")
            
        return lines
        
    def format_status(self) -> str:
        """Format status output for display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        status_output = []
        status_output.extend([
            "Adaptive Market Making Strategy",
            f"Exchange: {self.exchange}" + (" (Paper Trading)" if self.paper_trade else ""),
            f"Trading Pair: {self.trading_pair}",
            f"Risk Per Trade: {self.risk_factor * 100:.1f}%",
            f"Max Position Size: {self.max_position_size}",
            f"Stop Loss: {self.stop_loss * 100:.1f}%",
            f"Take Profit: {self.take_profit * 100:.1f}%",
            "-" * 50
        ])
        
        try:
            parent_status = super().format_status()
            status_output.append(parent_status)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error getting parent status: {e}")
            
        return "\n".join(status_output)
        
    async def on_stop(self):
        """
        Clean up resources when strategy is stopped
        """
        try:
            # Stop all candles to prevent network iterator issues
            for candle in self.candles:
                if hasattr(candle, 'stop'):
                    candle.stop()
            
            if self.btc_candles and hasattr(self.btc_candles, 'stop'):
                self.btc_candles.stop()
                
            # Call parent on_stop but handle possible errors
            try:
                await super().on_stop()
            except Exception as e:
                logging.getLogger(__name__).error(f"Error in parent on_stop: {e}")
                # Continue with cleanup even if parent on_stop fails
                
            logging.getLogger(__name__).info("Strategy stopped successfully")
        except Exception as e:
            logging.getLogger(__name__).error(f"Error stopping strategy: {e}")
            # Swallow the exception to prevent issues with the stop process