from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class PositionState(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    HOLDING = "holding"
    EXITING = "exiting"

@dataclass
class PositionConfig:
    max_position_size: Decimal
    entry_price: Decimal
    current_price: Decimal
    stop_loss_pct: Decimal
    take_profit_pct: Decimal
    position_value: Decimal = Decimal("0")
    
class PositionManager:
    """
    Manages trading positions based on market conditions and risk parameters.
    """
    
    def __init__(self,
                 max_position_size: Decimal,
                 stop_loss_pct: Decimal,
                 take_profit_pct: Decimal,
                 max_drawdown_pct: Decimal = Decimal("5.0"),
                 position_step_size: Decimal = Decimal("0.1")):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.position_step_size = position_step_size
        
        self.current_position = Decimal("0")
        self.position_state = PositionState.WAITING
        self.entry_prices: List[Decimal] = []
        self.last_position_update = None
        
    def calculate_position_size(self,
                              current_price: Decimal,
                              market_metrics: dict) -> Decimal:
        """
        Calculates appropriate position size based on market conditions.
        """
        # Base size on market stability
        confidence = Decimal(str(market_metrics.get("confidence", 0.5)))
        volatility = Decimal(str(market_metrics.get("volatility", 0.02)))
        
        # Reduce size during higher volatility
        volatility_factor = Decimal("1") / (Decimal("1") + volatility * Decimal("10"))
        
        # Scale with confidence
        confidence_factor = confidence * Decimal("0.8") + Decimal("0.2")
        
        # Calculate base position size
        base_size = self.max_position_size * confidence_factor * volatility_factor
        
        # Round to position step size
        steps = int(base_size / self.position_step_size)
        return self.position_step_size * Decimal(str(steps))
        
    def should_add_to_position(self,
                             current_price: Decimal,
                             market_metrics: dict) -> Tuple[bool, Decimal]:
        """
        Determines if we should add to the current position.
        """
        if self.position_state != PositionState.ACTIVE:
            return False, Decimal("0")
            
        if self.current_position >= self.max_position_size:
            return False, Decimal("0")
            
        # Check if market conditions are favorable
        if market_metrics.get("is_stable", False) and \
           market_metrics.get("confidence", 0) > 0.7:
            
            # Calculate additional position size
            available_size = self.max_position_size - self.current_position
            additional_size = self.calculate_position_size(current_price, market_metrics)
            additional_size = min(additional_size, available_size)
            
            return True, additional_size
            
        return False, Decimal("0")
        
    def should_reduce_position(self,
                             current_price: Decimal,
                             market_metrics: dict) -> Tuple[bool, Decimal]:
        """
        Determines if we should reduce the current position.
        """
        if self.current_position == 0:
            return False, Decimal("0")
            
        # Check for stop loss
        avg_entry = sum(self.entry_prices) / len(self.entry_prices) \
                   if self.entry_prices else Decimal("0")
        
        if avg_entry > 0:
            drawdown = (avg_entry - current_price) / avg_entry
            if drawdown >= self.stop_loss_pct:
                return True, self.current_position  # Full exit
                
        # Check market conditions
        if market_metrics.get("volatility", 0) > float(self.max_drawdown_pct) / 100:
            return True, self.current_position * Decimal("0.5")  # Partial exit
            
        return False, Decimal("0")
        
    def update_position(self,
                       size_delta: Decimal,
                       price: Decimal,
                       market_metrics: dict) -> None:
        """
        Updates the current position after a trade.
        """
        if size_delta > 0:  # Adding to position
            self.entry_prices.append(price)
            self.current_position += size_delta
            self.position_state = PositionState.ACTIVE
        else:  # Reducing position
            size_delta = abs(size_delta)
            self.current_position -= size_delta
            
            # Remove corresponding entry prices
            if self.current_position == 0:
                self.entry_prices = []
                self.position_state = PositionState.WAITING
            else:
                # Remove proportional amount of entry prices
                remove_count = int(len(self.entry_prices) * \
                                 (size_delta / (self.current_position + size_delta)))
                self.entry_prices = self.entry_prices[remove_count:]
                
        self.last_position_update = pd.Timestamp.now()
        
    def get_position_metrics(self) -> Dict:
        """
        Returns current position metrics for monitoring.
        """
        avg_entry = sum(self.entry_prices) / len(self.entry_prices) \
                   if self.entry_prices else Decimal("0")
                   
        return {
            "current_position": self.current_position,
            "position_state": self.position_state.value,
            "average_entry": avg_entry,
            "entry_prices": self.entry_prices.copy(),
            "last_update": self.last_position_update
        }