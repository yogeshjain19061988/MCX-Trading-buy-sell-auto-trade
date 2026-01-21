import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
import logging
import traceback

# Technical Analysis
import ta

# Zerodha API
from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import KiteException

# PyQt6 GUI
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# For alerts
import winsound
import platform

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

@dataclass
class TradingConfig:
    """Advanced trading configuration with auto entry/exit"""
    # Zerodha API
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    
    # MCX Configuration
    commodity: str = "GOLD"
    exchange: str = "MCX"
    
    # Contract Selection
    near_contract_symbol: str = ""
    mid_contract_symbol: str = ""
    near_contract_token: str = ""
    mid_contract_token: str = ""
    
    # Real-time Settings
    websocket_enabled: bool = True
    data_refresh_interval: int = 1000  # ms
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5  # seconds
    
    # Strategy Parameters
    lookback_period: int = 100
    std_dev_multiplier: float = 1.5
    atr_period: int = 10
    supertrend_factor: float = 3.0
    ma_fast: int = 5
    ma_slow: int = 20
    
    # Trading Parameters
    quantity: int = 1
    product_type: str = "NRML"
    order_type: str = "MARKET"
    
    # Auto Entry Settings
    auto_entry_enabled: bool = False
    entry_zscore_threshold: float = 2.0
    
    # Auto Exit Settings
    auto_exit_enabled: bool = False
    profit_target_percent: float = 1.0  # 1% profit target
    stop_loss_percent: float = 0.5      # 0.5% stop loss
    trailing_stop_enabled: bool = False
    trailing_stop_distance: float = 0.3  # 0.3% trailing stop
    
    # Alert Settings
    enable_audio_alerts: bool = True
    enable_visual_alerts: bool = True
    alert_zscore_threshold: float = 1.5
    
    # Auto Login Settings
    auto_login_enabled: bool = True
    
    def __post_init__(self):
        self.config_file = "config_advanced.json"
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(asdict(self), f, indent=4, default=str)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def load(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                
                for key, value in config_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                logger.info(f"Configuration loaded from {self.config_file}")
                return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
        return False

# ============================================================================
# ENHANCED MARKET DATA MANAGER WITH FIXED WEBSOCKET
# ============================================================================

class AdvancedMarketData(QObject):
    """Advanced market data with bid/ask support and fixed WebSocket"""
    
    tick_received = pyqtSignal(dict)
    connection_status = pyqtSignal(str, bool)
    
    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        self.kite = None
        self.kws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.should_reconnect = False
        self.subscribed_tokens = []
        
        # Market data storage
        self.market_data = {
            'near': {},
            'mid': {}
        }
        
        # Depth data (bid/ask)
        self.market_depth = {
            'near': {'bids': [], 'asks': []},
            'mid': {'bids': [], 'asks': []}
        }
        
        # Price history
        self.price_history = {
            'near': [],
            'mid': []
        }
        self.max_history = 500
    
    def initialize(self, kite: KiteConnect):
        """Initialize with KiteConnect"""
        self.kite = kite
        logger.info("AdvancedMarketData initialized")
    
    def start_websocket(self):
        """Start WebSocket connection with proper error handling"""
        try:
            if not self.kite:
                logger.error("KiteConnect not initialized")
                return False
            
            # Get instrument tokens
            if not self.config.near_contract_token or not self.config.mid_contract_token:
                logger.error("Contract tokens not set")
                return False
            
            try:
                near_token = int(self.config.near_contract_token)
                mid_token = int(self.config.mid_contract_token)
                self.subscribed_tokens = [near_token, mid_token]
            except ValueError as e:
                logger.error(f"Invalid token format: {e}")
                return False
            
            # Create KiteTicker with error handling
            try:
                self.kws = KiteTicker(
                    api_key=self.config.api_key,
                    access_token=self.config.access_token
                )
            except Exception as e:
                logger.error(f"Failed to create KiteTicker: {e}")
                return False
            
            # Set callbacks
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            # Start WebSocket in thread
            self.should_reconnect = True
            self.ws_thread = threading.Thread(target=self._ws_connect, daemon=True)
            self.ws_thread.start()
            
            logger.info(f"WebSocket thread started for tokens: {self.subscribed_tokens}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _ws_connect(self):
        """Internal WebSocket connection method"""
        try:
            logger.info("Attempting WebSocket connection...")
            self.kws.connect(threaded=True)
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._on_error(self.kws, e)
    
    def _on_connect(self, ws, response):
        """WebSocket connected callback"""
        try:
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("WebSocket connected successfully")
            
            # Subscribe to tokens and set mode
            if self.subscribed_tokens:
                self.kws.subscribe(self.subscribed_tokens)
                self.kws.set_mode(self.kws.MODE_FULL, self.subscribed_tokens)
                logger.info(f"Subscribed to tokens: {self.subscribed_tokens}")
            
            self.connection_status.emit("WebSocket Connected", True)
            
        except Exception as e:
            logger.error(f"Error in on_connect: {e}")
            self._on_error(ws, e)
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks"""
        try:
            for tick in ticks:
                token = str(tick['instrument_token'])
                timestamp = datetime.now()
                
                # Determine which contract this is
                contract_type = None
                if token == self.config.near_contract_token:
                    contract_type = 'near'
                elif token == self.config.mid_contract_token:
                    contract_type = 'mid'
                
                if contract_type:
                    # Update market data
                    self.market_data[contract_type] = {
                        'timestamp': timestamp,
                        'last_price': tick.get('last_price', 0),
                        'volume': tick.get('volume_traded', 0),
                        'oi': tick.get('oi', 0)
                    }
                    
                    # Update market depth (bid/ask)
                    if 'depth' in tick and tick['depth']:
                        bids = tick['depth'].get('buy', [])[:5]
                        asks = tick['depth'].get('sell', [])[:5]
                        self.market_depth[contract_type] = {
                            'bids': bids,
                            'asks': asks
                        }
                    
                    # Update price history
                    self.price_history[contract_type].append({
                        'timestamp': timestamp,
                        'price': tick.get('last_price', 0)
                    })
                    
                    # Trim history
                    if len(self.price_history[contract_type]) > self.max_history:
                        self.price_history[contract_type] = self.price_history[contract_type][-self.max_history:]
                    
                    # Emit tick data
                    tick_data = {
                        'contract': contract_type,
                        'timestamp': timestamp,
                        'last_price': tick.get('last_price', 0),
                        'best_bid': self.get_best_bid(contract_type),
                        'best_ask': self.get_best_ask(contract_type),
                        'volume': tick.get('volume_traded', 0),
                        'depth': self.market_depth[contract_type]
                    }
                    self.tick_received.emit(tick_data)
                    
        except Exception as e:
            logger.error(f"Error processing ticks: {e}")
    
    def _on_close(self, ws, code, reason):
        """WebSocket closed callback"""
        self.is_connected = False
        logger.warning(f"WebSocket closed: Code={code}, Reason={reason}")
        self.connection_status.emit(f"WebSocket Closed: {reason}", False)
        
        # Attempt reconnect if we should
        if self.should_reconnect:
            self._attempt_reconnect()
    
    def _on_error(self, ws, error):
        """WebSocket error callback"""
        logger.error(f"WebSocket error: {error}")
        self.connection_status.emit(f"WebSocket Error: {error}", False)
        self.is_connected = False
        
        # Attempt reconnect
        if self.should_reconnect:
            self._attempt_reconnect()
    
    def _attempt_reconnect(self):
        """Attempt to reconnect WebSocket"""
        if (self.reconnect_attempts < self.config.max_reconnect_attempts and 
            self.should_reconnect):
            
            self.reconnect_attempts += 1
            delay = self.config.reconnect_delay * self.reconnect_attempts
            
            logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})")
            time.sleep(delay)
            
            try:
                self.stop()
                self.start_websocket()
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
    
    def get_best_bid(self, contract_type: str) -> float:
        """Get best bid price"""
        if (contract_type in self.market_depth and 
            self.market_depth[contract_type]['bids']):
            return self.market_depth[contract_type]['bids'][0]['price']
        return 0.0
    
    def get_best_ask(self, contract_type: str) -> float:
        """Get best ask price"""
        if (contract_type in self.market_depth and 
            self.market_depth[contract_type]['asks']):
            return self.market_depth[contract_type]['asks'][0]['price']
        return 0.0
    
    def get_market_summary(self, contract_type: str) -> Dict:
        """Get market summary for contract"""
        if contract_type not in self.market_data or not self.market_data[contract_type]:
            return {}
        
        return {
            'last_price': self.market_data[contract_type].get('last_price', 0),
            'best_bid': self.get_best_bid(contract_type),
            'best_ask': self.get_best_ask(contract_type),
            'bid_ask_spread': self.get_best_ask(contract_type) - self.get_best_bid(contract_type),
            'volume': self.market_data[contract_type].get('volume', 0),
            'timestamp': self.market_data[contract_type].get('timestamp')
        }
    
    def stop(self):
        """Stop WebSocket connection"""
        try:
            self.should_reconnect = False
            
            if self.kws:
                # Unsubscribe before closing
                if self.subscribed_tokens:
                    self.kws.unsubscribe(self.subscribed_tokens)
                
                # Close WebSocket
                self.kws.close()
                self.kws = None
            
            self.is_connected = False
            logger.info("Market data stopped")
            
        except Exception as e:
            logger.error(f"Error stopping market data: {e}")

# ============================================================================
# FIXED POSITION MANAGER (Trailing Stop Fix)
# ============================================================================

class Position:
    """Trading position with P&L tracking"""
    
    def __init__(self, config):
        self.config = config
        self.entry_time = None
        self.exit_time = None
        self.position_type = None  # "BUY_SPREAD" or "SELL_SPREAD"
        self.entry_spread = 0.0
        self.current_spread = 0.0
        self.near_entry_price = 0.0
        self.mid_entry_price = 0.0
        self.near_current_price = 0.0
        self.mid_current_price = 0.0
        self.quantity = 0
        self.profit_target = 0.0
        self.stop_loss = 0.0
        self.trailing_stop = 0.0
        self.highest_profit = 0.0  # For trailing stop
        self.is_open = False
        self.order_ids = []
        
    def calculate_pnl(self):
        """Calculate current P&L"""
        if not self.is_open:
            return 0.0
        
        if self.position_type == "BUY_SPREAD":
            # Bought near, sold mid - profit when spread increases
            near_pnl = (self.near_current_price - self.near_entry_price) * self.quantity
            mid_pnl = (self.mid_entry_price - self.mid_current_price) * self.quantity
            return near_pnl + mid_pnl
        else:  # SELL_SPREAD
            # Sold near, bought mid - profit when spread decreases
            near_pnl = (self.near_entry_price - self.near_current_price) * self.quantity
            mid_pnl = (self.mid_current_price - self.mid_entry_price) * self.quantity
            return near_pnl + mid_pnl
    
    def calculate_pnl_percent(self):
        """Calculate P&L as percentage"""
        if not self.is_open or (self.near_entry_price + self.mid_entry_price) == 0:
            return 0.0
        
        total_investment = (self.near_entry_price + self.mid_entry_price) * self.quantity
        pnl = self.calculate_pnl()
        return (pnl / total_investment) * 100 if total_investment > 0 else 0.0
    
    def update_trailing_stop(self):
        """Update trailing stop based on highest profit"""
        if not self.config.trailing_stop_enabled or not self.is_open:
            return
        
        current_pnl_percent = self.calculate_pnl_percent()
        self.highest_profit = max(self.highest_profit, current_pnl_percent)
        
        # Update trailing stop if we have new high
        if current_pnl_percent >= self.highest_profit:
            self.trailing_stop = current_pnl_percent - self.config.trailing_stop_distance

class PositionManager:
    """Manage trading positions with auto entry/exit"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.current_position = None
        self.position_history = []
        self.max_positions = 10
        
    def open_position(self, position_type: str, entry_spread: float, 
                     near_price: float, mid_price: float, quantity: int):
        """Open a new trading position"""
        if self.current_position and self.current_position.is_open:
            logger.warning("Cannot open new position: position already open")
            return False
        
        self.current_position = Position(self.config)
        self.current_position.entry_time = datetime.now()
        self.current_position.position_type = position_type
        self.current_position.entry_spread = entry_spread
        self.current_position.current_spread = entry_spread
        self.current_position.near_entry_price = near_price
        self.current_position.mid_entry_price = mid_price
        self.current_position.near_current_price = near_price
        self.current_position.mid_current_price = mid_price
        self.current_position.quantity = quantity
        self.current_position.is_open = True
        
        # Calculate profit target and stop loss
        self._set_exit_levels(position_type, entry_spread, near_price, mid_price)
        
        logger.info(f"Opened {position_type} position at spread: {entry_spread:.2f}")
        return True
    
    def _set_exit_levels(self, position_type: str, entry_spread: float,
                        near_price: float, mid_price: float):
        """Set profit target and stop loss levels"""
        if position_type == "BUY_SPREAD":
            # For BUY_SPREAD, profit when spread increases
            self.current_position.profit_target = entry_spread * (1 + self.config.profit_target_percent / 100)
            self.current_position.stop_loss = entry_spread * (1 - self.config.stop_loss_percent / 100)
        else:  # SELL_SPREAD
            # For SELL_SPREAD, profit when spread decreases
            self.current_position.profit_target = entry_spread * (1 - self.config.profit_target_percent / 100)
            self.current_position.stop_loss = entry_spread * (1 + self.config.stop_loss_percent / 100)
        
        # Set initial trailing stop
        if self.config.trailing_stop_enabled:
            self.current_position.trailing_stop = -self.config.trailing_stop_distance
            self.current_position.highest_profit = 0.0
    
    def update_position(self, current_spread: float, near_price: float, mid_price: float):
        """Update current position with latest prices"""
        if not self.current_position or not self.current_position.is_open:
            return None
        
        self.current_position.current_spread = current_spread
        self.current_position.near_current_price = near_price
        self.current_position.mid_current_price = mid_price
        
        # Update trailing stop
        if self.config.trailing_stop_enabled:
            self.current_position.update_trailing_stop()
        
        # Check for exit conditions
        exit_reason = self._check_exit_conditions()
        
        return exit_reason
    
    def _check_exit_conditions(self):
        """Check if position should be exited"""
        if not self.current_position:
            return None
        
        current_pnl_percent = self.current_position.calculate_pnl_percent()
        current_spread = self.current_position.current_spread
        
        # Check profit target
        if self.current_position.position_type == "BUY_SPREAD":
            if current_spread >= self.current_position.profit_target:
                return "PROFIT_TARGET"
        else:
            if current_spread <= self.current_position.profit_target:
                return "PROFIT_TARGET"
        
        # Check stop loss
        if self.current_position.position_type == "BUY_SPREAD":
            if current_spread <= self.current_position.stop_loss:
                return "STOP_LOSS"
        else:
            if current_spread >= self.current_position.stop_loss:
                return "STOP_LOSS"
        
        # Check trailing stop
        if (self.config.trailing_stop_enabled and 
            current_pnl_percent <= self.current_position.trailing_stop):
            return "TRAILING_STOP"
        
        return None
    
    def close_position(self, exit_reason: str = "MANUAL"):
        """Close current position"""
        if not self.current_position or not self.current_position.is_open:
            return False
        
        self.current_position.exit_time = datetime.now()
        self.current_position.is_open = False
        
        # Add to history
        self.position_history.append(self.current_position)
        
        # Keep only recent history
        if len(self.position_history) > self.max_positions:
            self.position_history = self.position_history[-self.max_positions:]
        
        pnl = self.current_position.calculate_pnl()
        logger.info(f"Closed position: {exit_reason}, P&L: {pnl:.2f}")
        
        return True
    
    def get_position_summary(self):
        """Get current position summary"""
        if not self.current_position or not self.current_position.is_open:
            return None
        
        return {
            'type': self.current_position.position_type,
            'entry_time': self.current_position.entry_time,
            'entry_spread': self.current_position.entry_spread,
            'current_spread': self.current_position.current_spread,
            'pnl': self.current_position.calculate_pnl(),
            'pnl_percent': self.current_position.calculate_pnl_percent(),
            'profit_target': self.current_position.profit_target,
            'stop_loss': self.current_position.stop_loss,
            'trailing_stop': self.current_position.trailing_stop if self.config.trailing_stop_enabled else None
        }

# ============================================================================
# UPDATED ADVANCED TRADING DASHBOARD
# ============================================================================

class AdvancedTradingDashboard(QMainWindow):
    """Advanced trading dashboard with all features"""
    
    # Signals
    market_data_updated = pyqtSignal(dict)
    position_updated = pyqtSignal(dict)
    trading_signal = pyqtSignal(str, dict)
    
    def __init__(self):
        super().__init__()
        
        # Configuration
        self.config = TradingConfig()
        self.config.load()
        
        # Components
        self.kite = None
        self.market_data = None
        self.position_manager = None
        
        # UI State
        self.is_connected = False
        self.is_monitoring = False
        
        # Data storage for signals
        self.spread_history = []
        self.max_spread_history = 100
        
        # Initialize
        self.init_ui()
        self.init_components()
        
        # Try auto-connect
        if self.config.api_key and self.config.access_token:
            QTimer.singleShot(1000, self.auto_login_with_token)  # Delay to let UI load
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("MCX Calendar Spread Pro - Advanced Trading")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ===== STATUS BAR =====
        status_layout = QHBoxLayout()
        
        self.connection_status = QLabel("🔴 Disconnected")
        self.connection_status.setStyleSheet("font-weight: bold; padding: 5px;")
        
        self.data_status = QLabel("Data: Stopped")
        self.position_status = QLabel("Position: None")
        
        status_layout.addWidget(self.connection_status)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.data_status)
        status_layout.addWidget(QLabel(" | "))
        status_layout.addWidget(self.position_status)
        status_layout.addStretch()
        
        main_layout.addLayout(status_layout)
        
        # ===== CONTRACT SELECTION PANEL =====
        contract_group = QGroupBox("Contract Selection")
        contract_layout = QGridLayout()
        
        contract_layout.addWidget(QLabel("Commodity:"), 0, 0)
        self.commodity_label = QLabel(self.config.commodity)
        contract_layout.addWidget(self.commodity_label, 0, 1)
        
        contract_layout.addWidget(QLabel("Near Contract:"), 0, 2)
        self.near_contract_label = QLabel(self.config.near_contract_symbol or "--")
        contract_layout.addWidget(self.near_contract_label, 0, 3)
        
        contract_layout.addWidget(QLabel("Mid Contract:"), 0, 4)
        self.mid_contract_label = QLabel(self.config.mid_contract_symbol or "--")
        contract_layout.addWidget(self.mid_contract_label, 0, 5)
        
        contract_layout.addWidget(QLabel("Near Token:"), 1, 0)
        self.near_token_label = QLabel(str(self.config.near_contract_token) or "--")
        contract_layout.addWidget(self.near_token_label, 1, 1)
        
        contract_layout.addWidget(QLabel("Mid Token:"), 1, 2)
        self.mid_token_label = QLabel(str(self.config.mid_contract_token) or "--")
        contract_layout.addWidget(self.mid_token_label, 1, 3)
        
        # Contract selection button
        self.select_contracts_btn = QPushButton("Select Contracts")
        self.select_contracts_btn.clicked.connect(self.select_contracts)
        contract_layout.addWidget(self.select_contracts_btn, 1, 4, 1, 2)
        
        contract_group.setLayout(contract_layout)
        main_layout.addWidget(contract_group)
        
        # ===== REAL-TIME MARKET DATA PANEL =====
        market_group = QGroupBox("Real-Time Market Data")
        market_layout = QGridLayout()
        
        # Near Contract Data
        market_layout.addWidget(QLabel("Near Contract:"), 0, 0)
        self.near_price_label = QLabel("0.00")
        self.near_price_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        market_layout.addWidget(self.near_price_label, 0, 1)
        
        market_layout.addWidget(QLabel("Bid:"), 0, 2)
        self.near_bid_label = QLabel("0.00")
        self.near_bid_label.setStyleSheet("color: green;")
        market_layout.addWidget(self.near_bid_label, 0, 3)
        
        market_layout.addWidget(QLabel("Ask:"), 0, 4)
        self.near_ask_label = QLabel("0.00")
        self.near_ask_label.setStyleSheet("color: red;")
        market_layout.addWidget(self.near_ask_label, 0, 5)
        
        # Mid Contract Data
        market_layout.addWidget(QLabel("Mid Contract:"), 1, 0)
        self.mid_price_label = QLabel("0.00")
        self.mid_price_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        market_layout.addWidget(self.mid_price_label, 1, 1)
        
        market_layout.addWidget(QLabel("Bid:"), 1, 2)
        self.mid_bid_label = QLabel("0.00")
        self.mid_bid_label.setStyleSheet("color: green;")
        market_layout.addWidget(self.mid_bid_label, 1, 3)
        
        market_layout.addWidget(QLabel("Ask:"), 1, 4)
        self.mid_ask_label = QLabel("0.00")
        self.mid_ask_label.setStyleSheet("color: red;")
        market_layout.addWidget(self.mid_ask_label, 1, 5)
        
        # Spread Data
        market_layout.addWidget(QLabel("Spread:"), 2, 0)
        self.spread_label = QLabel("0.00")
        self.spread_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        market_layout.addWidget(self.spread_label, 2, 1)
        
        market_layout.addWidget(QLabel("Bid-Ask Spread:"), 2, 2)
        self.bid_ask_spread_label = QLabel("0.00")
        market_layout.addWidget(self.bid_ask_spread_label, 2, 3)
        
        market_layout.addWidget(QLabel("Volume:"), 2, 4)
        self.volume_label = QLabel("0")
        market_layout.addWidget(self.volume_label, 2, 5)
        
        market_group.setLayout(market_layout)
        main_layout.addWidget(market_group)
        
        # ===== POSITION & P&L PANEL =====
        pnl_group = QGroupBox("Position & P&L")
        pnl_layout = QGridLayout()
        
        pnl_layout.addWidget(QLabel("Position Type:"), 0, 0)
        self.position_type_label = QLabel("NONE")
        self.position_type_label.setStyleSheet("font-weight: bold;")
        pnl_layout.addWidget(self.position_type_label, 0, 1)
        
        pnl_layout.addWidget(QLabel("Entry Spread:"), 0, 2)
        self.entry_spread_label = QLabel("0.00")
        pnl_layout.addWidget(self.entry_spread_label, 0, 3)
        
        pnl_layout.addWidget(QLabel("Current Spread:"), 0, 4)
        self.current_spread_label = QLabel("0.00")
        pnl_layout.addWidget(self.current_spread_label, 0, 5)
        
        pnl_layout.addWidget(QLabel("P&L:"), 1, 0)
        self.pnl_label = QLabel("0.00")
        self.pnl_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        pnl_layout.addWidget(self.pnl_label, 1, 1)
        
        pnl_layout.addWidget(QLabel("P&L %:"), 1, 2)
        self.pnl_percent_label = QLabel("0.00%")
        pnl_layout.addWidget(self.pnl_percent_label, 1, 3)
        
        pnl_layout.addWidget(QLabel("Profit Target:"), 1, 4)
        self.profit_target_label = QLabel("0.00")
        pnl_layout.addWidget(self.profit_target_label, 1, 5)
        
        pnl_layout.addWidget(QLabel("Stop Loss:"), 2, 0)
        self.stop_loss_label = QLabel("0.00")
        pnl_layout.addWidget(self.stop_loss_label, 2, 1)
        
        pnl_layout.addWidget(QLabel("Trailing Stop:"), 2, 2)
        self.trailing_stop_label = QLabel("0.00")
        pnl_layout.addWidget(self.trailing_stop_label, 2, 3)
        
        pnl_layout.addWidget(QLabel("Duration:"), 2, 4)
        self.position_duration_label = QLabel("0s")
        pnl_layout.addWidget(self.position_duration_label, 2, 5)
        
        pnl_group.setLayout(pnl_layout)
        main_layout.addWidget(pnl_group)
        
        # ===== CONTROL PANEL =====
        control_group = QGroupBox("Trading Controls")
        control_layout = QHBoxLayout()
        
        # SETTINGS BUTTON
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        self.settings_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        
        # Connection controls
        self.connect_btn = QPushButton("Connect to Zerodha")
        self.connect_btn.clicked.connect(self.connect_to_zerodha)
        self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_zerodha)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        
        # Trading controls
        self.start_btn = QPushButton("▶ Start Trading")
        self.start_btn.clicked.connect(self.toggle_trading)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        
        self.auto_entry_btn = QPushButton("Auto Entry: OFF")
        self.auto_entry_btn.clicked.connect(self.toggle_auto_entry)
        self.auto_entry_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 10px;")
        
        self.auto_exit_btn = QPushButton("Auto Exit: OFF")
        self.auto_exit_btn.clicked.connect(self.toggle_auto_exit)
        self.auto_exit_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 10px;")
        
        # Manual trade buttons
        self.buy_spread_btn = QPushButton("BUY SPREAD")
        self.buy_spread_btn.clicked.connect(lambda: self.place_trade("BUY_SPREAD"))
        self.buy_spread_btn.setEnabled(False)
        self.buy_spread_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        
        self.sell_spread_btn = QPushButton("SELL SPREAD")
        self.sell_spread_btn.clicked.connect(lambda: self.place_trade("SELL_SPREAD"))
        self.sell_spread_btn.setEnabled(False)
        self.sell_spread_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        
        self.close_position_btn = QPushButton("CLOSE POSITION")
        self.close_position_btn.clicked.connect(self.close_position)
        self.close_position_btn.setEnabled(False)
        self.close_position_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 10px;")
        
        # Add to layout
        control_layout.addWidget(self.settings_btn)
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.disconnect_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.auto_entry_btn)
        control_layout.addWidget(self.auto_exit_btn)
        control_layout.addWidget(self.buy_spread_btn)
        control_layout.addWidget(self.sell_spread_btn)
        control_layout.addWidget(self.close_position_btn)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # ===== SIGNAL & STATISTICS PANEL =====
        stats_group = QGroupBox("Trading Signals & Statistics")
        stats_layout = QGridLayout()
        
        stats_layout.addWidget(QLabel("Signal:"), 0, 0)
        self.signal_label = QLabel("NEUTRAL")
        self.signal_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            padding: 10px;
            background-color: lightgray;
            border: 2px solid gray;
            border-radius: 5px;
        """)
        stats_layout.addWidget(self.signal_label, 0, 1, 1, 3)
        
        stats_layout.addWidget(QLabel("Z-Score:"), 1, 0)
        self.zscore_label = QLabel("0.00")
        stats_layout.addWidget(self.zscore_label, 1, 1)
        
        stats_layout.addWidget(QLabel("Upper Band:"), 1, 2)
        self.upper_band_label = QLabel("0.00")
        stats_layout.addWidget(self.upper_band_label, 1, 3)
        
        stats_layout.addWidget(QLabel("Lower Band:"), 2, 0)
        self.lower_band_label = QLabel("0.00")
        stats_layout.addWidget(self.lower_band_label, 2, 1)
        
        stats_layout.addWidget(QLabel("Spread Mean:"), 2, 2)
        self.spread_mean_label = QLabel("0.00")
        stats_layout.addWidget(self.spread_mean_label, 2, 3)
        
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)
        
        # ===== LOG PANEL =====
        log_group = QGroupBox("Trading Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 9pt;")
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        main_layout.addStretch()
    
    def init_components(self):
        """Initialize components"""
        self.market_data = AdvancedMarketData(self.config)
        self.position_manager = PositionManager(self.config)
        
        # Connect signals
        self.market_data.tick_received.connect(self.handle_tick_data)
        self.market_data.connection_status.connect(self.handle_connection_status)
        
        # Initialize UI state
        self.update_auto_entry_button()
        self.update_auto_exit_button()
    
    def connect_to_zerodha(self):
        """Connect to Zerodha API"""
        try:
            if not self.config.api_key:
                self.show_error("API Key Required", "Please set your API key in Settings")
                return
            
            self.kite = KiteConnect(api_key=self.config.api_key)
            
            if self.config.access_token:
                # Use existing access token
                self.kite.set_access_token(self.config.access_token)
                
                # Test connection
                profile = self.kite.profile()
                self.is_connected = True
                
                self.log_message(f"Connected to Zerodha as: {profile['user_name']}")
                self.update_connection_status(f"Connected as {profile['user_name']}", True)
                
                # Initialize market data
                self.market_data.initialize(self.kite)
                
                # Enable controls
                self.connect_btn.setEnabled(False)
                self.disconnect_btn.setEnabled(True)
                self.start_btn.setEnabled(True)
                self.select_contracts_btn.setEnabled(True)
                
                # Load contracts if not already set
                if not self.config.near_contract_symbol or not self.config.mid_contract_symbol:
                    QTimer.singleShot(1000, self.select_contracts)
                else:
                    self.update_contract_display()
                
            else:
                # Need to login
                self.manual_login()
                
        except KiteException as e:
            if "Invalid access token" in str(e):
                self.log_message("Access token expired. Please login again.")
                self.config.access_token = ""
                self.config.save()
                self.manual_login()
            else:
                self.show_error("Connection Failed", f"Error: {str(e)}")
        except Exception as e:
            self.show_error("Connection Error", f"Unexpected error: {str(e)}")
    
    def manual_login(self):
        """Manual login flow"""
        try:
            login_url = self.kite.login_url()
            self.log_message("Opening login URL in browser...")
            webbrowser.open(login_url)
            
            # Get request token
            request_token, ok = QInputDialog.getText(
                self, "Login Required",
                "After logging in, copy the 'request_token' from URL and paste below:"
            )
            
            if ok and request_token:
                self.log_message("Generating session...")
                data = self.kite.generate_session(
                    request_token=request_token,
                    api_secret=self.config.api_secret
                )
                
                self.config.access_token = data['access_token']
                self.config.save()
                
                self.kite.set_access_token(self.config.access_token)
                self.is_connected = True
                
                profile = self.kite.profile()
                self.log_message(f"Login successful! User: {profile['user_name']}")
                self.update_connection_status(f"Connected as {profile['user_name']}", True)
                
                # Initialize market data
                self.market_data.initialize(self.kite)
                
                # Enable controls
                self.connect_btn.setEnabled(False)
                self.disconnect_btn.setEnabled(True)
                self.start_btn.setEnabled(True)
                self.select_contracts_btn.setEnabled(True)
                
        except Exception as e:
            self.show_error("Login Failed", f"Error: {str(e)}")
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = AdvancedSettingsDialog(self.config, self)
        if dialog.exec():
            # Reload configuration if settings were saved
            self.config.load()
            
            # Update UI elements that might have changed
            self.update_auto_entry_button()
            self.update_auto_exit_button()
            
            # Update position manager with new config
            if self.position_manager:
                self.position_manager.config = self.config
            
            # Update market data with new config
            if self.market_data:
                self.market_data.config = self.config
            
            self.log_message("Settings updated and saved")
            
            # Auto-login if enabled and we have API credentials but not connected
            if (self.config.auto_login_enabled and 
                self.config.api_key and 
                self.config.api_secret and 
                not self.is_connected):
                
                # Check if we have access token
                if self.config.access_token:
                    # Try to connect with existing token
                    self.auto_login_with_token()
                else:
                    # Ask user if they want to login now
                    reply = QMessageBox.question(
                        self, "Auto Login",
                        "Would you like to login to Zerodha now?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        self.connect_to_zerodha()
    
    def auto_login_with_token(self):
        """Try to auto-login with existing access token"""
        try:
            if not self.config.api_key or not self.config.access_token:
                return False
            
            self.log_message("Attempting auto-login...")
            
            # Create KiteConnect instance
            self.kite = KiteConnect(api_key=self.config.api_key)
            self.kite.set_access_token(self.config.access_token)
            
            # Test connection
            profile = self.kite.profile()
            self.is_connected = True
            
            self.log_message(f"Auto-login successful! User: {profile['user_name']}")
            self.update_connection_status(f"Connected as {profile['user_name']}", True)
            
            # Initialize market data
            self.market_data.initialize(self.kite)
            
            # Enable controls
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.select_contracts_btn.setEnabled(True)
            
            # Update contract display if contracts are set
            if self.config.near_contract_symbol and self.config.mid_contract_symbol:
                self.update_contract_display()
            
            return True
            
        except KiteException as e:
            if "Invalid access token" in str(e):
                self.log_message("Auto-login failed: Access token expired")
                self.config.access_token = ""
                self.config.save()
                
                # Ask user to login manually
                reply = QMessageBox.question(
                    self, "Login Required",
                    "Access token expired. Would you like to login now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.connect_to_zerodha()
            else:
                self.log_message(f"Auto-login failed: {e}")
            return False
        except Exception as e:
            self.log_message(f"Auto-login error: {e}")
            return False
    
    def select_contracts(self):
        """Select MCX contracts"""
        if not self.kite:
            self.show_error("Not Connected", "Please connect to Zerodha first")
            return
        
        dialog = ContractSelectionDialog(self.kite, self.config, self)
        if dialog.exec():
            selected = dialog.selected_contracts
            
            # Update configuration
            self.config.near_contract_symbol = selected['near']['symbol']
            self.config.near_contract_token = selected['near']['token']
            self.config.mid_contract_symbol = selected['mid']['symbol']
            self.config.mid_contract_token = selected['mid']['token']
            
            # Extract commodity
            self.config.commodity = dialog._extract_commodity(self.config.near_contract_symbol)
            
            # Save configuration
            self.config.save()
            
            # Update display
            self.update_contract_display()
            
            self.log_message(f"Selected contracts: Near={self.config.near_contract_symbol}, Mid={self.config.mid_contract_symbol}")
    
    def update_contract_display(self):
        """Update contract display"""
        self.commodity_label.setText(self.config.commodity)
        self.near_contract_label.setText(self.config.near_contract_symbol)
        self.mid_contract_label.setText(self.config.mid_contract_symbol)
        self.near_token_label.setText(str(self.config.near_contract_token))
        self.mid_token_label.setText(str(self.config.mid_contract_token))
    
    def toggle_trading(self):
        """Toggle trading mode"""
        if not self.is_monitoring:
            self.start_trading()
        else:
            self.stop_trading()
    
    def start_trading(self):
        """Start trading"""
        try:
            # Start market data
            if self.market_data.start_websocket():
                self.is_monitoring = True
                self.start_btn.setText("⏸ Stop Trading")
                self.start_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
                self.data_status.setText("Data: Streaming")
                
                # Enable trading buttons
                self.buy_spread_btn.setEnabled(True)
                self.sell_spread_btn.setEnabled(True)
                
                self.log_message("Trading started")
                
        except Exception as e:
            self.log_message(f"Error starting trading: {e}")
    
    def stop_trading(self):
        """Stop trading"""
        self.is_monitoring = False
        self.market_data.stop()
        
        self.start_btn.setText("▶ Start Trading")
        self.start_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        self.data_status.setText("Data: Stopped")
        
        # Disable trading buttons
        self.buy_spread_btn.setEnabled(False)
        self.sell_spread_btn.setEnabled(False)
        
        self.log_message("Trading stopped")
    
    def toggle_auto_entry(self):
        """Toggle auto entry"""
        self.config.auto_entry_enabled = not self.config.auto_entry_enabled
        self.config.save()
        self.update_auto_entry_button()
        
        status = "ENABLED" if self.config.auto_entry_enabled else "DISABLED"
        self.log_message(f"Auto entry {status}")
    
    def update_auto_entry_button(self):
        """Update auto entry button display"""
        if self.config.auto_entry_enabled:
            self.auto_entry_btn.setText("Auto Entry: ON")
            self.auto_entry_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        else:
            self.auto_entry_btn.setText("Auto Entry: OFF")
            self.auto_entry_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 10px;")
    
    def toggle_auto_exit(self):
        """Toggle auto exit"""
        self.config.auto_exit_enabled = not self.config.auto_exit_enabled
        self.config.save()
        self.update_auto_exit_button()
        
        status = "ENABLED" if self.config.auto_exit_enabled else "DISABLED"
        self.log_message(f"Auto exit {status}")
    
    def update_auto_exit_button(self):
        """Update auto exit button display"""
        if self.config.auto_exit_enabled:
            self.auto_exit_btn.setText("Auto Exit: ON")
            self.auto_exit_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        else:
            self.auto_exit_btn.setText("Auto Exit: OFF")
            self.auto_exit_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 10px;")
    
    def handle_tick_data(self, tick_data):
        """Handle incoming tick data"""
        try:
            contract_type = tick_data['contract']
            
            # Update market data display
            if contract_type == 'near':
                self.near_price_label.setText(f"{tick_data['last_price']:.2f}")
                self.near_bid_label.setText(f"{tick_data['best_bid']:.2f}")
                self.near_ask_label.setText(f"{tick_data['best_ask']:.2f}")
            else:
                self.mid_price_label.setText(f"{tick_data['last_price']:.2f}")
                self.mid_bid_label.setText(f"{tick_data['best_bid']:.2f}")
                self.mid_ask_label.setText(f"{tick_data['best_ask']:.2f}")
            
            # Calculate spread when we have both prices
            near_summary = self.market_data.get_market_summary('near')
            mid_summary = self.market_data.get_market_summary('mid')
            
            if near_summary and mid_summary and near_summary.get('last_price', 0) > 0 and mid_summary.get('last_price', 0) > 0:
                near_price = near_summary['last_price']
                mid_price = mid_summary['last_price']
                spread = near_price - mid_price
                
                # Update spread display
                self.spread_label.setText(f"{spread:.2f}")
                
                # Update spread history for signal calculation
                self.spread_history.append(spread)
                if len(self.spread_history) > self.max_spread_history:
                    self.spread_history = self.spread_history[-self.max_spread_history:]
                
                # Calculate bid-ask spread
                near_bid_ask = near_summary.get('bid_ask_spread', 0)
                mid_bid_ask = mid_summary.get('bid_ask_spread', 0)
                total_bid_ask = near_bid_ask + mid_bid_ask
                
                self.bid_ask_spread_label.setText(f"{total_bid_ask:.2f}")
                
                # Update volume
                total_volume = near_summary.get('volume', 0) + mid_summary.get('volume', 0)
                self.volume_label.setText(f"{total_volume:,}")
                
                # Update position if open
                if self.position_manager.current_position and self.position_manager.current_position.is_open:
                    exit_reason = self.position_manager.update_position(spread, near_price, mid_price)
                    
                    if exit_reason and self.config.auto_exit_enabled:
                        self.close_position(exit_reason)
                    else:
                        self.update_position_display()
                
                # Calculate trading signal
                signal, stats = self.calculate_signal(spread)
                
                # Update signal display
                self.update_signal_display(signal, stats)
                
                # Check for trading signals (auto entry)
                if (self.config.auto_entry_enabled and 
                    not (self.position_manager.current_position and 
                         self.position_manager.current_position.is_open) and
                    signal != "NEUTRAL"):
                    
                    self.log_message(f"Auto entry signal: {signal}")
                    
                    # Check if we should auto enter
                    if self.should_auto_enter(signal, spread, stats):
                        self.place_trade(signal, auto=True)
            
        except Exception as e:
            self.log_message(f"Error handling tick data: {e}")
    
    def calculate_signal(self, spread: float):
        """Calculate trading signal based on spread history"""
        if len(self.spread_history) < 20:  # Need minimum data
            return "NEUTRAL", {'zscore': 0, 'mean': 0, 'std': 0}
        
        try:
            # Calculate statistics
            spread_mean = np.mean(self.spread_history)
            spread_std = np.std(self.spread_history)
            
            if spread_std > 0:
                z_score = (spread - spread_mean) / spread_std
                
                # Update statistics display
                self.zscore_label.setText(f"{z_score:.2f}")
                self.spread_mean_label.setText(f"{spread_mean:.2f}")
                
                # Calculate bands
                upper_band = spread_mean + (self.config.std_dev_multiplier * spread_std)
                lower_band = spread_mean - (self.config.std_dev_multiplier * spread_std)
                
                self.upper_band_label.setText(f"{upper_band:.2f}")
                self.lower_band_label.setText(f"{lower_band:.2f}")
                
                # Determine signal
                if z_score < -self.config.entry_zscore_threshold:
                    return "BUY_SPREAD", {'zscore': z_score, 'mean': spread_mean, 'std': spread_std}
                elif z_score > self.config.entry_zscore_threshold:
                    return "SELL_SPREAD", {'zscore': z_score, 'mean': spread_mean, 'std': spread_std}
                else:
                    return "NEUTRAL", {'zscore': z_score, 'mean': spread_mean, 'std': spread_std}
            
        except Exception as e:
            self.log_message(f"Error calculating signal: {e}")
        
        return "NEUTRAL", {'zscore': 0, 'mean': 0, 'std': 0}
    
    def update_signal_display(self, signal: str, stats: dict):
        """Update signal display based on signal"""
        if signal == "BUY_SPREAD":
            self.signal_label.setText("BUY SPREAD")
            self.signal_label.setStyleSheet("""
                font-size: 20px; 
                font-weight: bold; 
                padding: 10px;
                background-color: lightgreen;
                border: 2px solid green;
                border-radius: 5px;
            """)
        elif signal == "SELL_SPREAD":
            self.signal_label.setText("SELL SPREAD")
            self.signal_label.setStyleSheet("""
                font-size: 20px; 
                font-weight: bold; 
                padding: 10px;
                background-color: lightcoral;
                border: 2px solid red;
                border-radius: 5px;
            """)
        else:
            self.signal_label.setText("NEUTRAL")
            self.signal_label.setStyleSheet("""
                font-size: 20px; 
                font-weight: bold; 
                padding: 10px;
                background-color: lightgray;
                border: 2px solid gray;
                border-radius: 5px;
            """)
    
    def should_auto_enter(self, signal: str, spread: float, stats: dict) -> bool:
        """Determine if we should auto enter a trade"""
        # Add confirmation logic here
        # For example, require minimum spread history, check volatility, etc.
        
        if len(self.spread_history) < 50:  # Require more history
            return False
        
        # Check if z-score is beyond threshold
        zscore = stats.get('zscore', 0)
        threshold = self.config.entry_zscore_threshold
        
        if signal == "BUY_SPREAD" and zscore < -threshold * 1.2:  # Slightly stricter
            return True
        elif signal == "SELL_SPREAD" and zscore > threshold * 1.2:
            return True
        
        return False
    
    def handle_connection_status(self, status: str, connected: bool):
        """Handle connection status updates"""
        if connected:
            self.connection_status.setText(f"🟢 {status}")
            self.connection_status.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        else:
            self.connection_status.setText(f"🔴 {status}")
            self.connection_status.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
    
    def place_trade(self, position_type: str, auto: bool = False):
        """Place a spread trade"""
        if not self.kite:
            self.show_error("Not Connected", "Please connect to Zerodha first")
            return
        
        try:
            # Get current prices
            near_summary = self.market_data.get_market_summary('near')
            mid_summary = self.market_data.get_market_summary('mid')
            
            if not near_summary or not mid_summary:
                self.log_message("Cannot place trade: Market data not available")
                return
            
            near_price = near_summary['last_price']
            mid_price = mid_summary['last_price']
            spread = near_price - mid_price
            
            # Prepare orders
            if position_type == "BUY_SPREAD":
                orders = [
                    {
                        "tradingsymbol": self.config.near_contract_symbol,
                        "exchange": self.config.exchange,
                        "transaction_type": "BUY",
                        "quantity": self.config.quantity,
                        "order_type": self.config.order_type,
                        "product": self.config.product_type,
                        "variety": "regular"
                    },
                    {
                        "tradingsymbol": self.config.mid_contract_symbol,
                        "exchange": self.config.exchange,
                        "transaction_type": "SELL",
                        "quantity": self.config.quantity,
                        "order_type": self.config.order_type,
                        "product": self.config.product_type,
                        "variety": "regular"
                    }
                ]
            else:  # SELL_SPREAD
                orders = [
                    {
                        "tradingsymbol": self.config.near_contract_symbol,
                        "exchange": self.config.exchange,
                        "transaction_type": "SELL",
                        "quantity": self.config.quantity,
                        "order_type": self.config.order_type,
                        "product": self.config.product_type,
                        "variety": "regular"
                    },
                    {
                        "tradingsymbol": self.config.mid_contract_symbol,
                        "exchange": self.config.exchange,
                        "transaction_type": "BUY",
                        "quantity": self.config.quantity,
                        "order_type": self.config.order_type,
                        "product": self.config.product_type,
                        "variety": "regular"
                    }
                ]
            
            # Confirm if not auto trade
            if not auto:
                reply = QMessageBox.question(
                    self, "Confirm Trade",
                    f"Place {position_type} trade?\n\n"
                    f"Near: {self.config.near_contract_symbol}\n"
                    f"Mid: {self.config.mid_contract_symbol}\n"
                    f"Quantity: {self.config.quantity}\n"
                    f"Spread: {spread:.2f}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply != QMessageBox.StandardButton.Yes:
                    return
            
            # Place orders
            order_ids = []
            for order in orders:
                try:
                    order_id = self.kite.place_order(**order)
                    order_ids.append(order_id)
                    self.log_message(f"Order placed: {order['tradingsymbol']} - ID: {order_id}")
                except Exception as e:
                    self.log_message(f"Order failed for {order['tradingsymbol']}: {e}")
            
            # If both orders successful, open position
            if len(order_ids) == 2:
                self.position_manager.open_position(
                    position_type, spread, near_price, mid_price, self.config.quantity
                )
                
                # Update UI
                self.update_position_display()
                self.close_position_btn.setEnabled(True)
                
                log_type = "Auto" if auto else "Manual"
                self.log_message(f"{log_type} {position_type} trade executed successfully")
                
                # Show success message
                if not auto:
                    QMessageBox.information(self, "Success", "Spread trade executed!")
            
        except Exception as e:
            error_msg = f"Trade execution failed: {e}"
            self.log_message(error_msg)
            if not auto:
                self.show_error("Trade Failed", error_msg)
    
    def update_position_display(self):
        """Update position display"""
        summary = self.position_manager.get_position_summary()
        
        if not summary:
            self.position_type_label.setText("NONE")
            self.entry_spread_label.setText("0.00")
            self.current_spread_label.setText("0.00")
            self.pnl_label.setText("0.00")
            self.pnl_percent_label.setText("0.00%")
            self.profit_target_label.setText("0.00")
            self.stop_loss_label.setText("0.00")
            self.trailing_stop_label.setText("0.00")
            self.position_duration_label.setText("0s")
            self.position_status.setText("Position: None")
            
            # Update signal label
            self.signal_label.setText("NEUTRAL")
            self.signal_label.setStyleSheet("""
                font-size: 20px; 
                font-weight: bold; 
                padding: 10px;
                background-color: lightgray;
                border: 2px solid gray;
                border-radius: 5px;
            """)
            
            return
        
        # Update position info
        self.position_type_label.setText(summary['type'])
        self.entry_spread_label.setText(f"{summary['entry_spread']:.2f}")
        self.current_spread_label.setText(f"{summary['current_spread']:.2f}")
        
        # Update P&L with color coding
        pnl = summary['pnl']
        pnl_percent = summary['pnl_percent']
        
        self.pnl_label.setText(f"{pnl:+.2f}")
        self.pnl_percent_label.setText(f"{pnl_percent:+.2f}%")
        
        if pnl > 0:
            self.pnl_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
            self.pnl_percent_label.setStyleSheet("color: green;")
        elif pnl < 0:
            self.pnl_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
            self.pnl_percent_label.setStyleSheet("color: red;")
        else:
            self.pnl_label.setStyleSheet("color: black; font-size: 16px; font-weight: bold;")
            self.pnl_percent_label.setStyleSheet("color: black;")
        
        # Update exit levels
        self.profit_target_label.setText(f"{summary['profit_target']:.2f}")
        self.stop_loss_label.setText(f"{summary['stop_loss']:.2f}")
        
        if summary['trailing_stop'] is not None:
            self.trailing_stop_label.setText(f"{summary['trailing_stop']:.2f}%")
        else:
            self.trailing_stop_label.setText("N/A")
        
        # Update duration
        if summary['entry_time']:
            duration = datetime.now() - summary['entry_time']
            self.position_duration_label.setText(f"{duration.seconds}s")
        
        # Update position status
        self.position_status.setText(f"Position: {summary['type']} | P&L: {pnl:+.2f}")
        
        # Update signal label based on position
        if summary['type'] == "BUY_SPREAD":
            self.signal_label.setText("LONG")
            self.signal_label.setStyleSheet("""
                font-size: 20px; 
                font-weight: bold; 
                padding: 10px;
                background-color: lightgreen;
                border: 2px solid green;
                border-radius: 5px;
            """)
        else:
            self.signal_label.setText("SHORT")
            self.signal_label.setStyleSheet("""
                font-size: 20px; 
                font-weight: bold; 
                padding: 10px;
                background-color: lightcoral;
                border: 2px solid red;
                border-radius: 5px;
            """)
    
    def close_position(self, exit_reason: str = "MANUAL"):
        """Close current position"""
        try:
            if not self.position_manager.current_position:
                self.log_message("No position to close")
                return
            
            # Get current prices
            near_summary = self.market_data.get_market_summary('near')
            mid_summary = self.market_data.get_market_summary('mid')
            
            if not near_summary or not mid_summary:
                self.log_message("Cannot close position: Market data not available")
                return
            
            # Close position
            if self.position_manager.close_position(exit_reason):
                self.update_position_display()
                self.close_position_btn.setEnabled(False)
                
                self.log_message(f"Position closed: {exit_reason}")
                
                # Show P&L summary
                pos_summary = self.position_manager.get_position_summary()
                if pos_summary:
                    pnl_msg = f"P&L: {pos_summary['pnl']:+.2f} ({pos_summary['pnl_percent']:+.2f}%)"
                    self.log_message(pnl_msg)
            
        except Exception as e:
            self.log_message(f"Error closing position: {e}")
    
    def disconnect_zerodha(self):
        """Disconnect from Zerodha"""
        try:
            self.stop_trading()
            
            if self.market_data:
                self.market_data.stop()
            
            self.kite = None
            self.is_connected = False
            
            self.update_connection_status("Disconnected", False)
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.select_contracts_btn.setEnabled(False)
            self.buy_spread_btn.setEnabled(False)
            self.sell_spread_btn.setEnabled(False)
            self.close_position_btn.setEnabled(False)
            
            self.log_message("Disconnected from Zerodha")
            
        except Exception as e:
            self.log_message(f"Error disconnecting: {e}")
    
    def update_connection_status(self, status: str, connected: bool):
        """Update connection status display"""
        if connected:
            self.connection_status.setText(f"🟢 {status}")
            self.connection_status.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        else:
            self.connection_status.setText(f"🔴 {status}")
            self.connection_status.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
    
    def log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.log_text.append(log_entry)
        
        # Keep log manageable
        if self.log_text.toPlainText().count('\n') > 100:
            lines = self.log_text.toPlainText().split('\n')
            self.log_text.setPlainText('\n'.join(lines[-50:]))
        
        # Scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
        # Also print to console for debugging
        print(log_entry)
    
    def show_error(self, title: str, message: str):
        """Show error message"""
        QMessageBox.critical(self, title, message)
        self.log_message(f"ERROR: {title} - {message}")
    
    def closeEvent(self, event):
        """Handle application close"""
        self.stop_trading()
        self.disconnect_zerodha()
        event.accept()

# ============================================================================
# SETTINGS DIALOG (Keep as is)
# ============================================================================

class AdvancedSettingsDialog(QDialog):
    """Advanced settings dialog with auto-login option"""
    
    def __init__(self, config: TradingConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Advanced Trading Settings")
        self.setGeometry(300, 300, 600, 750)
        
        layout = QVBoxLayout()
        
        # Create tab widget
        tabs = QTabWidget()
        
        # API Settings
        api_tab = QWidget()
        api_layout = QFormLayout()
        
        self.api_key_input = QLineEdit(self.config.api_key)
        self.api_key_input.setPlaceholderText("Enter API Key")
        
        self.api_secret_input = QLineEdit(self.config.api_secret)
        self.api_secret_input.setPlaceholderText("Enter API Secret")
        self.api_secret_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        # Auto-login checkbox
        self.auto_login_check = QCheckBox("Enable Auto Login after settings")
        self.auto_login_check.setChecked(self.config.auto_login_enabled)
        
        api_layout.addRow("API Key:", self.api_key_input)
        api_layout.addRow("API Secret:", self.api_secret_input)
        api_layout.addRow("", self.auto_login_check)
        
        api_tab.setLayout(api_layout)
        tabs.addTab(api_tab, "API Settings")
        
        # Trading Settings
        trading_tab = QWidget()
        trading_layout = QFormLayout()
        
        self.quantity_spin = QSpinBox()
        self.quantity_spin.setRange(1, 1000)
        self.quantity_spin.setValue(self.config.quantity)
        
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["MARKET", "LIMIT"])
        self.order_type_combo.setCurrentText(self.config.order_type)
        
        trading_layout.addRow("Quantity:", self.quantity_spin)
        trading_layout.addRow("Order Type:", self.order_type_combo)
        
        trading_tab.setLayout(trading_layout)
        tabs.addTab(trading_tab, "Trading Settings")
        
        # Auto Entry Settings
        auto_entry_tab = QWidget()
        auto_entry_layout = QFormLayout()
        
        self.auto_entry_check = QCheckBox("Enable Auto Entry")
        self.auto_entry_check.setChecked(self.config.auto_entry_enabled)
        
        self.zscore_threshold_spin = QDoubleSpinBox()
        self.zscore_threshold_spin.setRange(1.0, 5.0)
        self.zscore_threshold_spin.setSingleStep(0.1)
        self.zscore_threshold_spin.setValue(self.config.entry_zscore_threshold)
        
        auto_entry_layout.addRow("", self.auto_entry_check)
        auto_entry_layout.addRow("Z-Score Threshold:", self.zscore_threshold_spin)
        
        auto_entry_tab.setLayout(auto_entry_layout)
        tabs.addTab(auto_entry_tab, "Auto Entry")
        
        # Auto Exit Settings
        auto_exit_tab = QWidget()
        auto_exit_layout = QFormLayout()
        
        self.auto_exit_check = QCheckBox("Enable Auto Exit")
        self.auto_exit_check.setChecked(self.config.auto_exit_enabled)
        
        self.profit_target_spin = QDoubleSpinBox()
        self.profit_target_spin.setRange(0.1, 10.0)
        self.profit_target_spin.setSingleStep(0.1)
        self.profit_target_spin.setValue(self.config.profit_target_percent)
        
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.1, 10.0)
        self.stop_loss_spin.setSingleStep(0.1)
        self.stop_loss_spin.setValue(self.config.stop_loss_percent)
        
        self.trailing_stop_check = QCheckBox("Enable Trailing Stop")
        self.trailing_stop_check.setChecked(self.config.trailing_stop_enabled)
        
        self.trailing_distance_spin = QDoubleSpinBox()
        self.trailing_distance_spin.setRange(0.1, 5.0)
        self.trailing_distance_spin.setSingleStep(0.1)
        self.trailing_distance_spin.setValue(self.config.trailing_stop_distance)
        
        auto_exit_layout.addRow("", self.auto_exit_check)
        auto_exit_layout.addRow("Profit Target (%):", self.profit_target_spin)
        auto_exit_layout.addRow("Stop Loss (%):", self.stop_loss_spin)
        auto_exit_layout.addRow("", self.trailing_stop_check)
        auto_exit_layout.addRow("Trailing Distance (%):", self.trailing_distance_spin)
        
        auto_exit_tab.setLayout(auto_exit_layout)
        tabs.addTab(auto_exit_tab, "Auto Exit")
        
        layout.addWidget(tabs)
        
        # Info label
        info_label = QLabel("Note: Auto-login will attempt to connect automatically after saving settings.")
        info_label.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(info_label)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def save_settings(self):
        """Save settings"""
        # API Settings
        self.config.api_key = self.api_key_input.text()
        self.config.api_secret = self.api_secret_input.text()
        self.config.auto_login_enabled = self.auto_login_check.isChecked()
        
        # Trading Settings
        self.config.quantity = self.quantity_spin.value()
        self.config.order_type = self.order_type_combo.currentText()
        
        # Auto Entry
        self.config.auto_entry_enabled = self.auto_entry_check.isChecked()
        self.config.entry_zscore_threshold = self.zscore_threshold_spin.value()
        
        # Auto Exit
        self.config.auto_exit_enabled = self.auto_exit_check.isChecked()
        self.config.profit_target_percent = self.profit_target_spin.value()
        self.config.stop_loss_percent = self.stop_loss_spin.value()
        self.config.trailing_stop_enabled = self.trailing_stop_check.isChecked()
        self.config.trailing_stop_distance = self.trailing_distance_spin.value()
        
        # Save to file
        self.config.save()
        
        self.accept()

# ============================================================================
# CONTRACT SELECTION DIALOG (Keep as is)
# ============================================================================

class ContractSelectionDialog(QDialog):
    """Dialog for selecting MCX contracts"""
    
    def __init__(self, kite: KiteConnect, current_config: TradingConfig, parent=None):
        super().__init__(parent)
        self.kite = kite
        self.config = current_config
        self.selected_contracts = {}
        self.init_ui()
        self.load_contracts()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Select MCX Contracts")
        self.setGeometry(300, 300, 800, 600)
        
        layout = QVBoxLayout()
        
        # Commodity filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by Commodity:"))
        
        self.commodity_filter = QComboBox()
        self.commodity_filter.addItems(["ALL", "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", 
                                        "COPPER", "ZINC", "LEAD", "ALUMINIUM", "NICKEL"])
        self.commodity_filter.setCurrentText(self.config.commodity if self.config.commodity else "GOLD")
        self.commodity_filter.currentTextChanged.connect(self.filter_contracts)
        
        filter_layout.addWidget(self.commodity_filter)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Contract tables
        tables_layout = QHBoxLayout()
        
        # Near month table
        near_group = QGroupBox("Near Month Contract")
        near_layout = QVBoxLayout()
        
        self.near_table = QTableWidget()
        self.near_table.setColumnCount(5)
        self.near_table.setHorizontalHeaderLabels(["Symbol", "Token", "Expiry", "Lot Size", "Select"])
        self.near_table.horizontalHeader().setStretchLastSection(True)
        self.near_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        near_layout.addWidget(self.near_table)
        near_group.setLayout(near_layout)
        
        # Mid month table
        mid_group = QGroupBox("Mid Month Contract")
        mid_layout = QVBoxLayout()
        
        self.mid_table = QTableWidget()
        self.mid_table.setColumnCount(5)
        self.mid_table.setHorizontalHeaderLabels(["Symbol", "Token", "Expiry", "Lot Size", "Select"])
        self.mid_table.horizontalHeader().setStretchLastSection(True)
        self.mid_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        mid_layout.addWidget(self.mid_table)
        mid_group.setLayout(mid_layout)
        
        tables_layout.addWidget(near_group)
        tables_layout.addWidget(mid_group)
        layout.addLayout(tables_layout)
        
        # Current selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Selected Near Contract:"))
        self.near_selection_label = QLabel("--")
        selection_layout.addWidget(self.near_selection_label)
        
        selection_layout.addWidget(QLabel("Selected Mid Contract:"))
        self.mid_selection_label = QLabel("--")
        selection_layout.addWidget(self.mid_selection_label)
        
        selection_layout.addStretch()
        layout.addLayout(selection_layout)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def load_contracts(self):
        """Load MCX contracts"""
        try:
            if not self.kite:
                logger.error("KiteConnect not available")
                return
            
            # Get all MCX instruments
            instruments = self.kite.instruments("MCX")
            
            # Filter for futures contracts
            futures = [i for i in instruments if "FUT" in i['tradingsymbol']]
            
            # Group by commodity
            self.contracts_by_commodity = {}
            for contract in futures:
                # Extract commodity from symbol
                symbol = contract['tradingsymbol']
                commodity = self._extract_commodity(symbol)
                
                if commodity not in self.contracts_by_commodity:
                    self.contracts_by_commodity[commodity] = []
                
                self.contracts_by_commodity[commodity].append({
                    'symbol': symbol,
                    'token': contract['instrument_token'],
                    'expiry': contract.get('expiry', ''),
                    'lot_size': contract.get('lot_size', 1)
                })
            
            # Apply initial filter
            self.filter_contracts(self.commodity_filter.currentText())
            
        except Exception as e:
            logger.error(f"Error loading contracts: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load contracts: {e}")
    
    def _extract_commodity(self, symbol: str) -> str:
        """Extract commodity from symbol"""
        for commodity in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", 
                         "COPPER", "ZINC", "LEAD", "ALUMINIUM", "NICKEL"]:
            if commodity in symbol:
                return commodity
        return "OTHER"
    
    def filter_contracts(self, commodity: str):
        """Filter contracts by commodity"""
        if commodity == "ALL":
            filtered_contracts = []
            for comm_contracts in self.contracts_by_commodity.values():
                filtered_contracts.extend(comm_contracts)
        else:
            filtered_contracts = self.contracts_by_commodity.get(commodity, [])
        
        # Sort by expiry
        filtered_contracts.sort(key=lambda x: x['expiry'])
        
        # Populate tables
        self._populate_table(self.near_table, filtered_contracts, "near")
        self._populate_table(self.mid_table, filtered_contracts, "mid")
    
    def _populate_table(self, table: QTableWidget, contracts: List, contract_type: str):
        """Populate contract table"""
        table.setRowCount(len(contracts))
        
        for i, contract in enumerate(contracts):
            # Symbol
            symbol_item = QTableWidgetItem(contract['symbol'])
            table.setItem(i, 0, symbol_item)
            
            # Token
            token_item = QTableWidgetItem(str(contract['token']))
            table.setItem(i, 1, token_item)
            
            # Expiry
            expiry_item = QTableWidgetItem(str(contract['expiry']))
            table.setItem(i, 2, expiry_item)
            
            # Lot Size
            lot_item = QTableWidgetItem(str(contract['lot_size']))
            table.setItem(i, 3, lot_item)
            
            # Select button
            select_btn = QPushButton("Select")
            select_btn.clicked.connect(lambda checked, c=contract, t=contract_type: 
                                      self.select_contract(c, t))
            table.setCellWidget(i, 4, select_btn)
        
        table.resizeColumnsToContents()
    
    def select_contract(self, contract: Dict, contract_type: str):
        """Select contract"""
        self.selected_contracts[contract_type] = contract
        
        if contract_type == "near":
            self.near_selection_label.setText(f"{contract['symbol']} (Exp: {contract['expiry']})")
        else:
            self.mid_selection_label.setText(f"{contract['symbol']} (Exp: {contract['expiry']})")
        
        logger.info(f"Selected {contract_type} contract: {contract['symbol']}")
    
    def accept_selection(self):
        """Accept selected contracts"""
        if 'near' in self.selected_contracts and 'mid' in self.selected_contracts:
            # Check if contracts are different
            near_symbol = self.selected_contracts['near']['symbol']
            mid_symbol = self.selected_contracts['mid']['symbol']
            
            if near_symbol == mid_symbol:
                QMessageBox.warning(self, "Invalid Selection", 
                                  "Near and Mid contracts must be different")
                return
            
            self.accept()
        else:
            QMessageBox.warning(self, "Selection Required", 
                              "Please select both near and mid contracts")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    print("=" * 70)
    print("MCX CALENDAR SPREAD PRO - ADVANCED TRADING SYSTEM")
    print("=" * 70)
    print("\nFeatures Included:")
    print("1. Real-time Bid/Ask prices with market depth")
    print("2. Comprehensive P&L tracking and position management")
    print("3. Flexible MCX instrument selection")
    print("4. Auto Entry with configurable thresholds")
    print("5. Auto Exit with Profit Target, Stop Loss & Trailing Stop")
    print("6. Manual trading controls")
    print("7. Auto-login after settings")
    print("\nNote: Requires Zerodha account with MCX subscription")
    print("=" * 70)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    dashboard = AdvancedTradingDashboard()
    dashboard.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()