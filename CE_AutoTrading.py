import logging
import json
import os
import threading
import time
import queue
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import pandas_ta as ta
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from kiteconnect import KiteConnect, KiteTicker

# ==================== 1. ENUMS AND CONSTANTS ====================
class TradeStatus(Enum):
    PENDING = "PENDING"
    ENTERED = "ENTERED"
    EXITED = "EXITED"
    CANCELLED = "CANCELLED"

class TradeDirection(Enum):
    BUY_SPREAD = "BUY_SPREAD"      # Buy near, sell next
    SELL_SPREAD = "SELL_SPREAD"    # Sell near, buy next

# ==================== 2. CHANDELIER EXIT INDICATOR ====================
def calculate_chandelier_exit(df, length=22, mult=3.0, use_close=True):
    """Calculates Chandelier Exit signals."""
    df = df.copy()
    
    df['atr'] = mult * ta.atr(df['high'], df['low'], df['close'], length=length)
    
    if use_close:
        highest_close = df['close'].rolling(window=length).max()
        lowest_close = df['close'].rolling(window=length).min()
    else:
        highest_close = df['high'].rolling(window=length).max()
        lowest_close = df['low'].rolling(window=length).min()
    
    df['long_stop'] = highest_close - df['atr']
    df['short_stop'] = lowest_close + df['atr']
    
    long_stop_prev = df['long_stop'].shift(1).fillna(df['long_stop'])
    short_stop_prev = df['short_stop'].shift(1).fillna(df['short_stop'])
    
    cond_long = df['close'].shift(1) > long_stop_prev
    df['long_stop'] = np.where(cond_long, 
                               np.maximum(df['long_stop'], long_stop_prev), 
                               df['long_stop'])
    
    cond_short = df['close'].shift(1) < short_stop_prev
    df['short_stop'] = np.where(cond_short, 
                                np.minimum(df['short_stop'], short_stop_prev), 
                                df['short_stop'])
    
    df['ce_dir'] = 1
    for i in range(1, len(df)):
        if df.loc[i, 'close'] > df.loc[i-1, 'short_stop']:
            df.loc[i, 'ce_dir'] = 1
        elif df.loc[i, 'close'] < df.loc[i-1, 'long_stop']:
            df.loc[i, 'ce_dir'] = -1
        else:
            df.loc[i, 'ce_dir'] = df.loc[i-1, 'ce_dir']
    
    df['buy_signal'] = (df['ce_dir'] == 1) & (df['ce_dir'].shift(1) == -1)
    df['sell_signal'] = (df['ce_dir'] == -1) & (df['ce_dir'].shift(1) == 1)
    
    return df

# ==================== 3. CALENDAR SPREAD STRATEGY ====================
class CalendarSpreadStrategy:
    """Implements calendar spread trading logic."""
    
    def __init__(self, lookback_period=200, std_dev_multiplier=1.0):
        self.lookback_period = lookback_period
        self.std_dev_multiplier = std_dev_multiplier
        self.spread_mean = None
        self.spread_std = None
        self.upper_band = None
        self.lower_band = None
        
    def calculate_spread_stats(self, near_month_prices, next_month_prices):
        """Calculate spread statistics."""
        if len(near_month_prices) != len(next_month_prices):
            raise ValueError("Price series must have same length")
        
        spread = next_month_prices - near_month_prices
        recent_spread = spread.tail(self.lookback_period)
        
        self.spread_mean = recent_spread.mean()
        self.spread_std = recent_spread.std()
        
        self.upper_band = self.spread_mean + (self.std_dev_multiplier * self.spread_std)
        self.lower_band = self.spread_mean - (self.std_dev_multiplier * self.spread_std)
        
        current_spread = spread.iloc[-1] if len(spread) > 0 else 0
        
        signal = "HOLD"
        action = None
        
        if current_spread > self.upper_band:
            signal = "SELL_SPREAD"
            action = {
                'near_month_action': 'SELL',
                'next_month_action': 'BUY',
                'reason': f'Spread {current_spread:.2f} > Upper Band {self.upper_band:.2f}'
            }
        elif current_spread < self.lower_band:
            signal = "BUY_SPREAD"
            action = {
                'near_month_action': 'BUY',
                'next_month_action': 'SELL',
                'reason': f'Spread {current_spread:.2f} < Lower Band {self.lower_band:.2f}'
            }
        
        return {
            'current_spread': current_spread,
            'mean': self.spread_mean,
            'std': self.spread_std,
            'upper_band': self.upper_band,
            'lower_band': self.lower_band,
            'signal': signal,
            'action': action
        }
    
    def calculate_realtime_spread(self, near_price, next_price):
        """Calculate spread for real-time prices."""
        if near_price is None or next_price is None:
            return None
        
        current_spread = next_price - near_price
        
        signal = "HOLD"
        action = None
        
        if self.upper_band is not None and current_spread > self.upper_band:
            signal = "SELL_SPREAD"
            action = {
                'near_month_action': 'SELL',
                'next_month_action': 'BUY',
                'reason': f'Spread {current_spread:.2f} > Upper Band {self.upper_band:.2f}'
            }
        elif self.lower_band is not None and current_spread < self.lower_band:
            signal = "BUY_SPREAD"
            action = {
                'near_month_action': 'BUY',
                'next_month_action': 'SELL',
                'reason': f'Spread {current_spread:.2f} < Lower Band {self.lower_band:.2f}'
            }
        
        return {
            'current_spread': current_spread,
            'signal': signal,
            'action': action
        }
    
    def calculate_exit_conditions(self, trade_direction, entry_spread, current_spread):
        """Calculate exit conditions for a trade."""
        exit_conditions = {
            'target_hit': False,
            'stop_loss_hit': False,
            'mean_reversion_hit': False,
            'time_exit': False,
            'reason': ''
        }
        
        if self.spread_mean is None or self.spread_std is None:
            return exit_conditions
        
        # Calculate returns
        if trade_direction == TradeDirection.BUY_SPREAD:
            # We expect spread to increase (next - near gets bigger)
            profit = current_spread - entry_spread
            target_spread = entry_spread + (self.spread_std * 1.0)  # 1 std dev target
            stop_loss_spread = entry_spread - (self.spread_std * 0.5)  # 0.5 std dev stop
            
            # Check conditions
            if current_spread >= target_spread:
                exit_conditions['target_hit'] = True
                exit_conditions['reason'] = f'Target reached: {current_spread:.2f} >= {target_spread:.2f}'
            elif current_spread <= stop_loss_spread:
                exit_conditions['stop_loss_hit'] = True
                exit_conditions['reason'] = f'Stop loss hit: {current_spread:.2f} <= {stop_loss_spread:.2f}'
            elif current_spread >= self.spread_mean:
                exit_conditions['mean_reversion_hit'] = True
                exit_conditions['reason'] = f'Mean reversion: {current_spread:.2f} >= {self.spread_mean:.2f}'
                
        elif trade_direction == TradeDirection.SELL_SPREAD:
            # We expect spread to decrease (next - near gets smaller)
            profit = entry_spread - current_spread
            target_spread = entry_spread - (self.spread_std * 1.0)  # 1 std dev target
            stop_loss_spread = entry_spread + (self.spread_std * 0.5)  # 0.5 std dev stop
            
            # Check conditions
            if current_spread <= target_spread:
                exit_conditions['target_hit'] = True
                exit_conditions['reason'] = f'Target reached: {current_spread:.2f} <= {target_spread:.2f}'
            elif current_spread >= stop_loss_spread:
                exit_conditions['stop_loss_hit'] = True
                exit_conditions['reason'] = f'Stop loss hit: {current_spread:.2f} >= {stop_loss_spread:.2f}'
            elif current_spread <= self.spread_mean:
                exit_conditions['mean_reversion_hit'] = True
                exit_conditions['reason'] = f'Mean reversion: {current_spread:.2f} <= {self.spread_mean:.2f}'
        
        return exit_conditions

# ==================== 4. AUTO TRADE MANAGER ====================
class AutoTradeManager:
    """Manages automatic entry and exit of trades."""
    
    def __init__(self, trading_api, instrument_manager, strategy, log_callback=None):
        self.trading_api = trading_api
        self.instrument_manager = instrument_manager
        self.strategy = strategy
        self.active_trades = []
        self.completed_trades = []
        self.auto_entry_enabled = False
        self.auto_exit_enabled = False
        self.max_open_trades = 3
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.target_pct = 0.03     # 3% target
        self.trade_queue = queue.Queue()
        self.is_running = False
        self.trade_thread = None
        self.trade_id_counter = 1
        self.log_callback = log_callback if log_callback else print
        
    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        self.log_callback(log_msg)
        
    def start(self):
        """Start the auto trade manager."""
        if not self.is_running:
            self.is_running = True
            self.trade_thread = threading.Thread(target=self._trade_monitor, daemon=True)
            self.trade_thread.start()
            self.log("🚀 Auto trade manager started")
            return True
        return False
    
    def stop(self):
        """Stop the auto trade manager."""
        self.is_running = False
        if self.trade_thread:
            self.trade_thread.join(timeout=2)
        self.log("🛑 Auto trade manager stopped")
        return True
    
    def enable_auto_entry(self, enabled=True):
        """Enable or disable auto entry."""
        self.auto_entry_enabled = enabled
        self.log(f"⚙️ Auto entry {'enabled' if enabled else 'disabled'}")
        return enabled
    
    def enable_auto_exit(self, enabled=True):
        """Enable or disable auto exit."""
        self.auto_exit_enabled = enabled
        self.log(f"⚙️ Auto exit {'enabled' if enabled else 'disabled'}")
        return enabled
    
    def set_max_trades(self, max_trades):
        """Set maximum number of open trades."""
        self.max_open_trades = max_trades
        self.log(f"⚙️ Max open trades set to {max_trades}")
    
    def set_risk_parameters(self, stop_loss_pct=None, target_pct=None):
        """Set risk management parameters."""
        if stop_loss_pct is not None:
            self.stop_loss_pct = stop_loss_pct / 100  # Convert to decimal
            self.log(f"⚙️ Stop loss set to {stop_loss_pct}%")
        if target_pct is not None:
            self.target_pct = target_pct / 100  # Convert to decimal
            self.log(f"⚙️ Target profit set to {target_pct}%")
    
    def check_entry_signal(self, signal_data, near_symbol, next_symbol, near_price, next_price):
        """Check if we should enter a trade based on signal."""
        if not self.auto_entry_enabled:
            self.log("⏸️ Auto entry disabled, skipping signal")
            return False
        
        # Check if we have too many open trades
        if len(self.active_trades) >= self.max_open_trades:
            self.log(f"🚫 Max trades reached ({len(self.active_trades)}/{self.max_open_trades})")
            return False
        
        # Check if signal is valid
        if not signal_data or not signal_data.get('action'):
            self.log("❌ No valid signal data")
            return False
        
        self.log(f"📈 Checking entry signal: {signal_data['signal']}")
        self.log(f"   Spread: {signal_data['current_spread']:.2f}")
        
        # Check if we already have a similar active trade
        for trade in self.active_trades:
            if (trade['near_symbol'] == near_symbol and 
                trade['next_symbol'] == next_symbol and
                trade['direction'].value == signal_data['signal'] and
                trade['status'] == TradeStatus.ENTERED):
                self.log(f"⏭️ Similar trade already active: {trade['id']}")
                return False
        
        # Create trade object
        trade = {
            'id': self.trade_id_counter,
            'status': TradeStatus.PENDING,
            'entry_time': datetime.now(),
            'direction': TradeDirection(signal_data['signal']),
            'near_symbol': near_symbol,
            'next_symbol': next_symbol,
            'entry_near_price': near_price,
            'entry_next_price': next_price,
            'entry_spread': signal_data['current_spread'],
            'exit_near_price': None,
            'exit_next_price': None,
            'exit_spread': None,
            'exit_time': None,
            'reason': signal_data['action']['reason'],
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'exit_reason': None,
            'order_ids': None,
            'exit_order_ids': None
        }
        
        self.trade_id_counter += 1
        
        # Add to trade queue
        self.trade_queue.put(('ENTRY', trade))
        self.log(f"📥 Added entry to queue: Trade {trade['id']} - {signal_data['signal']}")
        return True
    
    def check_exit_conditions(self, current_near_price, current_next_price):
        """Check exit conditions for all active trades."""
        if not self.auto_exit_enabled:
            return []
        
        exits = []
        current_spread = current_next_price - current_near_price
        
        for trade in self.active_trades[:]:  # Use copy to allow removal
            if trade['status'] != TradeStatus.ENTERED:
                continue
            
            # Calculate exit conditions
            exit_conditions = self.strategy.calculate_exit_conditions(
                trade['direction'],
                trade['entry_spread'],
                current_spread
            )
            
            # Check if any exit condition is met
            should_exit = (exit_conditions['target_hit'] or 
                          exit_conditions['stop_loss_hit'] or 
                          exit_conditions['mean_reversion_hit'])
            
            if should_exit:
                self.log(f"🎯 Exit condition met for trade {trade['id']}: {exit_conditions['reason']}")
                
                # Prepare exit trade
                exit_trade = trade.copy()
                exit_trade['status'] = TradeStatus.PENDING
                exit_trade['exit_near_price'] = current_near_price
                exit_trade['exit_next_price'] = current_next_price
                exit_trade['exit_spread'] = current_spread
                exit_trade['exit_reason'] = exit_conditions['reason']
                
                # Calculate P&L
                if trade['direction'] == TradeDirection.BUY_SPREAD:
                    pnl = (current_spread - trade['entry_spread'])
                else:  # SELL_SPREAD
                    pnl = (trade['entry_spread'] - current_spread)
                
                exit_trade['pnl'] = pnl
                exit_trade['pnl_pct'] = (pnl / trade['entry_spread']) * 100 if trade['entry_spread'] != 0 else 0
                
                # Add to exit queue
                self.trade_queue.put(('EXIT', exit_trade))
                exits.append(exit_trade)
        
        return exits
    
    def update_active_trades_pnl(self, current_near_price, current_next_price):
        """Update current P&L for active trades."""
        if not current_near_price or not current_next_price:
            return
        
        current_spread = current_next_price - current_near_price
        
        for trade in self.active_trades:
            if trade['status'] == TradeStatus.ENTERED:
                # Calculate current P&L
                if trade['direction'] == TradeDirection.BUY_SPREAD:
                    pnl = (current_spread - trade['entry_spread'])
                else:  # SELL_SPREAD
                    pnl = (trade['entry_spread'] - current_spread)
                
                trade['current_near_price'] = current_near_price
                trade['current_next_price'] = current_next_price
                trade['current_spread'] = current_spread
                trade['current_pnl'] = pnl
                trade['current_pnl_pct'] = (pnl / trade['entry_spread']) * 100 if trade['entry_spread'] != 0 else 0
    
    def _trade_monitor(self):
        """Monitor trade queue and execute trades."""
        self.log("🔧 Auto trade monitor thread started")
        
        while self.is_running:
            try:
                # Process trade queue
                while not self.trade_queue.empty():
                    trade_type, trade = self.trade_queue.get_nowait()
                    
                    self.log(f"📋 Processing {trade_type} for trade {trade['id']}")
                    
                    if trade_type == 'ENTRY':
                        success = self._execute_entry(trade)
                        if success:
                            self.log(f"✅ Entry executed for trade {trade['id']}")
                        else:
                            self.log(f"❌ Entry failed for trade {trade['id']}")
                    elif trade_type == 'EXIT':
                        success = self._execute_exit(trade)
                        if success:
                            self.log(f"✅ Exit executed for trade {trade['id']}")
                        else:
                            self.log(f"❌ Exit failed for trade {trade['id']}")
                    
                    self.trade_queue.task_done()
                
                # Clean up old completed trades
                self._cleanup_old_trades()
                
                time.sleep(1)  # Check every second
                
            except queue.Empty:
                time.sleep(0.5)
            except Exception as e:
                self.log(f"❌ Error in trade monitor: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    def _execute_entry(self, trade):
        """Execute a trade entry."""
        try:
            # Map direction to actions
            if trade['direction'] == TradeDirection.BUY_SPREAD:
                action = {'near_month_action': 'BUY', 'next_month_action': 'SELL'}
            else:  # SELL_SPREAD
                action = {'near_month_action': 'SELL', 'next_month_action': 'BUY'}
            
            self.log(f"📤 Placing entry orders for trade {trade['id']}: {action}")
            
            # Place orders
            success, result = self.trading_api.place_calendar_spread_order(
                trade['near_symbol'],
                trade['next_symbol'],
                action
            )
            
            if success:
                trade['status'] = TradeStatus.ENTERED
                trade['order_ids'] = result
                trade['entry_time'] = datetime.now()
                self.active_trades.append(trade)
                self.log(f"✅ Auto entry executed: {trade['direction'].value} - {trade['near_symbol']}/{trade['next_symbol']}")
                return True
            else:
                trade['status'] = TradeStatus.CANCELLED
                self.log(f"❌ Auto entry failed: {result}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error executing auto entry: {e}")
            trade['status'] = TradeStatus.CANCELLED
            return False
    
    def _execute_exit(self, trade):
        """Execute a trade exit."""
        try:
            # Reverse the position to exit
            if trade['direction'] == TradeDirection.BUY_SPREAD:
                exit_action = {'near_month_action': 'SELL', 'next_month_action': 'BUY'}
            else:  # SELL_SPREAD
                exit_action = {'near_month_action': 'BUY', 'next_month_action': 'SELL'}
            
            self.log(f"📤 Placing exit orders for trade {trade['id']}: {exit_action}")
            
            # Place exit orders
            success, result = self.trading_api.place_calendar_spread_order(
                trade['near_symbol'],
                trade['next_symbol'],
                exit_action
            )
            
            if success:
                trade['status'] = TradeStatus.EXITED
                trade['exit_time'] = datetime.now()
                trade['exit_order_ids'] = result
                
                # Move from active to completed
                self.active_trades = [t for t in self.active_trades if t['id'] != trade['id']]
                self.completed_trades.append(trade)
                
                self.log(f"✅ Auto exit executed: {trade['direction'].value} - P&L: {trade['pnl']:.2f} ({trade['pnl_pct']:.1f}%)")
                return True
            else:
                self.log(f"❌ Auto exit failed: {result}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error executing auto exit: {e}")
            return False
    
    def _cleanup_old_trades(self):
        """Clean up old completed trades."""
        # Keep only last 100 completed trades
        if len(self.completed_trades) > 100:
            self.completed_trades = self.completed_trades[-100:]
    
    def get_active_trades(self):
        """Get list of active trades."""
        return self.active_trades.copy()
    
    def get_completed_trades(self):
        """Get list of completed trades."""
        return self.completed_trades.copy()
    
    def get_trade_summary(self):
        """Get trade performance summary."""
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl': 0.0
            }
        
        winning_trades = [t for t in self.completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.completed_trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in self.completed_trades)
        
        return {
            'total_trades': len(self.completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': total_pnl,
            'win_rate': (len(winning_trades) / len(self.completed_trades)) * 100 if self.completed_trades else 0,
            'avg_pnl': total_pnl / len(self.completed_trades) if self.completed_trades else 0
        }
    
    def close_all_trades(self, current_near_price, current_next_price):
        """Close all active trades at market."""
        if not self.active_trades:
            return []
        
        self.log(f"🔚 Closing all {len(self.active_trades)} active trades")
        exits = []
        current_spread = current_next_price - current_near_price
        
        for trade in self.active_trades[:]:
            exit_trade = trade.copy()
            exit_trade['status'] = TradeStatus.PENDING
            exit_trade['exit_near_price'] = current_near_price
            exit_trade['exit_next_price'] = current_next_price
            exit_trade['exit_spread'] = current_spread
            exit_trade['exit_reason'] = "Manual close all"
            
            # Calculate P&L
            if trade['direction'] == TradeDirection.BUY_SPREAD:
                pnl = (current_spread - trade['entry_spread'])
            else:  # SELL_SPREAD
                pnl = (trade['entry_spread'] - current_spread)
            
            exit_trade['pnl'] = pnl
            exit_trade['pnl_pct'] = (pnl / trade['entry_spread']) * 100 if trade['entry_spread'] != 0 else 0
            
            self.trade_queue.put(('EXIT', exit_trade))
            exits.append(exit_trade)
        
        return exits

# ==================== 5. ZERODHA INTEGRATION ====================
class ZerodhaTradingAPI:
    """Handles authentication and trading operations."""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token = None
        self.login_url = None
        self.ticker = None
        self.websocket_thread = None
        self.is_websocket_running = False
        
    def get_login_url(self):
        """Generate login URL for user authentication."""
        self.login_url = self.kite.login_url()
        return self.login_url
    
    def set_access_token(self, request_token):
        """Generate and set access token using request token."""
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            return True, "Authentication successful"
        except Exception as e:
            return False, f"Authentication failed: {str(e)}"
    
    def start_websocket(self, on_ticks, on_connect=None, on_close=None):
        """Start WebSocket connection for real-time data."""
        try:
            self.ticker = KiteTicker(api_key=self.api_key, access_token=self.access_token)
            
            if on_ticks:
                self.ticker.on_ticks = on_ticks
            if on_connect:
                self.ticker.on_connect = on_connect
            if on_close:
                self.ticker.on_close = on_close
            
            self.is_websocket_running = True
            self.websocket_thread = threading.Thread(target=self.ticker.connect, daemon=True)
            self.websocket_thread.start()
            return True, "WebSocket started"
        except Exception as e:
            return False, f"WebSocket failed: {str(e)}"
    
    def stop_websocket(self):
        """Stop WebSocket connection."""
        if self.ticker:
            self.ticker.close()
            self.is_websocket_running = False
    
    def subscribe_instruments(self, tokens):
        """Subscribe to instruments for real-time data."""
        if self.ticker and self.is_websocket_running:
            try:
                self.ticker.subscribe(tokens)
                # Use MODE_FULL for bid/ask data
                self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
                return True
            except Exception as e:
                print(f"Error subscribing to tokens: {e}")
                return False
        return False
    
    def get_quote(self, instrument_token):
        """Get current quote with bid/ask data."""
        try:
            quote = self.kite.quote([instrument_token])
            return quote.get(str(instrument_token), {})
        except Exception as e:
            print(f"Error getting quote: {e}")
            return {}
    
    def get_historical_data(self, instrument_token, interval="day", duration=200):
        """Fetch historical data for analysis."""
        try:
            from_date = (datetime.now() - timedelta(days=duration*2)).date()
            to_date = datetime.now().date()
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def place_calendar_spread_order(self, near_month_symbol, next_month_symbol, action):
        """Place a calendar spread order (two simultaneous orders)."""
        orders = []
        
        order_map = {
            ('BUY', 'SELL'): {
                'near': {'transaction_type': KiteConnect.TRANSACTION_TYPE_BUY},
                'next': {'transaction_type': KiteConnect.TRANSACTION_TYPE_SELL}
            },
            ('SELL', 'BUY'): {
                'near': {'transaction_type': KiteConnect.TRANSACTION_TYPE_SELL},
                'next': {'transaction_type': KiteConnect.TRANSACTION_TYPE_BUY}
            }
        }
        
        action_key = (action['near_month_action'], action['next_month_action'])
        if action_key not in order_map:
            return False, "Invalid action combination"
        
        params = order_map[action_key]
        
        try:
            # First get current quotes for limit prices
            near_quote = self.get_quote_for_symbol(near_month_symbol)
            next_quote = self.get_quote_for_symbol(next_month_symbol)
            
            # Place near-month order
            near_price = self.get_order_price(near_quote, params['near']['transaction_type'])
            near_order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                tradingsymbol=near_month_symbol,
                exchange=self.kite.EXCHANGE_MCX,
                transaction_type=params['near']['transaction_type'],
                quantity=1,
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=near_price,
                product=self.kite.PRODUCT_MIS,
                validity=self.kite.VALIDITY_DAY
            )
            orders.append({'symbol': near_month_symbol, 'order_id': near_order_id, 'price': near_price})
            
            # Place next-month order
            next_price = self.get_order_price(next_quote, params['next']['transaction_type'])
            next_order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                tradingsymbol=next_month_symbol,
                exchange=self.kite.EXCHANGE_MCX,
                transaction_type=params['next']['transaction_type'],
                quantity=1,
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=next_price,
                product=self.kite.PRODUCT_MIS,
                validity=self.kite.VALIDITY_DAY
            )
            orders.append({'symbol': next_month_symbol, 'order_id': next_order_id, 'price': next_price})
            
            return True, orders
            
        except Exception as e:
            return False, f"Order placement failed: {str(e)}"
    
    def get_quote_for_symbol(self, symbol):
        """Get quote for a specific symbol."""
        # This is a simplified version - in reality you'd need instrument token
        try:
            # Get all instruments and find the right one
            instruments = self.kite.instruments("MCX")
            for inst in instruments:
                if inst['tradingsymbol'] == symbol:
                    return self.get_quote(inst['instrument_token'])
        except:
            pass
        return {}
    
    def get_order_price(self, quote, transaction_type):
        """Get appropriate order price based on transaction type."""
        if not quote:
            return 0
        
        depth = quote.get('depth', {})
        
        if transaction_type == KiteConnect.TRANSACTION_TYPE_BUY:
            # For buy, use ask price
            if 'sell' in depth and depth['sell']:
                return depth['sell'][0]['price']  # Best ask
            return quote.get('last_price', 0)
        else:  # SELL
            # For sell, use bid price
            if 'buy' in depth and depth['buy']:
                return depth['buy'][0]['price']  # Best bid
            return quote.get('last_price', 0)

# ==================== 6. INSTRUMENT DATA MANAGER ====================
class MCXInstrumentManager:
    """Fetches and manages MCX futures instrument data."""
    
    def __init__(self, kite_client):
        self.kite = kite_client
        self.instruments_df = None
        self.mcx_futures_df = None
        self.expiry_dates = {}
        self.commodity_map = {}
        
    def fetch_instruments(self):
        """Fetch all instruments and filter for MCX futures."""
        try:
            instruments = self.kite.instruments()
            self.instruments_df = pd.DataFrame(instruments)
            
            self.mcx_futures_df = self.instruments_df[
                (self.instruments_df['exchange'] == 'MCX') &
                (self.instruments_df['instrument_type'] == 'FUT')
            ].copy()
            
            if self.mcx_futures_df.empty:
                return False, "No MCX futures instruments found"
            
            self.mcx_futures_df['expiry'] = pd.to_datetime(self.mcx_futures_df['expiry'])
            now = pd.Timestamp.now()
            
            for index, row in self.mcx_futures_df.iterrows():
                tradingsymbol = row['tradingsymbol']
                
                commodities = ['GOLD', 'SILVER', 'CRUDEOIL', 'NATURALGAS', 
                              'COPPER', 'LEAD', 'ZINC', 'ALUMINIUM', 'NICKEL']
                
                base_symbol = None
                for commodity in commodities:
                    if commodity in tradingsymbol:
                        base_symbol = commodity
                        break
                
                if base_symbol is None:
                    import re
                    match = re.match(r'([A-Z]+)', tradingsymbol)
                    if match:
                        base_symbol = match.group(1)
                
                if base_symbol:
                    if base_symbol not in self.expiry_dates:
                        self.expiry_dates[base_symbol] = []
                        self.commodity_map[base_symbol] = []
                    
                    if row['expiry'] > now:
                        self.expiry_dates[base_symbol].append(row['expiry'])
                        self.commodity_map[base_symbol].append({
                            'tradingsymbol': tradingsymbol,
                            'expiry': row['expiry'],
                            'instrument_token': row['instrument_token']
                        })
            
            for commodity in self.expiry_dates:
                self.expiry_dates[commodity] = sorted(set(self.expiry_dates[commodity]))
            
            return True, f"Fetched {len(self.mcx_futures_df)} MCX futures instruments"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return False, f"Failed to fetch instruments: {str(e)}\n{error_details}"
    
    def get_commodity_list(self):
        """Return sorted list of available commodities."""
        if self.expiry_dates:
            return sorted(self.expiry_dates.keys())
        return []
    
    def get_expiry_dates(self, commodity):
        """Get expiry dates for a specific commodity."""
        return sorted(self.expiry_dates.get(commodity, []))
    
    def get_tradingsymbol_by_expiry(self, commodity, expiry_date):
        """Get tradingsymbol for a specific commodity and expiry date."""
        if commodity in self.commodity_map:
            for item in self.commodity_map[commodity]:
                if item['expiry'] == expiry_date:
                    return item['tradingsymbol']
        return None
    
    def get_instrument_token_by_expiry(self, commodity, expiry_date):
        """Get instrument token for a specific commodity and expiry date."""
        if commodity in self.commodity_map:
            for item in self.commodity_map[commodity]:
                if item['expiry'] == expiry_date:
                    return item['instrument_token']
        return None
    
    def get_instrument_token(self, tradingsymbol):
        """Get instrument token for a trading symbol."""
        if self.instruments_df is not None:
            match = self.instruments_df[self.instruments_df['tradingsymbol'] == tradingsymbol]
            if not match.empty:
                return match.iloc[0]['instrument_token']
        return None

# ==================== 7. PRICE DATA MODEL ====================
class InstrumentPriceData:
    """Stores and manages price data for an instrument."""
    
    def __init__(self, symbol, instrument_token):
        self.symbol = symbol
        self.instrument_token = instrument_token
        self.last_price = 0.0
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0
        self.bid_price = 0.0
        self.bid_quantity = 0
        self.bid_orders = 0
        self.ask_price = 0.0
        self.ask_quantity = 0
        self.ask_orders = 0
        self.timestamp = None
        self.depth = {}
        
    def update_from_tick(self, tick):
        """Update price data from WebSocket tick."""
        if 'last_price' in tick:
            self.last_price = tick['last_price']
        
        if 'ohlc' in tick:
            ohlc = tick['ohlc']
            self.open = ohlc.get('open', self.open)
            self.high = ohlc.get('high', self.high)
            self.low = ohlc.get('low', self.low)
            self.close = ohlc.get('close', self.close)
        
        if 'volume' in tick:
            self.volume = tick['volume']
        
        if 'depth' in tick:
            self.depth = tick['depth']
            # Get best bid (highest buy price)
            if 'buy' in tick['depth'] and tick['depth']['buy']:
                best_bid = tick['depth']['buy'][0]  # First element is best bid
                self.bid_price = best_bid['price']
                self.bid_quantity = best_bid['quantity']
                self.bid_orders = best_bid['orders']
            
            # Get best ask (lowest sell price)
            if 'sell' in tick['depth'] and tick['depth']['sell']:
                best_ask = tick['depth']['sell'][0]  # First element is best ask
                self.ask_price = best_ask['price']
                self.ask_quantity = best_ask['quantity']
                self.ask_orders = best_ask['orders']
        
        self.timestamp = datetime.now()
    
    def get_mid_price(self):
        """Calculate mid price from bid and ask."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.last_price
    
    def get_spread(self):
        """Calculate bid-ask spread."""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0
    
    def get_spread_percentage(self):
        """Calculate bid-ask spread as percentage."""
        if self.bid_price > 0 and self.ask_price > 0:
            mid_price = self.get_mid_price()
            if mid_price > 0:
                return (self.get_spread() / mid_price) * 100
        return 0.0

# ==================== 8. AUTO TRADE GUI COMPONENTS ====================
class AutoTradeControlPanel:
    """Control panel for auto trading."""
    
    def __init__(self, parent, auto_trade_manager):
        self.parent = parent
        self.auto_trade_manager = auto_trade_manager
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the auto trade control panel."""
        # Main frame
        main_frame = tk.LabelFrame(self.parent, text="🤖 Auto Trade Controller", padx=10, pady=10)
        main_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Status indicators
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Auto Entry Status
        self.entry_status_var = tk.StringVar(value="⭕ OFF")
        tk.Label(status_frame, text="Auto Entry:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.entry_status_label = tk.Label(status_frame, textvariable=self.entry_status_var, 
                                          font=("Arial", 10), fg="red")
        self.entry_status_label.pack(side=tk.LEFT, padx=5)
        
        # Auto Exit Status
        self.exit_status_var = tk.StringVar(value="⭕ OFF")
        tk.Label(status_frame, text="Auto Exit:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20,5))
        self.exit_status_label = tk.Label(status_frame, textvariable=self.exit_status_var, 
                                         font=("Arial", 10), fg="red")
        self.exit_status_label.pack(side=tk.LEFT, padx=5)
        
        # Trade Count
        self.trade_count_var = tk.StringVar(value="Active: 0 | Total: 0")
        tk.Label(status_frame, textvariable=self.trade_count_var, 
                font=("Arial", 10), fg="#2196F3").pack(side=tk.RIGHT, padx=5)
        
        # Control Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.entry_button = tk.Button(button_frame, text="🚀 Enable Auto Entry", 
                                     command=self.toggle_auto_entry,
                                     bg="#f44336", fg="white", width=15)
        self.entry_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = tk.Button(button_frame, text="🛑 Enable Auto Exit", 
                                    command=self.toggle_auto_exit,
                                    bg="#f44336", fg="white", width=15)
        self.exit_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📊 View Trades", 
                 command=self.view_trades,
                 bg="#2196F3", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🧪 Test Auto Trading", 
                 command=self.test_auto_trading,
                 bg="#9C27B0", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🗑️ Close All Trades", 
                 command=self.close_all_trades,
                 bg="#FF9800", fg="white", width=15).pack(side=tk.RIGHT, padx=5)
        
        # Settings Frame
        settings_frame = tk.LabelFrame(main_frame, text="⚙️ Auto Trade Settings", padx=10, pady=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Max Trades
        tk.Label(settings_frame, text="Max Open Trades:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_trades_var = tk.StringVar(value="3")
        tk.Entry(settings_frame, textvariable=self.max_trades_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Stop Loss
        tk.Label(settings_frame, text="Stop Loss (%):").grid(row=0, column=2, padx=(20,5), pady=5, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value="2.0")
        tk.Entry(settings_frame, textvariable=self.stop_loss_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Target Profit
        tk.Label(settings_frame, text="Target Profit (%):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.target_var = tk.StringVar(value="3.0")
        tk.Entry(settings_frame, textvariable=self.target_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Apply Settings Button
        tk.Button(settings_frame, text="💾 Apply Settings", 
                 command=self.apply_settings,
                 bg="#4CAF50", fg="white").grid(row=1, column=3, padx=5, pady=5)
        
        # Performance Summary
        perf_frame = tk.Frame(main_frame)
        perf_frame.pack(fill=tk.X, pady=10)
        
        self.performance_var = tk.StringVar(value="No trades yet")
        tk.Label(perf_frame, textvariable=self.performance_var, 
                font=("Arial", 9), fg="#666666").pack()
        
        # Update thread
        self.update_status()
    
    def toggle_auto_entry(self):
        """Toggle auto entry on/off."""
        current = self.auto_trade_manager.auto_entry_enabled
        new_state = not current
        self.auto_trade_manager.enable_auto_entry(new_state)
        
        if new_state:
            self.entry_status_var.set("✅ ON")
            self.entry_status_label.config(fg="green")
            self.entry_button.config(text="⏸️ Disable Auto Entry", bg="#4CAF50")
        else:
            self.entry_status_var.set("⭕ OFF")
            self.entry_status_label.config(fg="red")
            self.entry_button.config(text="🚀 Enable Auto Entry", bg="#f44336")
    
    def toggle_auto_exit(self):
        """Toggle auto exit on/off."""
        current = self.auto_trade_manager.auto_exit_enabled
        new_state = not current
        self.auto_trade_manager.enable_auto_exit(new_state)
        
        if new_state:
            self.exit_status_var.set("✅ ON")
            self.exit_status_label.config(fg="green")
            self.exit_button.config(text="⏸️ Disable Auto Exit", bg="#4CAF50")
        else:
            self.exit_status_var.set("⭕ OFF")
            self.exit_status_label.config(fg="red")
            self.exit_button.config(text="🛑 Enable Auto Exit", bg="#f44336")
    
    def apply_settings(self):
        """Apply auto trade settings."""
        try:
            max_trades = int(self.max_trades_var.get())
            stop_loss = float(self.stop_loss_var.get())
            target = float(self.target_var.get())
            
            self.auto_trade_manager.set_max_trades(max_trades)
            self.auto_trade_manager.set_risk_parameters(stop_loss, target)
            
            messagebox.showinfo("Settings Applied", 
                              f"Settings updated:\nMax Trades: {max_trades}\nStop Loss: {stop_loss}%\nTarget: {target}%")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers")
    
    def test_auto_trading(self):
        """Test auto trading functionality."""
        # Create a test signal
        test_signal = {
            'current_spread': 50,
            'signal': 'BUY_SPREAD',
            'action': {
                'near_month_action': 'BUY',
                'next_month_action': 'SELL',
                'reason': 'Test signal for auto trading'
            }
        }
        
        # Check if auto trade manager is ready
        if self.auto_trade_manager:
            # Temporarily enable auto entry for test
            was_enabled = self.auto_trade_manager.auto_entry_enabled
            self.auto_trade_manager.enable_auto_entry(True)
            
            # Try to trigger an entry
            success = self.auto_trade_manager.check_entry_signal(
                test_signal,
                "TESTNEAR",
                "TESTNEXT",
                1000,
                1050
            )
            
            # Restore previous state
            self.auto_trade_manager.enable_auto_entry(was_enabled)
            
            if success:
                messagebox.showinfo("Test Successful", "Auto trading test passed! Entry was added to queue.")
            else:
                messagebox.showinfo("Test Info", "Auto trading test: Entry was not triggered. Check max trades or other conditions.")
        else:
            messagebox.showerror("Test Failed", "Auto trade manager not initialized")
    
    def view_trades(self):
        """Open trade history window."""
        trade_window = tk.Toplevel(self.parent)
        trade_window.title("📋 Trade History")
        trade_window.geometry("1000x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(trade_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Active Trades Tab
        active_frame = ttk.Frame(notebook)
        notebook.add(active_frame, text="📈 Active Trades")
        self.setup_active_trades_tab(active_frame)
        
        # Completed Trades Tab
        completed_frame = ttk.Frame(notebook)
        notebook.add(completed_frame, text="📊 Completed Trades")
        self.setup_completed_trades_tab(completed_frame)
        
        # Summary Tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📈 Performance Summary")
        self.setup_summary_tab(summary_frame)
        
        # Refresh immediately
        self.refresh_active_trades_display(active_frame)
        self.refresh_completed_trades_display(completed_frame)
        self.refresh_summary_display(summary_frame)
    
    def setup_active_trades_tab(self, parent):
        """Setup active trades tab."""
        # Create treeview
        columns = ('ID', 'Direction', 'Symbols', 'Entry Time', 'Entry Spread', 'Current Spread', 'Current P&L', 'Status')
        self.active_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # Define headings
        for col in columns:
            self.active_tree.heading(col, text=col)
            self.active_tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.active_tree.yview)
        self.active_tree.configure(yscrollcommand=scrollbar.set)
        
        self.active_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh button
        tk.Button(parent, text="🔄 Refresh", 
                 command=lambda: self.refresh_active_trades_display(parent),
                 bg="#2196F3", fg="white").pack(pady=5)
    
    def setup_completed_trades_tab(self, parent):
        """Setup completed trades tab."""
        # Create treeview
        columns = ('ID', 'Direction', 'Symbols', 'Entry Time', 'Exit Time', 'Entry Spread', 
                  'Exit Spread', 'P&L', 'P&L %', 'Exit Reason')
        self.completed_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # Define headings
        for col in columns:
            self.completed_tree.heading(col, text=col)
            self.completed_tree.column(col, width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.completed_tree.yview)
        self.completed_tree.configure(yscrollcommand=scrollbar.set)
        
        self.completed_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        button_frame = tk.Frame(parent)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="🔄 Refresh", 
                 command=lambda: self.refresh_completed_trades_display(parent),
                 bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📤 Export to CSV", 
                 command=self.export_trades,
                 bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
    
    def setup_summary_tab(self, parent):
        """Setup performance summary tab."""
        self.summary_text = scrolledtext.ScrolledText(parent, height=20, width=80)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Refresh button
        tk.Button(parent, text="🔄 Refresh Summary", 
                 command=lambda: self.refresh_summary_display(parent),
                 bg="#2196F3", fg="white").pack(pady=5)
    
    def refresh_active_trades_display(self, parent):
        """Refresh active trades display."""
        # Clear existing items
        for item in self.active_tree.get_children():
            self.active_tree.delete(item)
        
        # Get active trades
        active_trades = self.auto_trade_manager.get_active_trades()
        
        # Add trades to tree
        for trade in active_trades:
            # Get current values
            current_spread = trade.get('current_spread', trade['entry_spread'])
            current_pnl = trade.get('current_pnl', 0.0)
            
            self.active_tree.insert('', tk.END, values=(
                trade['id'],
                trade['direction'].value,
                f"{trade['near_symbol']}/{trade['next_symbol']}",
                trade['entry_time'].strftime("%H:%M:%S"),
                f"{trade['entry_spread']:.2f}",
                f"{current_spread:.2f}",
                f"{current_pnl:.2f}",
                trade['status'].value
            ))
    
    def refresh_completed_trades_display(self, parent):
        """Refresh completed trades display."""
        # Clear existing items
        for item in self.completed_tree.get_children():
            self.completed_tree.delete(item)
        
        # Get completed trades
        completed_trades = self.auto_trade_manager.get_completed_trades()
        
        # Add trades to tree
        for trade in completed_trades:
            self.completed_tree.insert('', tk.END, values=(
                trade['id'],
                trade['direction'].value,
                f"{trade['near_symbol']}/{trade['next_symbol']}",
                trade['entry_time'].strftime("%H:%M:%S"),
                trade['exit_time'].strftime("%H:%M:%S") if trade['exit_time'] else "",
                f"{trade['entry_spread']:.2f}",
                f"{trade.get('exit_spread', 0):.2f}",
                f"{trade.get('pnl', 0):.2f}",
                f"{trade.get('pnl_pct', 0):.1f}%",
                trade.get('exit_reason', '')
            ))
    
    def refresh_summary_display(self, parent):
        """Refresh performance summary."""
        summary = self.auto_trade_manager.get_trade_summary()
        
        self.summary_text.delete(1.0, tk.END)
        
        self.summary_text.insert(tk.END, "📊 TRADE PERFORMANCE SUMMARY\n")
        self.summary_text.insert(tk.END, "=" * 40 + "\n\n")
        
        self.summary_text.insert(tk.END, f"Total Trades: {summary['total_trades']}\n")
        self.summary_text.insert(tk.END, f"Winning Trades: {summary['winning_trades']}\n")
        self.summary_text.insert(tk.END, f"Losing Trades: {summary['losing_trades']}\n")
        self.summary_text.insert(tk.END, f"Win Rate: {summary['win_rate']:.1f}%\n")
        self.summary_text.insert(tk.END, f"Total P&L: {summary['total_pnl']:.2f}\n")
        self.summary_text.insert(tk.END, f"Average P&L: {summary['avg_pnl']:.2f}\n\n")
        
        # Add active trades info
        active_trades = self.auto_trade_manager.get_active_trades()
        if active_trades:
            self.summary_text.insert(tk.END, "📈 ACTIVE TRADES\n")
            self.summary_text.insert(tk.END, "=" * 40 + "\n")
            for trade in active_trades:
                current_pnl = trade.get('current_pnl', 0)
                pnl_color = "green" if current_pnl > 0 else "red" if current_pnl < 0 else "black"
                self.summary_text.insert(tk.END, 
                    f"Trade {trade['id']}: {trade['direction'].value} - P&L: {current_pnl:.2f}\n")
        
        # Add color coding
        if summary['total_pnl'] > 0:
            self.summary_text.tag_add("profit", "6.0", "6.end")
            self.summary_text.tag_config("profit", foreground="green")
        elif summary['total_pnl'] < 0:
            self.summary_text.tag_add("loss", "6.0", "6.end")
            self.summary_text.tag_config("loss", foreground="red")
    
    def export_trades(self):
        """Export trades to CSV file."""
        try:
            completed_trades = self.auto_trade_manager.get_completed_trades()
            
            if not completed_trades:
                messagebox.showinfo("No Data", "No completed trades to export")
                return
            
            # Convert to DataFrame
            data = []
            for trade in completed_trades:
                data.append({
                    'ID': trade['id'],
                    'Direction': trade['direction'].value,
                    'Near_Symbol': trade['near_symbol'],
                    'Next_Symbol': trade['next_symbol'],
                    'Entry_Time': trade['entry_time'],
                    'Exit_Time': trade.get('exit_time'),
                    'Entry_Spread': trade['entry_spread'],
                    'Exit_Spread': trade.get('exit_spread', 0),
                    'Entry_Near_Price': trade['entry_near_price'],
                    'Entry_Next_Price': trade['entry_next_price'],
                    'Exit_Near_Price': trade.get('exit_near_price'),
                    'Exit_Next_Price': trade.get('exit_next_price'),
                    'P&L': trade.get('pnl', 0),
                    'P&L_%': trade.get('pnl_pct', 0),
                    'Exit_Reason': trade.get('exit_reason', ''),
                    'Status': trade['status'].value
                })
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Successful", f"Trades exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Failed", f"Error exporting trades: {str(e)}")
    
    def close_all_trades(self):
        """Close all active trades."""
        if not self.auto_trade_manager.get_active_trades():
            messagebox.showinfo("No Trades", "No active trades to close")
            return
        
        confirm = messagebox.askyesno("Confirm Close All", 
                                     f"Close all {len(self.auto_trade_manager.get_active_trades())} active trades?")
        
        if confirm:
            # This will be implemented in the main app
            messagebox.showinfo("Close All", "Close all feature triggered - Please implement in main app")
    
    def update_status(self):
        """Update status display."""
        if self.auto_trade_manager:
            # Update trade counts
            active_count = len(self.auto_trade_manager.get_active_trades())
            completed_count = len(self.auto_trade_manager.get_completed_trades())
            self.trade_count_var.set(f"Active: {active_count} | Total: {completed_count}")
            
            # Update performance
            summary = self.auto_trade_manager.get_trade_summary()
            if summary['total_trades'] > 0:
                perf_text = f"Win Rate: {summary['win_rate']:.1f}% | Total P&L: {summary['total_pnl']:.2f}"
                self.performance_var.set(perf_text)
        
        # Schedule next update
        self.parent.after(2000, self.update_status)

# ==================== 9. MAIN GUI APPLICATION ====================
class CalendarSpreadTradingApp:
    """Main GUI application with auto trading features."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("📊 MCX Calendar Spread Trading Tool")
        self.root.geometry("1400x900")
        
        # Configuration
        self.config_file = "zerodha_config.json"
        
        # Initialize components
        self.zerodha_api = None
        self.strategy = CalendarSpreadStrategy()
        self.instrument_manager = None
        self.auto_trade_manager = None
        self.auto_trade_controls = None
        
        # Real-time data
        self.near_month_data = None
        self.next_month_data = None
        self.realtime_spread = None
        self.is_realtime_running = False
        
        # Load saved credentials
        self.load_config()
        
        # Setup GUI
        self.setup_gui()
        
        # Start real-time updates
        self.start_realtime_updates()
    
    def load_config(self):
        """Load saved API credentials."""
        self.saved_api_key = ""
        self.saved_access_token = ""
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.saved_api_key = config.get('api_key', '')
                    self.saved_access_token = config.get('access_token', '')
            except:
                pass
    
    def save_config(self, api_key, access_token):
        """Save API credentials for auto-login."""
        config = {
            'api_key': api_key,
            'access_token': access_token,
            'last_saved': datetime.now().isoformat()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except:
            return False
    
    def setup_gui(self):
        """Setup the main GUI layout with tabs."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Zerodha Authentication
        auth_frame = ttk.Frame(notebook)
        notebook.add(auth_frame, text="🔐 Zerodha Login")
        self.setup_auth_tab(auth_frame)
        
        # Tab 2: Strategy Configuration
        strategy_frame = ttk.Frame(notebook)
        notebook.add(strategy_frame, text="⚙️ Strategy Setup")
        self.setup_strategy_tab(strategy_frame)
        
        # Tab 3: Trading Panel (with real-time updates)
        trade_frame = ttk.Frame(notebook)
        notebook.add(trade_frame, text="📈 Trading Panel")
        self.setup_trading_tab(trade_frame)
        
        # Tab 4: Auto Trading
        auto_frame = ttk.Frame(notebook)
        notebook.add(auto_frame, text="🤖 Auto Trading")
        self.setup_auto_trading_tab(auto_frame)
        
        # Tab 5: Positions
        positions_frame = ttk.Frame(notebook)
        notebook.add(positions_frame, text="💼 Positions")
        self.setup_positions_tab(positions_frame)
        
        # Tab 6: Logs
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📋 Logs")
        self.setup_log_tab(log_frame)
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, bg="#e0e0e0")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_auth_tab(self, parent):
        """Setup authentication tab."""
        tk.Label(parent, text="Zerodha API Configuration", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # API Key
        tk.Label(parent, text="API Key:").pack(pady=(10, 0))
        self.api_key_var = tk.StringVar(value=self.saved_api_key)
        tk.Entry(parent, textvariable=self.api_key_var, width=50).pack()
        
        # API Secret
        tk.Label(parent, text="API Secret:").pack(pady=(10, 0))
        self.api_secret_var = tk.StringVar()
        tk.Entry(parent, textvariable=self.api_secret_var, width=50, show="*").pack()
        
        # Buttons Frame
        button_frame = tk.Frame(parent)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="🔗 Generate Login URL", 
                 command=self.generate_login_url, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="✅ Authenticate", 
                 command=self.authenticate, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🚀 Auto-Login", 
                 command=self.auto_login, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Login URL Display
        tk.Label(parent, text="Login URL:").pack(pady=(20, 0))
        self.login_url_var = tk.StringVar()
        tk.Entry(parent, textvariable=self.login_url_var, width=70, state='readonly').pack()
        
        # Request Token
        tk.Label(parent, text="Request Token (from redirect URL):").pack(pady=(10, 0))
        self.request_token_var = tk.StringVar()
        tk.Entry(parent, textvariable=self.request_token_var, width=50).pack()
        
        # Authentication Status
        tk.Label(parent, text="Status:").pack(pady=(10, 0))
        self.auth_status_var = tk.StringVar(value="Not authenticated")
        tk.Label(parent, textvariable=self.auth_status_var, 
                foreground="red").pack()
    
    def setup_strategy_tab(self, parent):
        """Setup strategy configuration tab."""
        # Calendar Spread Parameters
        tk.Label(parent, text="Calendar Spread Parameters", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        param_frame = tk.Frame(parent)
        param_frame.pack(pady=10)
        
        # Lookback Period
        tk.Label(param_frame, text="Lookback Period (days):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.lookback_var = tk.StringVar(value="200")
        tk.Entry(param_frame, textvariable=self.lookback_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Standard Deviation Multiplier
        tk.Label(param_frame, text="Std Dev Multiplier:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.std_dev_var = tk.StringVar(value="1.0")
        tk.Entry(param_frame, textvariable=self.std_dev_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Chandelier Exit Parameters
        tk.Label(parent, text="Chandelier Exit Parameters", 
                font=("Arial", 12, "bold")).pack(pady=(20, 10))
        
        ce_frame = tk.Frame(parent)
        ce_frame.pack(pady=10)
        
        # ATR Period
        tk.Label(ce_frame, text="ATR Period:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.atr_period_var = tk.StringVar(value="22")
        tk.Entry(ce_frame, textvariable=self.atr_period_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # ATR Multiplier
        tk.Label(ce_frame, text="ATR Multiplier:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.atr_mult_var = tk.StringVar(value="3.0")
        tk.Entry(ce_frame, textvariable=self.atr_mult_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Instrument Selection
        tk.Label(parent, text="MCX Instruments", 
                font=("Arial", 12, "bold")).pack(pady=(20, 10))
        
        # MCX Instrument Selection
        instrument_frame = tk.LabelFrame(parent, text="MCX Futures Instrument Selection", padx=10, pady=10)
        instrument_frame.pack(fill=tk.X, padx=10, pady=(10, 10))
        
        # Control frame for fetch button
        control_frame = tk.Frame(instrument_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(control_frame, text="📥 Fetch MCX Instruments", 
                 command=self.fetch_mcx_instruments,
                 bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.instrument_status_var = tk.StringVar(value="Not loaded yet")
        tk.Label(control_frame, textvariable=self.instrument_status_var).pack(side=tk.LEFT, padx=20)
        
        # Dropdown frame
        dropdown_frame = tk.Frame(instrument_frame)
        dropdown_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Commodity selection
        tk.Label(dropdown_frame, text="Select Commodity:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.commodity_var = tk.StringVar()
        self.commodity_combo = ttk.Combobox(dropdown_frame, textvariable=self.commodity_var, 
                                          state="disabled", width=20)
        self.commodity_combo.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.commodity_combo.bind('<<ComboboxSelected>>', self.on_commodity_selected)
        
        # Near Month Expiry
        tk.Label(dropdown_frame, text="Near Month Expiry:").grid(row=0, column=1, padx=(30,5), pady=5, sticky=tk.W)
        self.near_expiry_var = tk.StringVar()
        self.near_expiry_combo = ttk.Combobox(dropdown_frame, textvariable=self.near_expiry_var, 
                                            state="disabled", width=20)
        self.near_expiry_combo.grid(row=1, column=1, padx=(30,5), pady=5, sticky=tk.W)
        self.near_expiry_combo.bind('<<ComboboxSelected>>', lambda e: self.update_symbol_display())
        
        # Next Month Expiry
        tk.Label(dropdown_frame, text="Next Month Expiry:").grid(row=0, column=2, padx=(30,5), pady=5, sticky=tk.W)
        self.next_expiry_var = tk.StringVar()
        self.next_expiry_combo = ttk.Combobox(dropdown_frame, textvariable=self.next_expiry_var, 
                                            state="disabled", width=20)
        self.next_expiry_combo.grid(row=1, column=2, padx=(30,5), pady=5, sticky=tk.W)
        self.next_expiry_combo.bind('<<ComboboxSelected>>', lambda e: self.update_symbol_display())
        
        # Symbol display frame
        symbol_frame = tk.Frame(instrument_frame)
        symbol_frame.pack(fill=tk.X, padx=5, pady=10)
        
        tk.Label(symbol_frame, text="Near Month Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.near_symbol_var = tk.StringVar()
        tk.Entry(symbol_frame, textvariable=self.near_symbol_var, width=25, state="readonly").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        tk.Label(symbol_frame, text="Next Month Symbol:").grid(row=0, column=2, padx=(30,5), pady=5, sticky=tk.W)
        self.next_symbol_var = tk.StringVar()
        tk.Entry(symbol_frame, textvariable=self.next_symbol_var, width=25, state="readonly").grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Update Strategy Button
        tk.Button(parent, text="⚙️ Update Strategy Parameters", 
                 command=self.update_strategy_params, bg="#4CAF50", fg="white").pack(pady=20)
    
    def setup_trading_tab(self, parent):
        """Setup trading execution tab with real-time updates."""
        # Real-time Spread Monitor
        realtime_frame = tk.LabelFrame(parent, text="🔄 Real-time Market Data", padx=10, pady=10)
        realtime_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create a frame for the market data table
        market_data_frame = tk.Frame(realtime_frame)
        market_data_frame.pack(fill=tk.X, pady=5)
        
        # Table header
        headers = ["Instrument", "Last Price", "Bid Price", "Ask Price", "Bid Qty", "Ask Qty", "Spread", "Timestamp"]
        for i, header in enumerate(headers):
            tk.Label(market_data_frame, text=header, font=("Arial", 9, "bold"), 
                    bg="#e0e0e0", width=12, relief=tk.RAISED).grid(row=0, column=i, padx=1, pady=1, sticky="nsew")
        
        # Near Month Data Row
        tk.Label(market_data_frame, text="Near Month", font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=1, column=0, padx=1, pady=1, sticky="nsew")
        
        self.near_last_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.near_last_var, font=("Arial", 9), 
                bg="#f5f5f5", fg="#2196F3", width=12, relief=tk.SUNKEN).grid(row=1, column=1, padx=1, pady=1, sticky="nsew")
        
        self.near_bid_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.near_bid_var, font=("Arial", 9), 
                bg="#f5f5f5", fg="#4CAF50", width=12, relief=tk.SUNKEN).grid(row=1, column=2, padx=1, pady=1, sticky="nsew")
        
        self.near_ask_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.near_ask_var, font=("Arial", 9), 
                bg="#f5f5f5", fg="#F44336", width=12, relief=tk.SUNKEN).grid(row=1, column=3, padx=1, pady=1, sticky="nsew")
        
        self.near_bid_qty_var = tk.StringVar(value="0")
        tk.Label(market_data_frame, textvariable=self.near_bid_qty_var, font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=1, column=4, padx=1, pady=1, sticky="nsew")
        
        self.near_ask_qty_var = tk.StringVar(value="0")
        tk.Label(market_data_frame, textvariable=self.near_ask_qty_var, font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=1, column=5, padx=1, pady=1, sticky="nsew")
        
        self.near_spread_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.near_spread_var, font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=1, column=6, padx=1, pady=1, sticky="nsew")
        
        self.near_time_var = tk.StringVar(value="--:--:--")
        tk.Label(market_data_frame, textvariable=self.near_time_var, font=("Arial", 8), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=1, column=7, padx=1, pady=1, sticky="nsew")
        
        # Next Month Data Row
        tk.Label(market_data_frame, text="Next Month", font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=2, column=0, padx=1, pady=1, sticky="nsew")
        
        self.next_last_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.next_last_var, font=("Arial", 9), 
                bg="#f5f5f5", fg="#2196F3", width=12, relief=tk.SUNKEN).grid(row=2, column=1, padx=1, pady=1, sticky="nsew")
        
        self.next_bid_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.next_bid_var, font=("Arial", 9), 
                bg="#f5f5f5", fg="#4CAF50", width=12, relief=tk.SUNKEN).grid(row=2, column=2, padx=1, pady=1, sticky="nsew")
        
        self.next_ask_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.next_ask_var, font=("Arial", 9), 
                bg="#f5f5f5", fg="#F44336", width=12, relief=tk.SUNKEN).grid(row=2, column=3, padx=1, pady=1, sticky="nsew")
        
        self.next_bid_qty_var = tk.StringVar(value="0")
        tk.Label(market_data_frame, textvariable=self.next_bid_qty_var, font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=2, column=4, padx=1, pady=1, sticky="nsew")
        
        self.next_ask_qty_var = tk.StringVar(value="0")
        tk.Label(market_data_frame, textvariable=self.next_ask_qty_var, font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=2, column=5, padx=1, pady=1, sticky="nsew")
        
        self.next_spread_var = tk.StringVar(value="0.00")
        tk.Label(market_data_frame, textvariable=self.next_spread_var, font=("Arial", 9), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=2, column=6, padx=1, pady=1, sticky="nsew")
        
        self.next_time_var = tk.StringVar(value="--:--:--")
        tk.Label(market_data_frame, textvariable=self.next_time_var, font=("Arial", 8), 
                bg="#f5f5f5", width=12, relief=tk.SUNKEN).grid(row=2, column=7, padx=1, pady=1, sticky="nsew")
        
        # Configure grid weights
        for i in range(8):
            market_data_frame.grid_columnconfigure(i, weight=1)
        
        # Calendar Spread Summary
        spread_summary_frame = tk.LabelFrame(realtime_frame, text="📊 Calendar Spread Summary", padx=10, pady=10)
        spread_summary_frame.pack(fill=tk.X, pady=10)
        
        spread_grid = tk.Frame(spread_summary_frame)
        spread_grid.pack(fill=tk.X)
        
        # Real-time spread
        tk.Label(spread_grid, text="Real-time Spread:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.realtime_spread_var = tk.StringVar(value="0.00")
        self.realtime_spread_label = tk.Label(spread_grid, textvariable=self.realtime_spread_var, 
                                             font=("Arial", 14, "bold"), fg="#2196F3")
        self.realtime_spread_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Signal
        tk.Label(spread_grid, text="Signal:", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=(20,5), pady=5, sticky=tk.W)
        self.realtime_signal_var = tk.StringVar(value="HOLD")
        self.realtime_signal_label = tk.Label(spread_grid, textvariable=self.realtime_signal_var, 
                                             font=("Arial", 14, "bold"), fg="#ffffff", bg="#9E9E9E")
        self.realtime_signal_label.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Upper Band
        tk.Label(spread_grid, text="Upper Band:", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.upper_band_var = tk.StringVar(value="0.00")
        tk.Label(spread_grid, textvariable=self.upper_band_var, font=("Arial", 10), fg="#F44336").grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Lower Band
        tk.Label(spread_grid, text="Lower Band:", font=("Arial", 10)).grid(row=1, column=2, padx=(20,5), pady=2, sticky=tk.W)
        self.lower_band_var = tk.StringVar(value="0.00")
        tk.Label(spread_grid, textvariable=self.lower_band_var, font=("Arial", 10), fg="#4CAF50").grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        
        # Mean
        tk.Label(spread_grid, text="Mean Spread:", font=("Arial", 10)).grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.mean_spread_var = tk.StringVar(value="0.00")
        tk.Label(spread_grid, textvariable=self.mean_spread_var, font=("Arial", 10), fg="#FF9800").grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Control buttons for real-time
        realtime_control_frame = tk.Frame(realtime_frame)
        realtime_control_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(realtime_control_frame, text="▶️ Start Real-time", 
                 command=self.start_realtime_data, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(realtime_control_frame, text="⏸️ Pause Real-time", 
                 command=self.pause_realtime_data, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(realtime_control_frame, text="🔄 Refresh Quotes", 
                 command=self.refresh_quotes, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Spread Analysis Frame
        analysis_frame = tk.LabelFrame(parent, text="📊 Historical Spread Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Spread Statistics
        stats_frame = tk.Frame(analysis_frame)
        stats_frame.pack(pady=10)
        
        self.spread_stats_vars = {}
        stats_labels = [
            ("Current Spread:", "current_spread"),
            ("Mean:", "mean"),
            ("Std Dev:", "std"),
            ("Upper Band:", "upper_band"),
            ("Lower Band:", "lower_band"),
            ("Signal:", "signal")
        ]
        
        for i, (label, key) in enumerate(stats_labels):
            tk.Label(stats_frame, text=label, font=("Arial", 10)).grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
            self.spread_stats_vars[key] = tk.StringVar(value="N/A")
            tk.Label(stats_frame, textvariable=self.spread_stats_vars[key], 
                    font=("Arial", 10, "bold"), width=20).grid(row=i, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Action Frame
        action_frame = tk.LabelFrame(parent, text="🎯 Trading Action", padx=10, pady=10)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.action_text = tk.Text(action_frame, height=4, width=80)
        self.action_text.pack(padx=10, pady=10)
        self.action_text.insert(tk.END, "No action recommended yet.")
        self.action_text.config(state=tk.DISABLED)
        
        # Buttons Frame
        button_frame = tk.Frame(parent)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="📊 Analyze Spread", 
                 command=self.analyze_spread, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="⚡ Execute Trade", 
                 command=self.execute_trade, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📋 Get Positions", 
                 command=self.get_positions, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=5)
    
    def setup_auto_trading_tab(self, parent):
        """Setup auto trading tab."""
        # Initialize auto trade manager if we have API
        if self.zerodha_api and self.instrument_manager:
            self.auto_trade_manager = AutoTradeManager(
                self.zerodha_api, 
                self.instrument_manager, 
                self.strategy,
                log_callback=self.log_message
            )
            # Start the auto trade manager
            self.auto_trade_manager.start()
            
            # Create control panel
            self.auto_trade_controls = AutoTradeControlPanel(parent, self.auto_trade_manager)
        else:
            tk.Label(parent, text="Please authenticate and fetch instruments first", 
                    font=("Arial", 12), fg="red").pack(pady=50)
            # Create a button to retry initialization
            tk.Button(parent, text="🔄 Initialize Auto Trading", 
                     command=lambda: self.initialize_auto_trading(parent),
                     bg="#2196F3", fg="white").pack(pady=20)
    
    def initialize_auto_trading(self, parent):
        """Initialize auto trading after authentication."""
        if self.zerodha_api and self.instrument_manager:
            # Clear existing widgets
            for widget in parent.winfo_children():
                widget.destroy()
            
            # Initialize auto trade manager
            self.auto_trade_manager = AutoTradeManager(
                self.zerodha_api, 
                self.instrument_manager, 
                self.strategy,
                log_callback=self.log_message
            )
            self.auto_trade_manager.start()
            
            # Create control panel
            self.auto_trade_controls = AutoTradeControlPanel(parent, self.auto_trade_manager)
            self.log_message("✅ Auto trading initialized successfully")
        else:
            messagebox.showerror("Error", 
                               "Please authenticate and fetch instruments first.\n"
                               "1. Go to Zerodha Login tab and authenticate\n"
                               "2. Go to Strategy Setup tab and fetch MCX instruments")
    
    def setup_positions_tab(self, parent):
        """Setup positions tab."""
        self.positions_text = scrolledtext.ScrolledText(parent, height=20, width=100)
        self.positions_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.positions_text.insert(tk.END, "No positions data yet.")
        self.positions_text.config(state=tk.DISABLED)
        
        # Refresh button
        tk.Button(parent, text="🔄 Refresh Positions", 
                 command=self.refresh_positions, bg="#2196F3", fg="white").pack(pady=5)
    
    def setup_log_tab(self, parent):
        """Setup logging tab."""
        self.log_text = scrolledtext.ScrolledText(parent, height=30, width=100)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Clear Logs Button
        tk.Button(parent, text="🗑️ Clear Logs", 
                 command=lambda: self.log_text.delete(1.0, tk.END), bg="#f44336", fg="white").pack(pady=5)
    
    def log_message(self, message):
        """Add message to log tab."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.status_var.set(message)
    
    def start_realtime_updates(self):
        """Start the 1-second real-time update loop."""
        self.update_realtime_spread()
        self.root.after(1000, self.start_realtime_updates)
    
    def update_realtime_spread(self):
        """Update real-time spread display and check auto trading."""
        if self.near_month_data and self.next_month_data:
            # Update displays first
            current_spread = self.next_month_data.last_price - self.near_month_data.last_price
            self.realtime_spread = current_spread
            
            # Update near month data display
            self.near_last_var.set(f"{self.near_month_data.last_price:.2f}")
            self.near_bid_var.set(f"{self.near_month_data.bid_price:.2f}")
            self.near_ask_var.set(f"{self.near_month_data.ask_price:.2f}")
            self.near_bid_qty_var.set(f"{self.near_month_data.bid_quantity:,}")
            self.near_ask_qty_var.set(f"{self.near_month_data.ask_quantity:,}")
            self.near_spread_var.set(f"{self.near_month_data.get_spread():.2f}")
            if self.near_month_data.timestamp:
                self.near_time_var.set(self.near_month_data.timestamp.strftime("%H:%M:%S"))
            
            # Update next month data display
            self.next_last_var.set(f"{self.next_month_data.last_price:.2f}")
            self.next_bid_var.set(f"{self.next_month_data.bid_price:.2f}")
            self.next_ask_var.set(f"{self.next_month_data.ask_price:.2f}")
            self.next_bid_qty_var.set(f"{self.next_month_data.bid_quantity:,}")
            self.next_ask_qty_var.set(f"{self.next_month_data.ask_quantity:,}")
            self.next_spread_var.set(f"{self.next_month_data.get_spread():.2f}")
            if self.next_month_data.timestamp:
                self.next_time_var.set(self.next_month_data.timestamp.strftime("%H:%M:%S"))
            
            # Update spread display
            self.realtime_spread_var.set(f"{current_spread:.2f}")
            
            # Update active trades P&L
            if self.auto_trade_manager:
                self.auto_trade_manager.update_active_trades_pnl(
                    self.near_month_data.last_price,
                    self.next_month_data.last_price
                )
            
            # Calculate real-time signal if we have bands
            if self.strategy.upper_band is not None and self.strategy.lower_band is not None:
                realtime_stats = self.strategy.calculate_realtime_spread(
                    self.near_month_data.last_price, self.next_month_data.last_price
                )
                
                if realtime_stats:
                    signal = realtime_stats['signal']
                    self.realtime_signal_var.set(signal)
                    
                    # Update color based on signal
                    if signal == "BUY_SPREAD":
                        self.realtime_signal_label.config(bg="#4CAF50")
                    elif signal == "SELL_SPREAD":
                        self.realtime_signal_label.config(bg="#F44336")
                    else:
                        self.realtime_signal_label.config(bg="#9E9E9E")
                    
                    # Update band displays
                    self.upper_band_var.set(f"{self.strategy.upper_band:.2f}")
                    self.lower_band_var.set(f"{self.strategy.lower_band:.2f}")
                    self.mean_spread_var.set(f"{self.strategy.spread_mean:.2f}")
                    
                    # Check auto trading conditions
                    self.check_auto_trading(realtime_stats)
    
    def check_auto_trading(self, realtime_stats):
        """Check and execute auto trading."""
        if not self.auto_trade_manager:
            return
        
        # Check for entry signal
        if (realtime_stats['action'] and 
            self.auto_trade_manager.auto_entry_enabled and
            self.near_month_data and self.next_month_data):
            
            near_symbol = self.near_symbol_var.get()
            next_symbol = self.next_symbol_var.get()
            
            if near_symbol and next_symbol:
                # Check auto entry
                entry_triggered = self.auto_trade_manager.check_entry_signal(
                    realtime_stats,
                    near_symbol,
                    next_symbol,
                    self.near_month_data.last_price,
                    self.next_month_data.last_price
                )
                
                if entry_triggered:
                    self.log_message(f"🤖 Auto entry triggered: {realtime_stats['signal']}")
                    # Show notification
                    self.root.after(100, lambda: messagebox.showinfo("Auto Entry", 
                                  f"Auto entry triggered!\n{realtime_stats['signal']}\n"
                                  f"Near: {near_symbol}\nNext: {next_symbol}"))
        
        # Check exit conditions for all active trades
        if (self.auto_trade_manager.auto_exit_enabled and 
            self.near_month_data and self.next_month_data):
            
            exits = self.auto_trade_manager.check_exit_conditions(
                self.near_month_data.last_price,
                self.next_month_data.last_price
            )
            
            for exit_trade in exits:
                self.log_message(f"🤖 Auto exit triggered: {exit_trade['exit_reason']}")
                self.log_message(f"  P&L: {exit_trade['pnl']:.2f} ({exit_trade['pnl_pct']:.1f}%)")
                # Show notification
                self.root.after(100, lambda t=exit_trade: messagebox.showinfo("Auto Exit", 
                              f"Auto exit triggered!\n{t['exit_reason']}\n"
                              f"P&L: {t['pnl']:.2f} ({t['pnl_pct']:.1f}%)"))
    
    def start_realtime_data(self):
        """Start real-time WebSocket data."""
        if not self.zerodha_api or not self.zerodha_api.access_token:
            messagebox.showerror("Error", "Please authenticate first")
            return
        
        if not self.near_symbol_var.get() or not self.next_symbol_var.get():
            messagebox.showerror("Error", "Please select instruments first")
            return
        
        try:
            # Get instrument tokens
            near_token = self.instrument_manager.get_instrument_token(self.near_symbol_var.get())
            next_token = self.instrument_manager.get_instrument_token(self.next_symbol_var.get())
            
            if not near_token or not next_token:
                messagebox.showerror("Error", "Could not get instrument tokens")
                return
            
            # Initialize price data objects
            self.near_month_data = InstrumentPriceData(self.near_symbol_var.get(), near_token)
            self.next_month_data = InstrumentPriceData(self.next_symbol_var.get(), next_token)
            
            # Get initial quotes
            self.refresh_quotes()
            
            # Start WebSocket
            success, message = self.zerodha_api.start_websocket(
                on_ticks=self.on_websocket_ticks,
                on_connect=lambda ws, response: self.on_websocket_connect(ws, response, [near_token, next_token]),
                on_close=lambda ws, code, reason: self.on_websocket_close(ws, code, reason)
            )
            
            if success:
                self.is_realtime_running = True
                self.log_message("✅ Real-time WebSocket started")
                self.log_message(f"  Subscribed to: {self.near_symbol_var.get()}, {self.next_symbol_var.get()}")
            else:
                self.log_message(f"❌ Failed to start WebSocket: {message}")
                
        except Exception as e:
            self.log_message(f"❌ Error starting real-time data: {str(e)}")
    
    def refresh_quotes(self):
        """Refresh quotes for selected instruments."""
        if not self.zerodha_api or not self.zerodha_api.access_token:
            return
        
        if not self.near_symbol_var.get() or not self.next_symbol_var.get():
            return
        
        try:
            # Get instrument tokens
            near_token = self.instrument_manager.get_instrument_token(self.near_symbol_var.get())
            next_token = self.instrument_manager.get_instrument_token(self.next_symbol_var.get())
            
            if near_token:
                quote_data = self.zerodha_api.get_quote(near_token)
                if quote_data and self.near_month_data:
                    # Update near month data
                    tick = {
                        'last_price': quote_data.get('last_price', 0),
                        'ohlc': quote_data.get('ohlc', {}),
                        'volume': quote_data.get('volume_traded', 0),
                        'depth': quote_data.get('depth', {})
                    }
                    self.near_month_data.update_from_tick(tick)
            
            if next_token:
                quote_data = self.zerodha_api.get_quote(next_token)
                if quote_data and self.next_month_data:
                    # Update next month data
                    tick = {
                        'last_price': quote_data.get('last_price', 0),
                        'ohlc': quote_data.get('ohlc', {}),
                        'volume': quote_data.get('volume_traded', 0),
                        'depth': quote_data.get('depth', {})
                    }
                    self.next_month_data.update_from_tick(tick)
            
            self.log_message("✅ Quotes refreshed")
            
        except Exception as e:
            self.log_message(f"❌ Error refreshing quotes: {str(e)}")
    
    def pause_realtime_data(self):
        """Pause real-time WebSocket data."""
        if self.zerodha_api:
            self.zerodha_api.stop_websocket()
            self.is_realtime_running = False
            self.log_message("⏸️ Real-time WebSocket paused")
    
    def on_websocket_ticks(self, ws, ticks):
        """Handle WebSocket ticks."""
        for tick in ticks:
            if 'instrument_token' in tick:
                instrument_token = tick['instrument_token']
                
                # Update near month data
                if self.near_month_data and instrument_token == self.near_month_data.instrument_token:
                    self.near_month_data.update_from_tick(tick)
                
                # Update next month data
                elif self.next_month_data and instrument_token == self.next_month_data.instrument_token:
                    self.next_month_data.update_from_tick(tick)
    
    def on_websocket_connect(self, ws, response, tokens):
        """Handle WebSocket connection."""
        self.zerodha_api.subscribe_instruments(tokens)
        self.log_message(f"✅ WebSocket connected, subscribed to {len(tokens)} instruments")
    
    def on_websocket_close(self, ws, code, reason):
        """Handle WebSocket close."""
        self.log_message(f"🔌 WebSocket closed: {reason}")
        self.is_realtime_running = False
    
    def generate_login_url(self):
        """Generate Zerodha login URL."""
        api_key = self.api_key_var.get().strip()
        api_secret = self.api_secret_var.get().strip()
        
        if not api_key or not api_secret:
            messagebox.showerror("Error", "Please enter API Key and Secret")
            return
        
        try:
            self.zerodha_api = ZerodhaTradingAPI(api_key, api_secret)
            login_url = self.zerodha_api.get_login_url()
            self.login_url_var.set(login_url)
            self.log_message(f"Login URL generated. Please visit: {login_url}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate login URL: {str(e)}")
    
    def authenticate(self):
        """Authenticate with Zerodha."""
        if not self.zerodha_api:
            messagebox.showerror("Error", "Please generate login URL first")
            return
        
        request_token = self.request_token_var.get().strip()
        if not request_token:
            messagebox.showerror("Error", "Please enter request token from redirect URL")
            return
        
        success, message = self.zerodha_api.set_access_token(request_token)
        
        if success:
            self.auth_status_var.set("Authenticated")
            self.log_message("Authentication successful")
            self.save_config(self.api_key_var.get(), self.zerodha_api.access_token)
        else:
            self.auth_status_var.set("Authentication failed")
            self.log_message(f"Authentication failed: {message}")
    
    def auto_login(self):
        """Attempt auto-login using saved credentials."""
        if not self.saved_api_key or not self.saved_access_token:
            messagebox.showinfo("Info", "No saved credentials found. Please authenticate first.")
            return
        
        try:
            self.zerodha_api = ZerodhaTradingAPI(self.saved_api_key, "dummy_secret")
            self.zerodha_api.access_token = self.saved_access_token
            self.zerodha_api.kite.set_access_token(self.saved_access_token)
            
            profile = self.zerodha_api.kite.profile()
            self.auth_status_var.set(f"Auto-login: {profile['user_name']}")
            self.log_message(f"Auto-login successful as {profile['user_name']}")
            self.api_key_var.set(self.saved_api_key)
            
        except Exception as e:
            messagebox.showerror("Auto-Login Failed", 
                               f"Auto-login failed. Please authenticate manually.\nError: {str(e)}")
            self.log_message(f"Auto-login failed: {str(e)}")
    
    def fetch_mcx_instruments(self):
        """Fetch MCX instruments."""
        if not self.zerodha_api or not self.zerodha_api.access_token:
            messagebox.showerror("Error", "Please authenticate first")
            return

        try:
            self.instrument_status_var.set("Fetching instruments...")
            self.instrument_manager = MCXInstrumentManager(self.zerodha_api.kite)
            success, message = self.instrument_manager.fetch_instruments()

            if success:
                commodities = self.instrument_manager.get_commodity_list()
                if commodities:
                    self.commodity_combo['values'] = commodities
                    self.commodity_combo.config(state="readonly")
                    self.near_expiry_combo.config(state="readonly")
                    self.next_expiry_combo.config(state="readonly")
                    
                    for commodity in ['GOLD', 'SILVER', 'CRUDEOIL', 'NATURALGAS']:
                        if commodity in commodities:
                            self.commodity_var.set(commodity)
                            self.on_commodity_selected()
                            break
                    
                    self.instrument_status_var.set(f"Loaded {len(commodities)} commodities")
                    self.log_message(f"MCX instruments fetched: {len(commodities)} commodities")
                else:
                    self.instrument_status_var.set("No commodities found")
            else:
                self.instrument_status_var.set("Fetch failed")
                messagebox.showerror("Error", message)

        except Exception as e:
            self.instrument_status_var.set("Error occurred")
            self.log_message(f"Error fetching instruments: {str(e)}")
    
    def on_commodity_selected(self, event=None):
        """When commodity is selected, populate expiry dates."""
        commodity = self.commodity_var.get()
        if not commodity or not self.instrument_manager:
            return

        expiry_dates = self.instrument_manager.get_expiry_dates(commodity)
        if expiry_dates:
            formatted_dates = [date.strftime("%d-%b-%Y") for date in expiry_dates]
            self.near_expiry_combo['values'] = formatted_dates
            self.next_expiry_combo['values'] = formatted_dates
            
            if len(formatted_dates) >= 2:
                self.near_expiry_var.set(formatted_dates[0])
                self.next_expiry_var.set(formatted_dates[1])
                self.update_symbol_display()
        else:
            self.near_expiry_combo.set('')
            self.next_expiry_combo.set('')
    
    def update_symbol_display(self):
        """Update trading symbol display."""
        commodity = self.commodity_var.get()
        near_expiry_str = self.near_expiry_var.get()
        next_expiry_str = self.next_expiry_var.get()
        
        if not all([commodity, near_expiry_str, next_expiry_str]):
            return
        
        try:
            near_expiry = pd.to_datetime(near_expiry_str, format="%d-%b-%Y")
            next_expiry = pd.to_datetime(next_expiry_str, format="%d-%b-%Y")
            
            near_symbol = self.instrument_manager.get_tradingsymbol_by_expiry(commodity, near_expiry)
            next_symbol = self.instrument_manager.get_tradingsymbol_by_expiry(commodity, next_expiry)
            
            if near_symbol:
                self.near_symbol_var.set(near_symbol)
            if next_symbol:
                self.next_symbol_var.set(next_symbol)
            
            if near_symbol and next_symbol:
                self.log_message(f"Selected: {commodity} | Near: {near_symbol} | Next: {next_symbol}")
            
        except Exception as e:
            self.log_message(f"Error updating symbols: {str(e)}")
    
    def update_strategy_params(self):
        """Update strategy parameters."""
        try:
            self.strategy.lookback_period = int(self.lookback_var.get())
            self.strategy.std_dev_multiplier = float(self.std_dev_var.get())
            self.log_message(f"Strategy updated: Lookback={self.strategy.lookback_period}, "
                           f"StdDev Multiplier={self.strategy.std_dev_multiplier}")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
    
    def analyze_spread(self):
        """Analyze calendar spread with historical data."""
        if not self.zerodha_api or not self.zerodha_api.access_token:
            messagebox.showerror("Error", "Please authenticate first")
            return
        
        near_symbol = self.near_symbol_var.get()
        next_symbol = self.next_symbol_var.get()
        
        if not near_symbol or not next_symbol:
            messagebox.showerror("Error", "Please select instruments first")
            return
        
        try:
            self.log_message("Fetching historical data...")
            
            near_token = self.instrument_manager.get_instrument_token(near_symbol)
            next_token = self.instrument_manager.get_instrument_token(next_symbol)
            
            if not near_token or not next_token:
                messagebox.showerror("Error", "Could not get instrument tokens")
                return
            
            # Fetch historical data
            lookback_days = self.strategy.lookback_period
            interval = "day"
            to_date = datetime.now().date()
            from_date = to_date - timedelta(days=lookback_days * 3)
            
            near_data = self.zerodha_api.kite.historical_data(
                instrument_token=int(near_token),
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            next_data = self.zerodha_api.kite.historical_data(
                instrument_token=int(next_token),
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not near_data or not next_data:
                messagebox.showerror("Error", "Failed to fetch historical data")
                return
            
            near_df = pd.DataFrame(near_data)
            next_df = pd.DataFrame(next_data)
            
            near_df['date'] = pd.to_datetime(near_df['date'])
            next_df['date'] = pd.to_datetime(next_df['date'])
            
            merged = pd.merge(near_df[['date', 'close']], next_df[['date', 'close']], 
                             on='date', suffixes=('_near', '_next'))
            
            near_series = pd.Series(merged['close_near'].values, index=merged['date'])
            next_series = pd.Series(merged['close_next'].values, index=merged['date'])
            
            stats = self.strategy.calculate_spread_stats(near_series, next_series)
            
            for key, var in self.spread_stats_vars.items():
                if key in stats:
                    if isinstance(stats[key], float):
                        var.set(f"{stats[key]:.2f}")
                    else:
                        var.set(str(stats[key]))
            
            self.action_text.config(state=tk.NORMAL)
            self.action_text.delete(1.0, tk.END)
            
            if stats['action']:
                action = stats['action']
                action_text = f"📊 SPREAD ANALYSIS COMPLETE\n\n"
                action_text += f"Signal: {stats['signal']}\n"
                action_text += f"Recommended Action:\n"
                action_text += f"  • {near_symbol}: {action['near_month_action']}\n"
                action_text += f"  • {next_symbol}: {action['next_month_action']}\n"
                action_text += f"Reason: {action['reason']}\n\n"
                action_text += f"Statistics:\n"
                action_text += f"• Current Spread: {stats['current_spread']:.2f}\n"
                action_text += f"• Mean: {stats['mean']:.2f}\n"
                action_text += f"• Std Dev: {stats['std']:.2f}\n"
                action_text += f"• Bands: [{stats['lower_band']:.2f}, {stats['upper_band']:.2f}]"
                
                self.log_message(f"Spread analysis: {stats['signal']} signal")
            else:
                action_text = "No trade signal. Spread within normal range."
                self.log_message("Spread analysis: No trading signal")
            
            self.action_text.insert(tk.END, action_text)
            self.action_text.config(state=tk.DISABLED)
            
        except Exception as e:
            error_msg = f"Spread analysis failed: {str(e)}"
            self.log_message(f"Spread analysis error: {str(e)}")
            messagebox.showerror("Error", error_msg)
    
    def execute_trade(self):
        """Execute trade from main button."""
        if not self.zerodha_api or not self.zerodha_api.access_token:
            messagebox.showerror("Error", "Please authenticate first")
            return
        
        near_symbol = self.near_symbol_var.get()
        next_symbol = self.next_symbol_var.get()
        
        if not near_symbol or not next_symbol:
            messagebox.showerror("Error", "Please select instruments first")
            return
        
        # Check if we have a signal
        signal_text = self.action_text.get(1.0, tk.END).strip()
        if "No trade signal" in signal_text or "HOLD" in signal_text:
            messagebox.showinfo("Info", "No trade signal to execute")
            return
        
        # Determine action based on signal
        if "BUY_SPREAD" in signal_text:
            action = {'near_month_action': 'BUY', 'next_month_action': 'SELL'}
        elif "SELL_SPREAD" in signal_text:
            action = {'near_month_action': 'SELL', 'next_month_action': 'BUY'}
        else:
            messagebox.showinfo("Info", "Cannot determine trade action")
            return
        
        # Get current prices for confirmation
        near_price = self.near_month_data.last_price if self.near_month_data else 0
        next_price = self.next_month_data.last_price if self.next_month_data else 0
        current_spread = next_price - near_price
        
        confirm = messagebox.askyesno("Confirm Trade", 
                                     f"Execute Calendar Spread Trade?\n\n"
                                     f"Near Month ({near_symbol}):\n"
                                     f"  Action: {action['near_month_action']}\n"
                                     f"  Last Price: {near_price:.2f}\n"
                                     f"  Bid: {self.near_month_data.bid_price if self.near_month_data else 0:.2f}\n"
                                     f"  Ask: {self.near_month_data.ask_price if self.near_month_data else 0:.2f}\n\n"
                                     f"Next Month ({next_symbol}):\n"
                                     f"  Action: {action['next_month_action']}\n"
                                     f"  Last Price: {next_price:.2f}\n"
                                     f"  Bid: {self.next_month_data.bid_price if self.next_month_data else 0:.2f}\n"
                                     f"  Ask: {self.next_month_data.ask_price if self.next_month_data else 0:.2f}\n\n"
                                     f"Current Spread: {current_spread:.2f}")
        
        if confirm:
            try:
                success, result = self.zerodha_api.place_calendar_spread_order(
                    near_symbol, next_symbol, action
                )
                
                if success:
                    # If auto trade manager exists, add to its tracking
                    if self.auto_trade_manager:
                        trade = {
                            'id': len(self.auto_trade_manager.get_active_trades()) + 1,
                            'status': TradeStatus.ENTERED,
                            'entry_time': datetime.now(),
                            'direction': TradeDirection("BUY_SPREAD" if action['near_month_action'] == 'BUY' else "SELL_SPREAD"),
                            'near_symbol': near_symbol,
                            'next_symbol': next_symbol,
                            'entry_near_price': near_price,
                            'entry_next_price': next_price,
                            'entry_spread': current_spread,
                            'reason': "Manual entry"
                        }
                        self.auto_trade_manager.active_trades.append(trade)
                    
                    self.log_message(f"Trade executed successfully! Orders: {result}")
                    messagebox.showinfo("Success", "Trade executed successfully!")
                else:
                    self.log_message(f"Trade execution failed: {result}")
                    messagebox.showerror("Error", f"Trade failed: {result}")
                    
            except Exception as e:
                error_msg = f"Trade execution error: {str(e)}"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
    
    def get_positions(self):
        """Fetch current positions."""
        if not self.zerodha_api or not self.zerodha_api.access_token:
            messagebox.showerror("Error", "Please authenticate first")
            return
        
        try:
            positions = self.zerodha_api.kite.positions()
            self.log_message("Positions fetched successfully")
            
            self.positions_text.config(state=tk.NORMAL)
            self.positions_text.delete(1.0, tk.END)
            
            positions_text = "=== CURRENT POSITIONS ===\n\n"
            
            for pos_type in ['day', 'net']:
                positions_text += f"{pos_type.upper()} Positions:\n"
                for pos in positions[pos_type]:
                    if pos['quantity'] != 0:
                        positions_text += f"  {pos['tradingsymbol']}: {pos['quantity']} units @ {pos['average_price']}\n"
                positions_text += "\n"
            
            # Add auto trades if available
            if self.auto_trade_manager:
                active_trades = self.auto_trade_manager.get_active_trades()
                if active_trades:
                    positions_text += "\n=== AUTO TRADES ===\n\n"
                    for trade in active_trades:
                        positions_text += f"  {trade['direction'].value}: {trade['near_symbol']}/{trade['next_symbol']}\n"
                        positions_text += f"    Entry: {trade['entry_time'].strftime('%H:%M:%S')} | Spread: {trade['entry_spread']:.2f}\n"
                        if 'current_pnl' in trade:
                            positions_text += f"    Current P&L: {trade['current_pnl']:.2f}\n"
            
            self.positions_text.insert(tk.END, positions_text)
            self.positions_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log_message(f"Error fetching positions: {str(e)}")
    
    def refresh_positions(self):
        """Refresh positions display."""
        self.get_positions()

# ==================== 10. MAIN EXECUTION ====================
def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    
    root = tk.Tk()
    
    # Set window icon and style
    root.iconbitmap(default='')  # Add icon path if available
    
    app = CalendarSpreadTradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()