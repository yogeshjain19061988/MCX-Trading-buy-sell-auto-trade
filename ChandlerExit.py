"""Advanced Zerodha Multi-Indicator Trading Dashboard
Features: Auto-login, Auto-token generation, P&L monitoring, Advanced strategy
Author: Trading Bot
Date: 2024
Updated: Token generation with API key/secret only
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import pickle
import webbrowser
import tempfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import mplfinance as mpf
from matplotlib import style
import talib
import warnings
from kiteconnect import KiteConnect, KiteTicker
from dateutil import parser
import csv
from enum import Enum
import schedule
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socket

warnings.filterwarnings('ignore')

# Define Trading Mode Enum
class TradingMode(Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"

class IndicatorsAdvanced:
    """Advanced Technical Indicators Calculation"""
    
    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        """Calculate Supertrend indicator with improved logic"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize arrays
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)
        
        # Vectorized calculation
        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
        
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        df['supertrend_upper'] = upper_band
        df['supertrend_lower'] = lower_band
        return df
    
    @staticmethod
    def vwap(df):
        """Calculate VWAP with intraday reset"""
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['typical_price_volume'] = df['typical_price'] * df['volume']
        
        # Reset cumulative at each new day
        df['date_group'] = df.index.date
        df['cum_typical_price_volume'] = df.groupby('date_group')['typical_price_volume'].cumsum()
        df['cum_volume'] = df.groupby('date_group')['volume'].cumsum()
        
        df['vwap'] = df['cum_typical_price_volume'] / df['cum_volume']
        df.drop(['date_group', 'typical_price', 'typical_price_volume', 
                'cum_typical_price_volume', 'cum_volume'], axis=1, inplace=True, errors='ignore')
        return df
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators"""
        # RSI with multiple timeframes
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # ADX with DMI
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Moving Averages
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Volume indicators
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Supertrend
        df = IndicatorsAdvanced.supertrend(df)
        
        # VWAP
        df = IndicatorsAdvanced.vwap(df)
        
        # Calculate derived signals
        df['price_vs_vwap'] = df['close'] / df['vwap'] - 1  # Percentage above/below VWAP
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['ema_crossover'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
        
        return df

class AdvancedTradingStrategy:
    """Advanced Trading Strategy with Position Management"""
    
    def __init__(self):
        self.positions = {}
        self.trades = []
        self.entry_prices = {}
        self.stop_losses = {}
        self.take_profits = {}
        self.max_positions = 3
        self.max_loss_per_trade = 0.02  # 2% max loss per trade
        self.take_profit_multiplier = 2  # 2:1 risk-reward ratio
        self.trailing_stop_activation = 0.03  # 3% profit to activate trailing stop
        self.trailing_stop_distance = 0.015  # 1.5% trailing distance
        
    def calculate_signal_strength(self, df):
        """Calculate signal strength with weighted scoring"""
        if len(df) == 0:
            return 0
        
        latest = df.iloc[-1]
        signal_strength = 0
        
        # 1. Supertrend (25 points)
        if latest['supertrend_direction'] == 1:
            signal_strength += 25
        
        # 2. Price vs VWAP (20 points)
        if latest['price_vs_vwap'] > 0.01:  # 1% above VWAP
            signal_strength += 20
        elif latest['price_vs_vwap'] > 0:
            signal_strength += 10
        
        # 3. RSI (15 points)
        if 30 < latest['rsi_14'] < 70:
            if latest['rsi_14'] > 55:
                signal_strength += 15
            elif latest['rsi_14'] < 45:
                signal_strength -= 15
        
        # 4. MACD (15 points)
        if latest['macd_crossover'] == 1:
            signal_strength += 15
        
        # 5. ADX & DMI (15 points)
        if latest['adx'] > 25:
            if latest['plus_di'] > latest['minus_di']:
                signal_strength += 15
            else:
                signal_strength -= 15
        
        # 6. Volume confirmation (10 points)
        if latest['volume'] > latest['volume_sma'] * 1.2:
            signal_strength += 10
        
        return signal_strength
    
    def generate_signals(self, df):
        """Generate buy/sell signals with advanced logic"""
        if len(df) < 50:
            return df
        
        df['signal'] = 0
        df['signal_strength'] = 0
        df['position_size'] = 0
        
        # Calculate rolling signals
        for i in range(30, len(df)):
            window_df = df.iloc[:i+1]
            signal_strength = self.calculate_signal_strength(window_df)
            
            # Get the last few candles for confirmation
            recent = window_df.iloc[-5:]
            
            # Check for trend confirmation
            trend_confirmed = all(recent['supertrend_direction'] == 1) if signal_strength > 0 else \
                             all(recent['supertrend_direction'] == -1) if signal_strength < 0 else False
            
            # Check for volume confirmation
            volume_confirmed = any(recent['volume'] > recent['volume_sma'] * 1.5)
            
            # Set signal
            if signal_strength >= 60 and trend_confirmed and volume_confirmed:
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_strength'] = signal_strength
                # Calculate position size based on signal strength
                df.loc[df.index[i], 'position_size'] = min(3, int(signal_strength / 20))
            
            elif signal_strength <= -60 and trend_confirmed:
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'signal_strength'] = signal_strength
        
        return df
    
    def calculate_position_sizing(self, capital, risk_per_trade, stop_loss_pct):
        """Calculate optimal position size"""
        risk_amount = capital * risk_per_trade
        position_value = risk_amount / stop_loss_pct
        return position_value
    
    def update_trailing_stop(self, symbol, current_price):
        """Update trailing stop loss"""
        if symbol in self.entry_prices and symbol in self.stop_losses:
            entry = self.entry_prices[symbol]
            current_stop = self.stop_losses[symbol]
            
            # Check if trailing stop should be activated
            profit_pct = (current_price - entry) / entry
            
            if profit_pct >= self.trailing_stop_activation:
                # Calculate new trailing stop
                new_stop = current_price * (1 - self.trailing_stop_distance)
                if new_stop > current_stop:
                    self.stop_losses[symbol] = new_stop
                    return True
        
        return False
    
    def check_exit_conditions(self, symbol, current_price, current_data):
        """Check if exit conditions are met"""
        if symbol not in self.entry_prices:
            return None
        
        entry = self.entry_prices[symbol]
        stop_loss = self.stop_losses.get(symbol)
        take_profit = self.take_profits.get(symbol)
        
        exit_reason = None
        
        # Check stop loss
        if stop_loss and current_price <= stop_loss:
            exit_reason = "STOP_LOSS"
        
        # Check take profit
        elif take_profit and current_price >= take_profit:
            exit_reason = "TAKE_PROFIT"
        
        # Check trailing stop
        elif self.update_trailing_stop(symbol, current_price):
            if current_price <= self.stop_losses[symbol]:
                exit_reason = "TRAILING_STOP"
        
        # Check indicator reversal
        elif symbol in self.positions and self.positions[symbol] == 'BUY':
            if current_data['supertrend_direction'] == -1 or current_data['rsi_14'] > 70:
                exit_reason = "INDICATOR_REVERSAL"
        
        return exit_reason
    
    def record_trade(self, symbol, action, quantity, price, reason=""):
        """Record trade details"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'reason': reason,
            'pnl': 0  # Will be calculated on exit
        }
        
        if action in ['BUY', 'LONG']:
            self.positions[symbol] = 'BUY'
            self.entry_prices[symbol] = price
            # Set initial stop loss
            self.stop_losses[symbol] = price * (1 - self.max_loss_per_trade)
            # Set take profit
            self.take_profits[symbol] = price * (1 + self.max_loss_per_trade * self.take_profit_multiplier)
        
        elif action in ['SELL', 'SHORT', 'EXIT']:
            if symbol in self.positions:
                # Calculate P&L
                entry_price = self.entry_prices.get(symbol, price)
                pnl = (price - entry_price) * quantity
                trade['pnl'] = pnl
                
                # Clear position
                if symbol in self.positions:
                    del self.positions[symbol]
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                if symbol in self.stop_losses:
                    del self.stop_losses[symbol]
                if symbol in self.take_profits:
                    del self.take_profits[symbol]
        
        self.trades.append(trade)
        return trade
    
    def get_total_pnl(self):
        """Calculate total P&L from all trades"""
        total = sum(trade.get('pnl', 0) for trade in self.trades)
        return total
    
    def get_open_positions_pnl(self, current_prices):
        """Calculate P&L for open positions"""
        open_pnl = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                current_price = current_prices[symbol]
                quantity = next((t['quantity'] for t in reversed(self.trades) 
                               if t['symbol'] == symbol and t['action'] in ['BUY', 'LONG']), 0)
                
                if position == 'BUY':
                    open_pnl += (current_price - entry_price) * quantity
        
        return open_pnl

class TokenRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for capturing OAuth redirect"""
    
    def __init__(self, request, client_address, server):
        self.server = server
        super().__init__(request, client_address, server)
    
    def do_GET(self):
        """Handle GET request - capture request token"""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        
        if 'request_token' in query:
            self.server.request_token = query['request_token'][0]
            self.server.got_token = True
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
            <head><title>Zerodha Token Generated</title></head>
            <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
                <h2 style="color: #4CAF50;">✓ Access Token Generated Successfully!</h2>
                <p>You can close this window and return to the trading application.</p>
                <p style="margin-top: 30px; color: #666;">
                    The token has been automatically captured and saved.
                </p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(400)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass

class TokenCaptureServer:
    """Server to capture OAuth redirect"""
    
    def __init__(self, port=8080):
        self.port = port
        self.request_token = None
        self.got_token = False
        self.server = None
    
    def start(self):
        """Start the server"""
        self.server = HTTPServer(('localhost', self.port), TokenRequestHandler)
        self.server.request_token = None
        self.server.got_token = False
        
        # Start server in a thread
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        return True
    
    def wait_for_token(self, timeout=60):
        """Wait for token to be captured"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.server and hasattr(self.server, 'got_token') and self.server.got_token:
                self.request_token = self.server.request_token
                return True
            time.sleep(0.5)
        return False
    
    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
    
    def get_token(self):
        """Get captured token"""
        return self.request_token

class ZerodhaTokenGenerator:
    """Token generation for Zerodha using API key and secret only"""
    
    def __init__(self):
        self.credentials_file = "zerodha_credentials.json"
        self.redirect_port = 8080
        self.redirect_url = f"http://localhost:{self.redirect_port}/"
        self.credentials = self.load_credentials()
        self.token_server = TokenCaptureServer(self.redirect_port)
    
    def load_credentials(self):
        """Load saved credentials"""
        try:
            if os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def save_credentials(self):
        """Save credentials to file"""
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(self.credentials, f, indent=2)
            return True
        except:
            return False
    
    def is_port_free(self, port):
        """Check if port is available"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            return True
        except socket.error:
            return False
    
    def generate_access_token(self, api_key, api_secret):
        """Generate access token using API key and secret only"""
        try:
            # Check if redirect port is available
            if not self.is_port_free(self.redirect_port):
                return False, f"Port {self.redirect_port} is not available. Please close any application using this port."
            
            # Initialize KiteConnect
            kite = KiteConnect(api_key=api_key)
            
            # Generate login URL
            login_url = kite.login_url()
            
            # Start token capture server
            if not self.token_server.start():
                return False, "Failed to start token capture server"
            
            # Open browser for login
            self.log_message("Opening browser for Zerodha login...", "INFO")
            webbrowser.open(login_url)
            
            # Wait for token (with timeout)
            self.log_message("Waiting for token... Please log in to Zerodha in the browser.", "INFO")
            
            if not self.token_server.wait_for_token(timeout=120):
                self.token_server.stop()
                return False, "Token generation timeout. Please try again."
            
            # Get request token
            request_token = self.token_server.get_token()
            
            if not request_token:
                self.token_server.stop()
                return False, "Failed to capture request token"
            
            # Generate session
            self.log_message("Generating session with request token...", "INFO")
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            # Save credentials
            self.credentials['api_key'] = api_key
            self.credentials['access_token'] = access_token
            self.credentials['request_token'] = request_token
            self.save_credentials()
            
            # Stop server
            self.token_server.stop()
            
            return True, access_token
            
        except Exception as e:
            # Stop server if running
            if hasattr(self, 'token_server'):
                self.token_server.stop()
            
            error_msg = str(e)
            if "invalid_request_token" in error_msg.lower():
                return False, "Invalid request token. Please generate a new token."
            elif "timeout" in error_msg.lower():
                return False, "Operation timed out. Please try again."
            else:
                return False, f"Token generation error: {error_msg}"
    
    def validate_token(self, api_key, access_token):
        """Validate if token is still valid"""
        try:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            
            # Try to fetch profile (simple API call to validate)
            profile = kite.profile()
            return True, "Token is valid"
        except Exception as e:
            return False, f"Token validation failed: {str(e)}"
    
    def log_message(self, message, level="INFO"):
        """Log message (can be overridden by GUI)"""
        print(f"[{level}] {message}")

class AdvancedZerodhaAPI:
    """Enhanced Zerodha API with token generation"""
    
    def __init__(self):
        self.kite = None
        self.kws = None
        self.token_generator = ZerodhaTokenGenerator()
        self.connected = False
        self.instruments = {}
        
        # Popular instruments
        self.popular_instruments = {
            'NIFTY 50': {'token': 256265, 'exchange': 'NSE', 'tradingsymbol': 'NIFTY 50'},
            'BANKNIFTY': {'token': 260105, 'exchange': 'NSE', 'tradingsymbol': 'BANKNIFTY'},
            'RELIANCE': {'token': 738561, 'exchange': 'NSE', 'tradingsymbol': 'RELIANCE'},
            'TCS': {'token': 2953217, 'exchange': 'NSE', 'tradingsymbol': 'TCS'},
            'INFY': {'token': 408065, 'exchange': 'NSE', 'tradingsymbol': 'INFY'},
            'HDFCBANK': {'token': 341249, 'exchange': 'NSE', 'tradingsymbol': 'HDFCBANK'},
            'ICICIBANK': {'token': 1270529, 'exchange': 'NSE', 'tradingsymbol': 'ICICIBANK'},
            'SBIN': {'token': 779521, 'exchange': 'NSE', 'tradingsymbol': 'SBIN'},
            'WIPRO': {'token': 969473, 'exchange': 'NSE', 'tradingsymbol': 'WIPRO'},
            'HINDUNILVR': {'token': 356865, 'exchange': 'NSE', 'tradingsymbol': 'HINDUNILVR'},
        }
        
        self.positions = {}
        self.holdings = {}
        self.margins = {}
        self.order_history = []
        
    def connect(self, api_key=None, access_token=None, auto_generate=False):
        """Connect to Zerodha with optional auto token generation"""
        try:
            if auto_generate and api_key:
                # Auto-generate token
                api_secret = self.token_generator.credentials.get('api_secret')
                if not api_secret:
                    return False, "API secret required for auto-generation"
                
                success, message = self.token_generator.generate_access_token(api_key, api_secret)
                if not success:
                    return False, f"Token generation failed: {message}"
                
                # Get generated token
                access_token = self.token_generator.credentials.get('access_token')
            
            # Connect with API
            if api_key and access_token:
                # Validate token first
                valid, msg = self.token_generator.validate_token(api_key, access_token)
                if not valid:
                    return False, f"Invalid token: {msg}"
                
                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(access_token)
                self.connected = True
                
                # Load instruments
                self.load_instruments()
                
                # Load initial data
                self.refresh_all_data()
                
                return True, "Connection successful"
            else:
                return False, "API credentials required"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def load_instruments(self):
        """Load instrument tokens"""
        try:
            if self.connected and self.kite:
                # Fetch from Zerodha API
                instruments = self.kite.instruments()
                for inst in instruments:
                    key = f"{inst['tradingsymbol']}_{inst['exchange']}"
                    self.instruments[key] = {
                        'token': inst['instrument_token'],
                        'tradingsymbol': inst['tradingsymbol'],
                        'exchange': inst['exchange'],
                        'name': inst['name'],
                        'lot_size': inst['lot_size'],
                        'tick_size': inst['tick_size']
                    }
                
                # Update popular instruments with actual tokens from API
                for symbol in self.popular_instruments.keys():
                    tradingsymbol = self.popular_instruments[symbol]['tradingsymbol']
                    exchange = self.popular_instruments[symbol]['exchange']
                    key = f"{tradingsymbol}_{exchange}"
                    
                    if key in self.instruments:
                        # Update token from API response
                        self.popular_instruments[symbol]['token'] = self.instruments[key]['token']
            
        except Exception as e:
            print(f"Error loading instruments: {e}")
    
    def refresh_all_data(self):
        """Refresh all account data"""
        try:
            if self.connected:
                # Get positions
                self.positions = self.kite.positions()
                
                # Get holdings
                self.holdings = self.kite.holdings()
                
                # Get margins
                self.margins = self.kite.margins()
                
                # Get orders
                orders = self.kite.orders()
                self.order_history.extend(orders)
                
                return True
        except Exception as e:
            print(f"Error refreshing data: {e}")
        
        return False
    
    def get_historical_data(self, instrument_token, interval="5minute", days=5):
        """Fetch historical data with caching"""
        cache_key = f"{instrument_token}_{interval}_{days}"
        cache_file = f"cache_{cache_key}.pkl"
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is recent (less than 5 minutes old)
                if datetime.now().timestamp() - cached_data['timestamp'] < 300:
                    return cached_data['data']
            except:
                pass
        
        # Fetch fresh data
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=False,
                oi=False
            )
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Cache the data
                cache_data = {
                    'timestamp': datetime.now().timestamp(),
                    'data': df
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol, transaction_type, quantity, order_type="MARKET", 
                   product="CNC", variety="regular", price=None, trigger_price=None):
        """Place an order with advanced options"""
        try:
            if not self.connected:
                return False, "Not connected to Zerodha"
            
            # Find instrument
            instrument = None
            for key, inst in self.popular_instruments.items():
                if key == symbol or inst['tradingsymbol'] == symbol:
                    instrument = inst
                    break
            
            if not instrument:
                # Try to find in all instruments
                for key, inst in self.instruments.items():
                    if inst['tradingsymbol'] == symbol:
                        instrument = inst
                        break
            
            if not instrument:
                return False, f"Instrument {symbol} not found"
            
            # Place order
            order_id = self.kite.place_order(
                variety=variety,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=instrument['tradingsymbol'],
                transaction_type=transaction_type.upper(),
                quantity=quantity,
                order_type=order_type.upper(),
                product=product.upper(),
                price=price,
                trigger_price=trigger_price
            )
            
            # Record order
            order_record = {
                'order_id': order_id,
                'timestamp': datetime.now(),
                'symbol': symbol,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'status': 'PENDING'
            }
            self.order_history.append(order_record)
            
            return True, f"Order placed: {order_id}"
            
        except Exception as e:
            return False, f"Order placement failed: {str(e)}"
    
    def get_order_status(self, order_id):
        """Get status of an order"""
        try:
            orders = self.kite.orders()
            for order in orders:
                if order['order_id'] == order_id:
                    return order['status'], order
        except:
            pass
        return "UNKNOWN", {}
    
    def square_off_all(self):
        """Square off all positions"""
        results = []
        
        try:
            if 'net' in self.positions:
                for position in self.positions['net']:
                    if position['quantity'] > 0:
                        # Square off long positions
                        success, message = self.place_order(
                            symbol=position['tradingsymbol'],
                            transaction_type="SELL",
                            quantity=position['quantity'],
                            order_type="MARKET"
                        )
                        results.append((position['tradingsymbol'], success, message))
            
            return results
            
        except Exception as e:
            return [("ALL", False, f"Error: {str(e)}")]

class PnLMonitor:
    """Profit and Loss Monitoring System"""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = {}
        self.total_investment = 0
        self.total_realized_pnl = 0
        self.total_unrealized_pnl = 0
        self.win_rate = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.risk_reward_ratio = 0
        
    def add_trade(self, trade):
        """Add a trade to monitor"""
        self.trades.append(trade)
        self.update_metrics()
    
    def update_metrics(self):
        """Update all P&L metrics"""
        if not self.trades:
            return
        
        # Calculate realized P&L
        realized_trades = [t for t in self.trades if t.get('exit_price') is not None]
        if realized_trades:
            wins = sum(1 for t in realized_trades if t.get('pnl', 0) > 0)
            self.win_rate = (wins / len(realized_trades)) * 100
            
            self.total_realized_pnl = sum(t.get('pnl', 0) for t in realized_trades)
        
        # Calculate daily P&L
        today = datetime.now().date()
        today_trades = [t for t in self.trades 
                       if t.get('timestamp').date() == today]
        self.daily_pnl[today] = sum(t.get('pnl', 0) for t in today_trades)
        
        # Calculate max drawdown
        if len(self.trades) > 1:
            running_pnl = 0
            peak = 0
            max_dd = 0
            
            for trade in sorted(self.trades, key=lambda x: x['timestamp']):
                running_pnl += trade.get('pnl', 0)
                if running_pnl > peak:
                    peak = running_pnl
                drawdown = peak - running_pnl
                if drawdown > max_dd:
                    max_dd = drawdown
            
            self.max_drawdown = max_dd
        
        # Calculate risk-reward ratio
        winning_trades = [t for t in realized_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in realized_trades if t.get('pnl', 0) < 0]
        
        if losing_trades:
            avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t.get('pnl', 0) for t in losing_trades]))
            self.risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    def get_summary(self):
        """Get P&L summary"""
        return {
            'total_trades': len(self.trades),
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'risk_reward_ratio': self.risk_reward_ratio,
            'daily_pnl': self.daily_pnl.get(datetime.now().date(), 0),
            'monthly_pnl': sum(self.daily_pnl.values())
        }
    
    def export_to_csv(self, filename="trades_pnl.csv"):
        """Export trades to CSV"""
        if not self.trades:
            return False
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades[0].keys())
                writer.writeheader()
                writer.writerows(self.trades)
            return True
        except:
            return False

class AdvancedTradingDashboard:
    """Advanced Trading Dashboard with Token Generation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Zerodha Trading Bot")
        self.root.geometry("1600x1000")
        
        # Initialize components
        self.zerodha = AdvancedZerodhaAPI()
        self.indicators = IndicatorsAdvanced()
        self.strategy = AdvancedTradingStrategy()
        self.pnl_monitor = PnLMonitor()
        
        # Trading settings
        self.trading_active = False
        self.trading_mode = TradingMode.PAPER
        self.auto_trading = False
        self.paper_capital = 100000  # Paper trading capital
        self.live_capital = 0
        self.order_quantity = 1
        
        # Data storage
        self.current_data = {}
        self.current_prices = {}
        self.signal_history = []
        
        # Threading
        self.data_queue = queue.Queue()
        self.update_thread = None
        self.auto_trade_thread = None
        
        # Setup GUI
        self.setup_gui()
        
        # Load settings
        self.load_settings()
        
        # Start background tasks
        self.start_background_tasks()
    
    def setup_gui(self):
        """Setup the advanced GUI layout"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.setup_dashboard_tab()
        self.setup_chart_tab()
        self.setup_token_generation_tab()
        self.setup_trading_tab()
        self.setup_pnl_tab()
        self.setup_logs_tab()
        
        # Status bar
        self.setup_status_bar()
    
    def setup_dashboard_tab(self):
        """Setup dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Left panel - Quick Stats
        left_frame = ttk.Frame(dashboard_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Connection Status
        conn_frame = ttk.LabelFrame(left_frame, text="Connection Status", padding=10)
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.conn_status_var = tk.StringVar(value="Disconnected")
        ttk.Label(conn_frame, textvariable=self.conn_status_var, 
                 font=("Arial", 12, "bold"), foreground="red").pack()
        
        # Account Summary
        acc_frame = ttk.LabelFrame(left_frame, text="Account Summary", padding=10)
        acc_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.acc_vars = {}
        fields = [
            ("Available Margin:", "margin_var"),
            ("Used Margin:", "used_margin_var"),
            ("Net Value:", "net_value_var"),
            ("Realized P&L:", "realized_pnl_var"),
            ("Unrealized P&L:", "unrealized_pnl_var"),
            ("Total P&L:", "total_pnl_var")
        ]
        
        for text, var_name in fields:
            frame = ttk.Frame(acc_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=text, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value="₹0.00")
            self.acc_vars[var_name] = var
            ttk.Label(frame, textvariable=var, font=("Arial", 10)).pack(side=tk.LEFT)
        
        # Right panel - Quick Actions
        right_frame = ttk.Frame(dashboard_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Quick Actions
        action_frame = ttk.LabelFrame(right_frame, text="Quick Actions", padding=10)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        buttons = [
            ("Connect Zerodha", self.connect_zerodha),
            ("Generate Token", self.show_token_generation),
            ("Fetch All Data", self.fetch_all_data),
            ("Start Auto Trading", self.start_auto_trading),
            ("Stop Auto Trading", self.stop_auto_trading),
            ("Square Off All", self.square_off_all),
            ("Export Trades", self.export_trades),
            ("Backup Data", self.backup_data)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(action_frame, text=text, command=command)
            btn.pack(fill=tk.X, pady=2)
        
        # Market Status
        market_frame = ttk.LabelFrame(right_frame, text="Market Status", padding=10)
        market_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.market_status_var = tk.StringVar(value="Market Closed")
        ttk.Label(market_frame, textvariable=self.market_status_var,
                 font=("Arial", 10)).pack()
    
    def setup_chart_tab(self):
        """Setup chart tab"""
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="Charts")
        
        # Control panel
        control_frame = ttk.Frame(chart_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Symbol selection
        ttk.Label(control_frame, text="Symbol:").pack(side=tk.LEFT, padx=5)
        self.symbol_var = tk.StringVar(value="RELIANCE")
        self.symbol_combo = ttk.Combobox(control_frame, textvariable=self.symbol_var,
                                        values=list(self.zerodha.popular_instruments.keys()),
                                        width=15)
        self.symbol_combo.pack(side=tk.LEFT, padx=5)
        self.symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_change)
        
        # Interval selection
        ttk.Label(control_frame, text="Interval:").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="5minute")
        interval_combo = ttk.Combobox(control_frame, textvariable=self.interval_var,
                                     values=["1minute", "3minute", "5minute", "15minute",
                                             "30minute", "60minute", "day"], width=10)
        interval_combo.pack(side=tk.LEFT, padx=5)
        
        # Chart type
        ttk.Label(control_frame, text="Chart Type:").pack(side=tk.LEFT, padx=5)
        self.chart_type_var = tk.StringVar(value="Candlestick")
        chart_combo = ttk.Combobox(control_frame, textvariable=self.chart_type_var,
                                  values=["Candlestick", "Line", "OHLC"], width=10)
        chart_combo.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        ttk.Button(control_frame, text="Refresh Chart", 
                  command=self.refresh_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Add Indicator", 
                  command=self.add_indicator_dialog).pack(side=tk.LEFT, padx=5)
        
        # Chart area
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax_main = self.fig.add_subplot(311)
        self.ax_volume = self.fig.add_subplot(312)
        self.ax_indicators = self.fig.add_subplot(313)
        
        self.fig.subplots_adjust(hspace=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    

    def save_manual_token(self):
        """Save a manually entered access token"""
        api_key = self.api_key_entry.get()
        access_token = self.manual_token_entry.get().strip()
        
        if not api_key:
            messagebox.showwarning("Warning", "Please enter API Key first")
            return
        
        if not access_token:
            messagebox.showwarning("Warning", "Please enter an Access Token")
            return
        
        # Validate the token
        self.token_status_var.set("Validating token...")
        self.log_message("Validating manually entered token...", "INFO")
        
        # Run validation in thread to avoid GUI freeze
        thread = threading.Thread(target=self._validate_and_save_token_thread,
                                args=(api_key, access_token))
        thread.daemon = True
        thread.start()

    def _validate_and_save_token_thread(self, api_key, access_token):
        """Thread for validating and saving manual token"""
        try:
            # Validate token (reuse your existing validation logic)
            valid, message = self.zerodha.token_generator.validate_token(api_key, access_token)
            
            if valid:
                # Save to credentials
                self.zerodha.token_generator.credentials['api_key'] = api_key
                self.zerodha.token_generator.credentials['access_token'] = access_token
                self.zerodha.token_generator.credentials['token_source'] = 'manual'
                self.zerodha.token_generator.save_credentials()
                
                # Update GUI
                self.root.after(0, self._manual_token_saved_callback, True, 
                            "Manual token saved and validated successfully!")
            else:
                self.root.after(0, self._manual_token_saved_callback, False, 
                            f"Token validation failed: {message}")
                
        except Exception as e:
            self.root.after(0, self._manual_token_saved_callback, False, 
                        f"Error saving token: {str(e)}")

    def _manual_token_saved_callback(self, success, message):
        """Callback for manual token save operation"""
        if success:
            self.token_status_var.set("Token saved successfully!")
            truncated_token = self.manual_token_entry.get()[:20] + "..."
            self.token_display_var.set(f"Manual: {truncated_token}")
            self.log_message(message, "INFO")
            messagebox.showinfo("Success", message)
        else:
            self.token_status_var.set("Token save failed")
            self.log_message(message, "ERROR")
            messagebox.showerror("Error", message)
    def setup_token_generation_tab(self):
        """Setup token generation tab"""
        token_frame = ttk.Frame(self.notebook)
        self.notebook.add(token_frame, text="Token Generation")
        
        # Token Generation Form
        form_frame = ttk.LabelFrame(token_frame, text="API Credentials", padding=20)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # API Key
        ttk.Label(form_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=10)
        self.api_key_entry = ttk.Entry(form_frame, width=40)
        self.api_key_entry.grid(row=0, column=1, pady=10, padx=10, sticky=tk.W)
        
        # API Secret
        ttk.Label(form_frame, text="API Secret:").grid(row=1, column=0, sticky=tk.W, pady=10)
        self.api_secret_entry = ttk.Entry(form_frame, width=40, show="*")
        self.api_secret_entry.grid(row=1, column=1, pady=10, padx=10, sticky=tk.W)

        # --- Manual Token Entry ---
        ttk.Label(form_frame, text="Access Token (Optional):").grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        self.manual_token_entry = ttk.Entry(form_frame, width=40, show="*")
        self.manual_token_entry.grid(row=3, column=1, pady=(20, 5), padx=10, sticky=tk.W)

        ttk.Button(form_frame, text="Save Token", 
        command=self.save_manual_token, width=15).grid(row=3, column=2, pady=(20, 5), padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(form_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Generate Access Token", 
                command=self.generate_access_token, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Validate Token", 
                command=self.validate_token, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Saved Credentials",
                command=self.load_saved_credentials, width=20).pack(side=tk.LEFT, padx=5)
        
        # Status display
        self.token_status_var = tk.StringVar(value="Ready for token generation")
        ttk.Label(form_frame, textvariable=self.token_status_var,
                foreground="blue").grid(row=3, column=0, columnspan=2, pady=10)
        
        # Generated Token Info
        token_info_frame = ttk.LabelFrame(form_frame, text="Token Information", padding=10)
        token_info_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(token_info_frame, text="Access Token:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.token_display_var = tk.StringVar(value="Not generated")
        ttk.Label(token_info_frame, textvariable=self.token_display_var, 
                font=("Courier", 8), wraplength=500).grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Instructions
        instr_frame = ttk.LabelFrame(token_frame, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        instructions = """
        1. Get your API Key and Secret from Zerodha Developer Console (https://developers.kite.trade)
        2. Enter your API Key and Secret above
        3. Click 'Generate Access Token' - a browser will open for Zerodha login
        4. Log in to your Zerodha account in the browser
        5. After login, you will be redirected and the token will be automatically captured
        6. Click 'Validate Token' to verify the token is working
        7. Use 'Load Saved Credentials' to reload previously generated tokens
        """
        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT).pack()
    
    def setup_trading_tab(self):
        """Setup trading tab"""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="Trading")
        
        # Left panel - Strategy Configuration
        left_frame = ttk.Frame(trading_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Strategy Settings
        strategy_frame = ttk.LabelFrame(left_frame, text="Strategy Settings", padding=10)
        strategy_frame.pack(fill=tk.BOTH, expand=True)
        
        settings = [
            ("Max Positions:", "max_positions_var", "3"),
            ("Risk per Trade (%):", "risk_per_trade_var", "2"),
            ("Take Profit Ratio:", "tp_ratio_var", "2"),
            ("Trailing Stop (%):", "trailing_stop_var", "1.5"),
            ("Stop Loss (%):", "stop_loss_var", "2"),
            ("Position Sizing:", "position_sizing_var", "Fixed")
        ]
        
        self.strategy_vars = {}
        for i, (label, var_name, default) in enumerate(settings):
            frame = ttk.Frame(strategy_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            self.strategy_vars[var_name] = var
            if "Position Sizing" in label:
                combo = ttk.Combobox(frame, textvariable=var, 
                                    values=["Fixed", "Dynamic", "Kelly"], width=15)
                combo.pack(side=tk.LEFT)
            else:
                ttk.Entry(frame, textvariable=var, width=15).pack(side=tk.LEFT)
        
        # Trading Mode
        mode_frame = ttk.Frame(strategy_frame)
        mode_frame.pack(fill=tk.X, pady=10)
        ttk.Label(mode_frame, text="Trading Mode:").pack(side=tk.LEFT)
        self.trading_mode_var = tk.StringVar(value="Paper Trading")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.trading_mode_var,
                                 values=["Paper Trading", "Live Trading"], width=15)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind('<<ComboboxSelected>>', self.on_trading_mode_change)
        
        # Auto Trading Switch
        self.auto_trading_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(strategy_frame, text="Enable Auto Trading",
                       variable=self.auto_trading_var).pack(pady=10)
        
        # Right panel - Order Management
        right_frame = ttk.Frame(trading_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Order Form
        order_frame = ttk.LabelFrame(right_frame, text="Manual Order", padding=10)
        order_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(order_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.order_symbol_var = tk.StringVar(value="RELIANCE")
        ttk.Entry(order_frame, textvariable=self.order_symbol_var, width=15).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(order_frame, text="Quantity:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.order_qty_var = tk.StringVar(value="1")
        ttk.Entry(order_frame, textvariable=self.order_qty_var, width=15).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(order_frame, text="Order Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.order_type_var = tk.StringVar(value="MARKET")
        ttk.Combobox(order_frame, textvariable=self.order_type_var,
                    values=["MARKET", "LIMIT", "SL", "SL-M"]).grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(order_frame, text="Price:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.order_price_var = tk.StringVar()
        ttk.Entry(order_frame, textvariable=self.order_price_var, width=15).grid(row=3, column=1, pady=5, padx=5)
        
        # Order Buttons
        btn_frame = ttk.Frame(order_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="BUY", command=lambda: self.place_manual_order("BUY"),
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="SELL", command=lambda: self.place_manual_order("SELL"),
                  width=10).pack(side=tk.LEFT, padx=2)
        
        # Open Positions
        pos_frame = ttk.LabelFrame(right_frame, text="Open Positions", padding=10)
        pos_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Symbol", "Qty", "Entry", "Current", "P&L", "Stop Loss", "Action")
        self.positions_tree = ttk.Treeview(pos_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(pos_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.positions_tree.configure(yscrollcommand=scrollbar.set)
        self.positions_tree.pack(fill=tk.BOTH, expand=True)
        
        # Position action buttons
        action_frame = ttk.Frame(pos_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Refresh", command=self.refresh_positions).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Close Selected", command=self.close_selected_position).pack(side=tk.LEFT, padx=2)
    
    def setup_pnl_tab(self):
        """Setup P&L monitoring tab"""
        pnl_frame = ttk.Frame(self.notebook)
        self.notebook.add(pnl_frame, text="P&L Monitor")
        
        # Top panel - Summary
        summary_frame = ttk.LabelFrame(pnl_frame, text="Performance Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create summary metrics in grid
        metrics = [
            ("Total Trades:", "total_trades_var"),
            ("Win Rate:", "win_rate_var"),
            ("Total P&L:", "total_pnl_summary_var"),
            ("Daily P&L:", "daily_pnl_var"),
            ("Monthly P&L:", "monthly_pnl_var"),
            ("Max Drawdown:", "max_dd_var"),
            ("Sharpe Ratio:", "sharpe_var"),
            ("Risk/Reward:", "risk_reward_var")
        ]
        
        self.pnl_vars = {}
        for i, (label, var_name) in enumerate(metrics):
            row = i // 4
            col = (i % 4) * 2
            
            ttk.Label(summary_frame, text=label).grid(row=row, column=col, sticky=tk.W, pady=5, padx=5)
            var = tk.StringVar(value="0")
            self.pnl_vars[var_name] = var
            ttk.Label(summary_frame, textvariable=var, font=("Arial", 10, "bold")).grid(
                row=row, column=col+1, sticky=tk.W, pady=5, padx=5)
        
        # Middle panel - Charts
        chart_frame = ttk.Frame(pnl_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create P&L chart
        self.pnl_fig = Figure(figsize=(10, 4), dpi=100)
        self.pnl_ax = self.pnl_fig.add_subplot(111)
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_fig, master=chart_frame)
        self.pnl_canvas.draw()
        self.pnl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bottom panel - Trade History
        history_frame = ttk.LabelFrame(pnl_frame, text="Trade History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ("Time", "Symbol", "Action", "Qty", "Price", "P&L", "Reason")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
    
    def setup_logs_tab(self):
        """Setup logs tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=30, width=100)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log controls
        control_frame = ttk.Frame(logs_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(control_frame, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Logs", command=self.save_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Logs", command=self.export_logs).pack(side=tk.LEFT, padx=5)
        
        # Log level filter
        ttk.Label(control_frame, text="Log Level:").pack(side=tk.LEFT, padx=5)
        self.log_level_var = tk.StringVar(value="ALL")
        log_combo = ttk.Combobox(control_frame, textvariable=self.log_level_var,
                                values=["ALL", "INFO", "WARNING", "ERROR", "TRADE"], width=10)
        log_combo.pack(side=tk.LEFT, padx=5)
    
    def setup_status_bar(self):
        """Setup status bar at bottom"""
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status labels
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.time_label = ttk.Label(self.status_bar, text="")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # Update time
        self.update_time()
    
    def update_time(self):
        """Update time in status bar"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    def log_message(self, message, level="INFO"):
        """Add message to log"""
        if self.log_level_var.get() != "ALL" and level != self.log_level_var.get():
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # Color coding
        colors = {
            "INFO": "black",
            "WARNING": "orange",
            "ERROR": "red",
            "TRADE": "green"
        }
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.tag_add(level, "end-2c linestart", "end-2c lineend")
        self.log_text.tag_config(level, foreground=colors.get(level, "black"))
        self.log_text.see(tk.END)
        
        # Update status bar for important messages
        if level in ["ERROR", "TRADE"]:
            self.status_label.config(text=message[:50])
    
    def clear_logs(self):
        """Clear log messages"""
        self.log_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to file"""
        filename = f"trading_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(self.log_text.get(1.0, tk.END))
        self.log_message(f"Logs saved to {filename}")
    
    def export_logs(self):
        """Export logs"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"Logs exported to {filename}")
    
    # def connect_zerodha(self):
    #     """Connect to Zerodha"""
    #     api_key = self.api_key_entry.get()
    #     access_token = self.zerodha.token_generator.credentials.get('access_token')
        
    #     if not api_key or not access_token:
    #         messagebox.showwarning("Warning", "Please generate access token first")
    #         return
        
    #     success, message = self.zerodha.connect(api_key, access_token)
        
    #     if success:
    #         self.conn_status_var.set("Connected")
    #         self.log_message("Connected to Zerodha API", "INFO")
    #         self.refresh_account_info()
    #     else:
    #         self.log_message(f"Connection failed: {message}", "ERROR")
    #         messagebox.showerror("Error", message)
    def connect_zerodha(self):
        """Connect to Zerodha using saved token (auto or manual)"""
        api_key = self.api_key_entry.get()
        
        if not api_key:
            messagebox.showwarning("Warning", "Please enter API Key")
            return
        
        # Get token from saved credentials (could be auto or manual)
        access_token = self.zerodha.token_generator.credentials.get('access_token')
        
        if not access_token:
            messagebox.showwarning("Warning", 
                "No access token found. Please generate one or enter manually.")
            return
        
        success, message = self.zerodha.connect(api_key, access_token)
        
        if success:
            self.conn_status_var.set("Connected")
            # Display token source
            token_source = self.zerodha.token_generator.credentials.get('token_source', 'auto')
            self.log_message(f"Connected using {token_source} token", "INFO")
            self.refresh_account_info()
        else:
            self.log_message(f"Connection failed: {message}", "ERROR")
            messagebox.showerror("Error", message)
    
    def show_token_generation(self):
        """Switch to token generation tab"""
        self.notebook.select(2)  # Token generation tab
    
    def generate_access_token(self):
        """Generate access token"""
        api_key = self.api_key_entry.get()
        api_secret = self.api_secret_entry.get()
        
        if not api_key or not api_secret:
            messagebox.showwarning("Warning", "Please enter API Key and Secret")
            return
        
        self.token_status_var.set("Generating access token...")
        self.log_message("Starting token generation...", "INFO")
        
        # Save API secret for future use
        self.zerodha.token_generator.credentials['api_secret'] = api_secret
        self.zerodha.token_generator.save_credentials()
        
        # Run in thread to avoid freezing GUI
        thread = threading.Thread(target=self._generate_token_thread,
                                 args=(api_key, api_secret))
        thread.daemon = True
        thread.start()
    
    def _generate_token_thread(self, api_key, api_secret):
        """Thread for token generation"""
        try:
            success, token = self.zerodha.token_generator.generate_access_token(api_key, api_secret)
            self.root.after(0, lambda: self._token_callback(success, token))
        except Exception as e:
            self.root.after(0, lambda: self._token_callback(False, str(e)))
    
    def _token_callback(self, success, token):
        """Callback for token generation"""
        if success:
            self.token_status_var.set("Token generated successfully!")
            truncated_token = token[:20] + "..." if len(token) > 20 else token
            self.token_display_var.set(truncated_token)
            self.log_message(f"Access token generated: {truncated_token}", "INFO")
            messagebox.showinfo("Success", "Access token generated!\nToken saved to credentials file.")
        else:
            self.token_status_var.set("Token generation failed")
            self.log_message(f"Token generation failed: {token}", "ERROR")
            messagebox.showerror("Error", token)
    
    def validate_token(self):
        """Validate the access token"""
        api_key = self.api_key_entry.get()
        access_token = self.zerodha.token_generator.credentials.get('access_token')
        
        if not api_key:
            messagebox.showwarning("Warning", "Please enter API Key")
            return
        
        if not access_token:
            messagebox.showwarning("Warning", "No access token found. Please generate one first.")
            return
        
        self.token_status_var.set("Validating token...")
        self.log_message("Validating access token...", "INFO")
        
        thread = threading.Thread(target=self._validate_token_thread,
                                 args=(api_key, access_token))
        thread.daemon = True
        thread.start()
    
    def _validate_token_thread(self, api_key, access_token):
        """Thread for token validation"""
        try:
            success, message = self.zerodha.token_generator.validate_token(api_key, access_token)
            self.root.after(0, lambda: self._validate_token_callback(success, message))
        except Exception as e:
            self.root.after(0, lambda: self._validate_token_callback(False, str(e)))
    
    def _validate_token_callback(self, success, message):
        """Callback for token validation"""
        if success:
            self.token_status_var.set("Token is valid!")
            self.log_message("Token validation successful", "INFO")
            messagebox.showinfo("Success", message)
        else:
            self.token_status_var.set("Token validation failed")
            self.log_message(f"Token validation failed: {message}", "ERROR")
            messagebox.showerror("Error", message)
    
    def load_saved_credentials(self):
        """Load saved credentials"""
        if self.zerodha.token_generator.credentials:
            creds = self.zerodha.token_generator.credentials
            
            self.api_key_entry.delete(0, tk.END)
            self.api_key_entry.insert(0, creds.get('api_key', ''))
            
            self.api_secret_entry.delete(0, tk.END)
            self.api_secret_entry.insert(0, creds.get('api_secret', ''))
            
            if 'access_token' in creds:
                truncated_token = creds['access_token'][:20] + "..." if len(creds['access_token']) > 20 else creds['access_token']
                self.token_display_var.set(truncated_token)
            
            self.log_message("Loaded saved credentials", "INFO")
        else:
            self.log_message("No saved credentials found", "WARNING")
    
    def refresh_chart(self):
        """Refresh chart with current symbol"""
        symbol = self.symbol_var.get()
        if symbol not in self.zerodha.popular_instruments:
            self.log_message(f"Symbol {symbol} not found", "ERROR")
            return
        
        instrument = self.zerodha.popular_instruments[symbol]
        interval = self.interval_var.get()
        
        self.log_message(f"Fetching chart for {symbol}...", "INFO")
        
        thread = threading.Thread(target=self._refresh_chart_thread,
                                 args=(instrument['token'], interval))
        thread.daemon = True
        thread.start()
    
    def _refresh_chart_thread(self, instrument_token, interval):
        """Thread for refreshing chart"""
        try:
            data = self.zerodha.get_historical_data(instrument_token, interval)
            
            if not data.empty:
                # Calculate indicators
                data = self.indicators.calculate_all_indicators(data)
                
                # Store data
                symbol = self.symbol_var.get()
                self.current_data[symbol] = data
                
                # Update GUI
                self.root.after(0, self._update_chart_gui, data)
                self.root.after(0, lambda: self.log_message(
                    f"Chart updated with {len(data)} candles", "INFO"))
            else:
                self.root.after(0, lambda: self.log_message("No data received", "ERROR"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Chart error: {str(e)}", "ERROR"))
    
    def _update_chart_gui(self, data):
        """Update chart GUI"""
        if data.empty:
            return
        
        # Clear axes
        self.ax_main.clear()
        self.ax_volume.clear()
        self.ax_indicators.clear()
        
        # Get last 100 candles
        plot_data = data.tail(100)
        
        # Plot main chart
        chart_type = self.chart_type_var.get()
        if chart_type == "Candlestick":
            # Plot candlesticks
            mpf.plot(plot_data, type='candle', style='charles',
                    ax=self.ax_main, volume=False, show_nontrading=False)
        else:
            # Plot line or OHLC
            self.ax_main.plot(plot_data.index, plot_data['close'], label='Close')
        
        # Add indicators
        self.ax_main.plot(plot_data.index, plot_data['supertrend'], label='Supertrend', color='green', linewidth=2)
        self.ax_main.plot(plot_data.index, plot_data['vwap'], label='VWAP', color='cyan', linestyle='--')
        self.ax_main.plot(plot_data.index, plot_data['sma_20'], label='SMA 20', color='yellow', alpha=0.7)
        
        # Plot buy/sell signals
        buy_signals = plot_data[plot_data['signal'] == 1]
        sell_signals = plot_data[plot_data['signal'] == -1]
        
        if len(buy_signals) > 0:
            self.ax_main.scatter(buy_signals.index, buy_signals['close'],
                               color='green', marker='^', s=100, label='Buy', zorder=5)
        
        if len(sell_signals) > 0:
            self.ax_main.scatter(sell_signals.index, sell_signals['close'],
                               color='red', marker='v', s=100, label='Sell', zorder=5)
        
        self.ax_main.set_title(f"{self.symbol_var.get()} - Price Chart")
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
        # Plot volume
        self.ax_volume.bar(plot_data.index, plot_data['volume'], color='gray', alpha=0.7)
        self.ax_volume.set_ylabel('Volume')
        self.ax_volume.grid(True, alpha=0.3)
        
        # Plot indicators
        self.ax_indicators.plot(plot_data.index, plot_data['rsi_14'], label='RSI', color='yellow')
        self.ax_indicators.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        self.ax_indicators.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        self.ax_indicators.set_ylabel('RSI')
        self.ax_indicators.set_ylim(0, 100)
        
        # Add MACD to second y-axis
        ax2 = self.ax_indicators.twinx()
        ax2.plot(plot_data.index, plot_data['macd'], label='MACD', color='magenta')
        ax2.plot(plot_data.index, plot_data['macd_signal'], label='Signal', color='blue')
        ax2.set_ylabel('MACD')
        
        self.ax_indicators.set_xlabel('Time')
        self.ax_indicators.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Format x-axis
        for ax in [self.ax_main, self.ax_volume, self.ax_indicators]:
            ax.xaxis.set_tick_params(rotation=45)
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def add_indicator_dialog(self):
        """Dialog to add indicator"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Indicator")
        dialog.geometry("300x400")
        
        indicators = [
            "RSI", "MACD", "Bollinger Bands", "Stochastic", "ATR",
            "Ichimoku Cloud", "Parabolic SAR", "Williams %R", "CCI"
        ]
        
        for indicator in indicators:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(dialog, text=indicator, variable=var)
            chk.pack(anchor=tk.W, padx=20, pady=5)
        
        ttk.Button(dialog, text="Add Selected",
                  command=lambda: self.add_indicators(dialog)).pack(pady=10)
    
    def add_indicators(self, dialog):
        """Add selected indicators"""
        dialog.destroy()
        self.log_message("Adding indicators...", "INFO")
    
    def on_symbol_change(self, event):
        """Handle symbol change"""
        symbol = self.symbol_var.get()
        self.log_message(f"Symbol changed to {symbol}", "INFO")
        self.refresh_chart()
    
    def on_trading_mode_change(self, event):
        """Handle trading mode change"""
        mode = self.trading_mode_var.get()
        if mode == "Paper Trading":
            self.trading_mode = TradingMode.PAPER
            self.log_message("Switched to Paper Trading mode", "INFO")
        else:
            self.trading_mode = TradingMode.LIVE
            self.log_message("Switched to Live Trading mode", "WARNING")
            if not self.zerodha.connected:
                messagebox.showwarning("Warning", "Please connect to Zerodha for live trading")
    
    def refresh_account_info(self):
        """Refresh account information"""
        if self.zerodha.connected:
            try:
                self.zerodha.refresh_all_data()
                
                # Update account variables
                if 'equity' in self.zerodha.margins:
                    equity = self.zerodha.margins['equity']
                    self.acc_vars['margin_var'].set(f"₹{equity.get('available', {}).get('cash', 0):,.2f}")
                    self.acc_vars['net_value_var'].set(f"₹{equity.get('net', 0):,.2f}")
                
                # Calculate P&L
                total_pnl = 0
                if 'net' in self.zerodha.positions:
                    for pos in self.zerodha.positions['net']:
                        total_pnl += pos.get('pnl', 0)
                
                self.acc_vars['total_pnl_var'].set(f"₹{total_pnl:,.2f}")
                
                self.log_message("Account info refreshed", "INFO")
                
            except Exception as e:
                self.log_message(f"Error refreshing account info: {str(e)}", "ERROR")
        else:
            self.acc_vars['margin_var'].set("₹0.00")
            self.acc_vars['total_pnl_var'].set("₹0.00")
    
    def refresh_positions(self):
        """Refresh positions display"""
        # Clear tree
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Add strategy positions
        for symbol, position in self.strategy.positions.items():
            entry_price = self.strategy.entry_prices.get(symbol, 0)
            current_price = self.current_prices.get(symbol, entry_price)
            quantity = next((t['quantity'] for t in reversed(self.strategy.trades) 
                           if t['symbol'] == symbol and t['action'] in ['BUY', 'LONG']), 0)
            pnl = (current_price - entry_price) * quantity
            stop_loss = self.strategy.stop_losses.get(symbol, 0)
            
            self.positions_tree.insert("", tk.END, values=(
                symbol, quantity, f"₹{entry_price:.2f}", f"₹{current_price:.2f}",
                f"₹{pnl:.2f}", f"₹{stop_loss:.2f}", position
            ))
        
        # Add Zerodha positions if connected
        if self.zerodha.connected and 'net' in self.zerodha.positions:
            for pos in self.zerodha.positions['net']:
                if pos['quantity'] != 0:
                    self.positions_tree.insert("", tk.END, values=(
                        pos['tradingsymbol'], pos['quantity'],
                        f"₹{pos.get('average_price', 0):.2f}",
                        f"₹{pos.get('last_price', 0):.2f}",
                        f"₹{pos.get('pnl', 0):.2f}", "-", "Zerodha"
                    ))
    
    def close_selected_position(self):
        """Close selected position"""
        selection = self.positions_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a position to close")
            return
        
        item = self.positions_tree.item(selection[0])
        values = item['values']
        symbol = values[0]
        quantity = int(values[1])
        
        confirm = messagebox.askyesno("Confirm", f"Close position for {symbol}?")
        if confirm:
            self.execute_trade("EXIT", symbol, quantity)
    
    def place_manual_order(self, action):
        """Place manual order"""
        symbol = self.order_symbol_var.get()
        quantity = int(self.order_qty_var.get())
        order_type = self.order_type_var.get()
        price = float(self.order_price_var.get()) if self.order_price_var.get() else None
        
        if self.trading_mode == TradingMode.PAPER:
            # Paper trading
            current_price = self.current_prices.get(symbol, 1000)
            trade = self.strategy.record_trade(symbol, action, quantity, current_price, "MANUAL")
            self.log_message(f"[PAPER] {action} {quantity} {symbol} at ₹{current_price:.2f}", "TRADE")
            self.update_pnl_display()
        else:
            # Live trading
            success, message = self.zerodha.place_order(
                symbol=symbol,
                transaction_type=action,
                quantity=quantity,
                order_type=order_type,
                price=price
            )
            
            if success:
                self.log_message(f"[LIVE] Order placed: {message}", "TRADE")
                self.refresh_account_info()
            else:
                self.log_message(f"[LIVE] Order failed: {message}", "ERROR")
        
        self.refresh_positions()
    
    def execute_trade(self, action, symbol, quantity, price=None, reason=""):
        """Execute a trade"""
        if price is None:
            price = self.current_prices.get(symbol, 0)
        
        if self.trading_mode == TradingMode.PAPER:
            # Paper trading
            trade = self.strategy.record_trade(symbol, action, quantity, price, reason)
            self.pnl_monitor.add_trade(trade)
            self.log_message(f"[PAPER] {action} {quantity} {symbol} at ₹{price:.2f} ({reason})", "TRADE")
        else:
            # Live trading
            if action == "EXIT":
                action = "SELL"
            
            success, message = self.zerodha.place_order(
                symbol=symbol,
                transaction_type=action,
                quantity=quantity
            )
            
            if success:
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'reason': reason
                }
                self.strategy.trades.append(trade)
                self.pnl_monitor.add_trade(trade)
                self.log_message(f"[LIVE] {action} {quantity} {symbol}: {message}", "TRADE")
                self.refresh_account_info()
            else:
                self.log_message(f"[LIVE] Trade failed: {message}", "ERROR")
        
        self.update_pnl_display()
        self.refresh_positions()
    
    def start_auto_trading(self):
        """Start auto trading"""
        if self.auto_trading_var.get():
            self.auto_trading = True
            self.log_message("Auto trading started", "INFO")
            
            # Start auto trade thread
            self.auto_trade_thread = threading.Thread(target=self.auto_trade_loop)
            self.auto_trade_thread.daemon = True
            self.auto_trade_thread.start()
        else:
            messagebox.showinfo("Info", "Please enable auto trading first")
    
    def stop_auto_trading(self):
        """Stop auto trading"""
        self.auto_trading = False
        self.log_message("Auto trading stopped", "INFO")
    
    def auto_trade_loop(self):
        """Auto trading loop"""
        while self.auto_trading:
            try:
                # Check market hours (9:15 AM to 3:30 PM IST)
                now = datetime.now()
                if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour >= 15:
                    time.sleep(60)
                    continue
                
                # Monitor all symbols
                for symbol, instrument in self.zerodha.popular_instruments.items():
                    if not self.auto_trading:
                        break
                    
                    # Skip if max positions reached
                    if len(self.strategy.positions) >= self.strategy.max_positions:
                        continue
                    
                    # Fetch data
                    data = self.zerodha.get_historical_data(instrument['token'], "5minute", 1)
                    if data.empty:
                        continue
                    
                    # Calculate indicators
                    data = self.indicators.calculate_all_indicators(data)
                    
                    # Generate signals
                    data = self.strategy.generate_signals(data)
                    
                    # Store current price
                    current_price = data.iloc[-1]['close']
                    self.current_prices[symbol] = current_price
                    
                    # Check for entry signal
                    latest = data.iloc[-1]
                    if latest['signal'] == 1 and symbol not in self.strategy.positions:
                        # Calculate position size
                        capital = self.paper_capital if self.trading_mode == TradingMode.PAPER else self.live_capital
                        risk_pct = float(self.strategy_vars['risk_per_trade_var'].get()) / 100
                        position_value = self.strategy.calculate_position_sizing(capital, risk_pct, 
                                                                               self.strategy.max_loss_per_trade)
                        quantity = max(1, int(position_value / current_price))
                        
                        # Execute buy
                        self.execute_trade("BUY", symbol, quantity, current_price, "AUTO_ENTRY")
                    
                    # Check for exit conditions
                    elif symbol in self.strategy.positions:
                        exit_reason = self.strategy.check_exit_conditions(symbol, current_price, latest)
                        if exit_reason:
                            quantity = next((t['quantity'] for t in reversed(self.strategy.trades) 
                                          if t['symbol'] == symbol and t['action'] in ['BUY', 'LONG']), 0)
                            self.execute_trade("EXIT", symbol, quantity, current_price, exit_reason)
                
                # Wait for next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.log_message(f"Auto trade error: {str(e)}", "ERROR")
                time.sleep(10)
    
    def update_pnl_display(self):
        """Update P&L display"""
        # Update summary metrics
        summary = self.pnl_monitor.get_summary()
        
        self.pnl_vars['total_trades_var'].set(str(summary['total_trades']))
        self.pnl_vars['win_rate_var'].set(f"{summary['win_rate']:.1f}%")
        self.pnl_vars['total_pnl_summary_var'].set(f"₹{summary['total_realized_pnl']:,.2f}")
        self.pnl_vars['daily_pnl_var'].set(f"₹{summary['daily_pnl']:,.2f}")
        self.pnl_vars['monthly_pnl_var'].set(f"₹{summary['monthly_pnl']:,.2f}")
        self.pnl_vars['max_dd_var'].set(f"₹{summary['max_drawdown']:,.2f}")
        self.pnl_vars['risk_reward_var'].set(f"{summary['risk_reward_ratio']:.2f}")
        
        # Update trade history
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        for trade in sorted(self.strategy.trades, key=lambda x: x['timestamp'], reverse=True)[:50]:
            self.history_tree.insert("", tk.END, values=(
                trade['timestamp'].strftime("%H:%M:%S"),
                trade['symbol'],
                trade['action'],
                trade['quantity'],
                f"₹{trade['price']:.2f}",
                f"₹{trade.get('pnl', 0):.2f}",
                trade.get('reason', '')
            ))
        
        # Update P&L chart
        self.update_pnl_chart()
    
    def update_pnl_chart(self):
        """Update P&L chart"""
        if not self.strategy.trades:
            return
        
        # Calculate cumulative P&L over time
        trades_sorted = sorted(self.strategy.trades, key=lambda x: x['timestamp'])
        dates = []
        cumulative_pnl = []
        current_pnl = 0
        
        for trade in trades_sorted:
            dates.append(trade['timestamp'])
            current_pnl += trade.get('pnl', 0)
            cumulative_pnl.append(current_pnl)
        
        # Plot
        self.pnl_ax.clear()
        self.pnl_ax.plot(dates, cumulative_pnl, 'b-', linewidth=2)
        self.pnl_ax.fill_between(dates, 0, cumulative_pnl, where=np.array(cumulative_pnl) > 0,
                                facecolor='green', alpha=0.3)
        self.pnl_ax.fill_between(dates, 0, cumulative_pnl, where=np.array(cumulative_pnl) < 0,
                                facecolor='red', alpha=0.3)
        
        self.pnl_ax.set_title("Cumulative P&L")
        self.pnl_ax.set_xlabel("Time")
        self.pnl_ax.set_ylabel("P&L (₹)")
        self.pnl_ax.grid(True, alpha=0.3)
        
        # Format x-axis
        self.pnl_ax.xaxis.set_tick_params(rotation=45)
        
        self.pnl_fig.tight_layout()
        self.pnl_canvas.draw()
    
    def fetch_all_data(self):
        """Fetch all data"""
        self.log_message("Fetching all data...", "INFO")
        
        for symbol in self.zerodha.popular_instruments.keys():
            self.refresh_chart()
            time.sleep(1)  # Delay to avoid rate limiting
        
        self.refresh_account_info()
        self.refresh_positions()
        self.update_pnl_display()
    
    def square_off_all(self):
        """Square off all positions"""
        confirm = messagebox.askyesno("Confirm", "Square off all positions?")
        if not confirm:
            return
        
        if self.trading_mode == TradingMode.PAPER:
            # Paper trading - close all strategy positions
            for symbol in list(self.strategy.positions.keys()):
                quantity = next((t['quantity'] for t in reversed(self.strategy.trades) 
                              if t['symbol'] == symbol and t['action'] in ['BUY', 'LONG']), 0)
                if quantity > 0:
                    current_price = self.current_prices.get(symbol, 0)
                    self.execute_trade("EXIT", symbol, quantity, current_price, "SQUARE_OFF")
            
            self.log_message("All paper positions squared off", "INFO")
        
        else:
            # Live trading
            if self.zerodha.connected:
                results = self.zerodha.square_off_all()
                for symbol, success, message in results:
                    if success:
                        self.log_message(f"Squared off {symbol}: {message}", "TRADE")
                    else:
                        self.log_message(f"Failed to square off {symbol}: {message}", "ERROR")
                
                self.refresh_account_info()
                self.refresh_positions()
    
    def export_trades(self):
        """Export trades to CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            if self.pnl_monitor.export_to_csv(filename):
                self.log_message(f"Trades exported to {filename}", "INFO")
            else:
                self.log_message("Failed to export trades", "ERROR")
    
    def backup_data(self):
        """Backup all data"""
        backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup trades
        self.pnl_monitor.export_to_csv(f"{backup_dir}/trades.csv")
        
        # Backup logs
        with open(f"{backup_dir}/logs.txt", 'w') as f:
            f.write(self.log_text.get(1.0, tk.END))
        
        # Backup settings
        settings = {
            'trading_mode': self.trading_mode.value,
            'symbol': self.symbol_var.get(),
            'interval': self.interval_var.get(),
            'strategy_settings': {k: v.get() for k, v in self.strategy_vars.items()}
        }
        with open(f"{backup_dir}/settings.json", 'w') as f:
            json.dump(settings, f, indent=2)
        
        self.log_message(f"Data backed up to {backup_dir}", "INFO")
    
    def load_settings(self):
        """Load settings"""
        try:
            if os.path.exists("trading_settings.json"):
                with open("trading_settings.json", 'r') as f:
                    settings = json.load(f)
                
                # Load API credentials
                if 'credentials' in settings:
                    creds = settings['credentials']
                    self.api_key_entry.insert(0, creds.get('api_key', ''))
                
                # Load trading settings
                if 'trading' in settings:
                    trading = settings['trading']
                    self.symbol_var.set(trading.get('symbol', 'RELIANCE'))
                    self.interval_var.set(trading.get('interval', '5minute'))
                    self.trading_mode_var.set(trading.get('trading_mode', 'Paper Trading'))
                
                self.log_message("Settings loaded", "INFO")
                
        except Exception as e:
            self.log_message(f"Error loading settings: {str(e)}", "ERROR")
    
    def save_settings(self):
        """Save settings"""
        try:
            settings = {
                'credentials': self.zerodha.token_generator.credentials,
                'trading': {
                    'symbol': self.symbol_var.get(),
                    'interval': self.interval_var.get(),
                    'trading_mode': self.trading_mode_var.get()
                },
                'strategy': {k: v.get() for k, v in self.strategy_vars.items()}
            }
            
            with open("trading_settings.json", 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.log_message("Settings saved", "INFO")
            
        except Exception as e:
            self.log_message(f"Error saving settings: {str(e)}", "ERROR")
    
    def start_background_tasks(self):
        """Start background tasks"""
        # Update market status
        self.update_market_status()
        
        # Schedule periodic updates
        self.root.after(30000, self.periodic_update)  # Every 30 seconds
    
    def update_market_status(self):
        """Update market status"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = (hour > 9 or (hour == 9 and minute >= 15)) and (hour < 15 or (hour == 15 and minute <= 30))
        
        if market_open:
            self.market_status_var.set("Market Open")
        else:
            self.market_status_var.set("Market Closed")
        
        self.root.after(60000, self.update_market_status)  # Update every minute
    
    def periodic_update(self):
        """Periodic updates"""
        if self.zerodha.connected:
            self.refresh_account_info()
            self.refresh_positions()
            self.update_pnl_display()
        
        self.root.after(30000, self.periodic_update)  # Schedule next update
    
    def on_closing(self):
        """Handle application closing"""
        # Stop auto trading
        self.auto_trading = False
        
        # Save settings
        self.save_settings()
        
        # Close application
        self.root.destroy()

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set window icon and title
    root.title("Advanced Zerodha Trading Bot - Token Generation")
    
    # Create and run application
    app = AdvancedTradingDashboard(root)
    
    # Set closing protocol
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()