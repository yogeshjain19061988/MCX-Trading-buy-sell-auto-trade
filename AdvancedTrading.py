import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
import threading
import time
import webbrowser
import socket
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
import logging
import traceback
from urllib.parse import urlparse, parse_qs
import http.server
import socketserver
import urllib.request

# Technical Analysis
import ta

# Zerodha API
from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import KiteException

# PyQt6 GUI
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

logger = logging.getLogger(__name__)

# Try to import QtWebEngineWidgets, but provide fallback if not available
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
    logger.info("PyQt6-WebEngine is available")
except ImportError:
    WEBENGINE_AVAILABLE = False
    logger.warning("PyQt6-WebEngine not found. Using external browser for login.")
    QWebEngineView = None  # Placeholder for type consistency

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
    redirect_port: int = 8080  # Port for OAuth redirect
    use_embedded_browser: bool = False  # Default to external browser
    
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
# OAUTH SERVER FOR AUTO-LOGIN
# ============================================================================

class OAuthRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for OAuth redirect"""
    
    def __init__(self, *args, callback=None, **kwargs):
        self.callback = callback
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET request"""
        try:
            # Parse URL and query parameters
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            # Check if request token is in the URL
            if 'request_token' in query_params:
                request_token = query_params['request_token'][0]
                status = query_params.get('status', [''])[0]
                action = query_params.get('action', [''])[0]
                
                logger.info(f"Received request_token: {request_token}")
                logger.info(f"Status: {status}, Action: {action}")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                # Create HTML response
                html_response = """
                <html>
                <head>
                    <title>Login Successful</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            text-align: center;
                            margin-top: 50px;
                            background-color: #f5f5f5;
                        }
                        .container {
                            background-color: white;
                            padding: 30px;
                            border-radius: 10px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                            display: inline-block;
                        }
                        .success {
                            color: #4CAF50;
                            font-size: 24px;
                            margin-bottom: 20px;
                        }
                        .message {
                            color: #666;
                            margin-bottom: 20px;
                        }
                        .close-btn {
                            background-color: #4CAF50;
                            color: white;
                            border: none;
                            padding: 10px 20px;
                            border-radius: 5px;
                            cursor: pointer;
                            font-size: 16px;
                        }
                        .close-btn:hover {
                            background-color: #45a049;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="success">✅ Login Successful!</div>
                        <div class="message">You can now close this window and return to the trading application.</div>
                        <button class="close-btn" onclick="window.close()">Close Window</button>
                    </div>
                </body>
                </html>
                """
                
                self.wfile.write(html_response.encode('utf-8'))
                
                # Call the callback function with the request token
                if self.callback:
                    self.callback(request_token)
                
            else:
                # Send default response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                html_response = """
                <html>
                <body>
                    <h1>OAuth Callback Server</h1>
                    <p>Waiting for authentication...</p>
                </body>
                </html>
                """
                self.wfile.write(html_response.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error in OAuth handler: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass

class OAuthServer:
    """OAuth server to capture request token"""
    
    def __init__(self, port=8080, callback=None):
        self.port = port
        self.callback = callback
        self.server = None
        self.server_thread = None
        self.is_running = False
    
    def start(self):
        """Start the OAuth server"""
        try:
            # Check if port is available
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', self.port))
            sock.close()
            
            if result == 0:
                logger.warning(f"Port {self.port} is already in use, trying another port...")
                # Try to find an available port
                for p in range(self.port + 1, self.port + 10):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', p))
                    sock.close()
                    if result != 0:
                        self.port = p
                        logger.info(f"Using port {self.port} instead")
                        break
            
            # Create custom handler factory with callback
            def handler_factory():
                return lambda *args, **kwargs: OAuthRequestHandler(
                    *args, callback=self.callback, **kwargs
                )
            
            # Create and start server
            self.server = socketserver.TCPServer(
                ("localhost", self.port),
                handler_factory()
            )
            
            # Start server in a thread
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            self.is_running = True
            logger.info(f"OAuth server started on port {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start OAuth server: {e}")
            return False
    
    def stop(self):
        """Stop the OAuth server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
            logger.info("OAuth server stopped")
    
    def get_redirect_url(self):
        """Get the redirect URL for OAuth"""
        return f"http://localhost:{self.port}/"

# ============================================================================
# EMBEDDED BROWSER FOR LOGIN (OPTIONAL)
# ============================================================================

if WEBENGINE_AVAILABLE:
    class EmbeddedBrowserDialog(QDialog):
        """Embedded browser dialog for Zerodha login"""
        
        login_complete = pyqtSignal(str)  # Signal with request_token
        
        def __init__(self, login_url: str, parent=None):
            super().__init__(parent)
            self.login_url = login_url
            self.request_token = None
            self.init_ui()
        
        def init_ui(self):
            """Initialize UI"""
            self.setWindowTitle("Zerodha Login")
            self.setGeometry(300, 200, 900, 700)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel("Login to Zerodha")
            title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title_label)
            
            # Instructions
            instructions = QLabel(
                "1. Login with your Zerodha credentials\n"
                "2. Authorize the application\n"
                "3. You will be redirected automatically\n\n"
                "Do not close this window until login is complete."
            )
            instructions.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(instructions)
            
            # Web view
            self.web_view = QWebEngineView()
            self.web_view.setUrl(QUrl(self.login_url))
            
            # Connect URL changed signal to check for request token
            self.web_view.urlChanged.connect(self.check_url_for_token)
            
            layout.addWidget(self.web_view)
            
            # Status label
            self.status_label = QLabel("Loading login page...")
            self.status_label.setStyleSheet("padding: 5px; color: #666;")
            layout.addWidget(self.status_label)
            
            # Buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.rejected.connect(self.reject)
            layout.addWidget(button_box)
            
            self.setLayout(layout)
        
        def check_url_for_token(self, url: QUrl):
            """Check if URL contains request token"""
            url_str = url.toString()
            
            # Update status
            self.status_label.setText(f"Loading: {url.host()}")
            
            # Check for request token in URL
            if "request_token=" in url_str:
                try:
                    # Extract request token
                    parsed_url = urlparse(url_str)
                    query_params = parse_qs(parsed_url.query)
                    
                    if 'request_token' in query_params:
                        self.request_token = query_params['request_token'][0]
                        logger.info(f"Found request token: {self.request_token}")
                        
                        # Update status
                        self.status_label.setText("✅ Login successful! Processing...")
                        
                        # Emit signal and close dialog
                        self.login_complete.emit(self.request_token)
                        
                        # Close dialog after short delay
                        QTimer.singleShot(1000, self.accept)
                        
                except Exception as e:
                    logger.error(f"Error extracting request token: {e}")
        
        def closeEvent(self, event):
            """Handle close event"""
            if not self.request_token:
                reply = QMessageBox.question(
                    self, "Cancel Login",
                    "Are you sure you want to cancel the login process?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()
else:
    # Define a dummy class if WebEngine is not available
    EmbeddedBrowserDialog = type('EmbeddedBrowserDialog', (QDialog,), {})
    logger.info("Embedded browser functionality disabled (PyQt6-WebEngine not available)")

# ============================================================================
# AUTO LOGIN MANAGER
# ============================================================================

class AutoLoginManager(QObject):
    """Manages automatic login to Zerodha"""
    
    login_complete = pyqtSignal(str)  # Signal with request_token
    login_failed = pyqtSignal(str)    # Signal with error message
    
    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        self.oauth_server = None
        self.login_dialog = None
        self.request_token = None
        self.kite = None
    
    def start_auto_login(self, parent_widget=None):
        """Start automatic login process"""
        try:
            if not self.config.api_key:
                self.login_failed.emit("API key is not set. Please configure in Settings.")
                return False
            
            # Create KiteConnect instance
            self.kite = KiteConnect(api_key=self.config.api_key)
            
            # Determine redirect URL
            redirect_url = f"http://localhost:{self.config.redirect_port}/"
            
            # Generate login URL
            login_url = self.kite.login_url() + f"&redirect_uri={redirect_url}"
            
            logger.info(f"Login URL: {login_url}")
            
            # Check if we can use embedded browser
            can_use_embedded = WEBENGINE_AVAILABLE and self.config.use_embedded_browser
            
            if can_use_embedded:
                # Use embedded browser
                return self._login_with_embedded_browser(login_url, parent_widget)
            else:
                # Use external browser with OAuth server
                if self.config.use_embedded_browser and not WEBENGINE_AVAILABLE:
                    logger.warning("Embedded browser requested but PyQt6-WebEngine not available. Using external browser.")
                return self._login_with_external_browser(login_url, parent_widget)
                
        except Exception as e:
            error_msg = f"Failed to start auto login: {e}"
            logger.error(error_msg)
            self.login_failed.emit(error_msg)
            return False
    
    def _login_with_embedded_browser(self, login_url: str, parent_widget):
        """Login using embedded browser"""
        try:
            self.login_dialog = EmbeddedBrowserDialog(login_url, parent_widget)
            self.login_dialog.login_complete.connect(self._handle_login_complete)
            
            # Show dialog modally
            if self.login_dialog.exec():
                if self.request_token:
                    return True
                else:
                    self.login_failed.emit("Login was cancelled or failed.")
                    return False
            else:
                self.login_failed.emit("Login dialog was closed.")
                return False
                
        except Exception as e:
            error_msg = f"Embedded browser login failed: {e}"
            logger.error(error_msg)
            self.login_failed.emit(error_msg)
            return False
    
    def _login_with_external_browser(self, login_url: str, parent_widget):
        """Login using external browser with OAuth server"""
        try:
            # Start OAuth server
            self.oauth_server = OAuthServer(
                port=self.config.redirect_port,
                callback=self._handle_request_token
            )
            
            if not self.oauth_server.start():
                self.login_failed.emit("Failed to start OAuth server.")
                return False
            
            # Open browser
            logger.info("Opening login page in external browser...")
            webbrowser.open(login_url)
            
            # Show waiting dialog
            return self._show_waiting_dialog(parent_widget)
            
        except Exception as e:
            error_msg = f"External browser login failed: {e}"
            logger.error(error_msg)
            self.login_failed.emit(error_msg)
            return False
    
    def _show_waiting_dialog(self, parent_widget):
        """Show waiting dialog for OAuth callback"""
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle("Waiting for Login")
        dialog.setGeometry(350, 350, 400, 200)
        
        layout = QVBoxLayout()
        
        # Message
        message = QLabel(
            "Please complete the login in your browser.\n\n"
            "1. Login with your Zerodha credentials\n"
            "2. Authorize the application\n"
            "3. You will be redirected automatically\n\n"
            "Do not close this window."
        )
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)
        
        # Timer label
        self.timer_label = QLabel("Waiting...")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timer_label)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        layout.addWidget(cancel_btn)
        
        dialog.setLayout(layout)
        
        # Timer for countdown
        self.wait_seconds = 300  # 5 minutes timeout
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self._update_timer(dialog))
        self.timer.start(1000)  # Update every second
        
        # Store reference
        self.waiting_dialog = dialog
        
        # Show dialog
        result = dialog.exec()
        
        # Cleanup
        self.timer.stop()
        if self.oauth_server:
            self.oauth_server.stop()
        
        return result == QDialog.DialogCode.Accepted
    
    def _update_timer(self, dialog):
        """Update timer for waiting dialog"""
        self.wait_seconds -= 1
        
        if self.wait_seconds <= 0:
            dialog.reject()
            self.login_failed.emit("Login timeout. Please try again.")
            return
        
        minutes = self.wait_seconds // 60
        seconds = self.wait_seconds % 60
        
        self.timer_label.setText(f"Waiting... ({minutes:02d}:{seconds:02d} remaining)")
        
        # Check if we got the token
        if self.request_token:
            dialog.accept()
    
    def _handle_request_token(self, request_token: str):
        """Handle request token from OAuth server"""
        try:
            logger.info(f"Received request token: {request_token}")
            self.request_token = request_token
            
            # Generate session
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.config.api_secret
            )
            
            # Update config
            self.config.access_token = data['access_token']
            self.config.save()
            
            logger.info("Session generated successfully")
            
            # Emit signal
            self.login_complete.emit(request_token)
            
            # Stop OAuth server
            if self.oauth_server:
                self.oauth_server.stop()
            
        except Exception as e:
            error_msg = f"Failed to generate session: {e}"
            logger.error(error_msg)
            self.login_failed.emit(error_msg)
    
    def _handle_login_complete(self, request_token: str):
        """Handle login complete from embedded browser"""
        try:
            logger.info(f"Received request token from embedded browser: {request_token}")
            self.request_token = request_token
            
            # Generate session
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.config.api_secret
            )
            
            # Update config
            self.config.access_token = data['access_token']
            self.config.save()
            
            logger.info("Session generated successfully")
            
            # Emit signal
            self.login_complete.emit(request_token)
            
        except Exception as e:
            error_msg = f"Failed to generate session: {e}"
            logger.error(error_msg)
            self.login_failed.emit(error_msg)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.oauth_server:
            self.oauth_server.stop()
        
        if self.login_dialog:
            self.login_dialog.close()

# ============================================================================
# ENHANCED MARKET DATA MANAGER
# ============================================================================

# Update the AdvancedMarketData class with proper WebSocket handling

# Update the AdvancedMarketData class to properly emit tick data

class AdvancedMarketData(QObject):
    """Advanced market data with bid/ask support"""
    
    tick_received = pyqtSignal(dict)
    connection_status = pyqtSignal(str, bool)
    
    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        self.kite = None
        self.kws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self._stopping = False
        self.tokens = []
        
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
        
        # Last tick times to avoid duplicates
        self.last_tick_time = {'near': None, 'mid': None}
    
    def initialize(self, kite: KiteConnect):
        """Initialize with KiteConnect"""
        self.kite = kite
        logger.info("AdvancedMarketData initialized")
    
    def start_websocket(self):
        """Start WebSocket connection"""
        try:
            if not self.kite:
                logger.error("KiteConnect not initialized")
                return False
            
            # Get instrument tokens
            near_token = self.config.near_contract_token
            mid_token = self.config.mid_contract_token
            
            if not near_token or not mid_token:
                logger.error("Contract tokens not set")
                return False
            
            # Reset stopping flag
            self._stopping = False
            
            # Store tokens
            self.tokens = [int(near_token), int(mid_token)]
            
            # Create KiteTicker
            self.kws = KiteTicker(
                api_key=self.config.api_key,
                access_token=self.config.access_token
            )
            
            # Set callbacks
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            # Start in thread
            self.ws_thread = threading.Thread(target=self._websocket_worker, daemon=True)
            self.ws_thread.start()
            
            logger.info(f"WebSocket starting for tokens: {self.tokens}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _websocket_worker(self):
        """Worker function for WebSocket connection"""
        try:
            if self.kws and not self._stopping:
                self.kws.connect(threaded=True)
        except Exception as e:
            logger.error(f"WebSocket worker error: {e}")
    
    def _on_connect(self, ws, response):
        """WebSocket connected - subscribe to tokens"""
        try:
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # Subscribe to full mode for depth data
            if self.kws and self.tokens:
                self.kws.subscribe(self.tokens)
                self.kws.set_mode(self.kws.MODE_FULL, self.tokens)
                logger.info(f"Subscribed to tokens: {self.tokens}")
            
            self.connection_status.emit("WebSocket Connected", True)
            logger.info("WebSocket connected and subscribed")
            
        except Exception as e:
            logger.error(f"Error in on_connect: {e}")
            self.connection_status.emit(f"Connection Error: {e}", False)
    
    def _on_close(self, ws, code, reason):
        """WebSocket closed"""
        self.is_connected = False
        self.connection_status.emit(f"WebSocket Closed: {reason}", False)
        logger.warning(f"WebSocket closed: {reason} (code: {code})")
        
        # Attempt reconnect only if we're not stopping
        if not self._stopping:
            self._attempt_reconnect()
    
    def _on_error(self, ws, error):
        """WebSocket error"""
        logger.error(f"WebSocket error: {error}")
        self.connection_status.emit(f"WebSocket Error: {error}", False)
    
    def _attempt_reconnect(self):
        """Attempt to reconnect"""
        if (not self._stopping and 
            self.reconnect_attempts < self.config.max_reconnect_attempts):
            self.reconnect_attempts += 1
            delay = self.config.reconnect_delay * self.reconnect_attempts
            
            logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})")
            threading.Timer(delay, self._safe_reconnect).start()
    
    def _safe_reconnect(self):
        """Safely attempt reconnect"""
        if not self._stopping:
            try:
                # Clean up existing connection
                if self.kws:
                    try:
                        if hasattr(self.kws, '_ws') and self.kws._ws:
                            self.kws.close()
                    except:
                        pass
                    self.kws = None
                
                # Start new connection
                self.start_websocket()
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks"""
        try:
            for tick in ticks:
                token = tick['instrument_token']
                timestamp = datetime.now()
                
                # Determine which contract this is
                contract_type = None
                if token == int(self.config.near_contract_token):
                    contract_type = 'near'
                elif token == int(self.config.mid_contract_token):
                    contract_type = 'mid'
                
                if contract_type:
                    # Skip duplicate ticks (same timestamp)
                    current_time = time.time()
                    if (self.last_tick_time[contract_type] and 
                        current_time - self.last_tick_time[contract_type] < 0.1):
                        continue
                    
                    self.last_tick_time[contract_type] = current_time
                    
                    # Update market data
                    self.market_data[contract_type] = {
                        'timestamp': timestamp,
                        'last_price': tick['last_price'],
                        'last_traded_price': tick.get('last_traded_price', tick['last_price']),
                        'volume': tick.get('volume_traded', 0),
                        'oi': tick.get('oi', 0)
                    }
                    
                    # Update market depth (bid/ask)
                    if 'depth' in tick:
                        bids = tick['depth'].get('buy', [])
                        asks = tick['depth'].get('sell', [])
                        
                        self.market_depth[contract_type] = {
                            'bids': bids[:5],
                            'asks': asks[:5]
                        }
                    
                    # Update price history
                    self.price_history[contract_type].append({
                        'timestamp': timestamp,
                        'price': tick['last_price']
                    })
                    
                    # Trim history
                    if len(self.price_history[contract_type]) > self.max_history:
                        self.price_history[contract_type] = self.price_history[contract_type][-self.max_history:]
                    
                    # Emit tick data immediately
                    tick_data = {
                        'contract': contract_type,
                        'timestamp': timestamp,
                        'last_price': tick['last_price'],
                        'best_bid': self.get_best_bid(contract_type),
                        'best_ask': self.get_best_ask(contract_type),
                        'volume': tick.get('volume_traded', 0),
                        'depth': self.market_depth[contract_type]
                    }
                    
                    # Emit signal on main thread
                    self.tick_received.emit(tick_data)
                    
        except Exception as e:
            logger.error(f"Error processing ticks: {e}")
    
    def get_best_bid(self, contract_type: str) -> float:
        """Get best bid price"""
        if contract_type in self.market_depth and self.market_depth[contract_type]['bids']:
            return self.market_depth[contract_type]['bids'][0]['price']
        return 0.0
    
    def get_best_ask(self, contract_type: str) -> float:
        """Get best ask price"""
        if contract_type in self.market_depth and self.market_depth[contract_type]['asks']:
            return self.market_depth[contract_type]['asks'][0]['price']
        return 0.0
    
    def get_market_summary(self, contract_type: str) -> Dict:
        """Get market summary for contract"""
        if contract_type not in self.market_data:
            return {}
        
        return {
            'last_price': self.market_data[contract_type].get('last_price', 0),
            'last_traded_price': self.market_data[contract_type].get('last_traded_price', 0),
            'best_bid': self.get_best_bid(contract_type),
            'best_ask': self.get_best_ask(contract_type),
            'bid_ask_spread': self.get_best_ask(contract_type) - self.get_best_bid(contract_type),
            'volume': self.market_data[contract_type].get('volume', 0),
            'timestamp': self.market_data[contract_type].get('timestamp')
        }
    
    def stop(self):
        """Stop WebSocket safely"""
        try:
            self._stopping = True
            
            if self.kws:
                try:
                    # Unsubscribe first
                    if self.tokens and self.is_connected:
                        try:
                            self.kws.unsubscribe(self.tokens)
                        except:
                            pass
                    
                    # Close connection
                    self.kws.close()
                    
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
                finally:
                    self.kws = None
                    self.is_connected = False
            
            logger.info("Market data stopped")
            
        except Exception as e:
            logger.error(f"Error stopping market data: {e}")

# ============================================================================
# POSITION MANAGER
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
        
        logger.info(f"Closed position: {exit_reason}, P&L: {self.current_position.calculate_pnl():.2f}")
        
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
# SIMPLE TOKEN UPDATE DIALOG
# ============================================================================

class SimpleTokenUpdateDialog(QDialog):
    """Simple dialog to update access token"""
    
    def __init__(self, current_token: str = "", parent=None):
        super().__init__(parent)
        self.current_token = current_token
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Update Access Token")
        self.setGeometry(400, 300, 500, 250)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Update Zerodha Access Token")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "1. Login to Zerodha in browser\n"
            "2. Get new access token from console\n"
            "3. Paste below and click Update"
        )
        instructions.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(instructions)
        
        # Token input
        layout.addWidget(QLabel("Access Token:"))
        
        self.token_input = QLineEdit()
        if self.current_token:
            self.token_input.setText(self.current_token)
        self.token_input.setPlaceholderText("Paste your new access token here...")
        
        # Show/hide checkbox
        show_token = QCheckBox("Show token")
        show_token.stateChanged.connect(self.toggle_token_visibility)
        
        token_layout = QVBoxLayout()
        token_layout.addWidget(self.token_input)
        token_layout.addWidget(show_token)
        
        layout.addLayout(token_layout)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def toggle_token_visibility(self, state):
        if state:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
    
    def get_token(self):
        return self.token_input.text().strip()

# ============================================================================
# UPDATED ADVANCED TRADING DASHBOARD WITH AUTO-LOGIN
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
        
        # Adjust config based on availability
        if not WEBENGINE_AVAILABLE and self.config.use_embedded_browser:
            self.config.use_embedded_browser = False
            self.config.save()
            logger.info("Disabled embedded browser setting (PyQt6-WebEngine not available)")
        
        # Components
        self.kite = None
        self.market_data = None
        self.position_manager = None
        self.auto_login_manager = None
        
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
        
        # ===== PROFIT/LOSS MONITORING PANEL =====
        monitor_group = QGroupBox("Profit/Loss Monitoring")
        monitor_layout = QGridLayout()
        
        # Current P&L Status
        monitor_layout.addWidget(QLabel("Current P&L:"), 0, 0)
        self.current_pnl_label = QLabel("0.00")
        self.current_pnl_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        monitor_layout.addWidget(self.current_pnl_label, 0, 1)
        
        monitor_layout.addWidget(QLabel("P&L %:"), 0, 2)
        self.current_pnl_percent_label = QLabel("0.00%")
        monitor_layout.addWidget(self.current_pnl_percent_label, 0, 3)
        
        monitor_layout.addWidget(QLabel("Target Profit:"), 0, 4)
        self.target_profit_label = QLabel("0.00")
        self.target_profit_label.setStyleSheet("color: green; font-weight: bold;")
        monitor_layout.addWidget(self.target_profit_label, 0, 5)
        
        monitor_layout.addWidget(QLabel("Stop Loss:"), 1, 0)
        self.stop_loss_monitor_label = QLabel("0.00")
        self.stop_loss_monitor_label.setStyleSheet("color: red; font-weight: bold;")
        monitor_layout.addWidget(self.stop_loss_monitor_label, 1, 1)
        
        monitor_layout.addWidget(QLabel("Distance to Target:"), 1, 2)
        self.distance_to_target_label = QLabel("0.00")
        monitor_layout.addWidget(self.distance_to_target_label, 1, 3)
        
        monitor_layout.addWidget(QLabel("Distance to Stop:"), 1, 4)
        self.distance_to_stop_label = QLabel("0.00")
        monitor_layout.addWidget(self.distance_to_stop_label, 1, 5)
        
        # Progress Bar for P&L
        monitor_layout.addWidget(QLabel("P&L Progress:"), 2, 0)
        self.pnl_progress_bar = QProgressBar()
        self.pnl_progress_bar.setRange(-100, 100)  # -100% to +100%
        self.pnl_progress_bar.setValue(0)
        self.pnl_progress_bar.setFormat("%v%")
        monitor_layout.addWidget(self.pnl_progress_bar, 2, 1, 1, 5)
        
        # Risk/Reward Ratio
        monitor_layout.addWidget(QLabel("Risk/Reward Ratio:"), 3, 0)
        self.risk_reward_label = QLabel("1:2")
        self.risk_reward_label.setStyleSheet("font-weight: bold;")
        monitor_layout.addWidget(self.risk_reward_label, 3, 1)
        
        # Time to Target/Stop Estimate
        monitor_layout.addWidget(QLabel("Time in Trade:"), 3, 2)
        self.time_in_trade_label = QLabel("0:00")
        monitor_layout.addWidget(self.time_in_trade_label, 3, 3)
        
        # Exit Probability (estimated)
        monitor_layout.addWidget(QLabel("Exit Probability:"), 3, 4)
        self.exit_probability_label = QLabel("0%")
        monitor_layout.addWidget(self.exit_probability_label, 3, 5)
        
        monitor_group.setLayout(monitor_layout)
        main_layout.addWidget(monitor_group)

        # ===== CONTROL PANEL =====
        control_group = QGroupBox("Trading Controls")
        control_layout = QHBoxLayout()
        
        # SETTINGS BUTTON
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        self.settings_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        
        # UPDATE TOKEN BUTTON - NEW ADDITION
        self.update_token_btn = QPushButton("Update Token")
        self.update_token_btn.clicked.connect(self.simple_token_update)
        self.update_token_btn.setStyleSheet("background-color: #673AB7; color: white; padding: 8px;")
        self.update_token_btn.setToolTip("Quickly update access token when it expires")
        
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
        control_layout.addWidget(self.update_token_btn)  # NEW ADDITION
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
        self.auto_login_manager = AutoLoginManager(self.config)
        
        # Connect signals
        self.market_data.tick_received.connect(self.handle_tick_data)
        self.market_data.connection_status.connect(self.handle_connection_status)
        
        # Connect auto-login signals
        self.auto_login_manager.login_complete.connect(self._on_auto_login_complete)
        self.auto_login_manager.login_failed.connect(self._on_auto_login_failed)
        
        # Initialize UI state
        self.update_auto_entry_button()
        self.update_auto_exit_button()
    
    def simple_token_update(self):
        """Quick token update"""
        # Get current token
        current = self.config.access_token[:20] + "..." if self.config.access_token else "None"
        
        # Show input dialog
        token, ok = QInputDialog.getText(
            self,
            "Update Access Token",
            f"Current token: {current}\n\nEnter new access token:",
            QLineEdit.EchoMode.Password,
            ""
        )
        
        if ok and token:
            try:
                # Test token
                kite_test = KiteConnect(api_key=self.config.api_key)
                kite_test.set_access_token(token)
                profile = kite_test.profile()
                
                # Update config
                self.config.access_token = token
                self.config.save()
                
                # Update existing connection
                if self.kite:
                    self.kite.set_access_token(token)
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Token updated!\nUser: {profile['user_name']}"
                )
                
                # Restart WebSocket if running
                if self.is_monitoring:
                    self.stop_trading()
                    self.start_trading()
                    
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Invalid Token",
                    f"Token is invalid or expired:\n{str(e)}"
                )
    
    def update_access_token_simple(self, new_token: str):
        """Simple method to update access token"""
        try:
            # Update config
            self.config.access_token = new_token.strip()
            
            # Save to file
            self.config.save()
            
            # Update KiteConnect instance if exists
            if self.kite:
                self.kite.set_access_token(new_token)
                
            self.log_message(f"Access token updated successfully")
            QMessageBox.information(self, "Success", "Access token updated!")
            
            # Reconnect if needed
            if self.is_connected:
                self.log_message("Reconnecting with new token...")
                self.disconnect_zerodha()
                QTimer.singleShot(1000, self.connect_to_zerodha)
                
            return True
            
        except Exception as e:
            error_msg = f"Failed to update token: {e}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            return False
    
    def toggle_auto_exit(self):
        """Toggle auto exit"""
        self.config.auto_exit_enabled = not self.config.auto_exit_enabled
        self.config.save()
        self.update_auto_exit_button()
        
        status = "ENABLED" if self.config.auto_exit_enabled else "DISABLED"
        self.log_message(f"Auto exit {status}")

    def toggle_auto_entry(self):
        """Toggle auto entry"""
        self.config.auto_entry_enabled = not self.config.auto_entry_enabled
        self.config.save()
        self.update_auto_entry_button()
        
        status = "ENABLED" if self.config.auto_entry_enabled else "DISABLED"
        self.log_message(f"Auto entry {status}")

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
                # Start auto-login process
                self.log_message("Starting auto-login process...")
                
                # Disable connect button during login
                self.connect_btn.setEnabled(False)
                self.connect_btn.setText("Logging in...")
                
                # Start auto-login
                if not self.auto_login_manager.start_auto_login(self):
                    self.connect_btn.setEnabled(True)
                    self.connect_btn.setText("Connect to Zerodha")
                
        except KiteException as e:
            if "Invalid access token" in str(e):
                self.log_message("Access token expired. Please login again.")
                self.config.access_token = ""
                self.config.save()
                self.connect_to_zerodha()
            else:
                self.show_error("Connection Failed", f"Error: {str(e)}")
                self.connect_btn.setEnabled(True)
                self.connect_btn.setText("Connect to Zerodha")
        except Exception as e:
            self.show_error("Connection Error", f"Unexpected error: {str(e)}")
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("Connect to Zerodha")
    
    # def _on_auto_login_complete(self, request_token: str):
    #     """Handle successful auto-login"""
    #     try:
    #         logger.info(f"Auto-login complete with request token: {request_token}")
            
    #         # 1. Set the NEW access token
    #         self.config.access_token = self.config.access_token  # Already updated by login manager
    #         self.kite.set_access_token(self.config.access_token)
    #         self.is_connected = True
            
    #         # 2. Get profile (verifies the new token works)
    #         profile = self.kite.profile()
    #         self.log_message(f"Login successful! User: {profile['user_name']} (Token Updated)")
    #         self.update_connection_status(f"Connected as {profile['user_name']}", True)
            
    #         # 3. Restart market data if it was running
    #         if self.is_monitoring:
    #             # Stop current market data
    #             if self.market_data:
    #                 self.market_data.stop()
    #                 time.sleep(1)  # Brief pause for cleanup
                
    #             # Re-initialize market data with updated config
    #             self.market_data = AdvancedMarketData(self.config)
    #             self.market_data.initialize(self.kite)
    #             self.market_data.tick_received.connect(self.handle_tick_data)
    #             self.market_data.connection_status.connect(self.handle_connection_status)
                
    #             # Restart WebSocket
    #             self.log_message("Restarting market data stream with new token...")
    #             if self.market_data.start_websocket():
    #                 self.log_message("Market data restarted successfully")
    #             else:
    #                 self.log_message("Failed to restart market data")
    #                 self.is_monitoring = False
    #                 self.start_btn.setText("▶ Start Trading")
    #                 self.start_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
    #                 self.data_status.setText("Data: Stopped")
            
    #         # 4. Update UI state
    #         self.connect_btn.setEnabled(False)
    #         self.disconnect_btn.setEnabled(True)
    #         self.start_btn.setEnabled(True)
    #         self.select_contracts_btn.setEnabled(True)
    #         self.connect_btn.setText("Connect to Zerodha")
            
    #         # Re-enable trading buttons if market data is running
    #         if self.is_monitoring:
    #             self.buy_spread_btn.setEnabled(True)
    #             self.sell_spread_btn.setEnabled(True)
            
    #     except Exception as e:
    #         error_msg = f"Failed to complete login: {e}"
    #         logger.error(error_msg)
    #         self._on_auto_login_failed(error_msg)

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
                # Reset milestone alerts
                self._reset_milestone_alerts()
                
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

    def _reset_milestone_alerts(self):
        """Reset milestone alert flags"""
        if hasattr(self, '_half_target_alerted'):
            self._half_target_alerted = False
        if hasattr(self, '_three_quarter_target_alerted'):
            self._three_quarter_target_alerted = False
        if hasattr(self, '_full_target_alerted'):
            self._full_target_alerted = False
        if hasattr(self, '_near_stop_alerted'):
            self._near_stop_alerted = False


    # def handle_tick_data(self, tick_data):
    #     """Handle incoming tick data"""
    #     try:
    #         self.debug_tick_data(tick_data)
    #         contract_type = tick_data['contract']
            
    #         # Update market data display using QTimer to ensure GUI updates happen on main thread
    #         if contract_type == 'near':
    #             # Update near contract data
    #             price = tick_data['last_price']
    #             bid = tick_data.get('best_bid', price)
    #             ask = tick_data.get('best_ask', price)
                
    #             # Update labels using lambda to ensure proper thread handling
    #             QTimer.singleShot(0, lambda: self.near_price_label.setText(f"{price:.2f}"))
    #             QTimer.singleShot(0, lambda: self.near_bid_label.setText(f"{bid:.2f}"))
    #             QTimer.singleShot(0, lambda: self.near_ask_label.setText(f"{ask:.2f}"))
                
    #             # Store the price for spread calculation
    #             self.near_price = price
    #             self.near_bid = bid
    #             self.near_ask = ask
                
    #         else:  # 'mid' contract
    #             # Update mid contract data
    #             price = tick_data['last_price']
    #             bid = tick_data.get('best_bid', price)
    #             ask = tick_data.get('best_ask', price)
                
    #             # Update labels
    #             QTimer.singleShot(0, lambda: self.mid_price_label.setText(f"{price:.2f}"))
    #             QTimer.singleShot(0, lambda: self.mid_bid_label.setText(f"{bid:.2f}"))
    #             QTimer.singleShot(0, lambda: self.mid_ask_label.setText(f"{ask:.2f}"))
                
    #             # Store the price for spread calculation
    #             self.mid_price = price
    #             self.mid_bid = bid
    #             self.mid_ask = ask
            
    #         # Calculate and update spread when we have both prices
    #         if hasattr(self, 'near_price') and hasattr(self, 'mid_price'):
    #             spread = self.near_price - self.mid_price
                
    #             # Update spread display
    #             QTimer.singleShot(0, lambda: self.spread_label.setText(f"{spread:.2f}"))
                
    #             # Update spread history for signal calculation
    #             self.spread_history.append(spread)
    #             if len(self.spread_history) > self.max_spread_history:
    #                 self.spread_history = self.spread_history[-self.max_spread_history:]
                
    #             # Calculate bid-ask spread
    #             if hasattr(self, 'near_bid') and hasattr(self, 'near_ask') and hasattr(self, 'mid_bid') and hasattr(self, 'mid_ask'):
    #                 near_bid_ask = self.near_ask - self.near_bid
    #                 mid_bid_ask = self.mid_ask - self.mid_bid
    #                 total_bid_ask = near_bid_ask + mid_bid_ask
                    
    #                 QTimer.singleShot(0, lambda: self.bid_ask_spread_label.setText(f"{total_bid_ask:.2f}"))
                
    #             # Update volume if available
    #             volume = tick_data.get('volume', 0)
    #             if volume > 0:
    #                 QTimer.singleShot(0, lambda: self.volume_label.setText(f"{volume:,}"))
                
    #             # Update position if open
    #             if self.position_manager.current_position and self.position_manager.current_position.is_open:
    #                 exit_reason = self.position_manager.update_position(spread, self.near_price, self.mid_price)
                    
    #                 if exit_reason and self.config.auto_exit_enabled:
    #                     self.close_position(exit_reason)
    #                 else:
    #                     # Update position display in UI thread
    #                     QTimer.singleShot(0, self.update_position_display)
                
    #             # Calculate trading signal
    #             signal, stats = self.calculate_signal(spread)
                
    #             # Update signal display
    #             QTimer.singleShot(0, lambda: self.update_signal_display(signal, stats))
                
    #             # Check for trading signals (auto entry)
    #             if (self.config.auto_entry_enabled and 
    #                 not (self.position_manager.current_position and 
    #                     self.position_manager.current_position.is_open) and
    #                 signal != "NEUTRAL"):
                    
    #                 self.log_message(f"Auto entry signal: {signal}")
                    
    #                 # Check if we should auto enter
    #                 if self.should_auto_enter(signal, spread, stats):
    #                     QTimer.singleShot(0, lambda: self.place_trade(signal, auto=True))
            
    #     except Exception as e:
    #         logger.error(f"Error handling tick data: {e}")
    #         # Log the error but don't crash
    
    def handle_tick_data(self, tick_data):
        """Handle incoming tick data"""
        try:
            contract_type = tick_data['contract']
            
            # Update market data display
            if contract_type == 'near':
                price = tick_data['last_price']
                bid = tick_data.get('best_bid', price)
                ask = tick_data.get('best_ask', price)
                
                self.near_price_label.setText(f"{price:.2f}")
                self.near_bid_label.setText(f"{bid:.2f}")
                self.near_ask_label.setText(f"{ask:.2f}")
                
                self.near_price = price
                self.near_bid = bid
                self.near_ask = ask
                
            else:  # 'mid' contract
                price = tick_data['last_price']
                bid = tick_data.get('best_bid', price)
                ask = tick_data.get('best_ask', price)
                
                self.mid_price_label.setText(f"{price:.2f}")
                self.mid_bid_label.setText(f"{bid:.2f}")
                self.mid_ask_label.setText(f"{ask:.2f}")
                
                self.mid_price = price
                self.mid_bid = bid
                self.mid_ask = ask
            
            # Calculate and update spread when we have both prices
            if hasattr(self, 'near_price') and hasattr(self, 'mid_price'):
                spread = self.near_price - self.mid_price
                self.spread_label.setText(f"{spread:.2f}")
                
                # Update spread history
                self.spread_history.append(spread)
                if len(self.spread_history) > self.max_spread_history:
                    self.spread_history = self.spread_history[-self.max_spread_history:]
                
                # Update position if open
                if self.position_manager.current_position and self.position_manager.current_position.is_open:
                    exit_reason = self.position_manager.update_position(spread, self.near_price, self.mid_price)
                    
                    # Check P&L milestones
                    summary = self.position_manager.get_position_summary()
                    if summary:
                        self.check_pnl_milestones(summary)
                    
                    if exit_reason and self.config.auto_exit_enabled:
                        self.close_position(exit_reason)
                    else:
                        self.update_position_display()
                
                # Calculate trading signal
                signal, stats = self.calculate_signal(spread)
                self.update_signal_display(signal, stats)
                
                # Check for trading signals (auto entry)
                if (self.config.auto_entry_enabled and 
                    not (self.position_manager.current_position and 
                        self.position_manager.current_position.is_open) and
                    signal != "NEUTRAL"):
                    
                    if self.should_auto_enter(signal, spread, stats):
                        self.place_trade(signal, auto=True)
            
        except Exception as e:
            logger.error(f"Error handling tick data: {e}")

    def update_auto_exit_button(self):
        """Update auto exit button display"""
        if self.config.auto_exit_enabled:
            self.auto_exit_btn.setText("Auto Exit: ON")
            self.auto_exit_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        else:
            self.auto_exit_btn.setText("Auto Exit: OFF")
            self.auto_exit_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 10px;")
            
    def update_auto_entry_button(self):
        """Update auto entry button display"""
        if self.config.auto_entry_enabled:
            self.auto_entry_btn.setText("Auto Entry: ON")
            self.auto_entry_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        else:
            self.auto_entry_btn.setText("Auto Entry: OFF")
            self.auto_entry_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 10px;")

    def handle_connection_status(self, status: str, connected: bool):
        """Handle connection status updates"""
        if connected:
            self.connection_status.setText(f"🟢 {status}")
            self.connection_status.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        else:
            self.connection_status.setText(f"🔴 {status}")
            self.connection_status.setStyleSheet("color: red; font-weight: bold; padding: 5px;")

    def _on_auto_login_complete(self, request_token: str):
        """Handle successful auto-login"""
        try:
            logger.info(f"Auto-login complete with request token: {request_token}")
            
            # 1. Stop trading if active
            was_trading = self.is_monitoring
            if was_trading:
                self.stop_trading()
                time.sleep(2)  # Give time for cleanup
            
            # 2. Set the NEW access token
            self.kite.set_access_token(self.config.access_token)
            self.is_connected = True
            
            # 3. Get profile (verifies the new token works)
            profile = self.kite.profile()
            self.log_message(f"Login successful! User: {profile['user_name']} (Token Updated)")
            self.update_connection_status(f"Connected as {profile['user_name']}", True)
            
            # 4. Re-initialize market data
            self.market_data = AdvancedMarketData(self.config)
            self.market_data.initialize(self.kite)
            self.market_data.tick_received.connect(self.handle_tick_data)
            self.market_data.connection_status.connect(self.handle_connection_status)
            
            # 5. Update UI state
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.select_contracts_btn.setEnabled(True)
            self.connect_btn.setText("Connect to Zerodha")
            
            # 6. Restart trading if it was active
            if was_trading:
                self.log_message("Restarting market data with new token...")
                QTimer.singleShot(3000, self.start_trading)  # Delay to ensure cleanup
            
        except Exception as e:
            error_msg = f"Failed to complete login: {e}"
            logger.error(error_msg)
            self._on_auto_login_failed(error_msg)
    
    def _on_auto_login_failed(self, error_msg: str):
        """Handle auto-login failure"""
        self.show_error("Login Failed", error_msg)
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect to Zerodha")
    
    # def open_settings(self):
    #     """Open settings dialog"""
    #     dialog = AdvancedSettingsDialog(self.config, self)
    #     if dialog.exec():
    #         # Reload configuration if settings were saved
    #         self.config.load()
            
    #         # Update UI elements that might have changed
    #         self.update_auto_entry_button()
    #         self.update_auto_exit_button()
            
    #         # Update position manager with new config
    #         if self.position_manager:
    #             self.position_manager.config = self.config
            
    #         # Update market data with new config
    #         if self.market_data:
    #             self.market_data.config = self.config
            
    #         # Update auto-login manager
    #         if self.auto_login_manager:
    #             self.auto_login_manager.config = self.config
            
    #         self.log_message("Settings updated and saved")
            
    #         # Auto-login if enabled and we have API credentials but not connected
    #         if (self.config.auto_login_enabled and 
    #             self.config.api_key and 
    #             self.config.api_secret and 
    #             not self.is_connected):
                
    #             # Check if we have access token
    #             if self.config.access_token:
    #                 # Try to connect with existing token
    #                 self.auto_login_with_token()
    #             else:
    #                 # Ask user if they want to login now
    #                 reply = QMessageBox.question(
    #                     self, "Auto Login",
    #                     "Would you like to login to Zerodha now?",
    #                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    #                     QMessageBox.StandardButton.Yes
    #                 )
                    
    #                 if reply == QMessageBox.StandardButton.Yes:
    #                     self.connect_to_zerodha()
    
    def open_settings(self):
        """Open settings dialog"""
        try:
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
                
                # Update auto-login manager
                if self.auto_login_manager:
                    self.auto_login_manager.config = self.config
                
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
        except Exception as e:
            self.log_message(f"Error opening settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open settings: {e}")

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
            # Check if we have contracts selected
            if not self.config.near_contract_token or not self.config.mid_contract_token:
                self.log_message("Cannot start trading: No contracts selected")
                return
            
            # Check if connected
            if not self.kite:
                self.log_message("Cannot start trading: Not connected to Zerodha")
                return
            
            # Start market data
            if self.market_data.start_websocket():
                self.is_monitoring = True
                self.start_btn.setText("⏸ Stop Trading")
                self.start_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
                self.data_status.setText("Data: Streaming")
                
                # Enable trading buttons after a delay to ensure connection
                QTimer.singleShot(3000, lambda: self._enable_trading_buttons(True))
                
                self.log_message("Trading started")
            else:
                self.log_message("Failed to start market data stream")
                self.is_monitoring = False
                
        except Exception as e:
            self.log_message(f"Error starting trading: {e}")
            self.is_monitoring = False
            self.start_btn.setText("▶ Start Trading")
            self.start_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
            self.data_status.setText("Data: Error")

    def _enable_trading_buttons(self, enable: bool):
        """Enable or disable trading buttons"""
        self.buy_spread_btn.setEnabled(enable and self.is_monitoring)
        self.sell_spread_btn.setEnabled(enable and self.is_monitoring)
    
    def stop_trading(self):
        """Stop trading"""
        try:
            self.is_monitoring = False
            if self.market_data:
                self.market_data.stop()
            
            self.start_btn.setText("▶ Start Trading")
            self.start_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
            self.data_status.setText("Data: Stopped")
            
            # Disable trading buttons
            self.buy_spread_btn.setEnabled(False)
            self.sell_spread_btn.setEnabled(False)
            
            self.log_message("Trading stopped")
        except Exception as e:
            self.log_message(f"Error stopping trading: {e}")
    
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
                
                # Update statistics display using QTimer
                QTimer.singleShot(0, lambda: self.zscore_label.setText(f"{z_score:.2f}"))
                QTimer.singleShot(0, lambda: self.spread_mean_label.setText(f"{spread_mean:.2f}"))
                
                # Calculate bands
                upper_band = spread_mean + (self.config.std_dev_multiplier * spread_std)
                lower_band = spread_mean - (self.config.std_dev_multiplier * spread_std)
                
                QTimer.singleShot(0, lambda: self.upper_band_label.setText(f"{upper_band:.2f}"))
                QTimer.singleShot(0, lambda: self.lower_band_label.setText(f"{lower_band:.2f}"))
                
                # Determine signal
                if z_score < -self.config.entry_zscore_threshold:
                    return "BUY_SPREAD", {'zscore': z_score, 'mean': spread_mean, 'std': spread_std}
                elif z_score > self.config.entry_zscore_threshold:
                    return "SELL_SPREAD", {'zscore': z_score, 'mean': spread_mean, 'std': spread_std}
                else:
                    return "NEUTRAL", {'zscore': z_score, 'mean': spread_mean, 'std': spread_std}
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
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
    
    def debug_tick_data(self, tick_data):
        """Debug method to log tick data"""
        logger.debug(f"Tick data received: {tick_data}")
        print(f"[DEBUG] Contract: {tick_data['contract']}, Price: {tick_data['last_price']:.2f}")

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
        """Update position display with monitoring"""
        summary = self.position_manager.get_position_summary()
        
        if not summary:
            # No position - reset all monitoring labels
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
            
            # Reset monitoring labels
            self.current_pnl_label.setText("0.00")
            self.current_pnl_percent_label.setText("0.00%")
            self.target_profit_label.setText("0.00")
            self.stop_loss_monitor_label.setText("0.00")
            self.distance_to_target_label.setText("0.00")
            self.distance_to_stop_label.setText("0.00")
            self.pnl_progress_bar.setValue(0)
            self.risk_reward_label.setText("0:0")
            self.time_in_trade_label.setText("0:00")
            self.exit_probability_label.setText("0%")
            
            # Reset signal label
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
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            seconds = duration.seconds % 60
            self.position_duration_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
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
    
        # ===== UPDATE MONITORING PANEL =====
        self.update_monitoring_panel(summary)
    

    def update_monitoring_panel(self, summary):
        """Update profit/loss monitoring panel"""
        try:
            # Get current values
            entry_spread = summary['entry_spread']
            current_spread = summary['current_spread']
            profit_target = summary['profit_target']
            stop_loss = summary['stop_loss']
            pnl = summary['pnl']
            pnl_percent = summary['pnl_percent']
            position_type = summary['type']
            
            # Update current P&L in monitoring panel
            self.current_pnl_label.setText(f"{pnl:+.2f}")
            self.current_pnl_percent_label.setText(f"{pnl_percent:+.2f}%")
            
            # Set P&L color
            if pnl > 0:
                self.current_pnl_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
                self.current_pnl_percent_label.setStyleSheet("color: green;")
            elif pnl < 0:
                self.current_pnl_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
                self.current_pnl_percent_label.setStyleSheet("color: red;")
            else:
                self.current_pnl_label.setStyleSheet("color: black; font-size: 16px; font-weight: bold;")
                self.current_pnl_percent_label.setStyleSheet("color: black;")
            
            # Update target and stop loss
            self.target_profit_label.setText(f"{profit_target:.2f}")
            self.stop_loss_monitor_label.setText(f"{stop_loss:.2f}")
            
            # Calculate distances to target and stop
            if position_type == "BUY_SPREAD":
                # For BUY_SPREAD, profit when spread increases
                distance_to_target = profit_target - current_spread
                distance_to_stop = current_spread - stop_loss
                
                # Calculate progress percentage (0-100%)
                total_range = profit_target - stop_loss
                if total_range > 0:
                    progress = ((current_spread - stop_loss) / total_range) * 100
                else:
                    progress = 50  # Default to middle
                
            else:  # SELL_SPREAD
                # For SELL_SPREAD, profit when spread decreases
                distance_to_target = current_spread - profit_target
                distance_to_stop = stop_loss - current_spread
                
                # Calculate progress percentage (0-100%)
                total_range = stop_loss - profit_target
                if total_range > 0:
                    progress = ((stop_loss - current_spread) / total_range) * 100
                else:
                    progress = 50  # Default to middle
            
            # Update distance labels
            self.distance_to_target_label.setText(f"{distance_to_target:+.2f}")
            self.distance_to_stop_label.setText(f"{distance_to_stop:+.2f}")
            
            # Set distance colors
            if distance_to_target > 0:
                self.distance_to_target_label.setStyleSheet("color: green;")
            else:
                self.distance_to_target_label.setStyleSheet("color: red;")
                
            if distance_to_stop > 0:
                self.distance_to_stop_label.setStyleSheet("color: green;")
            else:
                self.distance_to_stop_label.setStyleSheet("color: red;")
            
            # Update progress bar
            self.pnl_progress_bar.setValue(int(progress))
            
            # Update progress bar color based on value
            if progress >= 75:
                self.pnl_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid grey;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: green;
                        border-radius: 3px;
                    }
                """)
            elif progress >= 50:
                self.pnl_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid grey;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: orange;
                        border-radius: 3px;
                    }
                """)
            else:
                self.pnl_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid grey;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: red;
                        border-radius: 3px;
                    }
                """)
            
            # Calculate risk/reward ratio
            if position_type == "BUY_SPREAD":
                risk = entry_spread - stop_loss
                reward = profit_target - entry_spread
            else:
                risk = stop_loss - entry_spread
                reward = entry_spread - profit_target
            
            if risk > 0 and reward > 0:
                rr_ratio = reward / risk
                self.risk_reward_label.setText(f"1:{rr_ratio:.1f}")
            else:
                self.risk_reward_label.setText("N/A")
            
            # Update time in trade
            if summary['entry_time']:
                duration = datetime.now() - summary['entry_time']
                minutes = duration.seconds // 60
                seconds = duration.seconds % 60
                self.time_in_trade_label.setText(f"{minutes}:{seconds:02d}")
            
            # Calculate exit probability (simple estimation)
            exit_probability = self.calculate_exit_probability(summary)
            self.exit_probability_label.setText(f"{exit_probability:.0f}%")
            
            # Color code exit probability
            if exit_probability >= 70:
                self.exit_probability_label.setStyleSheet("color: green; font-weight: bold;")
            elif exit_probability >= 40:
                self.exit_probability_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.exit_probability_label.setStyleSheet("color: red; font-weight: bold;")
                
        except Exception as e:
            logger.error(f"Error updating monitoring panel: {e}")


    def calculate_exit_probability(self, summary):
        """Calculate estimated exit probability based on current position"""
        try:
            if not summary:
                return 0
            
            entry_spread = summary['entry_spread']
            current_spread = summary['current_spread']
            profit_target = summary['profit_target']
            stop_loss = summary['stop_loss']
            pnl_percent = summary['pnl_percent']
            position_type = summary['type']
            
            if position_type == "BUY_SPREAD":
                # For BUY_SPREAD, we want spread to increase
                total_range = profit_target - stop_loss
                
                if total_range > 0:
                    # How far are we from stop loss (0% at stop, 100% at target)
                    progress = ((current_spread - stop_loss) / total_range) * 100
                else:
                    progress = 50
            else:  # SELL_SPREAD
                # For SELL_SPREAD, we want spread to decrease
                total_range = stop_loss - profit_target
                
                if total_range > 0:
                    # How far are we from stop loss (0% at stop, 100% at target)
                    progress = ((stop_loss - current_spread) / total_range) * 100
                else:
                    progress = 50
            
            # Factor in P&L percentage
            pnl_factor = min(max(pnl_percent / self.config.profit_target_percent * 50, 0), 50)
            
            # Factor in time decay (positions held longer have higher exit probability)
            if summary['entry_time']:
                duration = datetime.now() - summary['entry_time']
                time_factor = min(duration.seconds / 3600 * 5, 20)  # Max 20% for time
            else:
                time_factor = 0
            
            # Total probability
            probability = progress * 0.4 + pnl_factor + time_factor
            
            # Cap at 95% (never 100% certainty)
            return min(probability, 95)
            
        except Exception as e:
            logger.error(f"Error calculating exit probability: {e}")
            return 0
        

    def check_pnl_milestones(self, summary):
        """Check and alert for P&L milestones"""
        try:
            if not summary:
                return
            
            pnl_percent = summary['pnl_percent']
            profit_target_percent = self.config.profit_target_percent
            stop_loss_percent = self.config.stop_loss_percent
            
            # Check if we hit profit targets
            if pnl_percent >= profit_target_percent * 0.5 and pnl_percent < profit_target_percent * 0.75:
                if not hasattr(self, '_half_target_alerted') or not self._half_target_alerted:
                    self.log_message(f"🎯 50% of profit target reached: {pnl_percent:.2f}%")
                    self.show_alert("Half Target", "50% of profit target reached!")
                    self._half_target_alerted = True
            
            if pnl_percent >= profit_target_percent * 0.75 and pnl_percent < profit_target_percent:
                if not hasattr(self, '_three_quarter_target_alerted') or not self._three_quarter_target_alerted:
                    self.log_message(f"🎯 75% of profit target reached: {pnl_percent:.2f}%")
                    self.show_alert("Three Quarter Target", "75% of profit target reached!")
                    self._three_quarter_target_alerted = True
            
            if pnl_percent >= profit_target_percent:
                if not hasattr(self, '_full_target_alerted') or not self._full_target_alerted:
                    self.log_message(f"🎯🎯 Profit target reached: {pnl_percent:.2f}%!")
                    self.show_alert("Target Hit", "Profit target reached! Consider closing position.")
                    self._full_target_alerted = True
            
            # Check if we're near stop loss
            if pnl_percent <= -stop_loss_percent * 0.75:
                if not hasattr(self, '_near_stop_alerted') or not self._near_stop_alerted:
                    self.log_message(f"⚠️ Near stop loss: {pnl_percent:.2f}%")
                    self.show_alert("Near Stop Loss", "Approaching stop loss level!")
                    self._near_stop_alerted = True
            
        except Exception as e:
            logger.error(f"Error checking P&L milestones: {e}")

    def show_alert(self, title, message):
        """Show visual and audio alert"""
        # Visual alert
        QMessageBox.warning(self, title, message)
        
        # Audio alert (Windows only)
        if platform.system() == "Windows" and self.config.enable_audio_alerts:
            try:
                winsound.Beep(1000, 500)  # Frequency 1000Hz, duration 500ms
            except:
                pass
    # def update_position_display(self):
    #     """Update position display"""
    #     summary = self.position_manager.get_position_summary()
        
    #     if not summary:
    #         self.position_type_label.setText("NONE")
    #         self.entry_spread_label.setText("0.00")
    #         self.current_spread_label.setText("0.00")
    #         self.pnl_label.setText("0.00")
    #         self.pnl_percent_label.setText("0.00%")
    #         self.profit_target_label.setText("0.00")
    #         self.stop_loss_label.setText("0.00")
    #         self.trailing_stop_label.setText("0.00")
    #         self.position_duration_label.setText("0s")
    #         self.position_status.setText("Position: None")
            
    #         # Update signal label
    #         self.signal_label.setText("NEUTRAL")
    #         self.signal_label.setStyleSheet("""
    #             font-size: 20px; 
    #             font-weight: bold; 
    #             padding: 10px;
    #             background-color: lightgray;
    #             border: 2px solid gray;
    #             border-radius: 5px;
    #         """)
            
    #         return
        
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
# ENHANCED SETTINGS DIALOG WITH ACCESS TOKEN MANAGEMENT
# ============================================================================

# class AdvancedSettingsDialog(QDialog):
#     """Advanced settings dialog with access token management"""
    
#     def __init__(self):
#         super().__init__()
        
#         # Configuration
#         self.config = TradingConfig()
#         self.config.load()
        
#         # Adjust config based on availability
#         if not WEBENGINE_AVAILABLE and self.config.use_embedded_browser:
#             self.config.use_embedded_browser = False
#             self.config.save()
#             logger.info("Disabled embedded browser setting (PyQt6-WebEngine not available)")
        
#         # Components
#         self.kite = None
#         self.market_data = None
#         self.position_manager = None
#         self.auto_login_manager = None
        
#         # UI State
#         self.is_connected = False
#         self.is_monitoring = False
        
#         # Data storage for signals
#         self.spread_history = []
#         self.max_spread_history = 100
        
#         # Current prices storage - ADD THESE
#         self.near_price = 0.0
#         self.near_bid = 0.0
#         self.near_ask = 0.0
#         self.mid_price = 0.0
#         self.mid_bid = 0.0
#         self.mid_ask = 0.0
        
#         # Initialize
#         self.init_ui()
#         self.init_components()
        
#         # Try auto-connect
#         if self.config.api_key and self.config.access_token:
#             QTimer.singleShot(1000, self.auto_login_with_token)  # Delay to let UI load
    
#     def init_ui(self):
#         self.setWindowTitle("Advanced Trading Settings")
#         self.setGeometry(300, 300, 700, 850)  # Increased height for new section
        
#         layout = QVBoxLayout()
        
#         # Create tab widget
#         tabs = QTabWidget()
        
#         # API Settings (Updated with Token Management)
#         api_tab = QWidget()
#         api_layout = QFormLayout()
        
#         # API Credentials
#         api_layout.addWidget(QLabel("<b>API Credentials</b>"))
        
#         self.api_key_input = QLineEdit(self.config.api_key)
#         self.api_key_input.setPlaceholderText("Enter API Key")
        
#         self.api_secret_input = QLineEdit(self.config.api_secret)
#         self.api_secret_input.setPlaceholderText("Enter API Secret")
#         self.api_secret_input.setEchoMode(QLineEdit.EchoMode.Password)
        
#         api_layout.addRow("API Key:", self.api_key_input)
#         api_layout.addRow("API Secret:", self.api_secret_input)
        
#         # Access Token Management Section
#         api_layout.addWidget(QLabel(""))
#         api_layout.addWidget(QLabel("<b>Access Token Management</b>"))
        
#         # Current Token Status
#         self.token_status_label = QLabel("Checking token status...")
#         self.token_status_label.setStyleSheet("font-weight: bold; padding: 5px;")
#         api_layout.addRow("Current Status:", self.token_status_label)
        
#         # Token Expiry Info
#         self.token_expiry_label = QLabel("Expiry: Unknown")
#         api_layout.addRow("Token Expiry:", self.token_expiry_label)
        
#         # Token Actions Group
#         token_actions_group = QGroupBox("Token Actions")
#         token_actions_layout = QHBoxLayout()
        
#         # View Token Button (with show/hide toggle)
#         self.view_token_btn = QPushButton("👁 View Token")
#         self.view_token_btn.clicked.connect(self.toggle_token_visibility)
#         self.view_token_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
#         token_actions_layout.addWidget(self.view_token_btn)
        
#         # Copy Token Button
#         self.copy_token_btn = QPushButton("📋 Copy Token")
#         self.copy_token_btn.clicked.connect(self.copy_token_to_clipboard)
#         self.copy_token_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
#         token_actions_layout.addWidget(self.copy_token_btn)
        
#         token_actions_group.setLayout(token_actions_layout)
#         api_layout.addRow("", token_actions_group)
        
#         # Manual Token Input
#         api_layout.addWidget(QLabel("<b>Update Access Token</b>"))
        
#         token_input_group = QGroupBox("Manual Token Update")
#         token_input_layout = QVBoxLayout()
        
#         # Token Input with visibility toggle
#         token_input_row = QHBoxLayout()
#         self.token_input = QLineEdit()
#         self.token_input.setPlaceholderText("Paste or enter new access token here")
#         self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        
#         self.show_token_check = QCheckBox("Show")
#         self.show_token_check.stateChanged.connect(self.toggle_input_token_visibility)
        
#         token_input_row.addWidget(self.token_input)
#         token_input_row.addWidget(self.show_token_check)
#         token_input_layout.addLayout(token_input_row)
        
#         # Generate from Request Token
#         request_token_layout = QHBoxLayout()
#         self.request_token_input = QLineEdit()
#         self.request_token_input.setPlaceholderText("Enter request_token from Zerodha")
        
#         self.generate_token_btn = QPushButton("Generate Access Token")
#         self.generate_token_btn.clicked.connect(self.generate_access_token)
#         self.generate_token_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 5px;")
        
#         request_token_layout.addWidget(QLabel("Request Token:"))
#         request_token_layout.addWidget(self.request_token_input)
#         request_token_layout.addWidget(self.generate_token_btn)
#         token_input_layout.addLayout(request_token_layout)
        
#         # Save Token Button
#         self.save_token_btn = QPushButton("💾 Save & Test Token")
#         self.save_token_btn.clicked.connect(self.save_and_test_token)
#         self.save_token_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 8px; font-weight: bold;")
#         token_input_layout.addWidget(self.save_token_btn)
        
#         token_input_group.setLayout(token_input_layout)
#         api_layout.addWidget(token_input_group)
        
#         # Test buttons
#         test_layout = QHBoxLayout()
        
#         self.test_api_btn = QPushButton("Test API Credentials")
#         self.test_api_btn.clicked.connect(self.test_api_credentials)
#         self.test_api_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        
#         self.test_token_btn = QPushButton("Test Current Token")
#         self.test_token_btn.clicked.connect(self.test_current_token)
#         self.test_token_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        
#         test_layout.addWidget(self.test_api_btn)
#         test_layout.addWidget(self.test_token_btn)
        
#         api_layout.addRow("", test_layout)
        
#         # Info text
#         info_text = QLabel(
#             "<b>Important Notes:</b><br>"
#             "1. Access tokens expire daily (~6 AM IST)<br>"
#             "2. Use 'Generate Access Token' with request_token for new tokens<br>"
#             "3. Or paste an existing valid token manually<br>"
#             "4. Token changes require reconnection"
#         )
#         info_text.setStyleSheet("color: #666; font-size: 9pt; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
#         api_layout.addRow("", info_text)
        
#         api_tab.setLayout(api_layout)
#         tabs.addTab(api_tab, "API & Token Settings")
        
#         # Auto Login Settings (updated)
#         auto_login_tab = QWidget()
#         auto_login_layout = QFormLayout()
        
#         self.auto_login_check = QCheckBox("Enable Auto Login on Startup")
#         self.auto_login_check.setChecked(self.config.auto_login_enabled)
        
#         # Token refresh reminder
#         token_refresh_group = QGroupBox("Token Refresh Reminder")
#         token_refresh_layout = QVBoxLayout()
        
#         self.token_reminder_check = QCheckBox("Show reminder 30 minutes before token expiry")
#         self.token_reminder_check.setChecked(True)
        
#         self.auto_refresh_check = QCheckBox("Attempt auto-refresh if token expires during session")
#         self.auto_refresh_check.setChecked(True)
        
#         token_refresh_layout.addWidget(self.token_reminder_check)
#         token_refresh_layout.addWidget(self.auto_refresh_check)
#         token_refresh_group.setLayout(token_refresh_layout)
        
#         # Embedded browser option
#         if WEBENGINE_AVAILABLE:
#             self.embedded_browser_check = QCheckBox("Use Embedded Browser for Login")
#             self.embedded_browser_check.setChecked(self.config.use_embedded_browser)
#             auto_login_layout.addRow("", self.embedded_browser_check)
#         else:
#             embedded_browser_info = QLabel(
#                 "⚠ Embedded browser not available.\n"
#                 "Install PyQt6-WebEngine for embedded browser login."
#             )
#             embedded_browser_info.setStyleSheet("color: #ff9800; font-size: 9pt; padding: 10px; background-color: #fff3e0; border-radius: 5px;")
#             auto_login_layout.addRow("", embedded_browser_info)
        
#         self.redirect_port_spin = QSpinBox()
#         self.redirect_port_spin.setRange(1024, 65535)
#         self.redirect_port_spin.setValue(self.config.redirect_port)
        
#         auto_login_layout.addRow("", self.auto_login_check)
#         auto_login_layout.addRow("", token_refresh_group)
#         auto_login_layout.addRow("Redirect Port:", self.redirect_port_spin)
        
#         auto_login_tab.setLayout(auto_login_layout)
#         tabs.addTab(auto_login_tab, "Auto Login")
        
#         # Trading Settings (existing)
#         trading_tab = QWidget()
#         trading_layout = QFormLayout()
        
#         self.quantity_spin = QSpinBox()
#         self.quantity_spin.setRange(1, 1000)
#         self.quantity_spin.setValue(self.config.quantity)
        
#         self.entry_zscore_spin = QDoubleSpinBox()
#         self.entry_zscore_spin.setRange(0.1, 5.0)
#         self.entry_zscore_spin.setSingleStep(0.1)
#         self.entry_zscore_spin.setValue(self.config.entry_zscore_threshold)
        
#         self.profit_target_spin = QDoubleSpinBox()
#         self.profit_target_spin.setRange(0.1, 10.0)
#         self.profit_target_spin.setSingleStep(0.1)
#         self.profit_target_spin.setValue(self.config.profit_target_percent)
        
#         self.stop_loss_spin = QDoubleSpinBox()
#         self.stop_loss_spin.setRange(0.1, 10.0)
#         self.stop_loss_spin.setSingleStep(0.1)
#         self.stop_loss_spin.setValue(self.config.stop_loss_percent)
        
#         trading_layout.addRow("Quantity:", self.quantity_spin)
#         trading_layout.addRow("Entry Z-Score Threshold:", self.entry_zscore_spin)
#         trading_layout.addRow("Profit Target (%):", self.profit_target_spin)
#         trading_layout.addRow("Stop Loss (%):", self.stop_loss_spin)
        
#         # Auto entry/exit checkboxes
#         self.auto_entry_check = QCheckBox("Enable Auto Entry")
#         self.auto_entry_check.setChecked(self.config.auto_entry_enabled)
        
#         self.auto_exit_check = QCheckBox("Enable Auto Exit")
#         self.auto_exit_check.setChecked(self.config.auto_exit_enabled)
        
#         trading_layout.addRow("", self.auto_entry_check)
#         trading_layout.addRow("", self.auto_exit_check)
        
#         trading_tab.setLayout(trading_layout)
#         tabs.addTab(trading_tab, "Trading Settings")
        
#         layout.addWidget(tabs)
        
#         # Buttons
#         button_box = QDialogButtonBox(
#             QDialogButtonBox.StandardButton.Ok | 
#             QDialogButtonBox.StandardButton.Cancel |
#             QDialogButtonBox.StandardButton.Apply
#         )
#         button_box.accepted.connect(self.save_settings)
#         button_box.rejected.connect(self.reject)
#         button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.save_settings)
        
#         layout.addWidget(button_box)
#         self.setLayout(layout)
        
#         # Load token status on startup
#         QTimer.singleShot(100, self.load_current_token_status)
    
#     def load_current_token_status(self):
#         """Load and display current token status"""
#         try:
#             if not self.config.api_key:
#                 self.token_status_label.setText("⚠ API Key Not Set")
#                 self.token_status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
#                 self.token_expiry_label.setText("Expiry: N/A")
#                 return
            
#             if not self.config.access_token:
#                 self.token_status_label.setText("❌ No Access Token")
#                 self.token_status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
#                 self.token_expiry_label.setText("Expiry: No token set")
#                 return
            
#             # Try to test the token
#             self.kite = KiteConnect(api_key=self.config.api_key)
#             self.kite.set_access_token(self.config.access_token)
            
#             # Get user profile to test token
#             profile = self.kite.profile()
            
#             # Token is valid
#             self.token_status_label.setText(f"✅ Valid (User: {profile['user_name']})")
#             self.token_status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
#             # Estimate expiry (tokens expire daily at ~6 AM IST)
#             now = datetime.now()
#             tomorrow_6am = now.replace(hour=6, minute=0, second=0, microsecond=0)
#             if now.hour >= 6:
#                 tomorrow_6am += timedelta(days=1)
            
#             time_remaining = tomorrow_6am - now
#             hours_remaining = time_remaining.total_seconds() / 3600
            
#             if hours_remaining > 0:
#                 self.token_expiry_label.setText(f"Expires in: ~{hours_remaining:.1f} hours (~6 AM IST)")
#             else:
#                 self.token_expiry_label.setText("⚠ May be expired")
                
#         except KiteException as e:
#             if "Invalid access token" in str(e) or "expired" in str(e).lower():
#                 self.token_status_label.setText("❌ Token Expired/Invalid")
#                 self.token_status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
#                 self.token_expiry_label.setText("Expiry: Token needs renewal")
#             else:
#                 self.token_status_label.setText("⚠ Connection Error")
#                 self.token_status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
#                 self.token_expiry_label.setText(f"Error: {str(e)[:50]}...")
#         except Exception as e:
#             self.token_status_label.setText("⚠ Error checking token")
#             self.token_status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
#             self.token_expiry_label.setText(f"Error: {str(e)[:50]}...")
    
#     def toggle_token_visibility(self):
#         """Toggle between showing and hiding the current token"""
#         if not self.config.access_token:
#             QMessageBox.warning(self, "No Token", "No access token is currently set.")
#             return
        
#         # Create a dialog to view token
#         dialog = QDialog(self)
#         dialog.setWindowTitle("Current Access Token")
#         dialog.setGeometry(400, 300, 500, 200)
        
#         layout = QVBoxLayout()
        
#         layout.addWidget(QLabel("<b>Current Access Token</b>"))
#         layout.addWidget(QLabel("Copy this token for backup or manual renewal:"))
        
#         token_text = QTextEdit()
#         token_text.setText(self.config.access_token)
#         token_text.setReadOnly(True)
#         token_text.setMaximumHeight(80)
#         token_text.setStyleSheet("font-family: monospace; background-color: #f9f9f9;")
        
#         layout.addWidget(token_text)
        
#         # Copy button
#         copy_btn = QPushButton("📋 Copy to Clipboard")
#         copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.config.access_token))
#         layout.addWidget(copy_btn)
        
#         # Close button
#         close_btn = QPushButton("Close")
#         close_btn.clicked.connect(dialog.accept)
#         layout.addWidget(close_btn)
        
#         dialog.setLayout(layout)
#         dialog.exec()
    
#     def copy_token_to_clipboard(self):
#         """Copy current token to clipboard"""
#         if not self.config.access_token:
#             QMessageBox.warning(self, "No Token", "No access token to copy.")
#             return
        
#         clipboard = QApplication.clipboard()
#         clipboard.setText(self.config.access_token)
        
#         QMessageBox.information(self, "Copied", "Access token copied to clipboard!")
    
#     def copy_to_clipboard(self, text):
#         """Generic copy to clipboard"""
#         clipboard = QApplication.clipboard()
#         clipboard.setText(text)
    
#     def toggle_input_token_visibility(self):
#         """Toggle visibility of token input field"""
#         if self.show_token_check.isChecked():
#             self.token_input.setEchoMode(QLineEdit.EchoMode.Normal)
#         else:
#             self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
    
#     def generate_access_token(self):
#         """Generate access token from request token"""
#         api_key = self.api_key_input.text()
#         api_secret = self.api_secret_input.text()
#         request_token = self.request_token_input.text().strip()
        
#         if not api_key or not api_secret:
#             QMessageBox.warning(self, "Missing Info", "Please enter API Key and Secret first.")
#             return
        
#         if not request_token:
#             QMessageBox.warning(self, "Missing Request Token", 
#                               "Please enter the request_token obtained from Zerodha login.")
#             return
        
#         try:
#             # Create KiteConnect instance
#             kite = KiteConnect(api_key=api_key)
            
#             # Generate session
#             data = kite.generate_session(
#                 request_token=request_token,
#                 api_secret=api_secret
#             )
            
#             access_token = data['access_token']
            
#             # Update the input field
#             self.token_input.setText(access_token)
            
#             # Also update config immediately
#             self.config.access_token = access_token
            
#             # Test the new token
#             kite.set_access_token(access_token)
#             profile = kite.profile()
            
#             QMessageBox.information(
#                 self, 
#                 "Token Generated Successfully",
#                 f"Access token generated!\n\n"
#                 f"User: {profile['user_name']}\n"
#                 f"Token: {access_token[:20]}...\n\n"
#                 f"Click 'Save & Test Token' to use this token."
#             )
            
#             # Update status
#             self.load_current_token_status()
            
#         except Exception as e:
#             QMessageBox.critical(
#                 self, 
#                 "Token Generation Failed", 
#                 f"Error generating access token:\n\n{str(e)}\n\n"
#                 f"Make sure:\n"
#                 f"1. API Key and Secret are correct\n"
#                 f"2. Request token is valid and not expired\n"
#                 f"3. Redirect URL is correctly set in Zerodha console"
#             )
    
#     def save_and_test_token(self):
#         """Save and test the manually entered token"""
#         new_token = self.token_input.text().strip()
        
#         if not new_token:
#             QMessageBox.warning(self, "Empty Token", "Please enter an access token first.")
#             return
        
#         # Validate token format (basic check)
#         if len(new_token) < 20:
#             reply = QMessageBox.question(
#                 self, 
#                 "Short Token", 
#                 "The token seems very short. Are you sure this is a valid Zerodha access token?",
#                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
#             )
#             if reply == QMessageBox.StandardButton.No:
#                 return
        
#         # Test the token
#         api_key = self.api_key_input.text()
#         if not api_key:
#             QMessageBox.warning(self, "Missing API Key", "Please enter API Key to test the token.")
#             return
        
#         try:
#             # Test the token
#             kite = KiteConnect(api_key=api_key)
#             kite.set_access_token(new_token)
#             profile = kite.profile()
            
#             # Token is valid - update config
#             self.config.access_token = new_token
            
#             # Save to config file
#             self.config.save()
            
#             QMessageBox.information(
#                 self,
#                 "Token Saved Successfully",
#                 f"Access token saved and validated!\n\n"
#                 f"User: {profile['user_name']}\n"
#                 f"Login Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
#                 f"Note: You may need to reconnect in the main application."
#             )
            
#             # Update status display
#             self.load_current_token_status()
            
#             # Clear input field
#             self.token_input.clear()
            
#         except Exception as e:
#             QMessageBox.critical(
#                 self,
#                 "Invalid Token",
#                 f"The access token is invalid:\n\n{str(e)}\n\n"
#                 f"Please check:\n"
#                 f"1. Token is correct and not expired\n"
#                 f"2. API Key matches the token\n"
#                 f"3. You have necessary permissions"
#             )
    
#     def test_api_credentials(self):
#         """Test API credentials only"""
#         api_key = self.api_key_input.text()
#         api_secret = self.api_secret_input.text()
        
#         if not api_key or not api_secret:
#             QMessageBox.warning(self, "Missing Information", "Please enter both API Key and API Secret")
#             return
        
#         try:
#             kite = KiteConnect(api_key=api_key)
            
#             # Generate login URL to test
#             redirect_url = f"http://localhost:{self.config.redirect_port}/"
#             login_url = kite.login_url() + f"&redirect_uri={redirect_url}"
            
#             # Save credentials temporarily for token generation
#             self.config.api_key = api_key
#             self.config.api_secret = api_secret
            
#             # Show success with options
#             msg_box = QMessageBox(self)
#             msg_box.setWindowTitle("API Credentials Valid")
#             msg_box.setText(
#                 f"API credentials are valid!\n\n"
#                 f"Login URL generated successfully.\n"
#                 f"Redirect URL: {redirect_url}\n\n"
#                 f"What would you like to do?"
#             )
            
#             open_browser_btn = msg_box.addButton("Open Login Page", QMessageBox.ButtonRole.ActionRole)
#             copy_url_btn = msg_box.addButton("Copy Login URL", QMessageBox.ButtonRole.ActionRole)
#             msg_box.addButton(QMessageBox.StandardButton.Ok)
            
#             msg_box.exec()
            
#             if msg_box.clickedButton() == open_browser_btn:
#                 webbrowser.open(login_url)
#             elif msg_box.clickedButton() == copy_url_btn:
#                 clipboard = QApplication.clipboard()
#                 clipboard.setText(login_url)
#                 QMessageBox.information(self, "Copied", "Login URL copied to clipboard!")
            
#         except Exception as e:
#             QMessageBox.critical(self, "API Test Failed", f"Error: {str(e)}")
    
#     def test_current_token(self):
#         """Test the currently saved token"""
#         if not self.config.api_key:
#             QMessageBox.warning(self, "Missing API Key", "Please set API Key first.")
#             return
        
#         if not self.config.access_token:
#             QMessageBox.warning(self, "No Token", "No access token is currently saved.")
#             return
        
#         try:
#             kite = KiteConnect(api_key=self.config.api_key)
#             kite.set_access_token(self.config.access_token)
#             profile = kite.profile()
            
#             # Get additional info if available
#             try:
#                 margins = kite.margins()
#                 equity = margins.get('equity', {}).get('available', {}).get('cash', 0)
#             except:
#                 equity = "Unknown"
            
#             QMessageBox.information(
#                 self,
#                 "Token Test Successful",
#                 f"Access token is valid!\n\n"
#                 f"User: {profile['user_name']}\n"
#                 f"User ID: {profile['user_id']}\n"
#                 f"Email: {profile['email']}\n"
#                 f"Equity Available: {equity}\n\n"
#                 f"Token last character: ...{self.config.access_token[-5:]}"
#             )
            
#             # Update status display
#             self.load_current_token_status()
            
#         except KiteException as e:
#             if "Invalid access token" in str(e) or "expired" in str(e).lower():
#                 QMessageBox.critical(
#                     self,
#                     "Token Expired",
#                     "The access token has expired or is invalid.\n\n"
#                     "Please generate a new token using the 'Generate Access Token' button."
#                 )
#                 self.token_status_label.setText("❌ Token Expired")
#                 self.token_status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
#             else:
#                 QMessageBox.critical(self, "Token Test Failed", f"Error: {str(e)}")
#         except Exception as e:
#             QMessageBox.critical(self, "Token Test Failed", f"Error: {str(e)}")
    
#     def save_settings(self):
#         """Save all settings"""
#         try:
#             # Save API credentials
#             old_api_key = self.config.api_key
#             old_api_secret = self.config.api_secret
            
#             self.config.api_key = self.api_key_input.text()
#             self.config.api_secret = self.api_secret_input.text()
            
#             # Check if API credentials changed
#             api_creds_changed = (
#                 old_api_key != self.config.api_key or 
#                 old_api_secret != self.config.api_secret
#             )
            
#             # Auto Login Settings
#             self.config.auto_login_enabled = self.auto_login_check.isChecked()
            
#             # Only save embedded browser setting if available
#             if WEBENGINE_AVAILABLE:
#                 self.config.use_embedded_browser = self.embedded_browser_check.isChecked()
#             else:
#                 self.config.use_embedded_browser = False
            
#             self.config.redirect_port = self.redirect_port_spin.value()
            
#             # Trading Settings
#             self.config.quantity = self.quantity_spin.value()
#             self.config.entry_zscore_threshold = self.entry_zscore_spin.value()
#             self.config.profit_target_percent = self.profit_target_spin.value()
#             self.config.stop_loss_percent = self.stop_loss_spin.value()
#             self.config.auto_entry_enabled = self.auto_entry_check.isChecked()
#             self.config.auto_exit_enabled = self.auto_exit_check.isChecked()
            
#             # Save to file
#             self.config.save()
            
#             # Show appropriate message
#             if api_creds_changed:
#                 QMessageBox.information(
#                     self, 
#                     "Settings Saved", 
#                     "Settings have been saved successfully!\n\n"
#                     "Note: API credentials have changed. "
#                     "You may need to reconnect or generate a new access token."
#                 )
#             else:
#                 QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully!")
            
#             # Don't close the dialog on Apply
#             if self.sender().text() == "Apply":
#                 return
#             else:
#                 self.accept()
                
#         except Exception as e:
#             QMessageBox.critical(self, "Save Failed", f"Error saving settings: {str(e)}")

# ============================================================================
# ENHANCED SETTINGS DIALOG WITH ACCESS TOKEN MANAGEMENT
# ============================================================================

class AdvancedSettingsDialog(QDialog):
    """Advanced settings dialog with access token management"""
    
    def __init__(self, config: TradingConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.kite = None
        self.init_ui()
        self.load_current_token_status()
    
    def init_ui(self):
        self.setWindowTitle("Advanced Trading Settings")
        self.setGeometry(300, 300, 700, 850)
        
        layout = QVBoxLayout()
        
        # Create tab widget
        tabs = QTabWidget()
        
        # API Settings
        api_tab = QWidget()
        api_layout = QFormLayout()
        
        # API Credentials
        api_layout.addWidget(QLabel("<b>API Credentials</b>"))
        
        self.api_key_input = QLineEdit(self.config.api_key)
        self.api_key_input.setPlaceholderText("Enter API Key")
        
        self.api_secret_input = QLineEdit(self.config.api_secret)
        self.api_secret_input.setPlaceholderText("Enter API Secret")
        self.api_secret_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        api_layout.addRow("API Key:", self.api_key_input)
        api_layout.addRow("API Secret:", self.api_secret_input)
        
        # Access Token Management Section
        api_layout.addWidget(QLabel(""))
        api_layout.addWidget(QLabel("<b>Access Token Management</b>"))
        
        # Current Token Status
        self.token_status_label = QLabel("Checking token status...")
        self.token_status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        api_layout.addRow("Current Status:", self.token_status_label)
        
        # Token Expiry Info
        self.token_expiry_label = QLabel("Expiry: Unknown")
        api_layout.addRow("Token Expiry:", self.token_expiry_label)
        
        # Token Actions Group
        token_actions_group = QGroupBox("Token Actions")
        token_actions_layout = QHBoxLayout()
        
        # View Token Button
        self.view_token_btn = QPushButton("👁 View Token")
        self.view_token_btn.clicked.connect(self.toggle_token_visibility)
        self.view_token_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        token_actions_layout.addWidget(self.view_token_btn)
        
        # Copy Token Button
        self.copy_token_btn = QPushButton("📋 Copy Token")
        self.copy_token_btn.clicked.connect(self.copy_token_to_clipboard)
        self.copy_token_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        token_actions_layout.addWidget(self.copy_token_btn)
        
        token_actions_group.setLayout(token_actions_layout)
        api_layout.addRow("", token_actions_group)
        
        # Manual Token Input
        api_layout.addWidget(QLabel("<b>Update Access Token</b>"))
        
        token_input_group = QGroupBox("Manual Token Update")
        token_input_layout = QVBoxLayout()
        
        # Token Input with visibility toggle
        token_input_row = QHBoxLayout()
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Paste or enter new access token here")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.show_token_check = QCheckBox("Show")
        self.show_token_check.stateChanged.connect(self.toggle_input_token_visibility)
        
        token_input_row.addWidget(self.token_input)
        token_input_row.addWidget(self.show_token_check)
        token_input_layout.addLayout(token_input_row)
        
        # Generate from Request Token
        request_token_layout = QHBoxLayout()
        self.request_token_input = QLineEdit()
        self.request_token_input.setPlaceholderText("Enter request_token from Zerodha")
        
        self.generate_token_btn = QPushButton("Generate Access Token")
        self.generate_token_btn.clicked.connect(self.generate_access_token)
        self.generate_token_btn.setStyleSheet("background-color: #FF9800; color: black; padding: 5px;")
        
        request_token_layout.addWidget(QLabel("Request Token:"))
        request_token_layout.addWidget(self.request_token_input)
        request_token_layout.addWidget(self.generate_token_btn)
        token_input_layout.addLayout(request_token_layout)
        
        # Save Token Button
        self.save_token_btn = QPushButton("💾 Save & Test Token")
        self.save_token_btn.clicked.connect(self.save_and_test_token)
        self.save_token_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 8px; font-weight: bold;")
        token_input_layout.addWidget(self.save_token_btn)
        
        token_input_group.setLayout(token_input_layout)
        api_layout.addWidget(token_input_group)
        
        # Test buttons
        test_layout = QHBoxLayout()
        
        self.test_api_btn = QPushButton("Test API Credentials")
        self.test_api_btn.clicked.connect(self.test_api_credentials)
        self.test_api_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        
        self.test_token_btn = QPushButton("Test Current Token")
        self.test_token_btn.clicked.connect(self.test_current_token)
        self.test_token_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        
        test_layout.addWidget(self.test_api_btn)
        test_layout.addWidget(self.test_token_btn)
        
        api_layout.addRow("", test_layout)
        
        # Info text
        info_text = QLabel(
            "<b>Important Notes:</b><br>"
            "1. Access tokens expire daily (~6 AM IST)<br>"
            "2. Use 'Generate Access Token' with request_token for new tokens<br>"
            "3. Or paste an existing valid token manually<br>"
            "4. Token changes require reconnection"
        )
        info_text.setStyleSheet("color: #666; font-size: 9pt; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        api_layout.addRow("", info_text)
        
        api_tab.setLayout(api_layout)
        tabs.addTab(api_tab, "API & Token Settings")
        
        # Auto Login Settings
        auto_login_tab = QWidget()
        auto_login_layout = QFormLayout()
        
        self.auto_login_check = QCheckBox("Enable Auto Login on Startup")
        self.auto_login_check.setChecked(self.config.auto_login_enabled)
        
        # Token refresh reminder
        token_refresh_group = QGroupBox("Token Refresh Reminder")
        token_refresh_layout = QVBoxLayout()
        
        self.token_reminder_check = QCheckBox("Show reminder 30 minutes before token expiry")
        self.token_reminder_check.setChecked(True)
        
        self.auto_refresh_check = QCheckBox("Attempt auto-refresh if token expires during session")
        self.auto_refresh_check.setChecked(True)
        
        token_refresh_layout.addWidget(self.token_reminder_check)
        token_refresh_layout.addWidget(self.auto_refresh_check)
        token_refresh_group.setLayout(token_refresh_layout)
        
        # Embedded browser option
        if WEBENGINE_AVAILABLE:
            self.embedded_browser_check = QCheckBox("Use Embedded Browser for Login")
            self.embedded_browser_check.setChecked(self.config.use_embedded_browser)
            auto_login_layout.addRow("", self.embedded_browser_check)
        else:
            embedded_browser_info = QLabel(
                "⚠ Embedded browser not available.\n"
                "Install PyQt6-WebEngine for embedded browser login."
            )
            embedded_browser_info.setStyleSheet("color: #ff9800; font-size: 9pt; padding: 10px; background-color: #fff3e0; border-radius: 5px;")
            auto_login_layout.addRow("", embedded_browser_info)
        
        self.redirect_port_spin = QSpinBox()
        self.redirect_port_spin.setRange(1024, 65535)
        self.redirect_port_spin.setValue(self.config.redirect_port)
        
        auto_login_layout.addRow("", self.auto_login_check)
        auto_login_layout.addRow("", token_refresh_group)
        auto_login_layout.addRow("Redirect Port:", self.redirect_port_spin)
        
        auto_login_tab.setLayout(auto_login_layout)
        tabs.addTab(auto_login_tab, "Auto Login")
        
        # Trading Settings (existing)
        trading_tab = QWidget()
        trading_layout = QFormLayout()
        
        self.quantity_spin = QSpinBox()
        self.quantity_spin.setRange(1, 1000)
        self.quantity_spin.setValue(self.config.quantity)
        
        self.entry_zscore_spin = QDoubleSpinBox()
        self.entry_zscore_spin.setRange(0.1, 5.0)
        self.entry_zscore_spin.setSingleStep(0.1)
        self.entry_zscore_spin.setValue(self.config.entry_zscore_threshold)
        
        self.profit_target_spin = QDoubleSpinBox()
        self.profit_target_spin.setRange(0.1, 10.0)
        self.profit_target_spin.setSingleStep(0.1)
        self.profit_target_spin.setValue(self.config.profit_target_percent)
        
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.1, 10.0)
        self.stop_loss_spin.setSingleStep(0.1)
        self.stop_loss_spin.setValue(self.config.stop_loss_percent)
        
        trading_layout.addRow("Quantity:", self.quantity_spin)
        trading_layout.addRow("Entry Z-Score Threshold:", self.entry_zscore_spin)
        trading_layout.addRow("Profit Target (%):", self.profit_target_spin)
        trading_layout.addRow("Stop Loss (%):", self.stop_loss_spin)
        
        # Auto entry/exit checkboxes
        self.auto_entry_check = QCheckBox("Enable Auto Entry")
        self.auto_entry_check.setChecked(self.config.auto_entry_enabled)
        
        self.auto_exit_check = QCheckBox("Enable Auto Exit")
        self.auto_exit_check.setChecked(self.config.auto_exit_enabled)
        
        trading_layout.addRow("", self.auto_entry_check)
        trading_layout.addRow("", self.auto_exit_check)
        
        trading_tab.setLayout(trading_layout)
        tabs.addTab(trading_tab, "Trading Settings")
        
        layout.addWidget(tabs)
        
        # Buttons - FIXED: Handle Apply button separately
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        
        # Get the Apply button and connect it separately
        apply_button = button_box.button(QDialogButtonBox.StandardButton.Apply)
        apply_button.clicked.connect(self.apply_settings)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
        
        # Load token status on startup
        QTimer.singleShot(100, self.load_current_token_status)
    
    def load_current_token_status(self):
        """Load and display current token status"""
        try:
            if not self.config.api_key:
                self.token_status_label.setText("⚠ API Key Not Set")
                self.token_status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
                self.token_expiry_label.setText("Expiry: N/A")
                return
            
            if not self.config.access_token:
                self.token_status_label.setText("❌ No Access Token")
                self.token_status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
                self.token_expiry_label.setText("Expiry: No token set")
                return
            
            # Try to test the token
            self.kite = KiteConnect(api_key=self.config.api_key)
            self.kite.set_access_token(self.config.access_token)
            
            # Get user profile to test token
            profile = self.kite.profile()
            
            # Token is valid
            self.token_status_label.setText(f"✅ Valid (User: {profile['user_name']})")
            self.token_status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
            # Estimate expiry (tokens expire daily at ~6 AM IST)
            now = datetime.now()
            tomorrow_6am = now.replace(hour=6, minute=0, second=0, microsecond=0)
            if now.hour >= 6:
                tomorrow_6am += timedelta(days=1)
            
            time_remaining = tomorrow_6am - now
            hours_remaining = time_remaining.total_seconds() / 3600
            
            if hours_remaining > 0:
                self.token_expiry_label.setText(f"Expires in: ~{hours_remaining:.1f} hours (~6 AM IST)")
            else:
                self.token_expiry_label.setText("⚠ May be expired")
                
        except KiteException as e:
            if "Invalid access token" in str(e) or "expired" in str(e).lower():
                self.token_status_label.setText("❌ Token Expired/Invalid")
                self.token_status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
                self.token_expiry_label.setText("Expiry: Token needs renewal")
            else:
                self.token_status_label.setText("⚠ Connection Error")
                self.token_status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
                self.token_expiry_label.setText(f"Error: {str(e)[:50]}...")
        except Exception as e:
            self.token_status_label.setText("⚠ Error checking token")
            self.token_status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
            self.token_expiry_label.setText(f"Error: {str(e)[:50]}...")
    
    def toggle_token_visibility(self):
        """Toggle between showing and hiding the current token"""
        if not self.config.access_token:
            QMessageBox.warning(self, "No Token", "No access token is currently set.")
            return
        
        # Create a dialog to view token
        dialog = QDialog(self)
        dialog.setWindowTitle("Current Access Token")
        dialog.setGeometry(400, 300, 500, 200)
        
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Current Access Token</b>"))
        layout.addWidget(QLabel("Copy this token for backup or manual renewal:"))
        
        token_text = QTextEdit()
        token_text.setText(self.config.access_token)
        token_text.setReadOnly(True)
        token_text.setMaximumHeight(80)
        token_text.setStyleSheet("font-family: monospace; background-color: #f9f9f9;")
        
        layout.addWidget(token_text)
        
        # Copy button
        copy_btn = QPushButton("📋 Copy to Clipboard")
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.config.access_token))
        layout.addWidget(copy_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def copy_token_to_clipboard(self):
        """Copy current token to clipboard"""
        if not self.config.access_token:
            QMessageBox.warning(self, "No Token", "No access token to copy.")
            return
        
        clipboard = QApplication.clipboard()
        clipboard.setText(self.config.access_token)
        
        QMessageBox.information(self, "Copied", "Access token copied to clipboard!")
    
    def copy_to_clipboard(self, text):
        """Generic copy to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
    
    def toggle_input_token_visibility(self):
        """Toggle visibility of token input field"""
        if self.show_token_check.isChecked():
            self.token_input.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
    
    def generate_access_token(self):
        """Generate access token from request token"""
        api_key = self.api_key_input.text()
        api_secret = self.api_secret_input.text()
        request_token = self.request_token_input.text().strip()
        
        if not api_key or not api_secret:
            QMessageBox.warning(self, "Missing Info", "Please enter API Key and Secret first.")
            return
        
        if not request_token:
            QMessageBox.warning(self, "Missing Request Token", 
                              "Please enter the request_token obtained from Zerodha login.")
            return
        
        try:
            # Create KiteConnect instance
            kite = KiteConnect(api_key=api_key)
            
            # Generate session
            data = kite.generate_session(
                request_token=request_token,
                api_secret=api_secret
            )
            
            access_token = data['access_token']
            
            # Update the input field
            self.token_input.setText(access_token)
            
            # Also update config immediately
            self.config.access_token = access_token
            
            # Test the new token
            kite.set_access_token(access_token)
            profile = kite.profile()
            
            QMessageBox.information(
                self, 
                "Token Generated Successfully",
                f"Access token generated!\n\n"
                f"User: {profile['user_name']}\n"
                f"Token: {access_token[:20]}...\n\n"
                f"Click 'Save & Test Token' to use this token."
            )
            
            # Update status
            self.load_current_token_status()
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Token Generation Failed", 
                f"Error generating access token:\n\n{str(e)}\n\n"
                f"Make sure:\n"
                f"1. API Key and Secret are correct\n"
                f"2. Request token is valid and not expired\n"
                f"3. Redirect URL is correctly set in Zerodha console"
            )
    
    def save_and_test_token(self):
        """Save and test the manually entered token"""
        new_token = self.token_input.text().strip()
        
        if not new_token:
            QMessageBox.warning(self, "Empty Token", "Please enter an access token first.")
            return
        
        # Validate token format (basic check)
        if len(new_token) < 20:
            reply = QMessageBox.question(
                self, 
                "Short Token", 
                "The token seems very short. Are you sure this is a valid Zerodha access token?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Test the token
        api_key = self.api_key_input.text()
        if not api_key:
            QMessageBox.warning(self, "Missing API Key", "Please enter API Key to test the token.")
            return
        
        try:
            # Test the token
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(new_token)
            profile = kite.profile()
            
            # Token is valid - update config
            self.config.access_token = new_token
            
            # Save to config file
            self.config.save()
            
            QMessageBox.information(
                self,
                "Token Saved Successfully",
                f"Access token saved and validated!\n\n"
                f"User: {profile['user_name']}\n"
                f"Login Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Note: You may need to reconnect in the main application."
            )
            
            # Update status display
            self.load_current_token_status()
            
            # Clear input field
            self.token_input.clear()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Invalid Token",
                f"The access token is invalid:\n\n{str(e)}\n\n"
                f"Please check:\n"
                f"1. Token is correct and not expired\n"
                f"2. API Key matches the token\n"
                f"3. You have necessary permissions"
            )
    
    def test_api_credentials(self):
        """Test API credentials only"""
        api_key = self.api_key_input.text()
        api_secret = self.api_secret_input.text()
        
        if not api_key or not api_secret:
            QMessageBox.warning(self, "Missing Information", "Please enter both API Key and API Secret")
            return
        
        try:
            kite = KiteConnect(api_key=api_key)
            
            # Generate login URL to test
            redirect_url = f"http://localhost:{self.config.redirect_port}/"
            login_url = kite.login_url() + f"&redirect_uri={redirect_url}"
            
            # Save credentials temporarily for token generation
            self.config.api_key = api_key
            self.config.api_secret = api_secret
            
            # Show success with options
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("API Credentials Valid")
            msg_box.setText(
                f"API credentials are valid!\n\n"
                f"Login URL generated successfully.\n"
                f"Redirect URL: {redirect_url}\n\n"
                f"What would you like to do?"
            )
            
            open_browser_btn = msg_box.addButton("Open Login Page", QMessageBox.ButtonRole.ActionRole)
            copy_url_btn = msg_box.addButton("Copy Login URL", QMessageBox.ButtonRole.ActionRole)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            
            msg_box.exec()
            
            if msg_box.clickedButton() == open_browser_btn:
                webbrowser.open(login_url)
            elif msg_box.clickedButton() == copy_url_btn:
                clipboard = QApplication.clipboard()
                clipboard.setText(login_url)
                QMessageBox.information(self, "Copied", "Login URL copied to clipboard!")
            
        except Exception as e:
            QMessageBox.critical(self, "API Test Failed", f"Error: {str(e)}")
    
    def test_current_token(self):
        """Test the currently saved token"""
        if not self.config.api_key:
            QMessageBox.warning(self, "Missing API Key", "Please set API Key first.")
            return
        
        if not self.config.access_token:
            QMessageBox.warning(self, "No Token", "No access token is currently saved.")
            return
        
        try:
            kite = KiteConnect(api_key=self.config.api_key)
            kite.set_access_token(self.config.access_token)
            profile = kite.profile()
            
            # Get additional info if available
            try:
                margins = kite.margins()
                equity = margins.get('equity', {}).get('available', {}).get('cash', 0)
            except:
                equity = "Unknown"
            
            QMessageBox.information(
                self,
                "Token Test Successful",
                f"Access token is valid!\n\n"
                f"User: {profile['user_name']}\n"
                f"User ID: {profile['user_id']}\n"
                f"Email: {profile['email']}\n"
                f"Equity Available: {equity}\n\n"
                f"Token last character: ...{self.config.access_token[-5:]}"
            )
            
            # Update status display
            self.load_current_token_status()
            
        except KiteException as e:
            if "Invalid access token" in str(e) or "expired" in str(e).lower():
                QMessageBox.critical(
                    self,
                    "Token Expired",
                    "The access token has expired or is invalid.\n\n"
                    "Please generate a new token using the 'Generate Access Token' button."
                )
                self.token_status_label.setText("❌ Token Expired")
                self.token_status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
            else:
                QMessageBox.critical(self, "Token Test Failed", f"Error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Token Test Failed", f"Error: {str(e)}")
    
    def save_settings(self):
        """Save all settings and close dialog"""
        try:
            # Save API credentials
            old_api_key = self.config.api_key
            old_api_secret = self.config.api_secret
            
            self.config.api_key = self.api_key_input.text()
            self.config.api_secret = self.api_secret_input.text()
            
            # Check if API credentials changed
            api_creds_changed = (
                old_api_key != self.config.api_key or 
                old_api_secret != self.config.api_secret
            )
            
            # Auto Login Settings
            self.config.auto_login_enabled = self.auto_login_check.isChecked()
            
            # Only save embedded browser setting if available
            if WEBENGINE_AVAILABLE:
                self.config.use_embedded_browser = self.embedded_browser_check.isChecked()
            else:
                self.config.use_embedded_browser = False
            
            self.config.redirect_port = self.redirect_port_spin.value()
            
            # Trading Settings
            self.config.quantity = self.quantity_spin.value()
            self.config.entry_zscore_threshold = self.entry_zscore_spin.value()
            self.config.profit_target_percent = self.profit_target_spin.value()
            self.config.stop_loss_percent = self.stop_loss_spin.value()
            self.config.auto_entry_enabled = self.auto_entry_check.isChecked()
            self.config.auto_exit_enabled = self.auto_exit_check.isChecked()
            
            # Save to file
            self.config.save()
            
            # Show appropriate message
            if api_creds_changed:
                QMessageBox.information(
                    self, 
                    "Settings Saved", 
                    "Settings have been saved successfully!\n\n"
                    "Note: API credentials have changed. "
                    "You may need to reconnect or generate a new access token."
                )
            else:
                QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully!")
            
            # Don't close the dialog on Apply
            if self.sender().text() == "Apply":
                return
            else:
                self.accept()
                
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Error saving settings: {str(e)}")
    
    def apply_settings(self):
        """Apply settings without closing dialog"""
        try:
            # Save API credentials
            old_api_key = self.config.api_key
            old_api_secret = self.config.api_secret
            
            self.config.api_key = self.api_key_input.text()
            self.config.api_secret = self.api_secret_input.text()
            
            # Check if API credentials changed
            api_creds_changed = (
                old_api_key != self.config.api_key or 
                old_api_secret != self.config.api_secret
            )
            
            # Auto Login Settings
            self.config.auto_login_enabled = self.auto_login_check.isChecked()
            
            # Only save embedded browser setting if available
            if WEBENGINE_AVAILABLE:
                self.config.use_embedded_browser = self.embedded_browser_check.isChecked()
            else:
                self.config.use_embedded_browser = False
            
            self.config.redirect_port = self.redirect_port_spin.value()
            
            # Trading Settings
            self.config.quantity = self.quantity_spin.value()
            self.config.entry_zscore_threshold = self.entry_zscore_spin.value()
            self.config.profit_target_percent = self.profit_target_spin.value()
            self.config.stop_loss_percent = self.stop_loss_spin.value()
            self.config.auto_entry_enabled = self.auto_entry_check.isChecked()
            self.config.auto_exit_enabled = self.auto_exit_check.isChecked()
            
            # Save to file
            self.config.save()
            
            # Update token status
            self.load_current_token_status()
            
            # Show appropriate message
            if api_creds_changed:
                QMessageBox.information(
                    self, 
                    "Settings Applied", 
                    "Settings have been applied successfully!\n\n"
                    "Note: API credentials have changed. "
                    "You may need to reconnect or generate a new access token."
                )
            else:
                QMessageBox.information(self, "Settings Applied", "Settings have been applied successfully!")
                
        except Exception as e:
            QMessageBox.critical(self, "Apply Failed", f"Error applying settings: {str(e)}")

# ============================================================================
# CONTRACT SELECTION DIALOG
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
    print("7. Auto-login with external browser support")
    print("8. Simple Access Token Update Button")
    
    if WEBENGINE_AVAILABLE:
        print("9. Optional embedded browser login (PyQt6-WebEngine available)")
    else:
        print("9. Note: Install PyQt6-WebEngine for embedded browser login")
        print("   Command: pip install PyQt6-WebEngine")
    
    print("\nNote: Requires Zerodha account with MCX subscription")
    print("=" * 70)
    
    # Check for PyQt6-WebEngine
    if not WEBENGINE_AVAILABLE:
        print("\n⚠ PyQt6-WebEngine not found. Using external browser for login.")
        print("  To enable embedded browser login, install: pip install PyQt6-WebEngine")
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    dashboard = AdvancedTradingDashboard()
    dashboard.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()