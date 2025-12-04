"""
Technical Indicators Utility
Calculate technical indicators for stock trading signals using pandas-ta
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

from typing import Dict


class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    def __init__(self):
        """Initialize technical indicators calculator"""
        self.indicator_columns = []
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        print("Calculating technical indicators...")
        
        # Ensure data is sorted by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate each indicator category
        df = self.add_moving_averages(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_trend_indicators(df)
        
        print(f"✓ Calculated {len(self.indicator_columns)} technical indicators")
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators"""
        
        # Simple Moving Averages
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        
        # Exponential Moving Averages
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        
        # Moving average crossovers
        df['SMA_20_50_cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        df['golden_cross'] = ((df['SMA_50'] > df['SMA_200']) & 
                               (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))).astype(int)
        df['death_cross'] = ((df['SMA_50'] < df['SMA_200']) & 
                              (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))).astype(int)
        
        self.indicator_columns.extend([
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26', 'EMA_50',
            'SMA_20_50_cross', 'golden_cross', 'death_cross'
        ])
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # Relative Strength Index (RSI)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # MACD crossover signals
        df['MACD_cross_bullish'] = ((df['MACD'] > df['MACD_signal']) & 
                                     (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
        df['MACD_cross_bearish'] = ((df['MACD'] < df['MACD_signal']) & 
                                     (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))).astype(int)
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['STOCH_k'] = stoch['STOCHk_14_3_3']
        df['STOCH_d'] = stoch['STOCHd_14_3_3']
        
        # Rate of Change (ROC)
        df['ROC_10'] = ta.roc(df['close'], length=10)
        
        # Williams %R
        df['WILLR_14'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        self.indicator_columns.extend([
            'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
            'MACD_cross_bullish', 'MACD_cross_bearish',
            'STOCH_k', 'STOCH_d', 'ROC_10', 'WILLR_14'
        ])
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        # Get actual column names (they may vary by pandas_ta version)
        bb_cols = bbands.columns.tolist()
        df['BB_lower'] = bbands[bb_cols[0]]  # Lower band
        df['BB_middle'] = bbands[bb_cols[1]]  # Middle band
        df['BB_upper'] = bbands[bb_cols[2]]  # Upper band
        df['BB_width'] = bbands[bb_cols[3]]  # Bandwidth
        df['BB_percent'] = bbands[bb_cols[4]]  # Percent B
        
        # Average True Range (ATR)
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Historical Volatility
        df['HV_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        
        # === ENHANCED VOLATILITY FEATURES ===
        # Annualized volatility (20-day rolling)
        df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Volatility regime: ratio of current vs long-term volatility
        df['volatility_60'] = df['close'].pct_change().rolling(60).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_60']
        
        # ATR as percentage of price (normalized volatility)
        df['ATR_pct'] = df['ATR_14'] / df['close'] * 100
        
        # Volatility percentile (where is current vol vs historical)
        df['volatility_percentile'] = df['volatility_20'].rolling(252).apply(
            lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5, raw=False
        )
        
        # High/Low volatility regime (1 = high vol, 0 = normal/low vol)
        df['high_volatility_regime'] = (df['volatility_ratio'] > 1.2).astype(int)
        
        # Bollinger Band signals
        df['BB_squeeze'] = (df['BB_width'] < df['BB_width'].rolling(20).quantile(0.1)).astype(int)
        df['BB_breakout_upper'] = (df['close'] > df['BB_upper']).astype(int)
        df['BB_breakout_lower'] = (df['close'] < df['BB_lower']).astype(int)
        
        self.indicator_columns.extend([
            'BB_lower', 'BB_middle', 'BB_upper', 'BB_width', 'BB_percent',
            'ATR_14', 'HV_20', 'BB_squeeze', 'BB_breakout_upper', 'BB_breakout_lower',
            'volatility_20', 'volatility_60', 'volatility_ratio', 'ATR_pct',
            'volatility_percentile', 'high_volatility_regime'
        ])
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # On-Balance Volume (OBV)
        df['OBV'] = ta.obv(df['close'], df['volume'])
        
        # Volume Moving Average
        df['Volume_SMA_20'] = ta.sma(df['volume'], length=20)
        
        # Volume ratio
        df['Volume_ratio'] = df['volume'] / df['Volume_SMA_20']
        
        # Accumulation/Distribution
        df['AD'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['CMF_20'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        
        # Volume signals
        df['high_volume'] = (df['Volume_ratio'] > 1.5).astype(int)
        
        self.indicator_columns.extend([
            'OBV', 'Volume_SMA_20', 'Volume_ratio',
            'AD', 'CMF_20', 'high_volume'
        ])
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        
        # Average Directional Index (ADX)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX_14'] = adx['ADX_14']
        df['DMP_14'] = adx['DMP_14']
        df['DMN_14'] = adx['DMN_14']
        
        # Aroon Indicator
        aroon = ta.aroon(df['high'], df['low'], length=25)
        df['AROON_up'] = aroon['AROONU_25']
        df['AROON_down'] = aroon['AROOND_25']
        df['AROON_osc'] = aroon['AROONOSC_25']
        
        # Parabolic SAR
        df['SAR'] = ta.psar(df['high'], df['low'], df['close'])['PSARl_0.02_0.2']
        
        # Trend strength
        df['trend_strength'] = np.where(df['ADX_14'] > 25, 1, 0)
        
        # === ENHANCED FEATURES FOR BETTER ACCURACY ===
        # Price distance from moving averages (mean reversion signals)
        df['price_vs_SMA20'] = (df['close'] - df['SMA_20']) / df['SMA_20'] * 100
        df['price_vs_SMA50'] = (df['close'] - df['SMA_50']) / df['SMA_50'] * 100
        df['price_vs_EMA12'] = (df['close'] - df['EMA_12']) / df['EMA_12'] * 100
        
        # Momentum strength (combined momentum indicators)
        df['momentum_strength'] = (
            (df['RSI_14'] - 50) / 50 +  # Normalized RSI
            np.sign(df['MACD']) * (df['MACD'].abs() / df['close'] * 100) +  # Normalized MACD
            (df['ROC_10'] / 100)  # ROC as momentum
        ) / 3
        
        # Volume confirmation for signals
        df['volume_confirmed_move'] = (
            (df['Volume_ratio'] > 1.2) &  # Above average volume
            (df['close'].pct_change().abs() > 0.01)  # Significant price move
        ).astype(int)
        
        # Price position in Bollinger Bands (0-1 scale)
        df['BB_position_pct'] = (
            (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        ).clip(0, 1)
        
        # Multi-timeframe trend alignment
        df['trend_alignment'] = (
            (df['close'] > df['SMA_20']).astype(int) +
            (df['close'] > df['SMA_50']).astype(int) +
            (df['SMA_20'] > df['SMA_50']).astype(int)
        ) / 3  # 0 = bearish, 1 = bullish
        
        # === MARKET REGIME & PRICE STRUCTURE FEATURES ===
        # Market regime based on ADX (trending vs ranging)
        df['market_regime'] = np.where(df['ADX_14'] > 25, 1, 0)  # 1 = trending, 0 = ranging
        
        # Strong trend regime (very strong trend)
        df['strong_trend'] = np.where(df['ADX_14'] > 40, 1, 0)
        
        # Higher highs and lower lows (price structure)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int).rolling(3).sum()
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int).rolling(3).sum()
        
        # Bullish structure: consecutive higher highs
        df['bullish_structure'] = (df['higher_high'] >= 2).astype(int)
        # Bearish structure: consecutive lower lows
        df['bearish_structure'] = (df['lower_low'] >= 2).astype(int)
        
        # Price action: range expansion/contraction
        df['daily_range'] = (df['high'] - df['low']) / df['close'] * 100
        df['range_ratio'] = df['daily_range'] / df['daily_range'].rolling(20).mean()
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        
        # Close position relative to daily range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['close_position'] = df['close_position'].fillna(0.5)
        
        # Days since swing high/low
        df['days_since_high_5'] = df['close'].rolling(5).apply(lambda x: 4 - x.argmax(), raw=False)
        df['days_since_low_5'] = df['close'].rolling(5).apply(lambda x: 4 - x.argmin(), raw=False)
        
        self.indicator_columns.extend([
            'ADX_14', 'DMP_14', 'DMN_14',
            'AROON_up', 'AROON_down', 'AROON_osc',
            'SAR', 'trend_strength',
            'price_vs_SMA20', 'price_vs_SMA50', 'price_vs_EMA12',
            'momentum_strength', 'volume_confirmed_move',
            'BB_position_pct', 'trend_alignment',
            'market_regime', 'strong_trend',
            'higher_high', 'lower_low', 'bullish_structure', 'bearish_structure',
            'daily_range', 'range_ratio', 'gap_up', 'gap_down',
            'close_position', 'days_since_high_5', 'days_since_low_5'
        ])
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate composite trading signals from indicators
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            DataFrame with signal columns
        """
        df = df.copy()
        
        # RSI signals (oversold/overbought)
        df['signal_RSI'] = np.where(df['RSI_14'] < 30, 1,  # Oversold - Buy
                                     np.where(df['RSI_14'] > 70, -1, 0))  # Overbought - Sell
        
        # MACD signals
        df['signal_MACD'] = np.where(df['MACD_cross_bullish'] == 1, 1,
                                      np.where(df['MACD_cross_bearish'] == 1, -1, 0))
        
        # Bollinger Band signals
        df['signal_BB'] = np.where(df['BB_breakout_lower'] == 1, 1,  # Below lower band - Buy
                                    np.where(df['BB_breakout_upper'] == 1, -1, 0))  # Above upper band - Sell
        
        # Moving Average signals
        df['signal_MA'] = np.where(df['SMA_20_50_cross'] == 1, 1, -1)
        
        # Stochastic signals
        df['signal_STOCH'] = np.where(df['STOCH_k'] < 20, 1,
                                       np.where(df['STOCH_k'] > 80, -1, 0))
        
        # Composite signal (average of all signals)
        signal_cols = ['signal_RSI', 'signal_MACD', 'signal_BB', 'signal_MA', 'signal_STOCH']
        df['composite_signal'] = df[signal_cols].mean(axis=1)
        
        # Final signal: Buy (1), Hold (0), Sell (-1)
        df['final_signal'] = np.where(df['composite_signal'] > 0.3, 1,
                                       np.where(df['composite_signal'] < -0.3, -1, 0))
        
        print("✓ Generated trading signals")
        
        return df
    
    def create_target_labels(
        self, 
        df: pd.DataFrame, 
        threshold_buy: float = 0.01,
        threshold_sell: float = -0.01,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Create target labels for classification
        
        Args:
            df: DataFrame with price data
            threshold_buy: Threshold for buy signal (e.g., 1% = 0.01)
            threshold_sell: Threshold for sell signal (e.g., -1% = -0.01)
            horizon: Number of days ahead to calculate return
            
        Returns:
            DataFrame with target label column
        """
        df = df.copy()
        
        # Calculate forward returns
        df['future_return'] = df['close'].pct_change(periods=horizon).shift(-horizon)
        
        # Create labels: 1 (Buy), 0 (Hold), -1 (Sell)
        df['target'] = np.where(df['future_return'] > threshold_buy, 1,
                                np.where(df['future_return'] < threshold_sell, -1, 0))
        
        # Remove last rows with NaN targets
        valid_rows = df['target'].notna().sum()
        
        print("✓ Created target labels (Buy: 1, Hold: 0, Sell: -1)")
        print(f"  Valid rows: {valid_rows}")
        print("  Label distribution:")
        print(f"    Buy (1):  {(df['target'] == 1).sum()} ({(df['target'] == 1).sum()/valid_rows*100:.1f}%)")
        print(f"    Hold (0): {(df['target'] == 0).sum()} ({(df['target'] == 0).sum()/valid_rows*100:.1f}%)")
        print(f"    Sell (-1): {(df['target'] == -1).sum()} ({(df['target'] == -1).sum()/valid_rows*100:.1f}%)")
        
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of current indicator values"""
        latest = df.iloc[-1]
        
        summary = {
            'RSI': float(latest['RSI_14']),
            'MACD': float(latest['MACD']),
            'MACD_signal': float(latest['MACD_signal']),
            'BB_position': 'Above Upper' if latest['close'] > latest['BB_upper'] 
                          else 'Below Lower' if latest['close'] < latest['BB_lower']
                          else 'Middle',
            'Volume_ratio': float(latest['Volume_ratio']),
            'ADX': float(latest['ADX_14']),
            'composite_signal': float(latest.get('composite_signal', 0)),
            'final_signal': 'BUY' if latest.get('final_signal', 0) == 1 
                           else 'SELL' if latest.get('final_signal', 0) == -1 
                           else 'HOLD'
        }
        
        return summary
