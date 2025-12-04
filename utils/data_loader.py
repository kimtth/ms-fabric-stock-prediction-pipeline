"""
Data Loader Utility for Stock Data
Handles data ingestion from yfinance and Delta table operations in Fabric Lakehouse
"""

import yfinance as yf
import pandas as pd
import json
import os

from datetime import datetime, timedelta
from typing import Optional, List, Tuple


class StockDataLoader:
    """Load and manage stock data from various sources"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize data loader with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol (e.g., 'MSFT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = self.config['data_source']['start_date']
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Add metadata columns
            df['Ticker'] = ticker
            df['FetchTimestamp'] = datetime.now()
            
            # Rename columns to standard format
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add partition columns for efficient storage
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            print(f"✓ Fetched {len(df)} records for {ticker}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data for {ticker}: {str(e)}")
            raise
    
    def fetch_latest_data(self, ticker: str, lookback_days: int = 5) -> pd.DataFrame:
        """
        Fetch only the latest data (for daily updates)
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with recent OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        return self.fetch_stock_data(
            ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] < 0).any():
                issues.append(f"Negative values in {col}")
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            issues.append("High price less than low price detected")
        
        # Check for duplicate dates
        if df['date'].duplicated().any():
            issues.append("Duplicate dates found")
        
        # Check for zero volume (potential data issue)
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"{zero_volume} records with zero volume")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            print("✓ Data validation passed")
        else:
            print(f"✗ Data validation failed with {len(issues)} issues")
            for issue in issues:
                print(f"  - {issue}")
        
        return is_valid, issues
    
    def write_to_bronze(self, df: pd.DataFrame, lakehouse_path: str) -> str:
        """
        Write data to Bronze layer (Delta format)
        
        Args:
            df: DataFrame to write
            lakehouse_path: Base path to lakehouse
            
        Returns:
            Path where data was written
        """
        bronze_path = os.path.join(
            lakehouse_path,
            self.config['lakehouse']['bronze_path']
        )
        
        # Write as Delta table (partitioned by year/month)
        output_path = bronze_path
        
        # Show only relative path from lakehouse
        bronze_rel = self.config['lakehouse']['bronze_path']
        print(f"Writing {len(df)} records to Bronze layer: {bronze_rel}")
        
        try:
            # In Fabric, you would use:
            # df.write.format("delta").mode("append").partitionBy("year", "month").save(output_path)
            
            # For local testing, write as parquet
            os.makedirs(output_path, exist_ok=True)
            df.to_parquet(
                output_path + "/data.parquet",
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            print(f"✓ Data written successfully to {bronze_rel}")
            return output_path
            
        except Exception as e:
            print(f"✗ Error writing to Bronze layer: {str(e)}")
            raise
    
    def read_from_bronze(self, lakehouse_path: str, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from Bronze layer
        
        Args:
            lakehouse_path: Base path to lakehouse
            ticker: Optional ticker to filter
            
        Returns:
            DataFrame with bronze data
        """
        bronze_path = os.path.join(
            lakehouse_path,
            self.config['lakehouse']['bronze_path']
        )
        
        # Show only relative path from lakehouse
        bronze_rel = self.config['lakehouse']['bronze_path']
        print(f"Reading data from Bronze layer: {bronze_rel}")
        
        try:
            # In Fabric, you would use:
            # df = spark.read.format("delta").load(bronze_path)
            
            # For local testing, read parquet
            df = pd.read_parquet(bronze_path + "/data.parquet")
            
            if ticker:
                df = df[df['Ticker'] == ticker]
            
            print(f"✓ Read {len(df)} records from Bronze layer")
            return df
            
        except Exception as e:
            print(f"✗ Error reading from Bronze layer: {str(e)}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'record_count': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'price_range': {
                'min': float(df['low'].min()),
                'max': float(df['high'].max()),
                'latest_close': float(df.iloc[-1]['close'])
            },
            'volume': {
                'avg': float(df['volume'].mean()),
                'total': int(df['volume'].sum())
            },
            'returns': {
                'total': float((df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100),
                'daily_avg': float(df['close'].pct_change().mean() * 100)
            }
        }
        
        return summary


def print_data_summary(summary: dict):
    """Pretty print data summary"""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Record Count:     {summary['record_count']:,}")
    print(f"Date Range:       {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Price Range:      ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}")
    print(f"Latest Close:     ${summary['price_range']['latest_close']:.2f}")
    print(f"Avg Daily Volume: {summary['volume']['avg']:,.0f}")
    print(f"Total Return:     {summary['returns']['total']:.2f}%")
    print(f"Avg Daily Return: {summary['returns']['daily_avg']:.4f}%")
    print("="*60 + "\n")
