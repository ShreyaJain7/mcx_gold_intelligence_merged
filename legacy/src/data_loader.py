import os
import glob
import logging
import pandas as pd
import numpy as np
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
MASTER_PARQUET_PATH = os.path.join(PROCESSED_DATA_DIR, "master_gold_contracts_12m.parquet")

# Contract Lot 
# GOLDM = 10g lot size 
# GOLDTEN = 10g lot size
# GOLDGUINEA = 8g lot size
# GOLDPETAL = 1g lot size
CONTRACT_GRAM_SCALERS = {
    'GOLDM': 10.0,
    'GOLDTEN': 10.0,
    'GOLDGUINEA': 8.0,
    'GOLDPETAL': 1.0
}


def clean_bhavcopy_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names, cleans date formats, and normalizes contract prices.
    """
    
    df.columns = [str(col).strip().upper() for col in df.columns]
    
    symbol_col = None
    for candidate in ['SYMBOL', 'INSTRUMENT_NAME', 'CONTRACT_NAME', 'COMMODITY']:
        if candidate in df.columns:
            symbol_col = candidate
            break
            
    if symbol_col is None:
        return pd.DataFrame()
        
    df['Symbol'] = df[symbol_col].astype(str).str.strip().str.upper()
    
    target_symbols = ['GOLDM', 'GOLDTEN', 'GOLDGUINEA', 'GOLDPETAL']
    df = df[df['Symbol'].isin(target_symbols)].copy()
    if df.empty:
        return pd.DataFrame()

    date_col = None
    for candidate in ['DATE', 'TRADE_DATE', 'TIMESTAMP', 'EXPIRY_DATE']:
        if candidate in df.columns:
            date_col = candidate
            break
            
    if date_col is None:
        return pd.DataFrame()
        
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date'])

    close_col = None
    for candidate in ['CLOSE', 'CLOSE_PRICE', 'SETTLEMENT_PRICE', 'LAST_PRICE']:
        if candidate in df.columns:
            close_col = candidate
            break
            
    if close_col is None:
        return pd.DataFrame()

    df['Close'] = pd.to_numeric(df[close_col], errors='coerce')
    df = df.dropna(subset=['Close'])

    high_col = 'HIGH' if 'HIGH' in df.columns else close_col
    low_col = 'LOW' if 'LOW' in df.columns else close_col
    
    df['High'] = pd.to_numeric(df[high_col], errors='coerce').fillna(df['Close'])
    df['Low'] = pd.to_numeric(df[low_col], errors='coerce').fillna(df['Close'])
    
    df = df.sort_values(['Symbol', 'Date']).drop_duplicates(subset=['Symbol', 'Date'], keep='last')
    
    return df[['Symbol', 'Date', 'Close', 'High', 'Low']]


def process_raw_bhavcopies() -> pd.DataFrame:
    """
    Reads all Bhavcopy CSV files from data and compiles them into a clean DataFrame.
    """
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not csv_files:
        logging.warning(f"No CSV files found in directory: {RAW_DATA_DIR}")
        return pd.DataFrame()

    all_frames = []
    for file_path in csv_files:
        try:
            temp_df = pd.read_csv(file_path)
            cleaned = clean_bhavcopy_df(temp_df)
            if not cleaned.empty:
                all_frames.append(cleaned)
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

    if not all_frames:
        return pd.DataFrame()

    master_df = pd.concat(all_frames, ignore_index=True)
    master_df = master_df.sort_values(['Symbol', 'Date']).drop_duplicates(subset=['Symbol', 'Date'], keep='last')
    
    return master_df


def load_or_create_master_dataset() -> Dict[str, pd.DataFrame]:
    """
    Loads processed contract data from Parquet, or builds it fresh from raw Bhavcopies.
    Ensures price scaling per gram across all contracts.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Process fresh raw data if Parquet missing or force reload requested
    master_df = process_raw_bhavcopies()
    
    if master_df.empty and os.path.exists(MASTER_PARQUET_PATH):
        logging.info(f"Loading existing Master Parquet from: {MASTER_PARQUET_PATH}")
        master_df = pd.read_parquet(MASTER_PARQUET_PATH)
    elif not master_df.empty:
        master_df.to_parquet(MASTER_PARQUET_PATH, index=False)
        logging.info(f"Successfully generated fresh master Parquet at: {MASTER_PARQUET_PATH} ({len(master_df)} records)")
    else:
        logging.error("No valid Bhavcopy data available to construct master dataset.")
        return {}

    df_dict = {}
    symbols = master_df['Symbol'].unique()

    for sym in symbols:
        sub_df = master_df[master_df['Symbol'] == sym].copy()
        sub_df = sub_df.sort_values('Date').set_index('Date')
        
        scale_factor = CONTRACT_GRAM_SCALERS.get(sym, 1.0)
        sub_df['Price_Per_Gram'] = sub_df['Close'] / scale_factor
        
        sub_df['Daily_Return'] = sub_df['Close'].pct_change().fillna(0.0)
        
        df_dict[sym] = sub_df

    return df_dict