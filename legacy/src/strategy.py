import numpy as np
import pandas as pd

def generate_pairs_zscore_signals(
    df_asset: pd.DataFrame, 
    df_benchmark: pd.DataFrame, 
    lookback: int = 20, 
    entry_z: float = 1.5, 
    exit_z: float = 0.5,
    transaction_cost_pct: float = 0.0005,
    symbol: str = ""
) -> pd.DataFrame:
    """
    Generates Z-Score signals 
    """
    data = df_asset.copy()
    
    log_asset = np.log(data['Close'])
    log_bench = np.log(df_benchmark['Close'].reindex(data.index).ffill())
    
    # GOLDTEN has a 10g lot size like GOLDM
    if symbol == 'GOLDTEN':
        spread = log_asset - log_bench
    else:
        spread = log_bench - log_asset
        
    mean_spread = spread.rolling(window=lookback, min_periods=5).mean()
    std_spread = spread.rolling(window=lookback, min_periods=5).std().replace(0, 1e-6)
    
    z_score = (spread - mean_spread) / std_spread
    data['Z_Score'] = z_score.fillna(0.0)
    
    signals = np.zeros(len(data))
    current_pos = 0.0
    
    for i in range(1, len(data)):
        z = data['Z_Score'].iloc[i]
        if z > entry_z:
            current_pos = -1.0
        elif z < -entry_z:
            current_pos = 1.0
        elif abs(z) < exit_z:
            current_pos = 0.0
        signals[i] = current_pos
        
    data['Signal'] = signals
    data['Position'] = data['Signal'].shift(1).fillna(0.0)
    
    if 'Daily_Return' not in data.columns:
        data['Daily_Return'] = data['Close'].pct_change().fillna(0.0)
        
    position_changes = data['Position'].diff().abs().fillna(0.0)
    data['Trade_Cost'] = position_changes * transaction_cost_pct
    data['Strategy_Return'] = (data['Position'] * data['Daily_Return']) - data['Trade_Cost']
    
    cum_mkt = (1 + data['Daily_Return']).cumprod() - 1
    cum_strat = (1 + data['Strategy_Return']).cumprod() - 1
    
    data['Cumulative_Market_Return'] = cum_mkt
    data['Cumulative_Strategy_Return'] = cum_strat
    data['Cumulative_Market'] = cum_mkt
    data['Cumulative_Strategy'] = cum_strat
    
    return data

def generate_ema_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    data = df.copy()
    if 'Daily_Return' not in data.columns:
        data['Daily_Return'] = data['Close'].pct_change().fillna(0.0)
    
    data['EMA_Short'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_Long'] = data['Close'].ewm(span=30, adjust=False).mean()
    data['Signal'] = np.where(data['EMA_Short'] > data['EMA_Long'], 1.0, -1.0)
    data['Position'] = data['Signal'].shift(1).fillna(0.0)
    data['Trade_Cost'] = data['Position'].diff().abs().fillna(0.0) * 0.0005
    data['Strategy_Return'] = (data['Position'] * data['Daily_Return']) - data['Trade_Cost']
    
    cum_mkt = (1 + data['Daily_Return']).cumprod() - 1
    cum_strat = (1 + data['Strategy_Return']).cumprod() - 1
    data['Cumulative_Market_Return'] = cum_mkt
    data['Cumulative_Strategy_Return'] = cum_strat
    data['Cumulative_Market'] = cum_mkt
    data['Cumulative_Strategy'] = cum_strat
    return data

generate_ema_atr_signals = generate_ema_signals