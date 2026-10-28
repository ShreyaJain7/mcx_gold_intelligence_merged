import numpy as np
import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.stattools import coint

def calculate_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.05) -> Dict[str, float]:
    """
    Calculates annualized Sharpe Ratio, Max Drawdown, Alpha, and Beta.
    """
    metrics = {
        'Sharpe_Ratio': 0.0,
        'Max_Drawdown': 0.0,
        'Alpha': 0.0,
        'Beta': 0.0
    }
    
    if df.empty or 'Strategy_Return' not in df.columns:
        return metrics

    strat_ret = df['Strategy_Return'].fillna(0.0)
    mkt_ret = df['Daily_Return'].fillna(0.0) if 'Daily_Return' in df.columns else strat_ret

    # 1. Annualized Sharpe Ratio 
    mean_ret = strat_ret.mean() * 252
    std_ret = strat_ret.std() * np.sqrt(252)
    if std_ret > 1e-6:
        metrics['Sharpe_Ratio'] = float((mean_ret - risk_free_rate) / std_ret)

    # 2. Maximum Drawdown
    cum_ret = (1 + strat_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    metrics['Max_Drawdown'] = float(drawdown.min())

    # 3. Alpha and Beta calculation 
    if len(mkt_ret) > 10 and mkt_ret.std() > 1e-6:
        cov_matrix = np.cov(strat_ret, mkt_ret)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
        alpha = (mean_ret - risk_free_rate) - beta * (mkt_ret.mean() * 252 - risk_free_rate)
        
        metrics['Beta'] = float(beta)
        metrics['Alpha'] = float(alpha)

    return metrics


def run_cointegration_test(series_a: pd.Series, series_b: pd.Series) -> Dict[str, Any]:
    """
    Performs Engle-Granger two-step cointegration test on Log-Price series.
    """
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 20:
        return {'t_stat': 0.0, 'p_value': 1.0}

    try:
        # Use log prices to test cointegration 
        log_a = np.log(aligned.iloc[:, 0])
        log_b = np.log(aligned.iloc[:, 1])
        
        t_stat, p_val, _ = coint(log_a, log_b)
        return {
            't_stat': float(t_stat),
            'p_value': float(p_val)
        }
    except Exception:
        return {'t_stat': 0.0, 'p_value': 1.0}