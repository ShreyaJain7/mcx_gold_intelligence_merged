import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from typing import Dict, Union

def calculate_beta_separation(
    strategy_returns: pd.Series, 
    market_returns: pd.Series
) -> Dict[str, Union[float, str]]:
    """
    Performs OLS regression of strategy returns against benchmark market returns.
    Separates market beta exposure from true strategy alpha.
    """
    combined = pd.concat([strategy_returns, market_returns], axis=1).dropna()
    combined.columns = ['Strategy', 'Market']
    
    if combined.empty or len(combined) < 10:
        return {
            'Alpha (Annualized)': 0.0,
            'Beta': 0.0,
            'R-Squared': 0.0,
            'P-Value': 1.0,
            'Status': 'Insufficient Data'
        }

    Y = combined['Strategy']
    X = combined['Market']
    X_with_const = sm.add_constant(X)

    model = sm.OLS(Y, X_with_const).fit()

    alpha_daily = model.params.get('const', 0.0)
    beta = model.params.get('Market', 0.0)
    p_value = model.pvalues.get('Market', 1.0)
    r_squared = model.rsquared

    alpha_annualized = alpha_daily * 252

    return {
        'Alpha (Annualized)': float(alpha_annualized),
        'Beta': float(beta),
        'R-Squared': float(r_squared),
        'P-Value': float(p_value),
        'Status': 'Significant' if p_value < 0.05 else 'Insignificant'
    }


def calculate_cointegration(
    series1: pd.Series, 
    series2: pd.Series
) -> Dict[str, Union[float, str, bool]]:
    """
    Performs the Engle-Granger two-step cointegration test between two price series.
    A p-value < 0.05 indicates statistical cointegration (mean-reverting relationship).
    """
    combined = pd.concat([series1, series2], axis=1).dropna()
    if len(combined) < 20:
        return {
            't-statistic': 0.0,
            'p-value': 1.0,
            'Is Cointegrated': False,
            'Status': 'Insufficient Data'
        }

    score, p_value, _ = coint(combined.iloc[:, 0], combined.iloc[:, 1])

    return {
        't-statistic': float(score),
        'p-value': float(p_value),
        'Is Cointegrated': bool(p_value < 0.05),
        'Status': 'Cointegrated (Mean-Reverting)' if p_value < 0.05 else 'Not Cointegrated'
    }