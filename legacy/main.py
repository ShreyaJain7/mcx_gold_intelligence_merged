import pandas as pd
import numpy as np
import logging
import os

from src.data_loader import load_or_create_master_dataset
from src.strategy import generate_pairs_zscore_signals, generate_ema_signals
from src.metrics import calculate_risk_metrics, run_cointegration_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline():
    logging.info("Starting MCX Intelligence Pipeline...")
    
    # 1. Load 
    df_dict = load_or_create_master_dataset()
    if not df_dict or 'GOLDM' not in df_dict:
        logging.error("Failed to load master dataset or GOLDM benchmark missing.")
        return

    benchmark_df = df_dict['GOLDM']
    
    print("\n--- Running Baseline Pairs Backtests & Attribution (Z-Score Mean-Reversion) ---")
    print(f"{'Symbol':>10} {'Market Return':>15} {'Strategy Return':>17} {'Net PnL (INR)':>15} {'Alpha':>8} {'Beta':>6} {'Sharpe':>8} {'Max Drawdown':>14}")
    
    dashboard_records = []
    
    # Custom tuned entry/exit Z-scores per contract
    contract_params = {
        'GOLDGUINEA': {'entry_z': 1.5, 'exit_z': 0.5},
        'GOLDPETAL':  {'entry_z': 1.2, 'exit_z': 0.2},
        'GOLDTEN':    {'entry_z': 1.8, 'exit_z': 0.5}
    }

    # 2. Loop Through All Contracts
    for symbol, df_contract in df_dict.items():
        if symbol == 'GOLDM':
            results = generate_ema_signals(df_contract, fast_window=10, slow_window=30)
        else:
            params = contract_params.get(symbol, {'entry_z': 1.5, 'exit_z': 0.5})
            results = generate_pairs_zscore_signals(
                df_asset=df_contract,
                df_benchmark=benchmark_df,
                lookback=20,
                entry_z=params['entry_z'],
                exit_z=params['exit_z'],
                symbol=symbol  # Passes symbol to handle contract-specific spread orientation
            )
            
        metrics = calculate_risk_metrics(results)
        
        mkt_ret = results['Cumulative_Market_Return'].iloc[-1]
        strat_ret = results['Cumulative_Strategy_Return'].iloc[-1]
        net_pnl = results['Strategy_Return'].sum() * df_contract['Close'].mean()
        
        alpha = metrics.get('Alpha', 0.0)
        beta = metrics.get('Beta', 0.0)
        sharpe = metrics.get('Sharpe_Ratio', 0.0)
        max_dd = metrics.get('Max_Drawdown', 0.0)
        
        print(f"{symbol:>10} {mkt_ret:>14.2%} {strat_ret:>16.2%} Rs. {net_pnl:>12,.2f} {alpha:>7.2%} {beta:>6.2f} {sharpe:>8.2f} {max_dd:>13.2%}")
        
        results['Symbol'] = symbol
        dashboard_records.append(results)

    # 3. Statistical Cointegration Tests
    print("\n--- Statistical Cointegration Tests (Benchmark: GOLDM Log-Price) ---")
    print(f"{'Pair':>20} {'t-Statistic':>12} {'p-Value':>10} {'Result':>28}")
    
    for symbol, df_contract in df_dict.items():
        if symbol == 'GOLDM':
            continue
        coint_res = run_cointegration_test(benchmark_df['Close'], df_contract['Close'])
        t_stat = coint_res.get('t_stat', 0.0)
        p_val = coint_res.get('p_value', 1.0)
        status = "Cointegrated (Mean-Reverting)" if p_val < 0.05 else "Not Cointegrated"
        
        print(f"{'GOLDM vs ' + symbol:>20} {t_stat:>12.4f} {p_val:>10.4f} {status:>28}")

    # 4. Parameter Sensitivity Check on GOLDGUINEA
    print("\n--- Running Parameter Sensitivity Check (GOLDGUINEA) ---")
    print(f"{'Z-Config':>12} {'Strategy Return':>16} {'Sharpe Ratio':>13} {'Alpha':>8} {'Beta':>6}")
    
    if 'GOLDGUINEA' in df_dict:
        guinea_df = df_dict['GOLDGUINEA']
        z_configs = [
            ("Z (1.0 / 0.2)", 1.0, 0.2),
            ("Z (1.2 / 0.2)", 1.2, 0.2),
            ("Z (1.5 / 0.5)", 1.5, 0.5)
        ]
        
        for name, entry_z, exit_z in z_configs:
            res_sens = generate_pairs_zscore_signals(
                df_asset=guinea_df,
                df_benchmark=benchmark_df,
                lookback=20,
                entry_z=entry_z,
                exit_z=exit_z,
                symbol='GOLDGUINEA'
            )
            m_sens = calculate_risk_metrics(res_sens)
            s_ret = res_sens['Cumulative_Strategy_Return'].iloc[-1]
            s_sharpe = m_sens.get('Sharpe_Ratio', 0.0)
            s_alpha = m_sens.get('Alpha', 0.0)
            s_beta = m_sens.get('Beta', 0.0)
            
            print(f"{name:>12} {s_ret:>15.2%} {s_sharpe:>13.2f} {s_alpha:>7.2%} {s_beta:>6.2f}")

    # 5. Export Consolidated Dashboard Dataset
    if dashboard_records:
        full_dashboard_df = pd.concat(dashboard_records, ignore_index=True)
        os.makedirs("data/processed", exist_ok=True)
        export_path = "data/processed/dashboard_export.parquet"
        full_dashboard_df.to_parquet(export_path, index=False)
        print(f"\nSuccessfully exported dashboard dataset to: {export_path}")

if __name__ == "__main__":
    run_pipeline()