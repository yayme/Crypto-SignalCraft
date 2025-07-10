import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import collect_only, calculate_ic, calculate_r2, calulate_win, calculate_sharpe
import json
import importlib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from scipy.stats import spearmanr, pearsonr
from regression import run_indicators

with open('universal_config.json', 'r') as f:
    config = json.load(f)

symbols_dict = config['symbols']
most_frequent = sorted(symbols_dict.items(), key=lambda x: x[1], reverse=True)[:5]
symbols = [s[0] for s in most_frequent]

indicator_module = importlib.import_module('indicators')
indicator_names = [
    'vwa_mean_reversion', 'volatility_breakout', 'buy_volume_sentiment', 'interest_rate_carry',
    'spread_vwap_signal', 'wick_reversal_signal',
    'sma_50', 'sma_200',
    'ema_20', 'ema_50', 'ema_200',
    'rsi_14',
    'macd_12_26_9',
    'bollinger_bands_20_2',
    'obv',
    'adx_14',
    'stochastic_oscillator_14',
    'ichimoku_cloud_9_26_52',
    'vwap',
    'atr', 'cci', 'williams_r'
]

indicator_functions = {
    'vwa_mean_reversion': indicator_module.vwa_mean_reversion,
    'volatility_breakout': indicator_module.volatility_breakout,
    'buy_volume_sentiment': indicator_module.buy_volume_sentiment,
    'interest_rate_carry': indicator_module.interest_rate_carry,
    'spread_vwap_signal': indicator_module.spread_vwap_signal,
    'wick_reversal_signal': indicator_module.wick_reversal_signal,
    'sma_50': lambda df: indicator_module.sma(df, window=50),
    'sma_200': lambda df: indicator_module.sma(df, window=200),
    'ema_20': lambda df: indicator_module.ema(df, window=20),
    'ema_50': lambda df: indicator_module.ema(df, window=50),
    'ema_200': lambda df: indicator_module.ema(df, window=200),
    'rsi_14': lambda df: indicator_module.rsi(df, window=14),
    'macd_12_26_9': lambda df: indicator_module.macd(df, fast=12, slow=26, signal=9),
    'bollinger_bands_20_2': lambda df: indicator_module.bollinger_bands(df, window=20, num_std=2),
    'obv': indicator_module.obv,
    'adx_14': lambda df: indicator_module.adx(df, window=14),
    'stochastic_oscillator_14': lambda df: indicator_module.stochastic_oscillator(df, k_window=14, d_window=3),
    'ichimoku_cloud_9_26_52': lambda df: indicator_module.ichimoku_cloud(df, tenkan=9, kijun=26, senkou=52),
    'vwap': indicator_module.vwap,
    'atr': indicator_module.atr,
    'cci': indicator_module.cci,
    'williams_r': indicator_module.williams_r
}

for symbol in symbols:
    filename = f"{symbol}_spot_full.csv"
    df = pd.read_csv(filename)
    print(f"\n=== {symbol} ===")
    print(df.head())
    print(df.tail())
    print(df.columns)
    n = len(df)
    split_idx = n // 2
    print(f"First 50% ends at index: {split_idx}, date: {df.iloc[split_idx-1]['close_time'] if 'close_time' in df.columns else split_idx-1}")
    df = df.reset_index(drop=True)
    for trading_window in range(1, 25):
        df[f'return_{trading_window}h'] = df['bam_close'].shift(-trading_window) - df['bam_close']
        df[f'return_{trading_window}h_b'] = (df[f'return_{trading_window}h'] > 0).astype(int)
    test_df = df.iloc[:split_idx].copy()
    results = []
    # Example usage of run_indicators with indicator_functions
    # You can adjust indicator_names and other params as needed
    # Uncomment the following lines to run:
    # linear_model, binary_model = run_indicators(
    #     test_df, indicator_names, indicator_functions=indicator_functions,
    #     obj=None, time_window=1, target_col='bam_close',
    #     alpha=0.1, l1_ratio=0.5, max_iter=10000,
    #     initial_train_size=0.5, val_size=0.1, step_size=0.05)
    for indicator_name in indicator_names:
        if indicator_name not in indicator_functions:
            continue
        indicator_fn = indicator_functions[indicator_name]
        print(f"\n--- Indicator: {indicator_name} ---")
        for trading_window in range(1, 25):
            y_reg = test_df[f'return_{trading_window}h']
            y_bin = test_df[f'return_{trading_window}h_b']
            try:
                signal = indicator_fn(test_df)
                if isinstance(signal, tuple) or isinstance(signal, list):
                    signal = signal[0]
                signal = pd.Series(signal, index=test_df.index)
                valid = signal.notna() & y_reg.notna() & y_bin.notna()
                X = signal[valid].values.reshape(-1, 1)
                y_reg_valid = y_reg[valid]
                y_bin_valid = y_bin[valid]
                if len(np.unique(y_bin_valid)) > 1:
                    logreg = LogisticRegression(max_iter=1000)
                    logreg.fit(X, y_bin_valid)
                    y_pred_bin = logreg.predict(X)
                    acc = accuracy_score(y_bin_valid, y_pred_bin)
                else:
                    acc = np.nan
                linreg = LinearRegression()
                linreg.fit(X, y_reg_valid)
                y_pred_reg = linreg.predict(X)
                r2 = r2_score(y_reg_valid, y_pred_reg)
                pearson_corr = pearsonr(signal[valid], y_reg_valid)[0] if len(signal[valid]) > 1 else np.nan
                spearman_corr = spearmanr(signal[valid], y_reg_valid)[0] if len(signal[valid]) > 1 else np.nan
                ic = calculate_ic(pd.DataFrame({'position': signal[valid], 'bam_close': test_df['bam_close'][valid]}), signal_col='position', price_col='bam_close')
                win_rate = calulate_win(pd.DataFrame({'position': signal[valid], 'bam_close': test_df['bam_close'][valid]}), signal_col='position', price_col='bam_close')
                sharpe = calculate_sharpe(pd.DataFrame({'position': signal[valid], 'bam_close': test_df['bam_close'][valid]}), price_col='bam_close', position_col='position')
                results.append({
                    'window': trading_window,
                    'indicator': indicator_name,
                    'acc': acc,
                    'r2': r2,
                    'pearson': pearson_corr,
                    'spearman': spearman_corr,
                    'ic': ic,
                    'win': win_rate,
                    'sharpe': sharpe
                })
                print(f"Window: {trading_window}h | Acc: {acc:.3f} | R2: {r2:.3f} | Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f} | IC: {ic:.3f} | Win: {win_rate:.3f} | Sharpe: {sharpe:.3f}")
            except Exception as e:
                print(f"Window: {trading_window}h | Error: {e}")
    # Export results as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"indicator_results_{symbol}.csv", index=False)
