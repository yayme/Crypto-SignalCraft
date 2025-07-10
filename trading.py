import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
    'sma_5': lambda df: indicator_module.sma(df, window=5),
    'sma_10': lambda df: indicator_module.sma(df, window=10),
    'sma_20': lambda df: indicator_module.sma(df, window=20),
    'sma_50': lambda df: indicator_module.sma(df, window=50),
    'sma_200': lambda df: indicator_module.sma(df, window=200),
    'ema_5': lambda df: indicator_module.ema(df, window=5),
    'ema_10': lambda df: indicator_module.ema(df, window=10),
    'ema_15': lambda df: indicator_module.ema(df, window=15),
    'ema_20': lambda df: indicator_module.ema(df, window=20),
    'ema_30': lambda df: indicator_module.ema(df, window=30),
    'ema_50': lambda df: indicator_module.ema(df, window=50),
    'ema_200': lambda df: indicator_module.ema(df, window=200),
    'momentum_3': lambda df: df['bam_close'].diff(3),
    'momentum_5': lambda df: df['bam_close'].diff(5),
    'momentum_10': lambda df: df['bam_close'].diff(10),
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

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr

def simulate_trading(df, model, feature_cols, target_col='bam_close',
                     start_time=None, end_time=None, initial_capital=1,
                     trading_window=1, threshold=0.1, stop_loss=None, take_profit=None,
                     plot=True, strategy='reactive', verbose=True):
    df = df.copy()
    
    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")

    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()

    expanded_feature_cols = []
    for feature in feature_cols:
        if feature in df.columns:
            expanded_feature_cols.append(feature)
        elif feature in indicator_functions:
            result = indicator_functions[feature](df)
            if isinstance(result, (tuple, list)):
                for i, arr in enumerate(result):
                    colname = f'{feature}_{i}'
                    df[colname] = arr
                    expanded_feature_cols.append(colname)
            else:
                df[feature] = result
                expanded_feature_cols.append(feature)
        else:
            if verbose:
                print(f"Feature '{feature}' not found in df.columns or indicator_functions. Skipping.")

    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)

    try:
        df['signal_raw'] = model.predict(df[expanded_feature_cols])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # Use raw signal for all strategies
    df['signal'] = df['signal_raw'].shift(1)
    df['signal_raw'] = df['signal_raw'].shift(1)

    # (remove z-score normalization and clipping)

    df['capital'] = np.nan
    df['position'] = np.nan
    df['pnl'] = np.nan

    capital = initial_capital
    prev_position = 0
    num_trades = 0
    trade_pnls = []
    capital_history = np.full(len(df), np.nan)
    stop_index = None

    for i in range(1, len(df)):
        if i % trading_window != 0:
            continue

        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        signal = curr['signal']

        if pd.isna(signal):
            continue

        raw_signal = curr['signal_raw']
        if strategy == 'reactive':
            capital_ratio = capital / initial_capital
            if capital_ratio >= 1.5:
                utilization = 0.8
            elif capital_ratio >= 1.0:
                utilization = 0.6
            elif capital_ratio >= 0.7:
                utilization = 0.4
            else:
                utilization = 0.2
            position = capital * signal * utilization

        elif strategy == 'reverse_reactive':
            capital_ratio = capital / initial_capital
            if capital_ratio <= 0.5:
                utilization = 0.9
            elif capital_ratio <= 0.7:
                utilization = 0.8
            elif capital_ratio <= 1.0:
                utilization = 0.6
            else:
                utilization = 0.3
            position = capital * signal * utilization

        elif strategy == 'simple':
            if raw_signal > 0:
                position = capital * 0.5
            elif raw_signal < 0:
                position = capital * -0.5
            else:
                position = 0
        else:
            position = capital * signal

        ret = curr['return']
        if stop_loss is not None and ret < stop_loss:
            pnl = position * stop_loss
            position = 0
        elif take_profit is not None and ret > take_profit:
            pnl = position * take_profit
            position = 0
        else:
            pnl = position * ret

        capital += pnl

        if signal != 0 and prev_position != position:
            num_trades += 1
            trade_pnls.append(pnl)

        df.at[df.index[i], 'position'] = position
        df.at[df.index[i], 'pnl'] = pnl
        df.at[df.index[i], 'capital'] = capital
        capital_history[i] = capital

        prev_position = position

        if capital < 0.001:
            stop_index = i
            if verbose:
                print(f"Capital dropped below 0.001 at index {i}. Stopping simulation.")
            break

    if stop_index is not None:
        df = df.iloc[:stop_index + 1]
        capital_history = capital_history[:stop_index + 1]
    else:
        df = df.dropna(subset=['capital'])

    pnl_series = df['pnl']
    actual_trades = df[df['position'] != 0]
    hit_rate = (actual_trades['pnl'] > 0).sum() / len(actual_trades) if len(actual_trades) > 0 else 0.0

    # Optional extra metrics
    if len(df) > 1:
        sharpe_ratio = np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)
        ic, _ = spearmanr(df['signal_raw'].shift(1), df['return'])
    else:
        sharpe_ratio = np.nan
        ic = np.nan

    total_return = df['capital'].iloc[-1] - initial_capital
    metrics = {
        'Total Return': total_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'IC (Spearman)': ic,
        'Number of Trades': len(actual_trades)
    }

    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Strategy Capital', linewidth=1)
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Trading Strategy Performance (Window: {trading_window}h)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            today = datetime.now().strftime("%d%m%Y")
            plot_filename = f"images/trading_simulation_{today}.png"
            os.makedirs('images', exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Plot saved as: {plot_filename}")
            plt.show()
            plt.close()
        elif verbose:
            print("No valid capital data to plot")

    if verbose:
        print(metrics)

    return metrics


def simulate_trading_ensemble(df, linear_model, binary_model, feature_cols, target_col='bam_close',
                             start_time=None, end_time=None, initial_capital=1,
                             trading_window=1, threshold=0.1, stop_loss=None, take_profit=None,
                             plot=True, strategy='reactive', verbose=True):
    df = df.copy()

    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")

    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()

    expanded_feature_cols = []
    for feature in feature_cols:
        if feature in df.columns:
            expanded_feature_cols.append(feature)
        elif feature in indicator_functions:
            result = indicator_functions[feature](df)
            if isinstance(result, (tuple, list)):
                for i, arr in enumerate(result):
                    colname = f'{feature}_{i}'
                    df[colname] = arr
                    expanded_feature_cols.append(colname)
            else:
                df[feature] = result
                expanded_feature_cols.append(feature)
        else:
            if verbose:
                print(f"Feature '{feature}' not found in df.columns or indicator_functions. Skipping.")

    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)

    try:
        df['linear_signal_raw'] = linear_model.predict(df[expanded_feature_cols])
        df['binary_signal'] = binary_model.predict(df[expanded_feature_cols])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # Use raw linear signal for all strategies
    df['linear_signal'] = df['linear_signal_raw'].shift(1)
    df['linear_signal_raw'] = df['linear_signal_raw'].shift(1)

    # Ensemble signal: only trade when both models agree (using raw signal)
    df['ensemble_signal'] = 0.0
    positive_mask = (df['linear_signal'] > 0) & (df['binary_signal'] == 1)
    df.loc[positive_mask, 'ensemble_signal'] = df.loc[positive_mask, 'linear_signal']
    negative_mask = (df['linear_signal'] < 0) & (df['binary_signal'] == 0)
    df.loc[negative_mask, 'ensemble_signal'] = df.loc[negative_mask, 'linear_signal']
    df['ensemble_signal'] = df['ensemble_signal'].shift(1)

    # (remove z-score normalization and clipping)

    df['capital'] = np.nan
    df['position'] = np.nan
    df['pnl'] = np.nan

    capital = initial_capital
    prev_position = 0
    num_trades = 0
    trade_pnls = []
    capital_history = np.full(len(df), np.nan)
    stop_index = None

    for i in range(1, len(df)):
        if i % trading_window != 0:
            continue

        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        signal = curr['ensemble_signal']
        raw_signal = curr['linear_signal_raw']

        if pd.isna(signal):
            continue

        if strategy == 'reactive':
            capital_ratio = capital / initial_capital
            if capital_ratio >= 1.5:
                utilization = 0.8
            elif capital_ratio >= 1.0:
                utilization = 0.6
            elif capital_ratio >= 0.7:
                utilization = 0.4
            else:
                utilization = 0.2
            position = capital * signal * utilization
        elif strategy == 'reverse_reactive':
            capital_ratio = capital / initial_capital
            if capital_ratio <= 0.5:
                utilization = 0.9
            elif capital_ratio <= 0.7:
                utilization = 0.8
            elif capital_ratio <= 1.0:
                utilization = 0.6
            else:
                utilization = 0.3
            position = capital * signal * utilization
        elif strategy == 'simple':
            if raw_signal > 0:
                position = capital * 0.5
            elif raw_signal < 0:
                position = capital * -0.5
            else:
                position = 0
        else:
            position = capital * signal

        ret = curr['return']

        if stop_loss is not None and ret < stop_loss:
            pnl = position * stop_loss
            position = 0
        elif take_profit is not None and ret > take_profit:
            pnl = position * take_profit
            position = 0
        else:
            pnl = position * ret

        capital += pnl

        if signal != 0 and prev_position != position:
            num_trades += 1
            trade_pnls.append(pnl)

        df.at[df.index[i], 'position'] = position
        df.at[df.index[i], 'pnl'] = pnl
        df.at[df.index[i], 'capital'] = capital
        capital_history[i] = capital

        prev_position = position

        if capital < 0.001:
            stop_index = i
            if verbose:
                print(f"Capital dropped below 0.001 at index {i}. Stopping simulation.")
            break

    if stop_index is not None:
        df = df.iloc[:stop_index + 1]
        capital_history = capital_history[:stop_index + 1]
    else:
        df = df.dropna(subset=['capital'])

    pnl_series = df['pnl']
    actual_trades = df[df['position'] != 0]
    hit_rate = (actual_trades['pnl'] > 0).sum() / len(actual_trades) if len(actual_trades) > 0 else 0.0
    if len(df) > 1:
        sharpe_ratio = np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)
        ic, _ = spearmanr(df['linear_signal_raw'].shift(1), df['return'])
    else:
        sharpe_ratio = np.nan
        ic = np.nan

    total_return = df['capital'].iloc[-1] - initial_capital
    metrics = {
        'Total Return': total_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'IC (Spearman)': ic,
        'Number of Trades': len(actual_trades)
    }

    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Ensemble Strategy Capital', linewidth=1, color='green')
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Ensemble Trading Strategy Performance (Window: {trading_window}h)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            today = datetime.now().strftime("%d%m%Y")
            plot_filename = f"images/ensemble_trading_simulation_{today}.png"
            os.makedirs('images', exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Ensemble plot saved as: {plot_filename}")
            plt.show()
            plt.close()
        elif verbose:
            print("No valid capital data to plot")

    if verbose:
        print(f"Ensemble Strategy Metrics: {metrics}")

    return metrics
