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
                     plot=True, strategy='reactive', verbose=True, scaler=None):
    df = df.copy()
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
    
    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")

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

    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()

    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)

    try:
        # Scale features if scaler is provided
        if scaler is not None:
            X = df[expanded_feature_cols].values
            X_scaled = scaler.transform(X)
            df['signal_raw'] = model.predict(X_scaled)
        else:
            df['signal_raw'] = model.predict(df[expanded_feature_cols])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # Min-max standardization to [-1, 1]
    min_signal = df['signal_raw'].min()
    max_signal = df['signal_raw'].max()
    if max_signal - min_signal > 1e-8:
        df['signal'] = 2 * (df['signal_raw'] - min_signal) / (max_signal - min_signal) - 1
    else:
        df['signal'] = 0
    df['signal'] = df['signal'].shift(1)
    df['signal_raw'] = df['signal_raw'].shift(1)

    

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
        elif strategy == 'quartile':
            # Quartile-based: scale position by signal quartiles
            q75 = df['signal_raw'].quantile(0.75)
            q25 = df['signal_raw'].quantile(0.25)
            if raw_signal >= q75:
                position = capital * 0.8
            elif raw_signal > 0:
                position = capital * 0.5
            elif raw_signal <= q25:
                position = capital * -0.8
            elif raw_signal < 0:
                position = capital * -0.5
            else:
                position = 0
        elif strategy == 'aggressive':
            if raw_signal > 0:
                position = capital
            elif raw_signal < 0:
                position = -capital
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
    # Hit ratio for long and short trades
    long_trades = actual_trades[actual_trades['position'] > 0]
    short_trades = actual_trades[actual_trades['position'] < 0]
    hit_rate_long = (long_trades['pnl'] > 0).sum() / len(long_trades) if len(long_trades) > 0 else float('nan')
    hit_rate_short = (short_trades['pnl'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else float('nan')
    num_long_trades = len(long_trades)
    num_short_trades = len(short_trades)

    # Optional extra metrics
    if len(df) > 1:
        # Calculate annualization factor for Sharpe ratio
        periods_per_year = 8760 / trading_window if trading_window else 8760
        sharpe_ratio = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan
        # Remove IC assignment

    total_return = df['capital'].iloc[-1] - initial_capital
    # Calculate buy and hold return
    if 'bam_close' in df.columns:
        buy_and_hold_return = (df['bam_close'].iloc[-1] - df['bam_close'].iloc[0]) / df['bam_close'].iloc[0]
    else:
        buy_and_hold_return = np.nan
    metrics = {
        'Total Return': total_return,
        'Buy and Hold Return': buy_and_hold_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Number of Trades': len(actual_trades)
    }

    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            # Prepend initial capital at step 0
            valid_capital = np.insert(valid_capital, 0, initial_capital)
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Strategy Capital', linewidth=1)
            # Buy and Hold capital trajectory
            if 'bam_close' in df.columns:
                buy_and_hold = initial_capital * (df['bam_close'] / df['bam_close'].iloc[0])
                buy_and_hold = np.insert(buy_and_hold.values, 0, initial_capital)
                plt.plot(buy_and_hold, color='red', label='Buy & Hold', linewidth=1, alpha=0.7)
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Trading Strategy Performance\nFrequency: {trading_window}h  Dates: {df["close_time"].iloc[0].strftime("%Y-%m-%d")} to {df["close_time"].iloc[-1].strftime("%Y-%m-%d")}')
            plt.legend()
            plt.grid(True)
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
        print(f"Hit Rate (Long/Price Rise): {hit_rate_long:.2%}")
        print(f"Hit Rate (Short/Price Fall): {hit_rate_short:.2%}")
        print(f"Number of Long Trades: {num_long_trades}")
        print(f"Number of Short Trades: {num_short_trades}")

    return metrics


def simulate_trading_ensemble(df, linear_model, binary_model, feature_cols, target_col='bam_close',
                             start_time=None, end_time=None, initial_capital=1,
                             trading_window=1, threshold=0.1, stop_loss=None, take_profit=None,
                             plot=True, strategy='reactive', verbose=True, scaler=None):
    df = df.copy()
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])

    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")

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

    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()

    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)

    try:
        # Scale features if scaler is provided
        if scaler is not None:
            X = df[expanded_feature_cols].values
            X_scaled = scaler.transform(X)
            df['linear_signal_raw'] = linear_model.predict(X_scaled)
            df['binary_signal'] = binary_model.predict(X_scaled)
            # Precompute probabilities for proba_ensemble strategy
            if strategy == 'proba_ensemble' and hasattr(binary_model, 'predict_proba'):
                proba = binary_model.predict_proba(X_scaled)[:, 1]
                df['binary_proba'] = pd.Series(proba, index=df.index).shift(1)
            else:
                df['binary_proba'] = np.nan
        else:
            df['linear_signal_raw'] = linear_model.predict(df[expanded_feature_cols])
            df['binary_signal'] = binary_model.predict(df[expanded_feature_cols])
            # Precompute probabilities for proba_ensemble strategy
            if strategy == 'proba_ensemble' and hasattr(binary_model, 'predict_proba'):
                proba = binary_model.predict_proba(df[expanded_feature_cols])[:, 1]
                df['binary_proba'] = pd.Series(proba, index=df.index).shift(1)
            else:
                df['binary_proba'] = np.nan
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

    # Min-max standardization to [-1, 1]
    min_signal = df['ensemble_signal'].min()
    max_signal = df['ensemble_signal'].max()
    if max_signal - min_signal > 1e-8:
        df['ensemble_signal'] = 2 * (df['ensemble_signal'] - min_signal) / (max_signal - min_signal) - 1
    else:
        df['ensemble_signal'] = 0
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
        elif strategy == 'quartile':
            # Quartile-based: scale position by signal quartiles
            q75 = df['linear_signal_raw'].quantile(0.75)
            q25 = df['linear_signal_raw'].quantile(0.25)
            if raw_signal >= q75:
                position = capital * 0.8
            elif raw_signal > 0:
                position = capital * 0.5
            elif raw_signal <= q25:
                position = capital * -0.8
            elif raw_signal < 0:
                position = capital * -0.5
            else:
                position = 0
        elif strategy == 'aggressive':
            if raw_signal > 0:
                position = capital
            elif raw_signal < 0:
                position = -capital
            else:
                position = 0
        elif strategy == 'proba_ensemble':
            curr_proba = curr['binary_proba']
            agree_long = (raw_signal > 0) and (curr['binary_signal'] == 1)
            agree_short = (raw_signal < 0) and (curr['binary_signal'] == 0)
            if agree_long:
                position = capital * ((curr_proba-0.5)*2 if not pd.isna(curr_proba) else 0.5)
            elif agree_short:
                position = -capital * ((0.5 - curr_proba)*2 if not pd.isna(curr_proba) else 0.5)
            else:
                position = 0
        elif strategy == 'aggressive_ensemble':
            curr_proba = curr['binary_proba']
            agree_long = (raw_signal > 0) and (curr['binary_signal'] == 1)
            agree_short = (raw_signal < 0) and (curr['binary_signal'] == 0)
            
            if agree_long:
                if not pd.isna(curr_proba):
                    if curr_proba > 0.8:
                        position = capital  # All in
                    elif curr_proba >= 0.6:
                        position = capital * 0.5  # Half position
                    else:
                        position = 0  # Don't trade
                else:
                    position = capital * 0.5  # Default half position
            elif agree_short:
                if not pd.isna(curr_proba):
                    if curr_proba < 0.2:  # Probability of fall > 0.8
                        position = -capital  # All in short
                    elif curr_proba <= 0.4:  # Probability of fall >= 0.6
                        position = -capital * 0.5  # Half position short
                    else:
                        position = 0  # Don't trade
                else:
                    position = -capital * 0.5  # Default half position
            else:
                position = 0  # No agreement between models
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
    # Hit ratio for long and short trades
    long_trades = actual_trades[actual_trades['position'] > 0]
    short_trades = actual_trades[actual_trades['position'] < 0]
    hit_rate_long = (long_trades['pnl'] > 0).sum() / len(long_trades) if len(long_trades) > 0 else float('nan')
    hit_rate_short = (short_trades['pnl'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else float('nan')
    num_long_trades = len(long_trades)
    num_short_trades = len(short_trades)

    if len(df) > 1:
        # Calculate annualization factor for Sharpe ratio
        periods_per_year = 8760 / trading_window if trading_window else 8760
        sharpe_ratio = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan
        # Remove IC assignment

    total_return = df['capital'].iloc[-1] - initial_capital
    # Calculate buy and hold return
    if 'bam_close' in df.columns:
        buy_and_hold_return = (df['bam_close'].iloc[-1] - df['bam_close'].iloc[0]) / df['bam_close'].iloc[0]
    else:
        buy_and_hold_return = np.nan
    metrics = {
        'Total Return': total_return,
        'Buy and Hold Return': buy_and_hold_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Number of Trades': len(actual_trades)
    }

    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
            date_range_str = f"{pd.to_datetime(df['close_time'].iloc[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(df['close_time'].iloc[-1]).strftime('%Y-%m-%d')}"
        else:
            date_range_str = ""
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            # Prepend initial capital at step 0
            valid_capital = np.insert(valid_capital, 0, initial_capital)
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Ensemble Strategy Capital', linewidth=1, color='green')
            # Buy and Hold capital trajectory
            if 'bam_close' in df.columns:
                buy_and_hold = initial_capital * (df['bam_close'] / df['bam_close'].iloc[0])
                buy_and_hold = np.insert(buy_and_hold.values, 0, initial_capital)
                plt.plot(buy_and_hold, color='red', label='Buy & Hold', linewidth=1, alpha=0.7)
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Ensemble Trading Strategy Performance\nFrequency: {trading_window}h  Dates: {date_range_str}')
            plt.legend()
            plt.grid(True)
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
        print(f"Hit Rate (Long/Price Rise): {hit_rate_long:.2%}")
        print(f"Hit Rate (Short/Price Fall): {hit_rate_short:.2%}")
        print(f"Number of Long Trades: {num_long_trades}")
        print(f"Number of Short Trades: {num_short_trades}")

    return metrics


def simulate_trading_quantile(df, quantile_model, feature_cols, target_col='bam_close',
                     start_time=None, end_time=None, initial_capital=1,
                     trading_window=1, threshold=0.1, stop_loss=None, take_profit=None,
                     plot=True, strategy='reactive', verbose=True):
    df = df.copy()
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
    # Always calculate 'return' inside, just like other simulate_trading functions
    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")

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

    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()

    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)

    try:
        df['signal_raw'] = quantile_model.predict(df[expanded_feature_cols])
    except Exception as e:
        raise RuntimeError(f"Quantile model prediction failed: {e}")

    # Min-max standardization to [-1, 1]
    min_signal = df['signal_raw'].min()
    max_signal = df['signal_raw'].max()
    if max_signal - min_signal > 1e-8:
        df['signal'] = 2 * (df['signal_raw'] - min_signal) / (max_signal - min_signal) - 1
    else:
        df['signal'] = 0
    df['signal'] = df['signal'].shift(1)
    df['signal_raw'] = df['signal_raw'].shift(1)

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
        raw_signal = curr['signal_raw']

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
        elif strategy == 'quartile':
            q75 = df['signal_raw'].quantile(0.75)
            q25 = df['signal_raw'].quantile(0.25)
            if raw_signal >= q75:
                position = capital * 0.8
            elif raw_signal > 0:
                position = capital * 0.5
            elif raw_signal <= q25:
                position = capital * -0.8
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
    # Hit ratio for long and short trades
    long_trades = actual_trades[actual_trades['position'] > 0]
    short_trades = actual_trades[actual_trades['position'] < 0]
    hit_rate_long = (long_trades['pnl'] > 0).sum() / len(long_trades) if len(long_trades) > 0 else float('nan')
    hit_rate_short = (short_trades['pnl'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else float('nan')
    num_long_trades = len(long_trades)
    num_short_trades = len(short_trades)

    if len(df) > 1:
        # Calculate annualization factor for Sharpe ratio
        periods_per_year = 8760 / trading_window if trading_window else 8760
        sharpe_ratio = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan

    total_return = df['capital'].iloc[-1] - initial_capital
    if 'bam_close' in df.columns:
        buy_and_hold_return = (df['bam_close'].iloc[-1] - df['bam_close'].iloc[0]) / df['bam_close'].iloc[0]
    else:
        buy_and_hold_return = np.nan
    metrics = {
        'Total Return': total_return,
        'Buy and Hold Return': buy_and_hold_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Number of Trades': len(actual_trades)
    }

    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
            date_range_str = f"{pd.to_datetime(df['close_time'].iloc[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(df['close_time'].iloc[-1]).strftime('%Y-%m-%d')}"
        else:
            date_range_str = ""
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            valid_capital = np.insert(valid_capital, 0, initial_capital)
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Quantile Strategy Capital', linewidth=1, color='purple')
            # Buy and Hold capital trajectory
            if 'bam_close' in df.columns:
                buy_and_hold = initial_capital * (df['bam_close'] / df['bam_close'].iloc[0])
                buy_and_hold = np.insert(buy_and_hold.values, 0, initial_capital)
                plt.plot(buy_and_hold, color='red', label='Buy & Hold', linewidth=1, alpha=0.7)
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Quantile Trading Strategy Performance\nFrequency: {trading_window}h  Dates: {date_range_str}')
            plt.legend()
            plt.grid(True)
            today = datetime.now().strftime("%d%m%Y")
            plot_filename = f"images/quantile_trading_simulation_{today}.png"
            os.makedirs('images', exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Quantile plot saved as: {plot_filename}")
            plt.show()
            plt.close()
        elif verbose:
            print("No valid capital data to plot")

    if verbose:
        print(metrics)
        print(f"Hit Rate (Long/Price Rise): {hit_rate_long:.2%}")
        print(f"Hit Rate (Short/Price Fall): {hit_rate_short:.2%}")
        print(f"Number of Long Trades: {num_long_trades}")
        print(f"Number of Short Trades: {num_short_trades}")

    return metrics


def simulate_trading_binary(df, binary_model, feature_cols, target_col='bam_close', strategy='prob',
                     start_time=None, end_time=None, initial_capital=1,
                     trading_window=1, threshold=0.1, stop_loss=None, take_profit=None,
                     plot=True, verbose=True, scaler=None):
    df = df.copy()
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")

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

    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()

    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)

    try:
        # Scale features if scaler is provided
        if scaler is not None:
            X = df[expanded_feature_cols].values
            X_scaled = scaler.transform(X)
            df['binary_signal'] = binary_model.predict(X_scaled)
            # Use probabilities for position sizing if available
            if hasattr(binary_model, 'predict_proba'):
                proba = binary_model.predict_proba(X_scaled)[:, 1]
                df['binary_proba'] = pd.Series(proba, index=df.index).shift(1)
            else:
                df['binary_proba'] = np.nan
        else:
            df['binary_signal'] = binary_model.predict(df[expanded_feature_cols])
            # Use probabilities for position sizing if available
            if hasattr(binary_model, 'predict_proba'):
                proba = binary_model.predict_proba(df[expanded_feature_cols])[:, 1]
                df['binary_proba'] = pd.Series(proba, index=df.index).shift(1)
            else:
                df['binary_proba'] = np.nan
    except Exception as e:
        raise RuntimeError(f"Binary model prediction failed: {e}")

    df['binary_signal'] = df['binary_signal'].shift(1)

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
        signal = curr['binary_signal']
        proba = curr['binary_proba'] if 'binary_proba' in curr else np.nan

        if pd.isna(signal):
            continue

        if strategy == 'prob':
            # If probability is available, use it for position sizing
            if not pd.isna(proba):
                position = capital * 0.5 * (proba - 0.5) * 2  # scale [-0.5, 0.5]
            else:
                # Only trade based on binary prediction: 1 = long, 0 = short, else 0
                if signal == 1:
                    position = capital * 0.5
                elif signal == 0:
                    position = capital * -0.5
                else:
                    position = 0
        elif strategy == 'aggressive':
            # Use the whole capital for long/short based on binary prediction
            if signal == 1:
                position = capital
            elif signal == 0:
                position = -capital
            else:
                position = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

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
    # Hit ratio for long and short trades
    long_trades = actual_trades[actual_trades['position'] > 0]
    short_trades = actual_trades[actual_trades['position'] < 0]
    hit_rate_long = (long_trades['pnl'] > 0).sum() / len(long_trades) if len(long_trades) > 0 else float('nan')
    hit_rate_short = (short_trades['pnl'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else float('nan')
    num_long_trades = len(long_trades)
    num_short_trades = len(short_trades)

    if len(df) > 1:
        # Calculate annualization factor for Sharpe ratio
        periods_per_year = 8760 / trading_window if trading_window else 8760
        sharpe_ratio = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan

    total_return = df['capital'].iloc[-1] - initial_capital
    if 'bam_close' in df.columns:
        buy_and_hold_return = (df['bam_close'].iloc[-1] - df['bam_close'].iloc[0]) / df['bam_close'].iloc[0]
    else:
        buy_and_hold_return = np.nan
    metrics = {
        'Total Return': total_return,
        'Buy and Hold Return': buy_and_hold_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Number of Trades': len(actual_trades)
    }

    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
            date_range_str = f"{pd.to_datetime(df['close_time'].iloc[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(df['close_time'].iloc[-1]).strftime('%Y-%m-%d')}"
        else:
            date_range_str = ""
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            valid_capital = np.insert(valid_capital, 0, initial_capital)
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Binary Strategy Capital', linewidth=1, color='orange')
            # Buy and Hold capital trajectory
            if 'bam_close' in df.columns:
                buy_and_hold = initial_capital * (df['bam_close'] / df['bam_close'].iloc[0])
                buy_and_hold = np.insert(buy_and_hold.values, 0, initial_capital)
                plt.plot(buy_and_hold, color='red', label='Buy & Hold', linewidth=1, alpha=0.7)
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Binary Trading Strategy Performance\nFrequency: {trading_window}h  Dates: {date_range_str}')
            plt.legend()
            plt.grid(True)
            today = datetime.now().strftime("%d%m%Y")
            plot_filename = f"images/binary_trading_simulation_{today}.png"
            os.makedirs('images', exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Binary plot saved as: {plot_filename}")
            plt.show()
            plt.close()
        elif verbose:
            print("No valid capital data to plot")

    if verbose:
        print(metrics)
        print(f"Hit Rate (Long/Price Rise): {hit_rate_long:.2%}")
        print(f"Hit Rate (Short/Price Fall): {hit_rate_short:.2%}")
        print(f"Number of Long Trades: {num_long_trades}")
        print(f"Number of Short Trades: {num_short_trades}")

    return metrics


def simulate_trading_poly(df, poly_model, feature_cols, target_col='bam_close',
                     start_time=None, end_time=None, initial_capital=1,
                     trading_window=1, threshold=0.1, stop_loss=None, take_profit=None,
                     plot=True, strategy='reactive', verbose=True):
    df = df.copy()
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")
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
    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()
    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)
    # Polynomial transform
    X_poly = poly_model.poly.transform(df[expanded_feature_cols].values)
    X_poly = poly_model.scaler.transform(X_poly)
    try:
        df['signal_raw'] = poly_model.predict(X_poly)
    except Exception as e:
        raise RuntimeError(f"Polynomial model prediction failed: {e}")
    # Min-max standardization to [-1, 1]
    min_signal = df['signal_raw'].min()
    max_signal = df['signal_raw'].max()
    if max_signal - min_signal > 1e-8:
        df['signal'] = 2 * (df['signal_raw'] - min_signal) / (max_signal - min_signal) - 1
    else:
        df['signal'] = 0
    df['signal'] = df['signal'].shift(1)
    df['signal_raw'] = df['signal_raw'].shift(1)
    # --- rest is identical to simulate_trading ---
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
        raw_signal = curr['signal_raw']
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
        elif strategy == 'quartile':
            # Quartile-based: scale position by signal quartiles
            q75 = df['signal_raw'].quantile(0.75)
            q25 = df['signal_raw'].quantile(0.25)
            if raw_signal >= q75:
                position = capital * 0.8
            elif raw_signal > 0:
                position = capital * 0.5
            elif raw_signal <= q25:
                position = capital * -0.8
            elif raw_signal < 0:
                position = capital * -0.5
            else:
                position = 0
        elif strategy == 'aggressive':
            if raw_signal > 0:
                position = capital
            elif raw_signal < 0:
                position = -capital
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
    long_trades = actual_trades[actual_trades['position'] > 0]
    short_trades = actual_trades[actual_trades['position'] < 0]
    hit_rate_long = (long_trades['pnl'] > 0).sum() / len(long_trades) if len(long_trades) > 0 else float('nan')
    hit_rate_short = (short_trades['pnl'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else float('nan')
    num_long_trades = len(long_trades)
    num_short_trades = len(short_trades)
    if len(df) > 1:
        # Calculate annualization factor for Sharpe ratio
        periods_per_year = 8760 / trading_window if trading_window else 8760
        sharpe_ratio = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan
    total_return = df['capital'].iloc[-1] - initial_capital
    if 'bam_close' in df.columns:
        buy_and_hold_return = (df['bam_close'].iloc[-1] - df['bam_close'].iloc[0]) / df['bam_close'].iloc[0]
    else:
        buy_and_hold_return = np.nan
    metrics = {
        'Total Return': total_return,
        'Buy and Hold Return': buy_and_hold_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Number of Trades': len(actual_trades)
    }
    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            valid_capital = np.insert(valid_capital, 0, initial_capital)
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Poly Strategy Capital', linewidth=1, color='blue')
            if 'bam_close' in df.columns:
                buy_and_hold = initial_capital * (df['bam_close'] / df['bam_close'].iloc[0])
                buy_and_hold = np.insert(buy_and_hold.values, 0, initial_capital)
                plt.plot(buy_and_hold, color='red', label='Buy & Hold', linewidth=1, alpha=0.7)
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Polynomial Trading Strategy Performance\nFrequency: {trading_window}h  Dates: {df["close_time"].iloc[0].strftime("%Y-%m-%d")} to {df["close_time"].iloc[-1].strftime("%Y-%m-%d")})')
            plt.legend()
            plt.grid(True)
            today = datetime.now().strftime("%d%m%Y")
            plot_filename = f"images/poly_trading_simulation_{today}.png"
            os.makedirs('images', exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Poly plot saved as: {plot_filename}")
            plt.show()
            plt.close()
        elif verbose:
            print("No valid capital data to plot")
    if verbose:
        print(metrics)
        print(f"Hit Rate (Long/Price Rise): {hit_rate_long:.2%}")
        print(f"Hit Rate (Short/Price Fall): {hit_rate_short:.2%}")
        print(f"Number of Long Trades: {num_long_trades}")
        print(f"Number of Short Trades: {num_short_trades}")
    return metrics


def simulate_trading_majority(df, majority_vote_model, indicator_names, feature_cols, target_col='bam_close',
                     start_time=None, end_time=None, initial_capital=1,
                     trading_window=1, threshold=0.1, stop_loss=None, take_profit=None,
                     plot=True, strategy='reactive', verbose=True):
    df = df.copy()
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
    if target_col in df.columns:
        df['return'] = (df[target_col].shift(-trading_window) - df[target_col]) / df[target_col]
    else:
        raise ValueError(f"{target_col} column is required to compute 'return'.")
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
    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()
    if not expanded_feature_cols:
        raise ValueError("No valid features found in feature_cols.")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=expanded_feature_cols + ['return'], inplace=True)
    # Get majority vote signal
    try:
        df['majority_signal'] = majority_vote_model(df)
    except Exception as e:
        raise RuntimeError(f"Majority vote model prediction failed: {e}")
    df['majority_signal'] = pd.Series(df['majority_signal'], index=df.index).shift(1)
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
        signal = curr['majority_signal']
        if pd.isna(signal):
            continue
        if signal == 1:
            position = capital
        elif signal == 0:
            position = -capital
        else:
            position = 0
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
    long_trades = actual_trades[actual_trades['position'] > 0]
    short_trades = actual_trades[actual_trades['position'] < 0]
    hit_rate_long = (long_trades['pnl'] > 0).sum() / len(long_trades) if len(long_trades) > 0 else float('nan')
    hit_rate_short = (short_trades['pnl'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else float('nan')
    num_long_trades = len(long_trades)
    num_short_trades = len(short_trades)
    if len(df) > 1:
        # Calculate annualization factor for Sharpe ratio
        periods_per_year = 8760 / trading_window if trading_window else 8760
        sharpe_ratio = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-7)) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan
    total_return = df['capital'].iloc[-1] - initial_capital
    if 'bam_close' in df.columns:
        buy_and_hold_return = (df['bam_close'].iloc[-1] - df['bam_close'].iloc[0]) / df['bam_close'].iloc[0]
    else:
        buy_and_hold_return = np.nan
    metrics = {
        'Total Return': total_return,
        'Buy and Hold Return': buy_and_hold_return,
        'Hit Rate': hit_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Number of Trades': len(actual_trades)
    }
    if plot:
        if 'close_time' in df.columns and verbose:
            print(f"Date range: {df['close_time'].iloc[0]} to {df['close_time'].iloc[-1]}")
        valid_capital = capital_history[~np.isnan(capital_history)]
        if len(valid_capital) > 0:
            valid_capital = np.insert(valid_capital, 0, initial_capital)
            plt.figure(figsize=(10, 4))
            plt.plot(valid_capital, label='Majority Vote Capital', linewidth=1, color='magenta')
            if 'bam_close' in df.columns:
                buy_and_hold = initial_capital * (df['bam_close'] / df['bam_close'].iloc[0])
                buy_and_hold = np.insert(buy_and_hold.values, 0, initial_capital)
                plt.plot(buy_and_hold, color='red', label='Buy & Hold', linewidth=1, alpha=0.7)
            plt.ylabel('Capital')
            plt.xlabel('Trading Steps')
            plt.title(f'Majority Vote Trading Performance\nFrequency: {trading_window}h  Dates: {df["close_time"].iloc[0].strftime("%Y-%m-%d")} to {df["close_time"].iloc[-1].strftime("%Y-%m-%d")})')
            plt.legend()
            plt.grid(True)
            today = datetime.now().strftime("%d%m%Y")
            plot_filename = f"images/majority_trading_simulation_{today}.png"
            os.makedirs('images', exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Majority plot saved as: {plot_filename}")
            plt.show()
            plt.close()
        elif verbose:
            print("No valid capital data to plot")
    if verbose:
        print(metrics)
        print(f"Hit Rate (Long/Price Rise): {hit_rate_long:.2%}")
        print(f"Hit Rate (Short/Price Fall): {hit_rate_short:.2%}")
        print(f"Number of Long Trades: {num_long_trades}")
        print(f"Number of Short Trades: {num_short_trades}")
    return metrics
