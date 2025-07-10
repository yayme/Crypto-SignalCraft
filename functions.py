import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

spot_parquet_dir = os.path.join('binance', 'spot-1h-0509')
futures_parquet_dir = os.path.join('binance', 'futures-1h-0603')

def collect_only(symbol):
    files = [f for f in os.listdir(spot_parquet_dir) if f.endswith('parquet')]
    df_list=[]
    for filename in tqdm(files, desc='Processing parquet files'):
        df = pd.read_parquet(os.path.join(spot_parquet_dir, filename))
        df = df.reset_index()
        if 'base' in df.columns:
            df=df[df['base']==symbol]
            df_list.append(df)
        else:
            print(f"'base' doesn't exist in {filename}")
    df_combined = pd.concat(df_list)
    df_combined.to_csv(f'{symbol}_spot_full.csv')

def plot_candlestick_range(df, start_time=None, end_time=None):
    if not pd.api.types.is_datetime64_any_dtype(df['close_time']):
        df['close_time'] = pd.to_datetime(df['close_time'])
    mask = pd.Series([True] * len(df))
    if start_time is not None:
        mask &= (df['close_time'] >= pd.to_datetime(start_time))
    if end_time is not None:
        mask &= (df['close_time'] <= pd.to_datetime(end_time))
    filtered = df[mask].sort_values('close_time')
    if filtered.empty:
        print(f"No data found for {symbol if 'symbol' in locals() else ''} between {start_time} and {end_time}")
        return
    fig = go.Figure(data=[go.Candlestick(
        x=filtered['close_time'],
        open=filtered['bam_open'],
        high=filtered['bam_high'],
        low=filtered['bam_low'],
        close=filtered['bam_close'],
        name=symbol if 'symbol' in locals() else ''
    )])
    fig.update_layout(
        title=f'Candlestick from {start_time} to {end_time}',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    fig.show()

def test_indicator(indicator_fn, df, start_time, end_time, window_hours=3, symbol='BTC', want_return=False):
    df = df.copy()
    df['close_time'] = pd.to_datetime(df['close_time'])
    df = df[df['base'] == symbol].sort_values('close_time').reset_index(drop=True)
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    df = df[(df['close_time'] >= start_time) & (df['close_time'] <= end_time)]
    entry_idx = np.arange(0, len(df) - window_hours, window_hours)
    signal = indicator_fn(df.iloc[entry_idx])
    price_now = df['bam_close'].values[entry_idx]
    price_future = df['bam_close'].values[entry_idx + window_hours]
    valid_mask = ~pd.isna(signal)
    signal = signal[valid_mask]
    price_now = price_now[valid_mask]
    price_future = price_future[valid_mask]
    returns = price_future - price_now
    wins_long = np.sum((signal == 1) & (returns > 0))
    wins_short = np.sum((signal == -1) & (returns < 0))
    total_long = np.sum(signal == 1)
    total_short = np.sum(signal == -1)
    total = total_long + total_short
    wins = wins_long + wins_short
    if total == 0:
        print("No trades made in the given period.")
    else:
        win_pct = 100 * wins / total
        print(f"Win % for {symbol} from {start_time} to {end_time}: {win_pct:.2f}% ({wins}/{total})")
        print(f"   Long wins: {wins_long}/{total_long}, Short wins: {wins_short}/{total_short}")
    if want_return== True:
       return {
        'win_pct': win_pct if total > 0 else None,
        'total_trades': total,
        'long_wins': wins_long,
        'short_wins': wins_short,
        'long_total': total_long,
        'short_total': total_short,
    }

def calculate_ic(df, signal_col='position', price_col='bam_close', use_log_returns=False):
    df = df.copy()
    if use_log_returns:
        df['future_return'] = np.log(df[price_col].shift(-1) / df[price_col])
    else:
        df['future_return'] = df[price_col].pct_change().shift(-1)
    df = df.dropna(subset=['future_return', signal_col])
    if len(df) < 2:
        return 0.0
    try:
        # Check if arrays have variation (not constant)
        if df[signal_col].std() > 0 and df['future_return'].std() > 0:
            ic, _ = spearmanr(df[signal_col], df['future_return'])
            return ic if not np.isnan(ic) else 0.0
        else:
            return 0.0
    except:
        return 0.0

def calculate_r2(df, signal_col='position', price_col='bam_close'):
    df = df.copy()
    df['future_return'] = df[price_col].pct_change().shift(-1)
    df = df.dropna(subset=['future_return', signal_col])
    if len(df) < 2:
        return 0.0
    return r2_score(df['future_return'], df[signal_col])

def calulate_win(df, signal_col='position', price_col='bam_close'):
    df = df.copy()
    df['true_return'] = df[price_col].pct_change().shift(-1)
    df = df.dropna(subset=['true_return', signal_col])
    df['product'] = df['true_return'] * df[signal_col]
    if len(df) == 0:
        return 0.0
    return len(df[df['product'] > 0]) / len(df)

def calculate_sharpe(df, price_col='bam_close', position_col='position', risk_free_rate=0.0, use_log_returns=False):
    df = df.copy()
    if use_log_returns:
        df['returns'] = np.log(df[price_col] / df[price_col].shift(1))
    else:
        df['returns'] = df[price_col].pct_change()
    df['strategy_return'] = df['returns'] * df[position_col].shift(1).fillna(0)
    rf_hourly = risk_free_rate / 8760
    df['excess_return'] = df['strategy_return'] - rf_hourly
    mean = df['excess_return'].mean()
    std = df['excess_return'].std()
    if std < 1e-8:
        return 0.0
    sharpe = mean / std * np.sqrt(8760)
    return sharpe

def rank_features_by_ic(df, feature_cols, target_col='return 1h', min_obs=100):
    results = []
    for feature in feature_cols:
        valid = df[[feature, target_col]].dropna()
        if len(valid) < min_obs:
            continue
        ic = spearmanr(valid[feature], valid[target_col])[0]
        results.append((feature, ic))
    results_df = pd.DataFrame(results, columns=['feature', 'IC'])
    results_df['abs_IC'] = results_df['IC'].abs()
    results_df = results_df.sort_values(by='abs_IC', ascending=False).reset_index(drop=True)
    return results_df[['feature', 'IC']]
