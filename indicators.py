import pandas as pd
import numpy as np
import ta

def vwa_mean_reversion(df, threshold=0.002):
    rel_diff = (df['bam_close'] - df['bam_vwa']) / df['bam_vwa']
    signal = np.where(rel_diff < -threshold, 1,
             np.where(rel_diff > threshold, -1, 0))
    return signal

def volatility_breakout(df, high_thresh=0.01, low_thresh=0.002):
    vol = df['bam_volatility_1m_return'].values
    signal = np.where(vol > high_thresh, 1,
             np.where(vol < low_thresh, -1, 0))
    return signal

def buy_volume_sentiment(df, high=0.6, low=0.4):
    pct = df['pct_buy_volume'].values
    signal = np.where(pct > high, 1,
             np.where(pct < low, -1, 0))
    return signal

def interest_rate_carry(df, high_thresh=0.10, low_thresh=0.02):
    r = df['annual_interest_rate'].values
    signal = np.where(r < low_thresh, 1,
             np.where(r > high_thresh, -1, 0))
    return signal

def spread_vwap_signal(df, threshold=0.001):
    s = df['spread_1m_vwa_vwa'].values
    signal = np.where(s < -threshold, 1,
             np.where(s > threshold, -1, 0))
    return signal

def wick_reversal_signal(df, wick_ratio=2.0):
    high = df['bam_high'].values
    low = df['bam_low'].values
    open_ = df['bam_open'].values
    close = df['bam_close'].values

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    body = np.abs(close - open_)

    signal = np.where((lower_wick > wick_ratio * body), 1,
             np.where((upper_wick > wick_ratio * body), -1, 0))
    return signal

def rsi(df, window=14):
    delta = df['bam_close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def sma(df, window=14):
    return df['bam_close'].rolling(window=window).mean()

def ema(df, window=14):
    return df['bam_close'].ewm(span=window, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['bam_close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['bam_close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(df, window=20, num_std=2):
    sma = df['bam_close'].rolling(window=window).mean()
    std = df['bam_close'].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

def atr(df, window=14):
    high = df['bam_high']
    low = df['bam_low']
    close = df['bam_close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def cci(df, window=20):
    tp = (df['bam_high'] + df['bam_low'] + df['bam_close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-9)
    return cci

def stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df['bam_low'].rolling(window=k_window).min()
    high_max = df['bam_high'].rolling(window=k_window).max()
    k = 100 * (df['bam_close'] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(window=d_window).mean()
    return k, d

def williams_r(df, window=14):
    high_max = df['bam_high'].rolling(window=window).max()
    low_min = df['bam_low'].rolling(window=window).min()
    wr = -100 * (high_max - df['bam_close']) / (high_max - low_min + 1e-9)
    return wr

def add_momentum_features(df, windows=[5, 10, 20]):
    features = []
    for w in windows:
        name = f'mom_{w}'
        df[name] = df['bam_close'].pct_change(w)
        features.append(name)
    return df, features

def add_momentum_vwa_features(df, windows=[5, 10]):
    features = []
    for w in windows:
        name = f'mom_vwa_{w}'
        df[name] = df['bam_vwa'].pct_change(w)
        features.append(name)
    return df, features

def add_volatility_features(df):
    features = []
    df['volatility_5'] = df['bam_close'].pct_change().rolling(5).std()
    features.append('volatility_5')
    
    df['range_pct'] = (df['bam_high'] - df['bam_low']) / df['bam_close']
    features.append('range_pct')
    
    df['vol_1m_return_log'] = np.log1p(df['bam_volatility_1m_return'].abs())
    features.append('vol_1m_return_log')
    return df, features

def add_spread_features(df):
    features = []
    df['spread_diff'] = df['spread_1m_vwa_vwa'] - df['spread_twa']
    df['spread_ratio'] = df['spread_1m_vwa_vwa'] / (df['spread_twa'] + 1e-9)
    features += ['spread_diff', 'spread_ratio']
    return df, features

def add_volume_features(df):
    features = []
    df['log_base_volume'] = np.log1p(df['base_volume'])
    df['vol_change_5'] = df['base_volume'].pct_change(5)
    features += ['log_base_volume', 'vol_change_5']
    return df, features

def add_price_diff_features(df):
    features = []
    df['oc_return'] = (df['bam_close'] - df['bam_open']) / df['bam_open']
    df['close_vwa_spread'] = (df['bam_close'] - df['bam_vwa']) / df['bam_vwa']
    features += ['oc_return', 'close_vwa_spread']
    return df, features

def add_interest_rate_features(df):
    features = []
    df['interest_rate_change'] = df['annual_interest_rate'].diff().fillna(0)
    features.append('interest_rate_change')
    return df, features


def add_temporal_features(df, windows=[3, 5, 10], lags=[1, 2, 3]):
    df = df.copy()
    features = []

    cols_to_lag = ['bam_close', 'bam_vwa', 'base_volume', 'bam_volatility_1m_return']

    for col in cols_to_lag:
        # Lag features
        for lag in lags:
            name = f'{col}_lag{lag}'
            df[name] = df[col].shift(lag)
            features.append(name)

        # Rolling stats
        for win in windows:
            mean_col = f'{col}_mean{win}'
            std_col = f'{col}_std{win}'
            z_col = f'{col}_zscore{win}'

            df[mean_col] = df[col].rolling(win).mean()
            df[std_col] = df[col].rolling(win).std()
            df[z_col] = (df[col] - df[mean_col]) / (df[std_col] + 1e-9)

            features += [mean_col, std_col, z_col]

        # EMA
        for span in windows:
            ema_col = f'{col}_ema{span}'
            df[ema_col] = df[col].ewm(span=span).mean()
            features.append(ema_col)

    # 1-period return
    df['return_1'] = df['bam_close'].pct_change()

    # Autocorrelations of return
    for lag in lags:
        ac_col = f'return_acf_lag{lag}'
        df[ac_col] = df['return_1'].rolling(20).apply(lambda x: x.autocorr(lag=lag), raw=False)
        features.append(ac_col)

    # Cumulative returns over different horizons
    for win in windows:
        cumret_col = f'cumret_{win}'
        df[cumret_col] = df['return_1'].rolling(win).sum()
        features.append(cumret_col)

    df.drop(columns=['return_1'], inplace=True)

    # Time-of-day features (if timestamp is available)
    if 'close_time' in df.columns and np.issubdtype(df['close_time'].dtype, np.datetime64):
        df['hour'] = df['close_time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        features += ['hour_sin', 'hour_cos']

    return df, features


def create_trading_features(df):
    df = df.copy()
    df['mom_5'] = df['bam_close'].pct_change(5)
    df['mom_10'] = df['bam_close'].pct_change(10)
    df['mom_vwa_5'] = df['bam_vwa'].pct_change(5)
    df['volatility_5'] = df['bam_close'].pct_change().rolling(5).std()
    df['range_pct'] = (df['bam_high'] - df['bam_low']) / df['bam_close']
    df['vol_1m_return_log'] = np.log1p(df['bam_volatility_1m_return'].abs())
    df['spread_diff'] = df['spread_1m_vwa_vwa'] - df['spread_twa']
    df['spread_ratio'] = df['spread_1m_vwa_vwa'] / (df['spread_twa'] + 1e-9)
    df['log_base_volume'] = np.log1p(df['base_volume'])
    df['vol_change_5'] = df['base_volume'].pct_change(5)
    df['oc_return'] = (df['bam_close'] - df['bam_open']) / df['bam_open']
    df['close_vwa_spread'] = (df['bam_close'] - df['bam_vwa']) / df['bam_vwa']
    df['interest_rate_change'] = df['annual_interest_rate'].diff().fillna(0)
    
    feature_cols = [
        'mom_5', 'mom_10', 'mom_vwa_5', 'volatility_5', 'range_pct',
        'vol_1m_return_log', 'spread_diff', 'spread_ratio', 'log_base_volume',
        'vol_change_5', 'oc_return', 'close_vwa_spread', 'interest_rate_change'
    ]
    return df, feature_cols

def obv(df):
    return ta.volume.OnBalanceVolumeIndicator(df['bam_close'], df['base_volume']).on_balance_volume()

def adx(df, window=14):
    return ta.trend.ADXIndicator(df['bam_high'], df['bam_low'], df['bam_close'], window=window).adx()

def ichimoku_cloud(df, tenkan=9, kijun=26, senkou=52):
    ichimoku = ta.trend.IchimokuIndicator(df['bam_high'], df['bam_low'], window1=tenkan, window2=kijun, window3=senkou)
    return ichimoku.ichimoku_a(), ichimoku.ichimoku_b(), ichimoku.ichimoku_base_line(), ichimoku.ichimoku_conversion_line()

def vwap(df):
    pv = (df['bam_close'] * df['base_volume']).cumsum()
    vol = df['base_volume'].cumsum()
    return pv / (vol + 1e-9)

# Example wrappers for parameterized indicators (to use in main1.py):
# indicator_functions = {
#     'sma_50': lambda df: sma(df, window=50),
#     'sma_200': lambda df: sma(df, window=200),
#     'ema_20': lambda df: ema(df, window=20),
#     'ema_50': lambda df: ema(df, window=50),
#     'ema_200': lambda df: ema(df, window=200),
#     'rsi_14': lambda df: rsi(df, window=14),
#     'macd_12_26_9': lambda df: macd(df, fast=12, slow=26, signal=9),
#     'bollinger_bands_20_2': lambda df: bollinger_bands(df, window=20, num_std=2),
#     'obv': obv,
#     'adx_14': lambda df: adx(df, window=14),
#     'stochastic_oscillator_14': lambda df: stochastic_oscillator(df, k_window=14, d_window=3),
#     'ichimoku_cloud_9_26_52': lambda df: ichimoku_cloud(df, tenkan=9, kijun=26, senkou=52),
#     'vwap': vwap,
# }
