from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression, HuberRegressor, BayesianRidge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
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
# Popular choices for obj: 'elasticnet', 'ridge', 'lasso', 'linear', 'huber', 'bayesianridge', 'sgd', 
# 'weights1'

def tcv_linear(df, feature_cols, target_col, 
               alpha=0.1, l1_ratio=0.5, max_iter=10000, 
               initial_train_size=0.5, val_size=0.1, step_size=0.05, obj=None):

    df = df.dropna(subset=feature_cols + [target_col]).copy()
    X_all = df[feature_cols].values
    y_all = df[target_col].values

    n = len(df)
    train_size = int(n * initial_train_size)
    val_len = int(n * val_size)
    step = int(n * step_size)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    rmse_list = []
    r2_list = []
    diracc_list = []

    def get_model(obj):
        if obj is None or obj == 'linear':
            return LinearRegression()
        elif obj == 'elasticnet':
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
        elif obj == 'ridge':
            return Ridge(alpha=alpha, max_iter=max_iter)
        elif obj == 'lasso':
            return Lasso(alpha=alpha, max_iter=max_iter)
        elif obj == 'huber':
            return HuberRegressor(max_iter=max_iter)
        elif obj == 'bayesianridge':
            return BayesianRidge()
        elif obj == 'sgd':
            return SGDRegressor(max_iter=max_iter)
        elif obj == 'weights1':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown obj: {obj}")

    last_train_y = last_train_pred = last_val_y = last_val_pred = None

    for i, start in enumerate(range(0, n - train_size - val_len + 1, step)):
        train_X = X_all[start : start + train_size]
        train_y = y_all[start : start + train_size]
        val_X = X_all[start + train_size : start + train_size + val_len]
        val_y = y_all[start + train_size : start + train_size + val_len]

        model = get_model(obj)
        if obj == 'weights1':
            sample_weight = np.abs(train_y)
            # Avoid all-zero weights
            sample_weight = sample_weight + 1e-8
            model.fit(train_X, train_y, sample_weight=sample_weight)
        else:
            model.fit(train_X, train_y)
        pred_y = model.predict(val_X)
        train_pred = model.predict(train_X)

        rmse = root_mean_squared_error(val_y, pred_y)
        r2 = r2_score(val_y, pred_y)
        true_dir = np.sign(np.diff(val_y))
        pred_dir = np.sign(np.diff(pred_y))
        dir_acc = (true_dir == pred_dir).mean()

        rmse_list.append(rmse)
        r2_list.append(r2)
        diracc_list.append(dir_acc)

        last_train_y = train_y[:50]
        last_train_pred = train_pred[:50]
        last_val_y = val_y[:50]
        last_val_pred = pred_y[:50]

    # Plot only the last fold's training and validation
    today = datetime.now().strftime("%d%m%Y")
    os.makedirs('images', exist_ok=True)
    
    # Training plot
    plt.figure(figsize=(10, 4))
    plt.scatter(range(50), last_train_y, label='Train True (first 50)', color='blue', alpha=0.7)
    plt.scatter(range(50), last_train_pred, label='Train Pred (first 50)', color='red', alpha=0.7)
    plt.title(f'Regression ({obj if obj else "linear"}): Training True vs Predicted (Last Fold, 50 pts)')
    plt.xlabel('Point (0-49)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plot_filename = f"images/regression_train_{obj if obj else 'linear'}_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Training plot saved as: {plot_filename}")
    plt.show()
    
    # Validation plot
    plt.figure(figsize=(10, 4))
    plt.scatter(range(50), last_val_y, label='Val True (first 50)', color='green', alpha=0.7)
    plt.scatter(range(50), last_val_pred, label='Val Pred (first 50)', color='red', alpha=0.7)
    plt.title(f'Regression ({obj if obj else "linear"}): Validation True vs Predicted (Last Fold, 50 pts)')
    plt.xlabel('Point (0-49)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plot_filename = f"images/regression_val_{obj if obj else 'linear'}_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Validation plot saved as: {plot_filename}")
    plt.show()

    print(f"Regression ({obj if obj else 'linear'}) Walk-Forward Validation Results ({len(rmse_list)} folds):")
    print(f"Avg RMSE: {np.mean(rmse_list):.4f}")
    print(f"Avg R² Score: {np.mean(r2_list):.4f}")
    print(f"Avg Directional Accuracy: {np.mean(diracc_list)*100:.2f}%")

    model = get_model(obj)
    if obj == 'weights1':
        sample_weight = np.abs(y_all[:int(n * (1 - val_size))])
        sample_weight = sample_weight + 1e-8
        model.fit(X_all[:int(n * (1 - val_size))], y_all[:int(n * (1 - val_size))], sample_weight=sample_weight)
    else:
        model.fit(X_all[:int(n * (1 - val_size))], y_all[:int(n * (1 - val_size))])
    return model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

def tcv_binary(df, feature_cols, target_col, 
                                    max_iter=10000,
                                    initial_train_size=0.5, val_size=0.1, step_size=0.05):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    df['target_binary'] = (df[target_col] > 0).astype(int)
    X_all = df[feature_cols].values
    y_all = df['target_binary'].values

    n = len(df)
    train_size = int(n * initial_train_size)
    val_len = int(n * val_size)
    step = int(n * step_size)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    acc_list = []
    cm_total = np.array([[0, 0], [0, 0]])

    last_train_y = last_train_proba = last_val_y = last_val_proba = None

    for i, start in enumerate(range(0, n - train_size - val_len + 1, step)):
        train_X = X_all[start : start + train_size]
        train_y = y_all[start : start + train_size]
        val_X = X_all[start + train_size : start + train_size + val_len]
        val_y = y_all[start + train_size : start + train_size + val_len]

        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(train_X, train_y)
        pred_y = model.predict(val_X)
        train_proba = model.predict_proba(train_X)[:, 1]
        val_proba = model.predict_proba(val_X)[:, 1]

        # Check if we have balanced predictions
        unique_preds = np.unique(pred_y)
        if len(unique_preds) < 2:
            print(f"Warning: Model predicting only one class: {unique_preds}")
            # Add some regularization to encourage balanced predictions
            model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
            model.fit(train_X, train_y)
            pred_y = model.predict(val_X)
            train_proba = model.predict_proba(train_X)[:, 1]
            val_proba = model.predict_proba(val_X)[:, 1]

        acc = accuracy_score(val_y, pred_y)
        cm = confusion_matrix(val_y, pred_y)
        acc_list.append(acc)
        cm_total += cm

        last_train_y = train_y
        last_train_proba = train_proba
        last_val_y = val_y
        last_val_proba = val_proba

    # Plot only the last fold's ROC curve
    fpr_train, tpr_train, _ = roc_curve(last_train_y, last_train_proba)
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_val, tpr_val, _ = roc_curve(last_val_y, last_val_proba)
    roc_auc_val = auc(fpr_val, tpr_val)

    today = datetime.now().strftime("%d%m%Y")
    os.makedirs('images', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_val, tpr_val, color='red', lw=2, label=f'Val ROC (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Last Fold)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plot_filename = f"images/roc_curve_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved as: {plot_filename}")
    plt.show()

    print(f"Logistic Regression Walk-Forward Validation Results ({len(acc_list)} folds):")
    print(f"Avg Accuracy: {np.mean(acc_list)*100:.2f}%")
    print("Confusion Matrix (Total):")
    print(cm_total)
    print("Classification Report (Last Fold):")
    print(classification_report(last_val_y, (last_val_proba > 0.5).astype(int), target_names=["Down/Flat", "Up"], zero_division=0))

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_all[:int(n * (1 - val_size))], y_all[:int(n * (1 - val_size))])
    return model

def run_indicators(df, indicator_names, obj=None, time_window=1, target_col='bam_close',
                   alpha=0.1, l1_ratio=0.5, max_iter=10000,
                   initial_train_size=0.5, val_size=0.1, step_size=0.05):
 
    df = df.copy()
    if time_window > 1:
        df = df.iloc[::time_window].reset_index(drop=True)
    
    # Ensure all indicators are calculated and available as columns
    feature_cols = []
    for name in indicator_names:
        if name in df.columns:
            feature_cols.append(name)
        elif name in indicator_functions:
            try:
                result = indicator_functions[name](df)
                if isinstance(result, tuple) or isinstance(result, list):
                    for i, arr in enumerate(result):
                        colname = f'{name}_{i}'
                        df[colname] = arr
                        feature_cols.append(colname)
                else:
                    df[name] = result
                    feature_cols.append(name)
            except Exception as e:
                print(f"Error calculating {name}: {e}")
                continue
        else:
            print(f"Indicator '{name}' not found in indicator_functions or df.columns.")
            continue
    
    # Filter out features with all NaN values
    feature_cols = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]
    
    if not feature_cols:
        print("No valid features found!")
        return None, None
    
    df['target_return'] = df[target_col].shift(-1) - df[target_col]
    df['target_return_bin'] = (df['target_return'] > 0).astype(int)
    
    print(f"Using features: {feature_cols}")
    print("\n--- Linear Regression  ---")
    linear_model = tcv_linear(df, feature_cols, 'target_return', alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                             initial_train_size=initial_train_size, val_size=val_size, step_size=step_size, obj=obj)
    print("\n--- LogisticRegression ---")
    binary_model = tcv_binary(df, feature_cols, 'target_return',
                             max_iter=max_iter, initial_train_size=initial_train_size, val_size=val_size, step_size=step_size)
    return linear_model, binary_model
