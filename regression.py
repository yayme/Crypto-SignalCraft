from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression, HuberRegressor, BayesianRidge, SGDRegressor, QuantileRegressor
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

def get_model(obj, alpha=0.1, l1_ratio=0.5, max_iter=10000):
    from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression, HuberRegressor, BayesianRidge, SGDRegressor
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

# Popular choices for obj: 'elasticnet', 'ridge', 'lasso', 'linear', 'huber', 'bayesianridge', 'sgd', 
# 'weights1'

def tcv_linear(df, feature_cols, target_col, 
               alpha=0.1, l1_ratio=0.5, max_iter=10000, 
               initial_train_size=0.5, val_size=0.1, step_size=0.05, obj=None, plot=False):
    


    df = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f'tcv_linear_{df.shape}')
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

    # Plot only the last fold's training and validation (optional)
    if plot:
        today = datetime.now().strftime("%d%m%Y")
        os.makedirs('images', exist_ok=True)
        
        # Training plot
        plt.figure(figsize=(10, 4))
        plt.plot(range(50), last_train_y, label='Train True (first 50)', color='blue', linewidth=1)
        plt.plot(range(50), last_train_pred, label='Train Pred (first 50)', color='red', linewidth=1)
        plt.title(f'Regression ({obj if obj else "linear"}): Training True vs Predicted (Last Fold, 50 pts)')
        plt.xlabel('Point (0-49)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f"images/regression_train_{obj if obj else 'linear'}_{today}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Training plot saved as: {plot_filename}")
        plt.show()
        
        # Validation plot
        plt.figure(figsize=(10, 4))
        plt.plot(range(50), last_val_y, label='Val True (first 50)', color='green', linewidth=1)
        plt.plot(range(50), last_val_pred, label='Val Pred (first 50)', color='red', linewidth=1)
        plt.title(f'Regression ({obj if obj else "linear"}): Validation True vs Predicted (Last Fold, 50 pts)')
        plt.xlabel('Point (0-49)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
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
    return model, scaler

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

        # Use balanced class weights by default
        model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        model.fit(train_X, train_y)
        pred_y = model.predict(val_X)
        train_proba = model.predict_proba(train_X)[:, 1]
        val_proba = model.predict_proba(val_X)[:, 1]

        # Check if we have balanced predictions
        unique_preds = np.unique(pred_y)
        if len(unique_preds) < 2:
            print(f"Warning: Model predicting only one class: {unique_preds}")
            # Try with different class weights if still unbalanced
            model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight={0: 1, 1: 2})
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
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"images/roc_curve_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved as: {plot_filename}")
    plt.show()

    print(f"Logistic Regression (Balanced) Walk-Forward Validation Results ({len(acc_list)} folds):")
    print(f"Avg Accuracy: {np.mean(acc_list)*100:.2f}%")
    print("Confusion Matrix (Total):")
    print(cm_total)
    print("Classification Report (Last Fold):")
    print(classification_report(last_val_y, (last_val_proba > 0.5).astype(int), target_names=["Down/Flat", "Up"], zero_division=0))

    # Final model with balanced class weights
    model = LogisticRegression(max_iter=max_iter, class_weight='balanced')
    model.fit(X_all[:int(n * (1 - val_size))], y_all[:int(n * (1 - val_size))])
    return model

def tcv_quantile(df, feature_cols, target_col='bam_close', quantile=0.5, alpha=0.1, max_iter=10000, initial_train_size=0.5, val_size=0.1, step_size=0.05, time_window=None):
    """
    Walk-forward quantile regression validation. Returns fitted QuantileRegressor on all data.
    Mirrors tcv_linear in structure and logic.
    """
    df = df.copy()
    if time_window is not None and time_window > 1:
        df = df.iloc[::time_window].reset_index(drop=True)
    df['target_return'] = (df[target_col].shift(-1) - df[target_col]) / df[target_col]
    print('Mean target_return:', df['target_return'].mean())
    print('Fraction positive:', (df['target_return'] > 0).mean())
    print('Fraction negative:', (df['target_return'] < 0).mean())
    df = df.dropna(subset=feature_cols + ['target_return'])
    X_all = df[feature_cols].values
    y_all = df['target_return'].values

    n = len(df)
    train_size = int(n * initial_train_size)
    val_len = int(n * val_size)
    step = int(n * step_size)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    mae_list = []
    r2_list = []
    diracc_list = []

    last_train_y = last_train_pred = last_val_y = last_val_pred = None

    for i, start in enumerate(range(0, n - train_size - val_len + 1, step)):
        train_X = X_all[start : start + train_size]
        train_y = y_all[start : start + train_size]
        val_X = X_all[start + train_size : start + train_size + val_len]
        val_y = y_all[start + train_size : start + train_size + val_len]

        model = QuantileRegressor(quantile=quantile, alpha=alpha, solver='highs')
        model.fit(train_X, train_y)
        pred_y = model.predict(val_X)
        train_pred = model.predict(train_X)

        mae = np.mean(np.abs(val_y - pred_y))
        r2 = r2_score(val_y, pred_y)
        true_dir = np.sign(np.diff(val_y))
        pred_dir = np.sign(np.diff(pred_y))
        dir_acc = (true_dir == pred_dir).mean()

        mae_list.append(mae)
        r2_list.append(r2)
        diracc_list.append(dir_acc)

        last_train_y = train_y[:50]
        last_train_pred = train_pred[:50]
        last_val_y = val_y[:50]
        last_val_pred = pred_y[:50]

    today = datetime.now().strftime("%d%m%Y")
    os.makedirs('images', exist_ok=True)
    # Training plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(50), last_train_y, label='Train True (first 50)', color='blue', linewidth=1)
    plt.plot(range(50), last_train_pred, label='Train Pred (first 50)', color='red', linewidth=1)
    plt.title(f'Quantile Regression (q={quantile}): Training True vs Predicted (Last Fold, 50 pts)')
    plt.xlabel('Point (0-49)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"images/quantile_regression_train_q{quantile}_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Quantile Regression training plot saved as: {plot_filename}")
    plt.show()

    # Validation plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(50), last_val_y, label='Val True (first 50)', color='green', linewidth=1)
    plt.plot(range(50), last_val_pred, label='Val Pred (first 50)', color='red', linewidth=1)
    plt.title(f'Quantile Regression (q={quantile}): Validation True vs Predicted (Last Fold, 50 pts)')
    plt.xlabel('Point (0-49)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"images/quantile_regression_val_q{quantile}_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Quantile Regression validation plot saved as: {plot_filename}")
    plt.show()

    print(f"Quantile Regression Walk-Forward Validation Results (q={quantile}, {len(mae_list)} folds):")
    print(f"Avg MAE: {np.mean(mae_list):.4f}")
    print(f"Avg R² Score: {np.mean(r2_list):.4f}")
    print(f"Avg Directional Accuracy: {np.mean(diracc_list)*100:.2f}%")

    # Fit on all data (except last val_size for fair comparison)
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver='highs')
    model.fit(X_all[:int(n * (1 - val_size))], y_all[:int(n * (1 - val_size))])
    return model


def fit_quantile_regression(df, feature_cols, target_col, quantile=0.5, alpha=0.1, max_iter=10000):
    """
    Fit QuantileRegressor on all data and return the model.
    """
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    df['target_return'] = (df[target_col].shift(-1) - df[target_col]) / df[target_col]
    print('Mean target_return:', df['target_return'].mean())
    print('Fraction positive:', (df['target_return'] > 0).mean())
    print('Fraction negative:', (df['target_return'] < 0).mean())
    X = df[feature_cols].values
    y = df['target_return'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver='highs')
    model.fit(X, y)
    return model

def tcv_polynomial(df, feature_cols, target_col, degree=2, max_iter=10000, initial_train_size=0.5, val_size=0.1, step_size=0.05, plot=False):
    """
    Polynomial regression with walk-forward validation. Returns fitted model on all data.
    """
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    X_all = df[feature_cols].values
    y_all = df[target_col].values
    
    # Debug target variable
    print(f"Target variable stats: mean={y_all.mean():.6f}, std={y_all.std():.6f}")
    print(f"Target range: [{y_all.min():.6f}, {y_all.max():.6f}]")
    
    n = len(df)
    train_size = int(n * initial_train_size)
    val_len = int(n * val_size)
    step = int(n * step_size)

    rmse_list = []
    r2_list = []
    diracc_list = []

    last_train_y = last_train_pred = last_val_y = last_val_pred = None

    for i, start in enumerate(range(0, n - train_size - val_len + 1, step)):
        train_X = X_all[start : start + train_size]
        train_y = y_all[start : start + train_size]
        val_X = X_all[start + train_size : start + train_size + val_len]
        val_y = y_all[start + train_size : start + train_size + val_len]

        # Fit polynomial features on training data only
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        train_X_poly = poly.fit_transform(train_X)
        val_X_poly = poly.transform(val_X)
        
        # Scale on training data only
        scaler = StandardScaler()
        train_X_poly = scaler.fit_transform(train_X_poly)
        val_X_poly = scaler.transform(val_X_poly)

        # Use Ridge regression for regularization
        model = Ridge(alpha=1.0, max_iter=max_iter)
        model.fit(train_X_poly, train_y)
        pred_y = model.predict(val_X_poly)
        train_pred = model.predict(train_X_poly)

        rmse = root_mean_squared_error(val_y, pred_y)
        r2 = r2_score(val_y, pred_y)
        true_dir = np.sign(np.diff(val_y))
        pred_dir = np.sign(np.diff(pred_y))
        dir_acc = (true_dir == pred_dir).mean()

        # Debug information for first few folds
        if i < 3:
            print(f"Polynomial Fold {i}: RMSE={rmse:.6f}, R²={r2:.6f}, DirAcc={dir_acc:.3f}")
            print(f"  Val_y range: [{val_y.min():.6f}, {val_y.max():.6f}], std: {val_y.std():.6f}")
            print(f"  Pred_y range: [{pred_y.min():.6f}, {pred_y.max():.6f}], std: {pred_y.std():.6f}")

        rmse_list.append(rmse)
        r2_list.append(r2)
        diracc_list.append(dir_acc)

        last_train_y = train_y[:50]
        last_train_pred = train_pred[:50]
        last_val_y = val_y[:50]
        last_val_pred = pred_y[:50]

    # Plot only the last fold's training and validation (optional)
    if plot:
        today = datetime.now().strftime("%d%m%Y")
        os.makedirs('images', exist_ok=True)
        
        # Training plot
        plt.figure(figsize=(10, 4))
        plt.plot(range(50), last_train_y, label='Train True (first 50)', color='blue', linewidth=1)
        plt.plot(range(50), last_train_pred, label='Train Pred (first 50)', color='red', linewidth=1)
        plt.title(f'Polynomial Regression (degree={degree}): Training True vs Predicted (Last Fold, 50 pts)')
        plt.xlabel('Point (0-49)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f"images/polynomial_regression_train_degree{degree}_{today}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Polynomial training plot saved as: {plot_filename}")
        plt.show()
        
        # Validation plot
        plt.figure(figsize=(10, 4))
        plt.plot(range(50), last_val_y, label='Val True (first 50)', color='green', linewidth=1)
        plt.plot(range(50), last_val_pred, label='Val Pred (first 50)', color='red', linewidth=1)
        plt.title(f'Polynomial Regression (degree={degree}): Validation True vs Predicted (Last Fold, 50 pts)')
        plt.xlabel('Point (0-49)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f"images/polynomial_regression_val_degree{degree}_{today}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Polynomial validation plot saved as: {plot_filename}")
        plt.show()

    print(f"Polynomial Regression Walk-Forward Validation Results (degree={degree}, {len(rmse_list)} folds):")
    print(f"Avg RMSE: {np.mean(rmse_list):.6f}")
    print(f"Avg R² Score: {np.mean(r2_list):.6f}")
    print(f"Avg Directional Accuracy: {np.mean(diracc_list)*100:.2f}%")

    # Fit on all data (except last val_size for fair comparison)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_all[:int(n * (1 - val_size))])
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)
    model = Ridge(alpha=1.0, max_iter=max_iter)
    model.fit(X_poly, y_all[:int(n * (1 - val_size))])
    model.poly = poly
    model.scaler = scaler
    return model

def tcv_majority_vote(df, indicator_names, feature_cols, target_col, max_iter=10000):
    """
    Majority vote model: binary regression on original features, plus each indicator gets a vote (rise if indicator > 0, fall if < 0, else abstain). If majority says rise, output 1; if fall, output 0; if tie, abstain (output np.nan).
    Returns a callable that takes a DataFrame and outputs the majority vote signal.
    """
    from sklearn.linear_model import LogisticRegression
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    # Binary regression on original features
    df['target_bin'] = (df[target_col] > 0).astype(int)
    X = df[feature_cols].values
    y = df['target_bin'].values
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(X, y)
    def majority_vote_predict(X_df):
        # Binary regression vote
        bin_pred = clf.predict(X_df[feature_cols])
        # Indicator votes
        indicator_votes = []
        for ind in indicator_names:
            if ind in X_df.columns:
                v = np.sign(X_df[ind])
                indicator_votes.append(v)
        # Stack votes: shape (n_samples, n_votes)
        if indicator_votes:
            votes = np.column_stack([bin_pred] + indicator_votes)
        else:
            votes = bin_pred.reshape(-1, 1)
        # Majority vote: 1 if sum > 0, 0 if sum < 0, np.nan if tie
        maj = np.apply_along_axis(lambda row: 1 if np.sum(row) > 0 else (0 if np.sum(row) < 0 else np.nan), 1, votes)
        return maj
    return majority_vote_predict


def run_indicators(df, indicator_names, obj=None, time_window=1, target_col='bam_close',
                   alpha=0.1, l1_ratio=0.5, max_iter=10000,
                   initial_train_size=0.5, val_size=0.1, step_size=0.05,
                   quantile=False, plot=False):
    """
    Calculate indicators, add as features, and run linear, binary, and optionally quantile regression with time series CV.
    Returns linear_model, binary_model, quantile_model (or None if quantile=False), and fitted scaler.
    """
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
        if quantile:
            return None, None, None, None
        else:
            return None, None, None
    
    df['target_return'] = (df[target_col].shift(-1) - df[target_col]) / df[target_col]
    print('Mean target_return:', df['target_return'].mean())
    print('Fraction positive:', (df['target_return'] > 0).mean())
    print('Fraction negative:', (df['target_return'] < 0).mean())
    df['target_return_bin'] = (df['target_return'] > 0).astype(int)
    
    print(f"Using features: {feature_cols}")
    print("\n--- Linear Regression  ---")
    linear_model, scaler = tcv_linear(df, feature_cols, 'target_return', alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                             initial_train_size=initial_train_size, val_size=val_size, step_size=step_size, obj=obj, plot=plot)
    print("\n--- LogisticRegression ---")
    binary_model = tcv_binary(df, feature_cols, 'target_return',
                             max_iter=max_iter, initial_train_size=initial_train_size, val_size=val_size, step_size=step_size)
    if quantile:
        print("\n--- Quantile Regression (q=0.5) ---")
        quantile_model = tcv_quantile(df, feature_cols, 'target_return', quantile=0.5, max_iter=max_iter,
                                     initial_train_size=initial_train_size, val_size=val_size, step_size=step_size)
        return linear_model, binary_model, quantile_model, scaler
    else:
        return linear_model, binary_model, scaler


def run_indicators2(df, indicator_names, time_window=1, target_col='bam_close',
                   degree=2, max_iter=10000,
                   initial_train_size=0.5, val_size=0.1, step_size=0.05, plot=False):
    """
    Calculate indicators, add as features, and run polynomial regression and majority vote model.
    Returns poly_model, majority_vote_model.
    """
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
    df['target_return'] = (df[target_col].shift(-1) - df[target_col]) / df[target_col]
    print('Mean target_return:', df['target_return'].mean())
    print('Fraction positive:', (df['target_return'] > 0).mean())
    print('Fraction negative:', (df['target_return'] < 0).mean())
    print(f"Using features: {feature_cols}")
    print("\n--- Polynomial Regression  ---")
    poly_model = tcv_polynomial(df, feature_cols, 'target_return', degree=degree, max_iter=max_iter,
                             initial_train_size=initial_train_size, val_size=val_size, step_size=step_size, plot=plot)
    print("\n--- Majority Vote Model ---")
    majority_vote_model = tcv_majority_vote(df, indicator_names, feature_cols, 'target_return', max_iter=max_iter)
    return poly_model, majority_vote_model

def test_model(df, model, indicator_names, original_columns, target_col='bam_close', time_window=1, start_time=None, end_time=None, scaler=None):
    """
    Test a trained model on a DataFrame by calculating indicators and making predictions.
    Returns R² score.
    
    Args:
        df: DataFrame with price data (full dataset)
        model: Trained model (linear, binary, etc.)
        indicator_names: List of indicator names to calculate
        original_columns: List of original feature columns
        target_col: Target column name
        time_window: Time window for target calculation
        start_time: Start time for testing (like in simulate_trading)
        end_time: End time for testing (optional)
        scaler: Fitted StandardScaler from training (optional)
    
    Returns:
        r2_score: R² score of the model
    """
    df = df.copy()
    if time_window > 1:
        df = df.iloc[::time_window].reset_index(drop=True)
    
    # Calculate target return first
    df['target_return'] = (df[target_col].shift(-time_window) - df[target_col]) / df[target_col]
    
    # Calculate indicators on full dataset first
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
    
    # Add original columns
    for col in original_columns:
        if col in df.columns:
            feature_cols.append(col)
    
    # Filter out features with all NaN values
    feature_cols = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]
    
    if not feature_cols:
        print("No valid features found!")
        return None
    
    # NOW filter by time after calculating indicators
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
    
    if start_time:
        df = df[df['close_time'] >= pd.to_datetime(start_time)].copy()
    if end_time:
        df = df[df['close_time'] <= pd.to_datetime(end_time)].copy()
    
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols + ['target_return'], inplace=True)
    
    if len(df) == 0:
        print("No valid data after cleaning!")
        return None
    
    X = df[feature_cols].values
    y = df['target_return'].values
    
    # Scale features using the fitted scaler if provided, otherwise fit a new one
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        # Binary model
        y_pred = model.predict_proba(X_scaled)[:, 1]
        r2 = None  # R² not meaningful for binary classification
    else:
        # Linear model
        y_pred = model.predict(X_scaled)
        # Calculate R² score only for regression models
        r2 = r2_score(y, y_pred)
    
    # Classification-style metrics for rise/fall
    true_binary = (y > 0).astype(int)  # 1 for rise, 0 for fall
    if hasattr(model, 'predict_proba'):
        pred_binary = (y_pred > 0.5).astype(int)
    else:
        pred_binary = (y_pred > 0).astype(int)
    
    # Calculate classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)
    cm = confusion_matrix(true_binary, pred_binary)
    
    # Weighted accuracy by rise/fall class
    rise_mask = (true_binary == 1)
    fall_mask = (true_binary == 0)
    
    rise_accuracy = accuracy_score(true_binary[rise_mask], pred_binary[rise_mask]) if rise_mask.sum() > 0 else 0
    fall_accuracy = accuracy_score(true_binary[fall_mask], pred_binary[fall_mask]) if fall_mask.sum() > 0 else 0
    
    # Overall weighted accuracy
    n_rise = rise_mask.sum()
    n_fall = fall_mask.sum()
    total = len(true_binary)
    weighted_accuracy = (rise_accuracy * n_rise + fall_accuracy * n_fall) / total if total > 0 else 0
    
    print(f"Test Model Results:")
    if r2 is not None:
        print(f"  R² Score: {r2:.6f}")
    else:
        print(f"  R² Score: N/A (binary classification model)")
    print(f"  Test samples: {len(y)}")
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Weighted Accuracy: {weighted_accuracy:.4f} ({weighted_accuracy*100:.2f}%)")
    print(f"  Rise Accuracy: {rise_accuracy:.4f} ({rise_accuracy*100:.2f}%) - {n_rise} samples")
    print(f"  Fall Accuracy: {fall_accuracy:.4f} ({fall_accuracy*100:.2f}%) - {n_fall} samples")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix (Fall/Rise):")
    print(f"    [[{cm[0,0]:4d} {cm[0,1]:4d}]  # Predicted Fall")
    print(f"     [{cm[1,0]:4d} {cm[1,1]:4d}]]  # Predicted Rise")
    print(f"     # Actual Fall  Actual Rise")
    
    return r2 if r2 is not None else None
