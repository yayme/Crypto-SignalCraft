import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import importlib
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

def run_indicators_xgb(df, indicator_names, time_window=1, target_col='bam_close',
                       initial_train_size=0.5, val_size=0.1, step_size=0.05,
                       optimize_hyperparams=True, n_calls=20):
    """
    Calculate indicators, add as features, and run XGBoost regression and classification with time series CV.
    Mirrors regression.py's run_indicators but uses XGBoost models.
    """
    df = df.copy()
    if time_window > 1:
        df = df.iloc[::time_window].reset_index(drop=True)
    feature_cols = []
    for name in indicator_names:
        if name in df.columns:
            feature_cols.append(name)
        elif name in indicator_functions:
            result = indicator_functions[name](df)
            if isinstance(result, tuple) or isinstance(result, list):
                for i, arr in enumerate(result):
                    colname = f'{name}_{i}'
                    df[colname] = arr
                    feature_cols.append(colname)
            else:
                colname = name
                df[colname] = result
                feature_cols.append(colname)
        else:
            print(f"Indicator '{name}' not found in indicator_functions or df.columns. If this is a parameterized indicator, add it to indicator_functions.")
    feature_cols = [col for col in feature_cols if not df[col].isna().all()]
    df['target_return'] = df[target_col].shift(-1) - df[target_col]
    df['target_return_bin'] = (df['target_return'] > 0).astype(int)
    print(f"Using features: {feature_cols}")
    print("\n--- XGBoost Regression  ---")
    linear_model = tcv_xgb_linear(df, feature_cols, 'target_return',
                                 initial_train_size=initial_train_size, val_size=val_size, step_size=step_size,
                                 optimize_hyperparams=optimize_hyperparams, n_calls=n_calls)
    print("\n--- XGBoost Classification ---")
    binary_model = tcv_xgb_binary(df, feature_cols, 'target_return',
                                 initial_train_size=initial_train_size, val_size=val_size, step_size=step_size,
                                 optimize_hyperparams=optimize_hyperparams, n_calls=n_calls)
    return linear_model, binary_model

def tcv_xgb_linear(df, feature_cols, target_col, 
                   initial_train_size=0.5, val_size=0.1, step_size=0.05,
                   optimize_hyperparams=True, n_calls=20):
  
  
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    X_all = df[feature_cols].values
    y_all = df[target_col].values

    n = len(df)
    train_size = int(n * initial_train_size)
    val_len = int(n * val_size)
    step = int(n * step_size)

    # Define hyperparameter search space
    space = [
        Real(0.01, 1.0, name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(1, 10, name='min_child_weight'),
        Real(0.1, 1.0, name='subsample'),
        Real(0.1, 1.0, name='colsample_bytree'),
        Integer(50, 500, name='n_estimators')
    ]

    def objective(params):
        learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, n_estimators = params
        
        rmse_list = []
        
        for start in range(0, n - train_size - val_len + 1, step):
            train_X = X_all[start : start + train_size]
            train_y = y_all[start : start + train_size]
            val_X = X_all[start + train_size : start + train_size + val_len]
            val_y = y_all[start + train_size : start + train_size + val_len]

            model = xgb.XGBRegressor(
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                n_estimators=int(n_estimators),
                random_state=42,
                early_stopping_rounds=50,
                eval_metric='rmse'
            )
            
            model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
            pred_y = model.predict(val_X)
            rmse = root_mean_squared_error(val_y, pred_y)
            rmse_list.append(rmse)
        
        return np.mean(rmse_list)

    if optimize_hyperparams:
        print("Optimizing XGBoost hyperparameters...")
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
        best_params = result.x
        print(f"Best hyperparameters: {dict(zip(['learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'n_estimators'], best_params))}")
    else:
        # Default parameters
        best_params = [0.1, 6, 1, 0.8, 0.8, 100]

    # Final evaluation with best parameters
    learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, n_estimators = best_params
    
    rmse_list = []
    r2_list = []
    diracc_list = []
    last_train_y = last_train_pred = last_val_y = last_val_pred = None

    for i, start in enumerate(range(0, n - train_size - val_len + 1, step)):
        train_X = X_all[start : start + train_size]
        train_y = y_all[start : start + train_size]
        val_X = X_all[start + train_size : start + train_size + val_len]
        val_y = y_all[start + train_size : start + train_size + val_len]

        model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_estimators=int(n_estimators),
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
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
    plt.title('XGBoost Regression: Training True vs Predicted (Last Fold, 50 pts)')
    plt.xlabel('Point (0-49)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plot_filename = f"images/xgb_train_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"XGBoost training plot saved as: {plot_filename}")
    plt.show()
    
    # Validation plot
    plt.figure(figsize=(10, 4))
    plt.scatter(range(50), last_val_y, label='Val True (first 50)', color='green', alpha=0.7)
    plt.scatter(range(50), last_val_pred, label='Val Pred (first 50)', color='red', alpha=0.7)
    plt.title('XGBoost Regression: Validation True vs Predicted (Last Fold, 50 pts)')
    plt.xlabel('Point (0-49)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plot_filename = f"images/xgb_val_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"XGBoost validation plot saved as: {plot_filename}")
    plt.show()

    print(f"XGBoost Regression Walk-Forward Validation Results ({len(rmse_list)} folds):")
    print(f"Avg RMSE: {np.mean(rmse_list):.4f}")
    print(f"Avg R² Score: {np.mean(r2_list):.4f}")
    print(f"Avg Directional Accuracy: {np.mean(diracc_list)*100:.2f}%")

    # Train final model on most recent data
    final_model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=int(n_estimators),
        random_state=42
    )
    final_model.fit(X_all[:int(n * (1 - val_size))], y_all[:int(n * (1 - val_size))])
    return final_model

def tcv_xgb_binary(df, feature_cols, target_col, 
                   initial_train_size=0.5, val_size=0.1, step_size=0.05,
                   optimize_hyperparams=True, n_calls=20):
   
  
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    df['target_binary'] = (df[target_col] > 0).astype(int)
    X_all = df[feature_cols].values
    y_all = df['target_binary'].values

    n = len(df)
    train_size = int(n * initial_train_size)
    val_len = int(n * val_size)
    step = int(n * step_size)

    # Define hyperparameter search space
    space = [
        Real(0.01, 1.0, name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(1, 10, name='min_child_weight'),
        Real(0.1, 1.0, name='subsample'),
        Real(0.1, 1.0, name='colsample_bytree'),
        Integer(50, 500, name='n_estimators')
    ]

    def objective(params):
        learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, n_estimators = params
        
        auc_list = []
        
        for start in range(0, n - train_size - val_len + 1, step):
            train_X = X_all[start : start + train_size]
            train_y = y_all[start : start + train_size]
            val_X = X_all[start + train_size : start + train_size + val_len]
            val_y = y_all[start + train_size : start + train_size + val_len]

            model = xgb.XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                n_estimators=int(n_estimators),
                random_state=42,
                early_stopping_rounds=50,
                eval_metric='auc'
            )
            
            model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
            val_proba = model.predict_proba(val_X)[:, 1]
            fpr, tpr, _ = roc_curve(val_y, val_proba)
            auc_score = auc(fpr, tpr)
            auc_list.append(auc_score)
        
        return -np.mean(auc_list)  # Negative because we want to maximize AUC

    if optimize_hyperparams:
        print("Optimizing XGBoost hyperparameters...")
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
        best_params = result.x
        print(f"Best hyperparameters: {dict(zip(['learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'n_estimators'], best_params))}")
    else:
        # Default parameters
        best_params = [0.1, 6, 1, 0.8, 0.8, 100]

    # Final evaluation with best parameters
    learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, n_estimators = best_params
    
    acc_list = []
    auc_list = []
    cm_total = np.array([[0, 0], [0, 0]])
    last_train_y = last_train_proba = last_val_y = last_val_proba = None

    for i, start in enumerate(range(0, n - train_size - val_len + 1, step)):
        train_X = X_all[start : start + train_size]
        train_y = y_all[start : start + train_size]
        val_X = X_all[start + train_size : start + train_size + val_len]
        val_y = y_all[start + train_size : start + train_size + val_len]

        model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_estimators=int(n_estimators),
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='auc',
            scale_pos_weight=1.0  # Add class balancing
        )
        
        model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
        pred_y = model.predict(val_X)
        train_proba = model.predict_proba(train_X)[:, 1]
        val_proba = model.predict_proba(val_X)[:, 1]

        # Check if we have balanced predictions
        unique_preds = np.unique(pred_y)
        if len(unique_preds) < 2:
            print(f"Warning: XGBoost predicting only one class: {unique_preds}")
            # Try with different scale_pos_weight
            model = xgb.XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                n_estimators=int(n_estimators),
                random_state=42,
                early_stopping_rounds=50,
                eval_metric='auc',
                scale_pos_weight=2.0  # Increase weight for minority class
            )
            model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
            pred_y = model.predict(val_X)
            train_proba = model.predict_proba(train_X)[:, 1]
            val_proba = model.predict_proba(val_X)[:, 1]

        acc = accuracy_score(val_y, pred_y)
        cm = confusion_matrix(val_y, pred_y)
        fpr, tpr, _ = roc_curve(val_y, val_proba)
        auc_score = auc(fpr, tpr)
        
        acc_list.append(acc)
        auc_list.append(auc_score)
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
    plt.title('XGBoost ROC Curve (Last Fold)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plot_filename = f"images/xgb_roc_{today}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"XGBoost ROC curve saved as: {plot_filename}")
    plt.show()

    print(f"XGBoost Classification Walk-Forward Validation Results ({len(acc_list)} folds):")
    print(f"Avg Accuracy: {np.mean(acc_list)*100:.2f}%")
    print(f"Avg AUC: {np.mean(auc_list):.4f}")
    print("Confusion Matrix (Total):")
    print(cm_total)
    print("Classification Report (Last Fold):")
    print(classification_report(last_val_y, (last_val_proba > 0.5).astype(int), target_names=["Down/Flat", "Up"], zero_division=0))

    # Train final model on most recent data
    final_model = xgb.XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=int(n_estimators),
        random_state=42
    )
    final_model.fit(X_all[:int(n * (1 - val_size))], y_all[:int(n * (1 - val_size))])
    return final_model
