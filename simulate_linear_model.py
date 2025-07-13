import os
import pandas as pd
import numpy as np
from regression import run_indicators, test_model
from trading import simulate_trading, simulate_trading_ensemble
import importlib

indicator_module = importlib.import_module('indicators')

original_columns = [
    'bam_vwa', 'bam_open', 'bam_close',
    'bam_high', 'bam_low', 'bam_volatility_1m_return', 'base_volume',
    'pct_buy_volume', 'annual_interest_rate', 'spread_1m_vwa_vwa',
    'spread_twa'
]
symbols = ['BNB', 'BTC', 'ETH', 'TRX', 'XRP']
linear_objectives = [None, 'elasticnet', 'ridge', 'lasso', 'linear', 'huber', 'bayesianridge', 'sgd', 'weights1']

for symbol in symbols:
    for trading_window in range(1, 2):
        print(f"Testing {symbol} with trading window {trading_window}")
        try:
            df = pd.read_csv(f'indicator_results_{symbol}.csv')
            df1 = df[df['window'] == trading_window]

            selected_indicators = df1[df1['win'] > 0.50]['indicator'].unique()
            if len(selected_indicators) == 0:
                print(f"No indicators with win rate > 50% for {symbol} - {trading_window}h window")
                continue
            print(f"Selected indicators: {selected_indicators}")

            df_symbol = pd.read_csv(f'{symbol}_spot_full.csv')
            half = len(df_symbol) // 2
            df_train = df_symbol.iloc[:half]
            
            # Get start time for testing (second half of data)
            start_time = df_symbol.iloc[half]['close_time']
            
            print(f"Training on: {df_train.iloc[0]['close_time']} to {df_train.iloc[-1]['close_time']}")
            print(f"Testing on: {start_time} to {df_symbol.iloc[-1]['close_time']}")

            for obj in linear_objectives:
                print(f"\n=== Testing {obj.upper() if obj else 'NONE'} model ===")
                try:
                    linear, binary, scaler = run_indicators(df_train, selected_indicators.tolist() + original_columns, 
                                                   time_window=trading_window, quantile=False, obj=obj)
                    
                    if linear is not None and binary is not None and scaler is not None:
                        print(f"Testing linear model on test set...")
                        r2_linear = test_model(df_symbol, linear, selected_indicators.tolist(), original_columns, 
                                             target_col='bam_close', time_window=trading_window, start_time=start_time, scaler=scaler)
                        
                        print(f"aggressive strategy (linear model)")
                        simulate_trading(df_symbol, linear, selected_indicators.tolist() + original_columns,
                                       start_time=start_time,
                                       trading_window=trading_window, plot=True, strategy='aggressive', scaler=scaler)
                        
                        print(f"proba_ensemble strategy (linear+binary model)")
                        simulate_trading_ensemble(df_symbol, linear, binary, selected_indicators.tolist() + original_columns,
                                               start_time=start_time,
                                               trading_window=trading_window, plot=True, strategy='proba_ensemble', scaler=scaler)
                    else:
                        print(f"Failed to create {obj} model")
                        
                except Exception as e:
                    print(f"Error with {obj} model: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing {symbol} - {trading_window}h: {str(e)}")
            continue

print("Linear model testing complete!")
