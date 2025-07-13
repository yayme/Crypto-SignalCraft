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
from regression import run_indicators, tcv_quantile
from trading import simulate_trading, simulate_trading_ensemble, simulate_trading_quantile
from datetime import datetime

original_columns=['bam_vwa', 'bam_open', 'bam_close',
       'bam_high', 'bam_low', 'bam_volatility_1m_return', 'base_volume',
       'pct_buy_volume', 'annual_interest_rate', 'spread_1m_vwa_vwa',
       'spread_twa']
symbols=['BNB', 'BTC', 'ETH','TRX','XRP']

for symbol in symbols:
    for trading_window in range(1,25):
        print(f"working with {symbol} and {trading_window}")
        
        try:
            # Read indicator results
            df = pd.read_csv(f'indicator_results_{symbol}.csv')
            df1 = df[df['window']==trading_window]
           
            # Select indicators with win rate > 50%
            selected_indicators = df1[df1['win']>0.50]['indicator'].unique()
            
            if len(selected_indicators) == 0:
                print(f"No indicators with win rate > 50% for {symbol} - {trading_window}h window")
                continue
                
            print(f"Selected indicators: {selected_indicators}")
            
            # Read symbol data
            df_symbol = pd.read_csv(f'{symbol}_spot_full.csv')
            half = len(df_symbol) // 2
            df1 = df_symbol.iloc[:half]
            
            # Print info about the data
            if 'close_time' in df1.columns:
                last_time = df1.iloc[-1]['close_time'] if len(df1) > 0 else 'N/A'
                print(f"Last close time: {last_time}, Shape: {df1.shape}")
            else:
                print(f"Shape: {df1.shape}")
            
            # Run regression and trading
            linear, binary, quantile = run_indicators(df1, selected_indicators.tolist() + original_columns, time_window=trading_window)
            
            if linear is not None:
                print("reactive")
                simulate_trading(df1[:1000], linear, selected_indicators.tolist() + original_columns, 
                               trading_window=trading_window, plot=True, strategy='reactive')
                print("non-reactive")
                simulate_trading(df1[:1000], linear, selected_indicators.tolist() + original_columns, 
                               trading_window=trading_window, plot=True, strategy='none')
                print("reverse-reactive (martingale)")
                simulate_trading(df1[:1000], linear, selected_indicators.tolist() + original_columns, 
                               trading_window=trading_window, plot=True, strategy='reverse_reactive')
                print("simple")
                simulate_trading(df1[:1000], linear, selected_indicators.tolist() + original_columns, 
                               trading_window=trading_window, plot=True, strategy='simple')
                
                # Ensemble strategy
                print("ensemble")
                simulate_trading_ensemble(df1[:1000], linear, binary, selected_indicators.tolist() + original_columns, 
                                       trading_window=trading_window, plot=True, strategy='reactive')
                print("ensemble simple")
                simulate_trading_ensemble(df1[:1000], linear, binary, selected_indicators.tolist() + original_columns, 
                                       trading_window=trading_window, plot=True, strategy='simple')
            else:
                print("Failed to create linear model")
            # Quantile regression strategy (always run if quantile_model is not None)
            if quantile is not None:
                print("quantile regression (median, q=0.5)")
                simulate_trading_quantile(df1[:1000], quantile, selected_indicators.tolist() + original_columns, 
                                         trading_window=trading_window, plot=True, strategy='reactive')
                print("quantile regression simple")
                simulate_trading_quantile(df1[:1000], quantile, selected_indicators.tolist() + original_columns, 
                                         trading_window=trading_window, plot=True, strategy='simple')
                print("quantile regression quartile")
                simulate_trading_quantile(df1[:1000], quantile, selected_indicators.tolist() + original_columns, 
                                         trading_window=trading_window, plot=True, strategy='quartile')
            # Binary-only trading strategy
            if binary is not None:
                print("binary only")
                from trading import simulate_trading_binary
                simulate_trading_binary(df1[:1000], binary, selected_indicators.tolist() + original_columns, 
                                       trading_window=trading_window, plot=True)
                
        except Exception as e:
            print(f"Error processing {symbol} - {trading_window}h: {str(e)}")
            continue

print("Processing complete!")


      
