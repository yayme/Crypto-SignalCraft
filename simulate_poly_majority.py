import os
import pandas as pd
from regression import run_indicators2
from trading import simulate_trading_poly, simulate_trading_majority

original_columns = [
    'bam_vwa', 'bam_open', 'bam_close',
    'bam_high', 'bam_low', 'bam_volatility_1m_return', 'base_volume',
    'pct_buy_volume', 'annual_interest_rate', 'spread_1m_vwa_vwa',
    'spread_twa'
]
symbols = ['BNB', 'BTC', 'ETH', 'TRX', 'XRP']

for symbol in symbols:
    for trading_window in range(1, 25):
        print(f"Testing {symbol} with trading window {trading_window} (polynomial only)")
        try:
            # Read indicator results
            df = pd.read_csv(f'indicator_results_{symbol}.csv')
            df1 = df[df['window'] == trading_window]

            # Select indicators with win rate > 50%
            selected_indicators = df1[df1['win'] > 0.50]['indicator'].unique()
            if len(selected_indicators) == 0:
                print(f"No indicators with win rate > 50% for {symbol} - {trading_window}h window")
                continue
            print(f"Selected indicators: {selected_indicators}")

            # Read symbol data
            df_symbol = pd.read_csv(f'{symbol}_spot_full.csv')
            half = len(df_symbol) // 2
            df1 = df_symbol.iloc[:half]

            # Fit polynomial model only (comment out majority vote)
            poly_model, majority_vote_model = run_indicators2(df1, selected_indicators.tolist() + original_columns, time_window=trading_window, degree=1)
            if poly_model is not None:
                # Test all polynomial strategies
                strategies = ["reactive", "reverse_reactive", "simple", "quartile", "aggressive"]
                for strategy in strategies:
                    print(f"Testing {strategy} strategy (polynomial model)")
                    simulate_trading_poly(df1[:1000], poly_model, selected_indicators.tolist() + original_columns,
                                          trading_window=trading_window, plot=True, strategy=strategy)
                
                # Comment out majority vote testing
                # print("majority vote strategy")
                # simulate_trading_majority(df1[:1000], majority_vote_model, selected_indicators.tolist(), selected_indicators.tolist() + original_columns,
                #                               trading_window=trading_window, plot=True)
            else:
                print("Failed to create polynomial model")
        except Exception as e:
            print(f"Error processing {symbol} - {trading_window}h: {str(e)}")
            continue

print("Polynomial strategy testing complete!")
