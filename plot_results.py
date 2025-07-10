import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
metrics = ['pearson', 'r2', 'spearman', 'ic', 'win', 'sharpe']
colors = {
    'pearson': 'blue',
    'r2': 'red',
    'spearman': 'green',
    'ic': 'purple',
    'win': 'orange',
    'sharpe': 'brown'
}

csv_files = glob.glob('indicator_results_*.csv')

for csv_file in csv_files:
    symbol = csv_file.split('_')[-1].replace('.csv', '')
    df = pd.read_csv(csv_file)
    indicators = df['indicator'].unique()
    for indicator in indicators:
        plt.figure(figsize=(12, 6))
        plt.title(f'{symbol} - {indicator} Performance vs. Time Window')
        for metric in metrics:
            plt.plot(
                df[df['indicator'] == indicator]['window'],
                df[df['indicator'] == indicator][metric],
                label=metric,
                color=colors[metric]
            )
        plt.xlabel('Time Window (h)')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join('images',f'plot_{symbol}_{indicator}.png'))
        plt.close()
