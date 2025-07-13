# SignalCraft: A Modular System for Crypto Alpha Discovery

A modular system for developing and testing predictive trading strategies on crypto spot data, designed with a research-first mindset. Built to identify statistically strong signals and validate their profitability through robust, realistic backtests.

## 🎯 Purpose

This project answers a fundamental quant question:  
**Can we extract repeatable, tradable signals from historical crypto data using simple models and rigorous validation?**

The pipeline provides a structured approach to:
- Score and filter technical indicators by predictive strength
- Train interpretable models on informative signals
- Simulate multiple trading strategies with real-world constraints
- Prevent data leakage via strict train/test separation
- Evaluate final strategies using capital growth and risk metrics

## 🔧 Core Components

- `main1.py`: Central pipeline — data split, feature selection, training, evaluation  
- `regression.py`: Linear, logistic, quantile, ensemble models  
- `trading.py`: Strategy simulator with capital allocation logic  
- `simulate_*.py`: Batch simulations across assets/timeframes  
- `functions.py`: IC, Sharpe, win rate, normalization, etc.  
- `indicators.py`: Technical signal generation (RSI, momentum, moving averages)  

## 🔁 Workflow

1. **Data Split**  
   Each symbol’s data is split:  
   - First half → feature selection, training  
   - Second half → final testing (strictly untouched)

2. **Indicator Evaluation**  
   Compute metrics:  
   - Win rate  
   - Weighted win rate (adjusts for class imbalance)  
   - Information Coefficient (IC)  
   - Sharpe ratio, R²  
   Top indicators are selected based on thresholds (e.g. win rate > 51%)

3. **Model Training**  
   Fit models on selected indicators:  
   - Linear regression  
   - Logistic regression  
   - Polynomial features  
   - Quantile regression  
   - Voting ensembles  
   All features are **standardized using training set statistics**.

4. **Trading Strategy Simulation**  
   Simulate using different capital allocation logics:  
   - **Reactive / Reverse Reactive**: Capital scales with recent performance  
   - **Quartile-Based**: Position size grows with signal strength  
   - **Aggressive / Simple**: Full or fixed capital trades  
   - **Probability-weighted**: Use model confidence to size positions  
   - **Majority Vote**: Trade only when model and indicators agree

5. **Final Testing**  
   Evaluate only on the second half of each symbol’s data — held out from all prior steps.  
   Metrics: cumulative return, Sharpe, win rate, drawdown.

6. **Signal Evaluation & Portfolio Sizing**  
   Analyze consistency and predictive strength of signals across coins.  
   Use **Kelly criterion** for capital allocation across assets/models.

## ✅ Key Modeling Principles

- **Time Series Cross-Validation**: Used to tune and test models while respecting temporal order  
- **No Data Leakage**: Indicator and model selection is done entirely on training half  
- **Universal Scaling**: StandardScaler fit only on train set; reused on test  
- **Interpretable Models**: Focus on models with explainable decision rules and signal accountability  
- **Benchmarking**: Buy-and-hold comparison included for every strategy

## ✅ Supported Models

| Model       | Purpose                                |
|-------------|-----------------------------------------|
| Linear      | Predict return                          |
| Logistic    | Predict direction (up/down)             |
| Polynomial  | Capture non-linear price interactions   |
| Quantile    | Predict return distribution tails       |
| Ensemble    | Robust voting or joint decision models  |

## 📈 Results

- **~53% hit rate** across coins using the **Linear + Binary Ensemble** model  
- **Consistently outperformed buy-and-hold** baselines in final testing  
- **Sharpe ratio > 1** in most test periods, with lower drawdowns  

