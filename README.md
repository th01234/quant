### quant ideas

`stratefy_final` is a comprehensive Python toolkit designed for quantitative analysis and strategy development on Bitcoin price data.

## Workflow Overview

1. **Data Acquisition**
   - Load historical Bitcoin OHLCV data.
   - Preprocess and clean data.

2. **Feature Engineering**
   - Calculate advanced indicators (EMA, RSI, etc.).
   - Generate rolling statistics.

3. **Machine Learning-Based Signal Generation**
   - Quantile classification of price action.
   - Train XGBoost and neural networks.
   - Hyperparameter optimization.

4. **Risk Modeling**
   - Model future volatility with LASSO regression.
   - Integrate risk predictions into signals.

5. **Walk-Forward Validation**
   - Out-of-sample testing and signal refinement.

6. **Strategy Backtesting**
   - Simulate trades and manage risk/money.
   - Measure performance (return, Sharpe, drawdown).

7. **Results Analysis**
   - Visualize portfolio performance.
   - Summarize backtest results.

The modular design enables rigorous research and systematic Bitcoin trading strategy development.







## quant_test.py

`quant_test.py` is an advanced quantitative analysis and pairs trading script for BAE Systems (BA.L) and Lockheed Martin (LMT). It includes:

- **Data Acquisition:** Downloads daily close prices via `yfinance`.
- **Feature Engineering:** Calculates:
  - 60-day rolling correlation
  - 20-day rolling volatility of returns
- **Cointegration Analysis:**
  - Engle-Granger test for cointegration
  - Augmented Dickey-Fuller test for stationarity
  - Hedge ratio estimation via OLS regression
  - Spread and Z-score computation
- **Statistical Testing:**
  - Shapiro-Wilk test for normality
  - Levene’s test for equal variance
  - Pearson correlation
- **Backtesting Framework:**
  - Mean-reversion strategy using Z-score thresholds
  - Entry/exit logic for simulated trading
  - Portfolio value tracking
- **Performance Metrics:**
  - Sharpe ratio
  - Maximum drawdown
  - Cumulative returns
- **Visualization:**
  - Price charts, spread and Z-score plots
  - Rolling correlation and volatility plots
  - Portfolio equity
