import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind, shapiro, levene
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm


# STEP 1: Download and Align Data

ticker1 = "BA.L"  # BAE Systems
ticker2 = "LMT"    # Lockheed Martin
start_date = "2000-10-01"
end_date = "2025-03-20"

# Download historical data with 1d interval
data = yf.download([ticker1, ticker2], start=start_date, end=end_date, interval="1d")
df = data['Close'].rename(columns={ticker1: 'BAE', ticker2: 'LMT'}).dropna().reset_index(drop=True)


# STEP 2: Advanced Feature Engineering

# Rolling correlation (60-day window)
df['Rolling_Corr'] = df['BAE'].rolling(60).corr(df['LMT'])

# Volatility (20-day rolling std of returns)
df['BAE_Vol'] = df['BAE'].pct_change().rolling(20).std()
df['LMT_Vol'] = df['LMT'].pct_change().rolling(20).std()


# STEP 3: Cointegration Analysis

# Engle-Granger Test
coint_stat, p_value_coint, crit_values = coint(df['BAE'], df['LMT'])

# Hedge ratio using OLS
X = sm.add_constant(df['LMT'])
model = sm.OLS(df['BAE'], X).fit()
beta = model.params['LMT']
intercept = model.params['const']

# Spread calculations
df['Spread'] = df['BAE'] - (beta * df['LMT'] + intercept)
df['Z'] = (df['Spread'] - df['Spread'].mean()) / df['Spread'].std()

# ADF Test
adf_result = adfuller(df['Spread'].dropna())


# STEP 4: Statistical Tests

# Returns calculations
df['BAE_Returns'] = df['BAE'].pct_change()
df['LMT_Returns'] = df['LMT'].pct_change()
df = df.dropna().reset_index(drop=True)

# Normality tests
_, sw_p_bae = shapiro(df['BAE_Returns'])
_, sw_p_lmt = shapiro(df['LMT_Returns'])


# STEP 5: Backtesting Framework (Corrected)

# Trading rules
entry_z = 2.0
exit_z = 0.5

# Initialize positions
df['Position'] = 0
df['Portfolio'] = 1.0  # Starting with $1

# Backtest logic using .loc
for i in range(1, len(df)):
    # Entry signal
    if df.loc[i-1, 'Z'] > entry_z:
        df.loc[i, 'Position'] = -1  # Short BAE, Long LMT
    elif df.loc[i-1, 'Z'] < -entry_z:
        df.loc[i, 'Position'] = 1   # Long BAE, Short LMT
        
    # Exit signal
    if abs(df.loc[i-1, 'Z']) < exit_z:
        df.loc[i, 'Position'] = 0
        
    # Calculate returns
    if df.loc[i, 'Position'] != 0:
        ret = df.loc[i, 'Position'] * (df.loc[i, 'BAE_Returns'] - df.loc[i, 'LMT_Returns'])
        df.loc[i, 'Portfolio'] = df.loc[i-1, 'Portfolio'] * (1 + ret)
    else:
        df.loc[i, 'Portfolio'] = df.loc[i-1, 'Portfolio']


# STEP 6: Performance Metrics

# Sharpe ratio
returns = df['Portfolio'].pct_change().dropna()
sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

# Max drawdown
cum_returns = (1 + returns).cumprod()
peak = cum_returns.expanding(min_periods=1).max()
drawdown = (cum_returns/peak - 1).min()


# STEP 7: Enhanced Visualizations

plt.figure(figsize=(18, 20))
gs = plt.GridSpec(6, 1)

# Price plot
ax0 = plt.subplot(gs[0])
ax0.plot(df['BAE'], label='BAE Systems', color='navy')
ax0.plot(df['LMT'], label='Lockheed Martin', color='darkgreen')
ax0.set_title('Stock Prices')
ax0.legend()

# Spread and Z-score
ax1 = plt.subplot(gs[1])
ax1.plot(df['Spread'], color='purple', label='Spread')
ax1.axhline(df['Spread'].mean(), color='black', ls='--')
ax1.set_title('Price Spread')
ax1.legend()

ax2 = plt.subplot(gs[2])
ax2.plot(df['Z'], color='teal', label='Z-Score')
ax2.axhline(entry_z, color='r', ls='--')
ax2.axhline(-entry_z, color='r', ls='--')
ax2.axhline(0, color='k', alpha=0.5)
ax2.set_title('Z-Score of Spread')

# Rolling statistics
ax3 = plt.subplot(gs[3])
ax3.plot(df['Rolling_Corr'], color='darkorange', label='60D Rolling Correlation')
ax3.set_ylim(-1, 1)
ax3.axhline(0, color='k', alpha=0.5)
ax3.set_title('Rolling Correlation')

# Portfolio performance
ax4 = plt.subplot(gs[4])
ax4.plot(df['Portfolio'], color='darkblue', label='Strategy Equity')
ax4.set_title(f'Backtest Performance | Sharpe: {sharpe_ratio:.2f} | Max DD: {drawdown*100:.1f}%')

# Volatility comparison
ax5 = plt.subplot(gs[5])
ax5.plot(df['BAE_Vol'], color='navy', label='BAE Volatility')
ax5.plot(df['LMT_Vol'], color='darkgreen', label='LMT Volatility')
ax5.set_title('20D Rolling Volatility')

plt.tight_layout()
plt.show()


# STEP 8: Print Full Report

print(f'''
=== Advanced Pair Analysis Report ===
[Cointegration Analysis]
Engle-Granger p-value: {p_value_coint:.4f}
ADF Test p-value: {adf_result[1]:.4f}

[Correlation Analysis]
Pearson Correlation: {pearsonr(df['BAE_Returns'], df['LMT_Returns'])[0]:.3f}
Rolling Correlation Range: {df['Rolling_Corr'].min():.2f} to {df['Rolling_Corr'].max():.2f}

[Statistical Tests]
Shapiro-Wilk (Normality): BAE={sw_p_bae:.3f}, LMT={sw_p_lmt:.3f}
Levene's Test (Equal Variance): p={levene(df['BAE_Returns'], df['LMT_Returns'])[1]:.3f}

[Performance Metrics]
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {drawdown*100:.1f}%
Cumulative Return: {(df['Portfolio'].iloc[-1]-1)*100:.1f}%
''')
