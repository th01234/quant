import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import GridSearchCV
from bayes_opt_xgb import bayesian_opt_xgb
from svj_model import simulate_svj
from walk_forward import walk_forward_validation
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# EMA and EMSD functions
def compute_ema(series, tau, lam):
    weights = np.array([lam**i for i in range(tau-1, -1, -1)])
    norm_factor = (1 - lam) / (1 - lam**tau)
    return series.rolling(tau).apply(lambda x: norm_factor * np.sum(weights * x[-tau:]), raw=True)

def compute_emsd(series, tau, lam):
    ema = compute_ema(series, tau, lam)
    squared_diff = (series - ema)**2
    return np.sqrt(compute_ema(squared_diff, tau, lam))

def compute_rsi(series, tau):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(tau).mean()
    avg_loss = loss.rolling(tau).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_features(df, window_size=1):
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['R'] = df['price'].pct_change()
    df['R_bar'] = df['R'].rolling(window_size).mean()
    df['R_tilde'] = df['R'] - df['R_bar']
    df['sigma'] = df['R_tilde'].rolling(window_size).std()
    df['R_hat'] = df['R_tilde'] / df['sigma']

    tau_ema = [window_size, window_size*2, window_size*4, window_size*8]
    tau_rsi = [window_size, window_size*2, window_size*4]
    lam_vals = [0.98, 0.95, 0.92, 0.9]

    for i, tau in enumerate(tau_ema):
        lam = lam_vals[i]
        df[f'EMA_{tau}'] = compute_ema(df['R_hat'], tau, lam)
        df[f'EMSD_{tau}'] = compute_emsd(df['R_hat'], tau, lam)

    for tau in tau_rsi:
        df[f'RSI_{tau}'] = compute_rsi(df['R_hat'], tau)

    features = ['R_hat'] + [f'EMA_{tau}' for tau in tau_ema] + [f'EMSD_{tau}' for tau in tau_ema] + [f'RSI_{tau}' for tau in tau_rsi]
    return df, features

def generate_ann_signals(df, features, K=3):
    df['target'] = df['R_hat'].shift(-1)
    df = df.dropna(subset=['target'] + features)

    train_size = int(0.7 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    if len(train_df) < K or train_df['target'].nunique() < K:
        print("Not enough training data or unique target values.")
        df['signal'] = 0
        return df

    # Compute quantile thresholds on training set
    quantiles = np.quantile(train_df['target'], np.linspace(0, 1, K+1)[1:-1])
    train_df['class'] = assign_quantile_class(train_df['target'].values, quantiles)
    X_train = train_df[features]
    y_train = train_df['class']

    # XGBoost GridSearchCV as before
    param_grid = {
        'xgbclassifier__n_estimators': [1000],
        'xgbclassifier__max_depth': [3],
        'xgbclassifier__learning_rate': [0.1]
    }
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=K,
            random_state=42,
            eval_metric='mlogloss'
        )
    )
    grid = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
    try:
        grid.fit(X_train, y_train)
        print("Best XGBoost params:", grid.best_params_)
        model = grid.best_estimator_
    except Exception as e:
        print(f"Model training error: {e}")
        df['signal'] = 0
        return df

    # Predict class for test set
    test_df['class'] = model.predict(test_df[features])

    # Trading rule: Buy if class == K-1 (top quantile), Short if class == 0 (bottom quantile)
    test_df['signal'] = 0
    test_df.loc[test_df['class'] == K-1, 'signal'] = 1
    test_df.loc[test_df['class'] == 0, 'signal'] = -1

    df['signal'] = 0
    df.update(test_df[['signal']])
    df['signal'] = df['signal'].shift(1).fillna(0)
    print("Signal Distribution:\n", df['signal'].value_counts())
    return df

def add_lasso_risk_feature(df, features, target_col='future_vol', window=12, lasso_sample=5000):
    df[target_col] = df['price'].pct_change().rolling(window).std().shift(-window)
    df = df.fillna(0)  # Replace NaNs with zeros
    X = df[features]
    y = df[target_col]
    if len(X) > lasso_sample:
        X_sub = X.iloc[:lasso_sample]
        y_sub = y.iloc[:lasso_sample]
    else:
        X_sub = X
        y_sub = y
    lasso = LassoCV(cv=3).fit(X_sub, y_sub)
    df['lasso_pred_risk'] = lasso.predict(X)
    print("LASSO selected features:", [f for f, c in zip(features, lasso.coef_) if abs(c) > 1e-6])
    return df, lasso

def compute_normalized_returns(df, T1):
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['R'] = df['price'].pct_change()
    df['R_bar'] = df['R'].rolling(T1).mean()
    df['R_tilde'] = df['R'] - df['R_bar']
    df['sigma'] = df['R_tilde'].rolling(T1).std()
    df['R_hat'] = df['R_tilde'] / df['sigma']
    return df

def compute_nu(df, tau):
    df[f'nu_plus_{tau}'] = df['R_hat'].rolling(window=tau, min_periods=1).apply(
        lambda x: np.sum(np.maximum(x, 0)), raw=True).shift(-tau+1)
    df[f'nu_minus_{tau}'] = df['R_hat'].rolling(window=tau, min_periods=1).apply(
        lambda x: np.sum(np.maximum(-x, 0)), raw=True).shift(-tau+1)
    return df

def backtest_strategy(df, initial_capital=100000, transaction_cost=0.001, hold_period=4, risk_pct=0.005):
    df = df.sort_values('timestamp').reset_index(drop=True)
    if df.empty or 'signal' not in df.columns or df['signal'].dropna().empty:
        print("No signals for backtest.")
        return df, {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'final_value': initial_capital, 'num_trades': 0}

    cash = initial_capital
    position = 0
    shares = 0
    entry_idx = None
    entry_price = None
    portfolio_value = []
    returns = []
    num_trades = 0

    df['rolling_std'] = df['price'].rolling(hold_period).std().fillna(method='bfill')

    for i, row in df.iterrows():
        price = row['price']
        signal = 0 if pd.isna(row['signal']) else row['signal']

        if position == 0 and signal != 0:
            day = pd.to_datetime(row['timestamp'], unit='s').date()
            trades_today = df.loc[(df.index < i) & (pd.to_datetime(df['timestamp'], unit='s').dt.date == day) & (df['signal'] != 0)].shape[0]
            if trades_today >= 20:
                signal = 0

        if position == 0 and signal != 0:
            entry_idx = i
            entry_price = price
            stop_loss = entry_price - 2 * row['rolling_std'] if signal == 1 else entry_price + 2 * row['rolling_std']
            position = signal
            risk_amount = cash * risk_pct
            shares = risk_amount / (2 * row['rolling_std']) if row['rolling_std'] > 0 else 0
            cash -= shares * price * (1 + transaction_cost)
            num_trades += 1

        if position != 0:
            hold_time = i - entry_idx
            if (hold_time >= hold_period) or \
               (position == 1 and price <= stop_loss) or \
               (position == -1 and price >= stop_loss):
                cash += shares * price * (1 - transaction_cost)
                position = 0
                shares = 0
                entry_idx = None
                entry_price = None

        value = cash + (shares * price if position != 0 else 0)
        portfolio_value.append(value)
        returns.append((value - portfolio_value[-2]) / portfolio_value[-2] if i > 0 and portfolio_value[-2] > 0 else 0)

    df['portfolio_value'] = portfolio_value
    df['returns'] = returns
    total_return = (portfolio_value[-1] - initial_capital) / initial_capital
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252*24) if np.std(returns) else 0
    turnover = np.mean(np.abs(np.diff(df['signal'].fillna(0))))
    fitness = np.sqrt(np.abs(total_return)/max(turnover, 0.125)) * sharpe_ratio
    max_drawdown = calculate_max_drawdown(np.array(portfolio_value))

    return df, {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'fitness': fitness,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_value[-1],
        'num_trades': num_trades
    }

def calculate_max_drawdown(values):
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return drawdown.min()

def at_nan2zero(arr):
    return np.nan_to_num(arr, nan=0.0)

def run_backtest_example():
    data_file = "btcusd_1-min_data.csv"
    print("Loading historical data from btcusd_1-min_data.csv...")

    df = pd.DataFrame()
    raw_data = pd.read_csv(data_file)

    # Correct column assignments
    df['timestamp'] = raw_data['Timestamp']
    df['price'] = raw_data['Close']
    df['High'] = raw_data['High']
    df['Low'] = raw_data['Low']
    df['Close'] = raw_data['Close']
    df['Open'] = raw_data['Open']
    df['Volume'] = raw_data['Volume']

    print(f"\nData Summary:")
    print(f"Number of candles: {len(df)}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # --- Use the most recent 4,000,000 rows ---
    df = df.iloc[-4000000:]  # Use the freshest data

    T1 = 4
    df = compute_normalized_returns(df, T1=T1)
    window_size = 16  # or any value you want
    df, features = compute_features(df, window_size=window_size)
    df, lasso = add_lasso_risk_feature(df, features, window=window_size)
    df = df.fillna(0)  # Ensure all NaNs are replaced with zeros

    for col in df.columns:
        df[col] = pd.Series(at_nan2zero(df[col].values), index=df.index)

    walk_forward_validation(df, features, K=3, n_splits=5)
    df = generate_ann_signals_nn(df, features, K=3)

    print("\nSample Signals:", df[['timestamp', 'price', 'signal']].tail(10))

    backtest_df, stats = backtest_strategy(df, initial_capital=1000000, transaction_cost=0.001, hold_period=1, risk_pct=0.02)

    if backtest_df is not None and not backtest_df.empty and 'portfolio_value' in backtest_df.columns:
        print(backtest_df[['timestamp', 'price', 'R_hat', 'signal', 'portfolio_value']].head(10))
        print("\nFinal Portfolio Value:", backtest_df['portfolio_value'].iloc[-1])
        print("\nBacktest Stats:", stats)
    else:
        print("No backtest data to display.")

class QuantileANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=1)
        return probs

def assign_quantile_class(series, quantiles):
    classes = np.zeros(len(series), dtype=int)
    for i, x in enumerate(series):
        for alpha, q in enumerate(quantiles):
            if x <= q:
                classes[i] = alpha
                break
        else:
            classes[i] = len(quantiles)
    return classes

def trading_signal_from_probs(probs, K, prob_threshold=0.7):
    pred_class = probs.argmax(axis=1)
    top_prob = probs.max(axis=1)
    signals = np.zeros(len(pred_class))
    signals[(pred_class >= K-2) & (top_prob > prob_threshold)] = 1
    signals[(pred_class <= 1) & (top_prob > prob_threshold)] = -1
    return signals

def generate_ann_signals_nn(df, features, K=5, hidden_dims=[64, 32], epochs=20, batch_size=256, prob_threshold=0.7):
    df['target'] = df['R_hat'].shift(-1)
    df = df.dropna(subset=['target'] + features)

    train_size = int(0.7 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    if len(train_df) < K or train_df['target'].nunique() < K:
        print("Not enough training data or unique target values.")
        df['signal'] = 0
        return df

    quantiles = np.quantile(train_df['target'], np.linspace(0, 1, K+1)[1:-1])
    train_df['class'] = assign_quantile_class(train_df['target'].values, quantiles)
    test_df['class'] = assign_quantile_class(test_df['target'].values, quantiles)

    X_train = train_df[features].values
    y_train = train_df['class'].values
    X_test = test_df[features].values

    input_dim = X_train.shape[1]
    model = QuantileANN(input_dim, hidden_dims, K)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            optimizer.zero_grad()
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = model(X_test_tensor).cpu().numpy()
        signals = trading_signal_from_probs(probs, K, prob_threshold)

    test_df['signal'] = signals
    df['signal'] = 0
    df.update(test_df[['signal']])
    df['signal'] = df['signal'].shift(1).fillna(0)
    print("Signal Distribution:\n", df['signal'].value_counts())
    return df

def compute_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(window=period).mean()
    return atr

def backtest_strategy(df, initial_capital=1000000, transaction_cost=0.001, hold_period=4, risk_pct=0.01, atr_period=14, atr_mult=2):
    df = df.sort_values('timestamp').reset_index(drop=True)
    if df.empty or 'signal' not in df.columns or df['signal'].dropna().empty:
        print("No signals for backtest.")
        return df, {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'final_value': initial_capital, 'num_trades': 0}

    df['ATR'] = compute_atr(df, period=atr_period).fillna(method='bfill')

    cash = initial_capital
    position = 0
    shares = 0
    entry_idx = None
    entry_price = None
    portfolio_value = []
    returns = []
    num_trades = 0

    for i, row in df.iterrows():
        price = row['price']
        signal = 0 if pd.isna(row['signal']) else row['signal']

        if position == 0 and signal != 0:
            day = pd.to_datetime(row['timestamp'], unit='s').date()
            trades_today = df.loc[(df.index < i) & (pd.to_datetime(df['timestamp'], unit='s').dt.date == day) & (df['signal'] != 0)].shape[0]
            if trades_today >= 20:
                signal = 0

        if position == 0 and signal != 0:
            entry_idx = i
            entry_price = price
            stop_loss = entry_price - atr_mult * row['ATR'] if signal == 1 else entry_price + atr_mult * row['ATR']
            position = signal
            risk_amount = cash * risk_pct
            shares = risk_amount / (atr_mult * row['ATR']) if row['ATR'] > 0 else 0
            cash -= shares * price * (1 + transaction_cost)
            num_trades += 1

        if position != 0:
            hold_time = i - entry_idx
            if (hold_time >= hold_period) or \
               (position == 1 and price <= stop_loss) or \
               (position == -1 and price >= stop_loss):
                cash += shares * price * (1 - transaction_cost)
                position = 0
                shares = 0
                entry_idx = None
                entry_price = None

        value = cash + (shares * price if position != 0 else 0)
        portfolio_value.append(value)
        returns.append((value - portfolio_value[-2]) / portfolio_value[-2] if i > 0 and portfolio_value[-2] > 0 else 0)

    df['portfolio_value'] = portfolio_value
    df['returns'] = returns
    total_return = (portfolio_value[-1] - initial_capital) / initial_capital
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252*24) if np.std(returns) else 0
    turnover = np.mean(np.abs(np.diff(df['signal'].fillna(0))))
    fitness = np.sqrt(np.abs(total_return)/max(turnover, 0.125)) * sharpe_ratio
    max_drawdown = calculate_max_drawdown(np.array(portfolio_value))

    return df, {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'fitness': fitness,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_value[-1],
        'num_trades': num_trades
    }

if __name__ == "__main__":
    run_backtest_example()
