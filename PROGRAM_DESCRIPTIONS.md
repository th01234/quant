# 🌟 Quant Repository Program Overview & Relations

Welcome to **Quant**, your all-in-one quantitative analysis and algorithmic trading toolkit!  
This repository is built for professionals, researchers, and innovators who demand modularity, reliability, and performance.

---

## 🗂️ Program Breakdown & Relationships

---

### `strategy_final.py`  
**🧠 Core Strategy Engine**

- **Purpose:**  
  The central hub for strategy design, machine learning signal generation, and backtesting.
- **Features:**  
  - Loads and preprocesses financial time-series (e.g. Bitcoin).
  - Extracts advanced features (EMA, RSI, rolling stats).
  - Generates buy/sell signals via XGBoost and neural networks (ANN).
  - Hyperparameter optimization via Bayesian search.
  - Robust risk modeling (LASSO regression).
  - Walk-forward validation for out-of-sample performance.
  - Comprehensive backtesting with statistics and money management.
- **Relations:**  
  - **Imports:** `bayes_opt_xgb.py`, `walk_forward.py`
  - **Extends:** Easily integrates new models and features.

---

### `bayes_opt_xgb.py`  
**🔬 Bayesian Optimization for XGBoost**

- **Purpose:**  
  Hyperparameter tuning for optimal model performance.
- **Features:**  
  - Automated search for best XGBoost parameters.
  - Data preprocessing pipeline (imputation, scaling).
  - Cross-validated results for robustness.
- **Relations:**  
  - **Called by:** `strategy_final.py`

---

### `walk_forward.py`  
**⏩ Walk-Forward Validation**

- **Purpose:**  
  Rigorous time-series cross-validation and signal refinement.
- **Features:**  
  - Splits data into training/testing folds respecting temporal order.
  - Quantile-based classification for target generation.
  - Fold-wise accuracy diagnostics.
- **Relations:**  
  - **Used by:** `strategy_final.py` (for out-of-sample evaluation)

---

### `quant_test.py`  
**📊 Advanced Equity Pairs Analysis**

- **Purpose:**  
  Standalone script for statistical arbitrage and pairs trading (e.g., BAE Systems & Lockheed Martin).
- **Features:**  
  - Downloads equity data with `yfinance`.
  - Computes rolling correlation, spread, Z-score, and cointegration.
  - Normality/variance tests and portfolio performance analytics.
  - Implements mean-reversion strategy and visualizes results.
- **Relations:**  
  - **Independent:** Shares analytical foundations but operates separately.

---

### `README.md`  
**📚 Repository Overview**

- **Purpose:**  
  Workflow summary and documentation.
- **Features:**  
  - Step-by-step guide from data to results.
  - Highlights modular design and extensibility.
  - Quick reference for all core scripts.

---

## 🎯 Professional Highlights

- **Modular**: Easily plug in new models, features, or datasets.
- **Robust**: Bayesian optimization and walk-forward validation ensure reliability.
- **Extensible**: Designed for research and rapid prototyping.
- **Transparent**: Every step is documented for reproducibility.

---

> **Quant is the professional’s choice for quantitative research, trading strategy development, and advanced analytics.  
> Unlock the future of finance and data science—one module at a time!**