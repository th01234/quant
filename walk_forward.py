import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def walk_forward_validation(df, features, n_splits=5, K=5):
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    # Prepare target as in your pipeline
    df['target'] = df['R_hat'].shift(-1)
    df = df.dropna(subset=['target'] + features).reset_index(drop=True)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        X_train, y_train = train[features], train['target']
        X_test, y_test = test[features], test['target']

        # Quantile binning for classification
        quantiles = np.quantile(y_train, np.linspace(0, 1, K+1)[1:-1])
        y_train_class = np.digitize(y_train, quantiles)
        y_test_class = np.digitize(y_test, quantiles)

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

        pipeline.fit(X_train, y_train_class)
        preds = pipeline.predict(X_test)

        # Example: Calculate accuracy
        accuracy = np.mean(preds == y_test_class)
        print(f"Fold {fold}: Accuracy={accuracy:.4f}, Train {train.index[0]}-{train.index[-1]}, Test {test.index[0]}-{test.index[-1]}")
        results.append({
            'fold': fold,
            'train_start': train.index[0],
            'train_end': train.index[-1],
            'test_start': test.index[0],
            'test_end': test.index[-1],
            'accuracy': accuracy
        })

    return results
