from skopt import BayesSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def bayesian_opt_xgb(X, y, n_iter=25, cv=3, n_jobs=-1, num_class=3):
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=num_class,
            random_state=42,
            eval_metric='mlogloss'
        )
    )

    param_space = {
        'xgbclassifier__n_estimators': (100, 2000),
        'xgbclassifier__max_depth': (2, 8),
        'xgbclassifier__learning_rate': (0.01, 0.3, 'log-uniform'),
        'xgbclassifier__subsample': (0.5, 1.0, 'uniform'),
        'xgbclassifier__colsample_bytree': (0.5, 1.0, 'uniform'),
        'xgbclassifier__gamma': (0, 5),
        'xgbclassifier__reg_alpha': (0, 2),
        'xgbclassifier__reg_lambda': (0, 2),
    }

    opt = BayesSearchCV(
        pipeline,
        param_space,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=2,
        random_state=42
    )
    opt.fit(X, y)
    return opt.best_estimator_
