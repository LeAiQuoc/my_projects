# project1/ml_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def preprocess_data(data: pd.DataFrame, target_col: str, ml_type: str):
    """Validate target column and preprocess data for ML."""
    if not target_col or target_col not in data.columns:
        raise ValueError(f"'{target_col}' not a valid column or empty.")

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    if ml_type == "classifier":
        if y.dtype in ['float64', 'int64'] and y.nunique() > 10:
            raise ValueError(f"'{target_col}' appears continuous but classifier selected.")
        if y.dtype == 'object' or y.dtype.name == 'category':
            label_map = {label: idx for idx, label in enumerate(y.unique())}
            y = y.map(label_map)
    else:
        if y.dtype == 'object' or y.nunique() <= 10:
            raise ValueError(f"'{target_col}' is categorical but regressor selected.")

    if X.isna().any().any():
        raise ValueError("Missing data in features. Fill before proceeding.")

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def run_classifiers(X_train, X_test, y_train, y_test):
    """Run classifier models and return results."""
    classifiers = {
        'LoR': LogisticRegression(random_state=101),
        'KNN': KNeighborsClassifier(),
        'SVC': SVC(random_state=101, probability=True)
    }
    param_grids = {
        'LoR': {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['liblinear']},
        'KNN': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']},
        'SVC': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
    }

    results = {}
    for name, clf in classifiers.items():
        grid = GridSearchCV(estimator=clf, param_grid=param_grids[name], cv=3, n_jobs=-1, scoring='f1')
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            "best_params": grid.best_params_,
            "classification_report": report,
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "model": grid.best_estimator_
        }

    return results


def run_regressors(X_train, X_test, y_train, y_test):
    """Run regressor models and return results."""
    regressors = {
        'LiR': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'SVR': SVR()
    }
    param_grids = {
        'LiR': {},
        'Lasso': {'alpha': [0.1, 1]},
        'Ridge': {'alpha': [0.1, 1]},
        'ElasticNet': {'alpha': [0.1, 1], 'l1_ratio': [0.2, 0.5]},
        'SVR': {'C': [0.1, 1], 'kernel': ['linear']}
    }

    results = {}
    for name, reg in regressors.items():
        grid = GridSearchCV(estimator=reg, param_grid=param_grids[name], cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {
            "best_params": grid.best_params_,
            "MAE": mae,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, y_pred),
            "model": grid.best_estimator_
        }

    return results
