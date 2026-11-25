"""
House Price Prediction - Model Training Pipeline

Trains and evaluates a Random Forest regression model
using the California Housing dataset.

Includes dataset validation, cross-validation,
performance evaluation, feature importance,
logging and model persistence.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET_SCALE = 100000

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


# ==================================================
# 2. DATA LOADING AND VALIDATION
# ==================================================

def load_dataset():
    """Load and validate the California Housing dataset."""

    logger.info("Loading California Housing dataset")

    housing = fetch_california_housing(as_frame=True)

    X = housing.data.copy()
    y = housing.target.copy() * TARGET_SCALE

    if X.empty or y.empty:
        raise ValueError("Dataset is empty")

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    X = X[FEATURE_COLUMNS]

    if len(X) != len(y):
        raise ValueError(
            "Feature and target lengths do not match"
        )

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError(
            "Dataset contains missing values"
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Dataset contains invalid feature values"
        )

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError(
            "Dataset contains invalid target values"
        )

    if (y < 0).any():
        raise ValueError(
            "Dataset contains negative house prices"
        )

    logger.info(
        "Dataset loaded successfully: %s rows, %s features",
        X.shape[0],
        X.shape[1],
    )

    return X, y


# ==================================================
# 3. MODEL CONFIGURATION
# ==================================================

def create_model():
    """Create a reproducible Random Forest model."""

    return RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ==================================================
# 4. CROSS-VALIDATION
# ==================================================

def evaluate_cross_validation(model, X_train, y_train):
    """Perform cross-validation on training data only."""

    logger.info(
        "Starting %s-fold cross-validation",
        CV_FOLDS,
    )

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        error_score="raise",
    )

    mae_scores = -scores

    results = {
        "folds": CV_FOLDS,
        "fold_mae_usd": [
            round(float(score), 2)
            for score in mae_scores
        ],
        "mean_mae_usd": round(
            float(np.mean(mae_scores)), 2
        ),
        "std_mae_usd": round(
            float(np.std(mae_scores)), 2
        ),
    }

    logger.info(
        "Cross-validation MAE: $%.2f",
        results["mean_mae_usd"],
    )

    return results


# ==================================================
# 5. MODEL PERFORMANCE EVALUATION
# ==================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on unseen test data."""

    predictions = model.predict(X_test)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model returned invalid predictions"
        )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    metrics = {
        "mae_usd": round(float(mae), 2),
        "mse_usd_squared": round(float(mse), 2),
        "rmse_usd": round(float(rmse), 2),
        "r2_score": round(float(r2), 6),
    }

    return metrics


# ==================================================
# 6. FEATURE IMPORTANCE
# ==================================================

def get_feature_importance(model, feature_names):
    """Extract and rank model feature importances."""

    importance = [
        {
            "feature": feature,
            "importance": round(float(score), 6),
        }
        for feature, score in zip(
            feature_names,
            model.feature_importances_,
        )
    ]

    return sorted(
        importance,
        key=lambda item: item["importance"],
        reverse=True,
    )


# ==================================================
# 7. MODEL AND METRICS PERSISTENCE
# ==================================================

def save_artifacts(model, report):
    """Save the trained model and evaluation report."""

    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    logger.info("Model saved to: %s", MODEL_PATH)

    logger.info(
        "Evaluation report saved to: %s",
        METRICS_PATH,
    )


# ==================================================
# 8. COMPLETE TRAINING PIPELINE
# ==================================================

def train_model():
    """Execute the complete model training pipeline."""

    start_time = time.perf_counter()

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    logger.info(
        "Training samples: %s",
        len(X_train),
    )

    logger.info(
        "Testing samples: %s",
        len(X_test),
    )

    model = create_model()

    # Cross-validation uses training data only.
    cv_results = evaluate_cross_validation(
        model,
        X_train,
        y_train,
    )

    logger.info("Training Random Forest model")

    model.fit(X_train, y_train)

    logger.info("Evaluating trained model")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    feature_importance = get_feature_importance(
        model,
        X.columns,
    )

    execution_time = round(
        time.perf_counter() - start_time,
        2,
    )

    report = {
        "project": "House Price Prediction",
        "dataset": "California Housing",
        "model": "RandomForestRegressor",
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "target_currency": "USD",
        "target_scale": TARGET_SCALE,
        "random_state": RANDOM_STATE,
        "dataset_samples": len(X),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "model_parameters": model.get_params(),
        "test_metrics": metrics,
        "cross_validation": cv_results,
        "feature_importance": feature_importance,
        "training_pipeline_seconds": execution_time,
    }

    save_artifacts(model, report)

    print("\n" + "=" * 55)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(f"Dataset samples : {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nMODEL PERFORMANCE")
    print("-" * 55)

    print(
        f"MAE  : ${metrics['mae_usd']:,.2f}"
    )

    print(
        f"MSE  : {metrics['mse_usd_squared']:,.2f} USD²"
    )

    print(
        f"RMSE : ${metrics['rmse_usd']:,.2f}"
    )

    print(
        f"R²   : {metrics['r2_score']:.4f}"
    )

    print(
        f"Cross-validation MAE: "
        f"${cv_results['mean_mae_usd']:,.2f}"
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 55)

    for item in feature_importance:
        print(
            f"{item['feature']:15s}: "
            f"{item['importance']:.6f}"
        )

    print(
        f"\nPipeline execution time: "
        f"{execution_time} seconds"
    )

    print(f"Model file: {MODEL_PATH.name}")
    print(f"Metrics file: {METRICS_PATH.name}")
    print("=" * 55)

    return report


# ==================================================
# 9. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    try:
        train_model()

    except Exception:
        logger.exception(
            "Model training pipeline failed"
        )
        raise
