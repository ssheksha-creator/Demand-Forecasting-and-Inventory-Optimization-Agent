"""
AI Demand & Inventory Optimizer
Phase 2: Forecasting V2

Goals
-----
1. Use the Phase-1 advanced features.
2. Avoid future-unknown operational variables in the primary forecast.
3. Evaluate chronologically using rolling time-series validation.
4. Compare against practical forecasting baselines.
5. Train LightGBM point + quantile models.
6. Produce P10 / P50 / P90 forecasts.
7. Preserve forecast_next_days() and forecast_all_products().
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import (
    DATA_PATH,
    TARGET_COLUMN,
    build_features,
    get_feature_columns,
    load_data,
)


# ============================================================================
# PATHS / CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "demand_model_v2.pkl"
LOWER_MODEL_PATH = MODEL_DIR / "demand_model_p10.pkl"
UPPER_MODEL_PATH = MODEL_DIR / "demand_model_p90.pkl"

METADATA_PATH = MODEL_DIR / "model_metadata_v2.json"

FORECAST_OUTPUT_PATH = OUTPUT_DIR / "forecasts_v2.csv"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "validation_predictions_v2.csv"
CV_OUTPUT_PATH = OUTPUT_DIR / "rolling_validation_v2.csv"

VALIDATION_DAYS = 30
FORECAST_HORIZON = 7


# ============================================================================
# FUTURE FEATURE POLICY
# ============================================================================

# These values are not reliably known for future dates.
# They are therefore excluded from the primary demand model.
DYNAMIC_FUTURE_UNKNOWN = {
    "Inventory Level",
    "Units Ordered",
    "Units Sold",
    "Weather Condition",
    "Competitor Pricing",
    "Epidemic",
}


# These values can reasonably be treated as known/planned inputs.
CATEGORICAL_COLUMNS = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Promotion",
    "Seasonality",
]


# ============================================================================
# MODEL FEATURES
# ============================================================================

MODEL_FEATURES = [
    col
    for col in get_feature_columns()
    if col not in DYNAMIC_FUTURE_UNKNOWN
]


# Remove features that depend on same-day operational observations.
MODEL_FEATURES = [
    col
    for col in MODEL_FEATURES
    if col not in {
        "inventory_to_demand_ratio",
        "order_to_demand_ratio",
        "sales_to_demand_ratio",
        "inventory_minus_recent_demand",
        "order_minus_recent_demand",
    }
]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare model input dataframe.
    """

    X = df[MODEL_FEATURES].copy()

    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X


# ============================================================================
# METRICS
# ============================================================================

def _metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate MAE, RMSE and MAPE.
    """

    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(
        y_true_np,
        y_pred_np,
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_true_np,
            y_pred_np,
        )
    )

    denominator = np.where(
        np.abs(y_true_np) < 1e-8,
        np.nan,
        np.abs(y_true_np),
    )

    mape = (
        np.nanmean(
            np.abs(
                (y_true_np - y_pred_np)
                / denominator
            )
        )
        * 100
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def _clip_predictions(
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Demand cannot be negative.
    """

    return np.maximum(
        np.asarray(predictions, dtype=float),
        0.0,
    )


# ============================================================================
# BASELINES
# ============================================================================

def make_baseline_predictions(
    feature_df: pd.DataFrame,
    validation_mask: pd.Series,
) -> pd.DataFrame:
    """
    Create leakage-safe baseline predictions.

    Baselines:
    - Yesterday
    - Same day last week
    - 7-day moving average
    """

    rows = feature_df.loc[
        validation_mask
    ].copy()

    rows["Naive Yesterday"] = (
        rows["demand_lag_1"]
    )

    rows["Seasonal Naive 7D"] = (
        rows["demand_lag_7"]
    )

    rows["Moving Average 7D"] = (
        rows["demand_rolling_mean_7"]
    )

    return rows[
        [
            "Date",
            "Store ID",
            "Product ID",
            TARGET_COLUMN,
            "Naive Yesterday",
            "Seasonal Naive 7D",
            "Moving Average 7D",
        ]
    ].copy()


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(
    objective: str = "regression",
    alpha: Optional[float] = None,
    n_estimators: int = 1800,
) -> LGBMRegressor:
    """
    Create LightGBM model.
    """

    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 0.25,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "objective": objective,
    }

    if objective == "quantile":
        params["alpha"] = alpha

    return LGBMRegressor(**params)


# ============================================================================
# VALIDATION WINDOWS
# ============================================================================

def _make_validation_windows(
    dates: pd.Series,
    validation_days: int = VALIDATION_DAYS,
    n_folds: int = 4,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create chronological rolling validation windows.
    """

    unique_dates = (
        pd.Series(
            pd.to_datetime(dates)
            .drop_duplicates()
            .sort_values()
        )
        .reset_index(drop=True)
    )

    required_dates = validation_days * (
        n_folds + 1
    )

    if len(unique_dates) < required_dates:
        raise ValueError(
            "Not enough dates for rolling validation."
        )

    windows = []

    for i in range(n_folds - 1, -1, -1):

        end_idx = (
            len(unique_dates)
            - 1
            - i * validation_days
        )

        start_idx = (
            end_idx
            - validation_days
            + 1
        )

        if start_idx <= 0:
            continue

        val_start = unique_dates.iloc[
            start_idx
        ]

        val_end = unique_dates.iloc[
            end_idx
        ]

        windows.append(
            (
                val_start,
                val_end,
            )
        )

    return windows


# ============================================================================
# ROLLING VALIDATION
# ============================================================================

def rolling_time_series_validation(
    feature_df: pd.DataFrame,
    n_folds: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological rolling validation.
    """

    df = (
        feature_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df = df.dropna(
        subset=[
            "demand_lag_90",
            "demand_rolling_mean_90",
        ]
    ).copy()

    windows = _make_validation_windows(
        df["Date"],
        validation_days=VALIDATION_DAYS,
        n_folds=n_folds,
    )

    fold_records = []
    prediction_records = []

    for fold_number, (
        val_start,
        val_end,
    ) in enumerate(
        windows,
        start=1,
    ):

        train_mask = (
            df["Date"] < val_start
        )

        val_mask = (
            (df["Date"] >= val_start)
            & (df["Date"] <= val_end)
        )

        train_df = df.loc[
            train_mask
        ].copy()

        val_df = df.loc[
            val_mask
        ].copy()

        if train_df.empty or val_df.empty:
            continue

        X_train = _prepare_X(
            train_df
        )

        y_train = train_df[
            TARGET_COLUMN
        ]

        X_val = _prepare_X(
            val_df
        )

        y_val = val_df[
            TARGET_COLUMN
        ]

        model = create_model()

        model.fit(
            X_train,
            y_train,
            categorical_feature=[
                c
                for c in CATEGORICAL_COLUMNS
                if c in X_train.columns
            ],
            eval_set=[
                (
                    X_val,
                    y_val,
                )
            ],
            callbacks=[
                early_stopping(
                    100,
                    verbose=False,
                )
            ],
        )

        predictions = _clip_predictions(
            model.predict(X_val)
        )

        model_metrics = _metrics(
            y_val,
            predictions,
        )

        baseline = (
            make_baseline_predictions(
                df,
                val_mask,
            )
        )

        for baseline_name in [
            "Naive Yesterday",
            "Seasonal Naive 7D",
            "Moving Average 7D",
        ]:

            baseline_values = (
                baseline[
                    baseline_name
                ]
                .astype(float)
                .values
            )

            valid = np.isfinite(
                baseline_values
            )

            if valid.sum() == 0:

                baseline_metrics = {
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                }

            else:

                baseline_metrics = _metrics(
                    y_val.iloc[
                        np.where(valid)[0]
                    ],
                    baseline_values[
                        valid
                    ],
                )

            fold_records.append(
                {
                    "Fold": fold_number,
                    "Validation Start": str(
                        val_start.date()
                    ),
                    "Validation End": str(
                        val_end.date()
                    ),
                    "Model": baseline_name,
                    **baseline_metrics,
                }
            )

        fold_records.append(
            {
                "Fold": fold_number,
                "Validation Start": str(
                    val_start.date()
                ),
                "Validation End": str(
                    val_end.date()
                ),
                "Model": "LightGBM V2",
                **model_metrics,
            }
        )

        fold_predictions = val_df[
            [
                "Date",
                "Store ID",
                "Product ID",
                TARGET_COLUMN,
            ]
        ].copy()

        # IMPORTANT:
        # Rename Demand -> Actual Demand here.
        fold_predictions = (
            fold_predictions.rename(
                columns={
                    TARGET_COLUMN:
                    "Actual Demand"
                }
            )
        )

        fold_predictions[
            "Predicted Demand"
        ] = predictions

        fold_predictions[
            "Fold"
        ] = fold_number

        fold_predictions[
            "Model"
        ] = "LightGBM V2"

        prediction_records.append(
            fold_predictions
        )

    fold_metrics = pd.DataFrame(
        fold_records
    )

    if prediction_records:
        predictions = pd.concat(
            prediction_records,
            ignore_index=True,
        )
    else:
        predictions = pd.DataFrame()

    return (
        fold_metrics,
        predictions,
    )


# ============================================================================
# FINAL HOLDOUT EVALUATION
# ============================================================================

def evaluate_final_holdout(
    feature_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Evaluate V2 using the final chronological 30-day holdout.
    """

    df = feature_df.copy()

    max_date = df["Date"].max()

    validation_start = (
        max_date
        - pd.Timedelta(
            days=VALIDATION_DAYS - 1
        )
    )

    train_df = df[
        df["Date"] < validation_start
    ].copy()

    val_df = df[
        df["Date"] >= validation_start
    ].copy()

    train_df = train_df.dropna(
        subset=[
            "demand_lag_90",
            "demand_rolling_mean_90",
        ]
    )

    val_df = val_df.dropna(
        subset=[
            "demand_lag_90",
            "demand_rolling_mean_90",
        ]
    )

    X_train = _prepare_X(
        train_df
    )

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_val = _prepare_X(
        val_df
    )

    y_val = val_df[
        TARGET_COLUMN
    ]

    model = create_model()

    model.fit(
        X_train,
        y_train,
        categorical_feature=[
            c
            for c in CATEGORICAL_COLUMNS
            if c in X_train.columns
        ],
        eval_set=[
            (
                X_val,
                y_val,
            )
        ],
        callbacks=[
            early_stopping(
                100,
                verbose=False,
            )
        ],
    )

    predictions = _clip_predictions(
        model.predict(X_val)
    )

    # ------------------------------------------------------------
    # IMPORTANT FIX
    # Convert Demand -> Actual Demand
    # ------------------------------------------------------------

    result = val_df[
        [
            "Date",
            "Store ID",
            "Product ID",
            TARGET_COLUMN,
        ]
    ].copy()

    result = result.rename(
        columns={
            TARGET_COLUMN:
            "Actual Demand"
        }
    )

    result[
        "Predicted Demand"
    ] = predictions

    result[
        "Residual"
    ] = (
        result["Actual Demand"]
        - result["Predicted Demand"]
    )

    result[
        "Absolute Error"
    ] = result[
        "Residual"
    ].abs()

    result["APE"] = np.where(
        result["Actual Demand"].abs()
        > 1e-8,
        (
            result[
                "Absolute Error"
            ]
            / result[
                "Actual Demand"
            ].abs()
            * 100
        ),
        np.nan,
    )

    metrics = _metrics(
        y_val,
        predictions,
    )

    baseline_predictions = (
        val_df["demand_lag_1"]
        .values
    )

    baseline_metrics = _metrics(
        y_val,
        baseline_predictions,
    )

    return {
        "metrics": metrics,
        "baseline_metrics":
            baseline_metrics,
        "predictions": result,
        "validation_start":
            str(
                validation_start.date()
            ),
        "validation_end":
            str(
                max_date.date()
            ),
        "best_iteration":
            getattr(
                model,
                "best_iteration_",
                None,
            ),
    }


# ============================================================================
# FINAL MODELS
# ============================================================================

def train_final_models(
    feature_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Train:
    - P50 point model
    - P10 lower quantile model
    - P90 upper quantile model
    """

    df = feature_df.copy()

    df = df.dropna(
        subset=[
            "demand_lag_90",
            "demand_rolling_mean_90",
        ]
    ).reset_index(drop=True)

    X = _prepare_X(df)

    y = df[
        TARGET_COLUMN
    ]

    categorical_features = [
        c
        for c in CATEGORICAL_COLUMNS
        if c in X.columns
    ]

    # ------------------------------------------------------------
    # P50 / Point Forecast
    # ------------------------------------------------------------

    point_model = create_model(
        objective="regression",
        n_estimators=1800,
    )

    point_model.fit(
        X,
        y,
        categorical_feature=
            categorical_features,
    )

    # ------------------------------------------------------------
    # P10
    # ------------------------------------------------------------

    p10_model = create_model(
        objective="quantile",
        alpha=0.10,
        n_estimators=1400,
    )

    p10_model.fit(
        X,
        y,
        categorical_feature=
            categorical_features,
    )

    # ------------------------------------------------------------
    # P90
    # ------------------------------------------------------------

    p90_model = create_model(
        objective="quantile",
        alpha=0.90,
        n_estimators=1400,
    )

    p90_model.fit(
        X,
        y,
        categorical_feature=
            categorical_features,
    )

    # ------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------

    joblib.dump(
        point_model,
        MODEL_PATH,
    )

    joblib.dump(
        p10_model,
        LOWER_MODEL_PATH,
    )

    joblib.dump(
        p90_model,
        UPPER_MODEL_PATH,
    )

    return {
        "point_model":
            point_model,
        "p10_model":
            p10_model,
        "p90_model":
            p90_model,
    }


# ============================================================================
# LOAD MODELS
# ============================================================================

def _load_models():
    """
    Load trained V2 models.
    """

    if not MODEL_PATH.exists():
        train_model()

    point_model = joblib.load(
        MODEL_PATH
    )

    if LOWER_MODEL_PATH.exists():
        p10_model = joblib.load(
            LOWER_MODEL_PATH
        )
    else:
        p10_model = point_model

    if UPPER_MODEL_PATH.exists():
        p90_model = joblib.load(
            UPPER_MODEL_PATH
        )
    else:
        p90_model = point_model

    return (
        point_model,
        p10_model,
        p90_model,
    )


# ============================================================================
# SERIES HELPER
# ============================================================================

def _get_latest_series(
    raw_df: pd.DataFrame,
    store_id: str,
    product_id: str,
) -> pd.DataFrame:
    """
    Get one Store × Product historical series.
    """

    series = raw_df[
        (
            raw_df["Store ID"]
            .astype(str)
            == str(store_id)
        )
        &
        (
            raw_df["Product ID"]
            .astype(str)
            == str(product_id)
        )
    ].copy()

    if series.empty:
        raise ValueError(
            f"No data found for "
            f"Store={store_id}, "
            f"Product={product_id}"
        )

    return series.sort_values(
        "Date"
    ).copy()


# ============================================================================
# FUTURE FORECAST
# ============================================================================

def forecast_next_days(
    store_id: str,
    product_id: str,
    horizon: int = FORECAST_HORIZON,
    price: Optional[float] = None,
    discount: Optional[float] = None,
    promotion: Optional[object] = None,
) -> pd.DataFrame:
    """
    Recursively forecast one Store × Product series.

    Price, discount and promotion can be supplied as planned future
    assumptions. Other future-unknown variables are carried forward only
    to keep the dataframe structurally complete; they are not model inputs.
    """

    if horizon < 1:
        raise ValueError(
            "horizon must be >= 1"
        )

    raw = load_data(
        DATA_PATH
    )

    series = _get_latest_series(
        raw,
        store_id,
        product_id,
    )

    (
        point_model,
        p10_model,
        p90_model,
    ) = _load_models()

    history = series.copy()

    last_date = history[
        "Date"
    ].max()

    last_row = history.iloc[
        -1
    ].copy()

    if price is None:
        price = float(
            last_row["Price"]
        )

    if discount is None:
        discount = float(
            last_row["Discount"]
        )

    if promotion is None:
        promotion = last_row[
            "Promotion"
        ]

    predictions = []

    for step in range(
        1,
        horizon + 1,
    ):

        future_date = (
            last_date
            + pd.Timedelta(
                days=step
            )
        )

        future_row = (
            last_row.copy()
        )

        future_row[
            "Date"
        ] = future_date

        future_row[
            "Price"
        ] = price

        future_row[
            "Discount"
        ] = discount

        future_row[
            "Promotion"
        ] = promotion

        # Future-unknown variables are carried forward but are NOT
        # part of MODEL_FEATURES.
        future_row[
            "Inventory Level"
        ] = last_row[
            "Inventory Level"
        ]

        future_row[
            "Units Ordered"
        ] = last_row[
            "Units Ordered"
        ]

        future_row[
            "Units Sold"
        ] = last_row[
            "Units Sold"
        ]

        future_row[
            "Weather Condition"
        ] = last_row[
            "Weather Condition"
        ]

        future_row[
            "Epidemic"
        ] = last_row[
            "Epidemic"
        ]

        future_row[
            "Competitor Pricing"
        ] = last_row[
            "Competitor Pricing"
        ]

        # Placeholder demand.
        future_row[
            TARGET_COLUMN
        ] = np.nan

        temp = pd.concat(
            [
                history,
                pd.DataFrame(
                    [future_row]
                ),
            ],
            ignore_index=True,
        )

        engineered = build_features(
            temp,
            drop_initial_missing=False,
        )

        current_features = (
            engineered.iloc[
                [-1]
            ].copy()
        )

        X_future = _prepare_X(
            current_features
        )

        # Fill unexpected numeric missing values.
        for col in X_future.columns:

            if X_future[
                col
            ].isna().any():

                if pd.api.types.is_numeric_dtype(
                    X_future[col]
                ):

                    X_future[
                        col
                    ] = X_future[
                        col
                    ].fillna(0)

        # ------------------------------------------------------------
        # Predictions
        # ------------------------------------------------------------

        p50 = float(
            _clip_predictions(
                point_model.predict(
                    X_future
                )
            )[0]
        )

        p10 = float(
            _clip_predictions(
                p10_model.predict(
                    X_future
                )
            )[0]
        )

        p90 = float(
            _clip_predictions(
                p90_model.predict(
                    X_future
                )
            )[0]
        )

        # Guarantee valid quantile ordering.
        lower = min(
            p10,
            p50,
            p90,
        )

        upper = max(
            p10,
            p50,
            p90,
        )

        predictions.append(
            {
                "Date":
                    future_date,
                "Store ID":
                    store_id,
                "Product ID":
                    product_id,
                "Forecast Type":
                    "Future",
                "Predicted Demand":
                    p50,
                "P10 Demand":
                    lower,
                "P50 Demand":
                    p50,
                "P90 Demand":
                    upper,
            }
        )

        # Recursive prediction becomes historical demand.
        future_row[
            TARGET_COLUMN
        ] = p50

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [future_row]
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(
        predictions
    )


# ============================================================================
# FORECAST ALL STORE × PRODUCT SERIES
# ============================================================================

def forecast_all_products(
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Generate forecasts for all Store × Product combinations.
    """

    raw = load_data(
        DATA_PATH
    )

    outputs = []

    combinations = (
        raw[
            [
                "Store ID",
                "Product ID",
            ]
        ]
        .drop_duplicates()
    )

    for (
        store_id,
        product_id,
    ) in combinations.itertuples(
        index=False,
        name=None,
    ):

        outputs.append(
            forecast_next_days(
                store_id=store_id,
                product_id=product_id,
                horizon=horizon,
            )
        )

    if not outputs:
        return pd.DataFrame()

    return pd.concat(
        outputs,
        ignore_index=True,
    )


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_model() -> Dict[str, object]:
    """
    Complete Phase-2 forecasting pipeline.
    """

    print("=" * 78)
    print("PHASE 2 - FORECASTING V2")
    print("=" * 78)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    raw = load_data(
        DATA_PATH
    )

    feature_df = build_features(
        raw,
        drop_initial_missing=False,
    )

    print(
        f"Raw shape: {raw.shape}"
    )

    print(
        f"Feature shape: "
        f"{feature_df.shape}"
    )

    print(
        f"V2 model features: "
        f"{len(MODEL_FEATURES)}"
    )

    # ------------------------------------------------------------
    # Rolling validation
    # ------------------------------------------------------------

    print("-" * 78)
    print(
        "Running rolling "
        "time-series validation..."
    )

    cv_metrics, cv_predictions = (
        rolling_time_series_validation(
            feature_df,
            n_folds=4,
        )
    )

    cv_metrics.to_csv(
        CV_OUTPUT_PATH,
        index=False,
    )

    if not cv_metrics.empty:

        print(
            cv_metrics.groupby(
                "Model"
            )[
                [
                    "MAE",
                    "RMSE",
                    "MAPE",
                ]
            ]
            .mean()
            .round(4)
        )

    # ------------------------------------------------------------
    # Final holdout
    # ------------------------------------------------------------

    print("-" * 78)
    print(
        "Running final 30-day "
        "chronological holdout..."
    )

    holdout = evaluate_final_holdout(
        feature_df
    )

    holdout[
        "predictions"
    ].to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print(
        "V2 holdout metrics:"
    )

    print(
        pd.Series(
            holdout[
                "metrics"
            ]
        )
        .round(4)
        .to_string()
    )

    print(
        "Seasonal/naive "
        "baseline metrics:"
    )

    print(
        pd.Series(
            holdout[
                "baseline_metrics"
            ]
        )
        .round(4)
        .to_string()
    )

    # ------------------------------------------------------------
    # Train final models
    # ------------------------------------------------------------

    print("-" * 78)
    print(
        "Training final point "
        "+ quantile models..."
    )

    models = train_final_models(
        feature_df
    )

    # ------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------

    point_model = models[
        "point_model"
    ]

    importance = pd.DataFrame(
        {
            "Feature":
                MODEL_FEATURES,
            "Importance":
                point_model
                .feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    # ------------------------------------------------------------
    # Future forecasts
    # ------------------------------------------------------------

    print("-" * 78)
    print(
        "Generating 7-day "
        "future forecasts..."
    )

    future = forecast_all_products(
        horizon=FORECAST_HORIZON
    )

    future_output = future[
        [
            "Date",
            "Store ID",
            "Product ID",
            "Forecast Type",
            "Predicted Demand",
            "P10 Demand",
            "P50 Demand",
            "P90 Demand",
        ]
    ].copy()

    future_output[
        "Actual Demand"
    ] = np.nan

    # ------------------------------------------------------------
    # Validation output
    # ------------------------------------------------------------

    validation_output = (
        holdout[
            "predictions"
        ].copy()
    )

    validation_output[
        "Forecast Type"
    ] = "Validation"

    validation_output[
        "P10 Demand"
    ] = np.nan

    validation_output[
        "P50 Demand"
    ] = validation_output[
        "Predicted Demand"
    ]

    validation_output[
        "P90 Demand"
    ] = np.nan

    # ------------------------------------------------------------
    # Combine validation + future
    # ------------------------------------------------------------

    validation_output = validation_output[
        [
            "Date",
            "Store ID",
            "Product ID",
            "Forecast Type",
            "Actual Demand",
            "Predicted Demand",
            "P10 Demand",
            "P50 Demand",
            "P90 Demand",
        ]
    ]

    future_output = future_output[
        [
            "Date",
            "Store ID",
            "Product ID",
            "Forecast Type",
            "Actual Demand",
            "Predicted Demand",
            "P10 Demand",
            "P50 Demand",
            "P90 Demand",
        ]
    ]

    combined = pd.concat(
        [
            validation_output,
            future_output,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        FORECAST_OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    metadata = {
        "model_version":
            "V2",

        "model_type":
            "LightGBM Global Demand Forecasting Model",

        "target":
            TARGET_COLUMN,

        "feature_count":
            len(MODEL_FEATURES),

        "feature_columns":
            MODEL_FEATURES,

        "categorical_columns":
            CATEGORICAL_COLUMNS,

        "excluded_future_unknown_features":
            sorted(
                DYNAMIC_FUTURE_UNKNOWN
            ),

        "validation_method":
            "4-fold chronological rolling validation",

        "validation_days":
            VALIDATION_DAYS,

        "forecast_horizon":
            FORECAST_HORIZON,

        "final_holdout": {
            "validation_start":
                holdout[
                    "validation_start"
                ],

            "validation_end":
                holdout[
                    "validation_end"
                ],

            "MAE":
                holdout[
                    "metrics"
                ]["MAE"],

            "RMSE":
                holdout[
                    "metrics"
                ]["RMSE"],

            "MAPE":
                holdout[
                    "metrics"
                ]["MAPE"],

            "baseline_MAE":
                holdout[
                    "baseline_metrics"
                ]["MAE"],

            "baseline_RMSE":
                holdout[
                    "baseline_metrics"
                ]["RMSE"],

            "baseline_MAPE":
                holdout[
                    "baseline_metrics"
                ]["MAPE"],

            "best_iteration":
                holdout[
                    "best_iteration"
                ],
        },

        "feature_importance":
            importance.to_dict(
                orient="records"
            ),

        "quantile_forecasts":
            True,

        "artifacts": {
            "point_model":
                str(MODEL_PATH),

            "p10_model":
                str(LOWER_MODEL_PATH),

            "p90_model":
                str(UPPER_MODEL_PATH),

            "forecast_output":
                str(
                    FORECAST_OUTPUT_PATH
                ),

            "validation_output":
                str(
                    VALIDATION_OUTPUT_PATH
                ),

            "rolling_cv_output":
                str(
                    CV_OUTPUT_PATH
                ),
        },
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------

    print("-" * 78)

    print(
        f"Saved point model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved P10 model:   "
        f"{LOWER_MODEL_PATH}"
    )

    print(
        f"Saved P90 model:   "
        f"{UPPER_MODEL_PATH}"
    )

    print(
        f"Saved forecasts:   "
        f"{FORECAST_OUTPUT_PATH}"
    )

    print(
        f"Saved CV results:  "
        f"{CV_OUTPUT_PATH}"
    )

    print(
        f"Saved metadata:    "
        f"{METADATA_PATH}"
    )

    print("=" * 78)
    print(
        "PHASE 2 TRAINING COMPLETE"
    )
    print("=" * 78)

    return {
        "metadata":
            metadata,

        "cv_metrics":
            cv_metrics,

        "holdout":
            holdout,

        "future_forecasts":
            future,

        "feature_importance":
            importance,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_model()