"""Train-only preprocessing and simple, robust freight-rate estimators."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

NUMERIC = [
    "distance",
    "weight",
    "market_index",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]


class Features(BaseEstimator, TransformerMixin):
    """Recover city coordinates from training only; never infer missing market signals."""

    def __init__(self, quote=False, core=False):
        self.quote = quote
        self.core = core

    def fit(self, X, y=None):
        self.coordinates_ = {}
        for city, lat, lon in [
            ("pickup", "pickup_lat", "pickup_lon"),
            ("delivery", "delivery_lat", "delivery_lon"),
        ]:
            if lat in X and lon in X:
                frame = X[[city, lat, lon]].rename(
                    columns={city: "city", lat: "lat", lon: "lon"}
                )
                self.coordinates_[city] = frame.groupby("city")[["lat", "lon"]].median()
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in ["pickup", "delivery", "equipment"]:
            out[col] = X[col].fillna("Unknown").astype(str).str.strip()
        date = pd.to_datetime(X["date"], errors="raise")
        out["weekday"] = date.dt.dayofweek.astype(str)
        angle = 2 * np.pi * (date.dt.dayofyear - 1) / 365.25
        # A single annual harmonic limits extrapolation variance with <1 year of labels.
        out["year_sin"] = np.sin(angle)
        out["year_cos"] = np.cos(angle)
        for col in NUMERIC:
            out[col] = pd.to_numeric(X[col], errors="coerce") if col in X else np.nan
        out.loc[out.weight <= 0, "weight"] = np.nan
        out.loc[out.distance <= 0, "distance"] = np.nan
        for city, lat, lon in [
            ("pickup", "pickup_lat", "pickup_lon"),
            ("delivery", "delivery_lat", "delivery_lon"),
        ]:
            if city in self.coordinates_:
                for col, axis in [(lat, "lat"), (lon, "lon")]:
                    out[col] = out[col].fillna(
                        out[city].map(self.coordinates_[city][axis])
                    )
        out["log_distance"] = np.log(out.distance)
        out["inverse_distance"] = 1 / out.distance
        out["log_weight"] = np.log(out.weight)
        out["weight_missing"] = out.weight.isna().astype(float)
        if self.quote:
            out["quote_signal"] = pd.to_numeric(
                X.get("quote_signal", pd.Series(np.nan, index=X.index)), errors="coerce"
            )
        if self.core:
            out = out.drop(columns="market_index")
        return out.replace([np.inf, -np.inf], np.nan)


def preprocessing(quote=False, core=False, dense=False):
    nums = [
        "log_distance",
        "inverse_distance",
        "weight",
        "log_weight",
        "weight_missing",
        "year_sin",
        "year_cos",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]
    if not core:
        nums += ["market_index"]
    if quote:
        nums += ["quote_signal"]
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    spline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=5, degree=2, extrapolation="linear")),
            ("scale", StandardScaler()),
        ]
    )
    return Pipeline(
        [
            ("features", Features(quote=quote, core=core)),
            (
                "encode",
                ColumnTransformer(
                    [
                        ("numeric", numeric, nums),
                        ("distance_shape", spline, ["log_distance"]),
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=not dense
                            ),
                            ["pickup", "delivery", "equipment", "weekday"],
                        ),
                    ],
                    sparse_threshold=0 if dense else 0.3,
                ),
            ),
        ]
    )


class RateModel:
    """Log-rate regression: robust IRLS is a small series of weighted Ridge fits."""

    def __init__(self, kind="robust", quote=False, core=False):
        self.kind, self.quote, self.core = kind, quote, core

    def fit(self, X, y):
        self.preprocessor_ = preprocessing(self.quote, self.core, dense=True)
        z = self.preprocessor_.fit_transform(X)
        ylog = np.log(np.asarray(y))
        if self.kind == "boosting":
            self.estimator_ = HistGradientBoostingRegressor(
                max_iter=200,
                max_leaf_nodes=15,
                learning_rate=0.07,
                l2_regularization=10,
                min_samples_leaf=60,
                early_stopping=False,
                random_state=42,
            )
            self.estimator_.fit(z, ylog)
        else:
            self.estimator_ = Ridge(alpha=10, solver="cholesky").fit(z, ylog)
            if self.kind == "robust":
                for _ in range(12):
                    residual = ylog - self.estimator_.predict(z)
                    scale = max(
                        1.4826 * np.median(np.abs(residual - np.median(residual))), 0.01
                    )
                    weights = np.minimum(
                        1, 1.35 * scale / np.maximum(np.abs(residual), 1e-10)
                    )
                    self.estimator_.fit(z, ylog, sample_weight=weights)
        self.n_train_ = len(X)
        return self

    def predict(self, X):
        return np.maximum(
            1, np.exp(self.estimator_.predict(self.preprocessor_.transform(X)))
        )


class EquipmentBaseline:
    def fit(self, X, y):
        rpm = pd.Series(np.asarray(y) / X.distance.to_numpy(), index=X.index)
        self.rates_ = rpm.groupby(X.equipment).median().to_dict()
        self.default_ = rpm.median()
        return self

    def predict(self, X):
        return (
            X.distance.to_numpy()
            * X.equipment.map(self.rates_).fillna(self.default_).to_numpy()
        )
