"""Post-selection diagnostics; these results do not choose or tune models."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train import load_data, metrics, CANDIDATES

out = Path("artifacts")
dev, val, _, _ = load_data()
holdout = pd.read_csv(out / "holdout_predictions.csv")
summary = json.loads((out / "holdout_metrics.json").read_text())
result = {}
# Day-cluster bootstrap respects dependence among loads on the same date.
daily = (
    holdout.assign(error=abs(holdout.actual - holdout.predicted))
    .groupby("date")
    .agg(error_sum=("error", "sum"), n=("error", "size"))
)
rng = np.random.default_rng(42)
indices = rng.integers(0, len(daily), size=(2000, len(daily)))
boot = daily.error_sum.to_numpy()[indices].sum(axis=1) / daily.n.to_numpy()[
    indices
].sum(axis=1)
result["day_cluster_bootstrap_mae_95pct"] = np.quantile(boot, [0.025, 0.975]).tolist()
error = np.abs(holdout.actual - holdout.predicted)
result["largest_1pct_error_share_pct"] = float(
    100 * error.nlargest(int(np.ceil(len(error) * 0.01))).sum() / error.sum()
)
result["median_absolute_error"] = float(error.median())
result["p90_absolute_error"] = float(error.quantile(0.9))
result["p95_absolute_error"] = float(error.quantile(0.95))
# Deterministically hide 8 TRAINING cities in an earlier rolling fold.
cities = sorted(
    set(dev.loc[dev.date < "2025-07-01", "pickup"])
    | set(dev.loc[dev.date < "2025-07-01", "delivery"])
)
hidden = cities[::8]
tr = dev[
    (dev.date < "2025-07-01") & ~dev.pickup.isin(hidden) & ~dev.delivery.isin(hidden)
]
te = dev[(dev.date >= "2025-07-01") & (dev.date < "2025-09-01")]
stress = CANDIDATES[summary["model"]]().fit(tr, tr.posted_rate)
p = stress.predict(te)
new = te.pickup.isin(hidden) | te.delivery.isin(hidden)
result["city_stress"] = {
    "hidden_cities": hidden,
    "train_n": len(tr),
    "unseen_city": metrics(te.loc[new, "posted_rate"], p[new]),
    "seen_city": metrics(te.loc[~new, "posted_rate"], p[~new]),
}
model = joblib.load(out / "freight_model.joblib")["main"]
names = model.preprocessor_.named_steps["encode"].get_feature_names_out()
coef = pd.DataFrame({"feature": names, "coefficient_log_rate": model.estimator_.coef_})
coef.to_csv(out / "model_coefficients.csv", index=False)
result["quote_standardized_coefficient"] = (
    float(
        coef.loc[coef.feature == "numeric__quote_signal", "coefficient_log_rate"].iloc[
            0
        ]
    )
    if model.quote
    else None
)
result["feature_count"] = len(names)
# Save diagnostic results as JSON.
(out / "diagnostics.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
