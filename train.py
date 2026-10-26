"""Reproducible forward validation, final refit, and ID-aligned prediction export."""

from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from freight_model import RateModel, EquipmentBaseline

ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    "equipment_rpm": lambda: EquipmentBaseline(),
    "ridge_log": lambda: RateModel(kind="ridge"),
    "robust_log": lambda: RateModel(kind="robust"),
    "robust_log_quote": lambda: RateModel(kind="robust", quote=True),
    "boosting_log": lambda: RateModel(kind="boosting"),
}


def data_path(stem):
    for name in [stem, stem.replace("_", "-")]:
        path = ROOT / "data" / f"{name}.csv"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "wape_pct": float(100 * np.abs(y - p).sum() / y.sum()),
        "mape_pct": float(100 * np.mean(np.abs(y - p) / y)),
        "bias": float(np.mean(p - y)),
        "r2": float(r2_score(y, p)),
        "n": len(y),
    }


def load_data():
    dev = pd.read_csv(data_path("train_test"))
    val = pd.read_csv(data_path("validation"))
    template = pd.read_csv(data_path("validation_predictions_template"))
    december = pd.read_csv(data_path("december_chart_inputs"))
    for frame in [dev, val, template]:
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError("Missing or duplicate load IDs")
    if len(val) != 12000 or set(val.load_id) != set(template.load_id):
        raise ValueError("Validation/template IDs must match exactly with 12,000 rows")
    if set(dev.load_id) & set(val.load_id):
        raise ValueError("Overlapping development and prediction IDs")
    y = pd.to_numeric(dev.posted_rate, errors="coerce")
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Invalid training target")
    for frame in [dev, val, december]:
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if (frame.distance <= 0).any():
            raise ValueError("Invalid distance")
    if dev.date.max() >= val.date.min():
        raise ValueError("Expected future-only validation data")
    return dev, val, template, december


def audit(dev, val):
    result = {}
    for name, frame in [("development", dev), ("prediction", val)]:
        result[name] = {
            "rows": len(frame),
            "date_min": str(frame.date.min().date()),
            "date_max": str(frame.date.max().date()),
            "missing": frame.isna().sum().astype(int).to_dict(),
            "negative_weight": int((frame.weight <= 0).sum()),
            "duplicates_excluding_id": int(
                frame.drop(columns="load_id").duplicated().sum()
            ),
            "monthly_counts": frame.groupby(frame.date.dt.strftime("%Y-%m"))
            .size()
            .astype(int)
            .to_dict(),
        }
    for col in ["pickup", "delivery", "equipment"]:
        unseen = set(val[col]) - set(dev[col])
        result[f"unseen_{col}"] = {
            "values": sorted(unseen),
            "rows": int(val[col].isin(unseen).sum()),
        }
    result["unseen_either_city_rows"] = int(
        (~val.pickup.isin(dev.pickup) | ~val.delivery.isin(dev.delivery)).sum()
    )
    result["unseen_lane_rows"] = int(
        (~(val.pickup + "|" + val.delivery).isin(dev.pickup + "|" + dev.delivery)).sum()
    )
    result["quote_correlations_by_month"] = {
        str(m): float(g.quote_signal.corr(g.posted_rate / g.distance))
        for m, g in dev.groupby(dev.date.dt.month)
    }
    result["target_quantiles"] = dev.posted_rate.quantile(
        [0, 0.01, 0.5, 0.95, 0.99, 1]
    ).to_dict()
    result["input_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            data_path("train_test"),
            data_path("validation"),
            data_path("validation_predictions_template"),
        ]
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    dev, val, template, december = load_data()
    if args.predict_only:
        bundle = joblib.load(out / "freight_model.joblib")
    else:
        audit_result = audit(dev, val)
        (out / "data_audit.json").write_text(json.dumps(audit_result, indent=2))
        rows = []
        folds = [
            ("May-Jun", "2025-05-01", "2025-07-01"),
            ("Jul-Aug", "2025-07-01", "2025-09-01"),
        ]
        for fold, start, end in folds:
            tr = dev[dev.date < pd.Timestamp(start)]
            te = dev[(dev.date >= pd.Timestamp(start)) & (dev.date < pd.Timestamp(end))]
            for name, factory in CANDIDATES.items():
                model = factory().fit(tr, tr.posted_rate)
                p = model.predict(te)
                row = {
                    "fold": fold,
                    "model": name,
                    "train_n": len(tr),
                    **metrics(te.posted_rate, p),
                }
                rows.append(row)
                print(row, flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(out / "model_comparison.csv", index=False)
        # Equal weight to the two future windows; primary objective is dollar MAE.
        ranking = results.groupby("model").mae.mean().sort_values()
        best = ranking.index[0]
        # Choose the simplest eligible regression within 2% of the best mean MAE.
        eligible = set(ranking[ranking <= ranking.iloc[0] * 1.02].index)
        selected = next(name for name in CANDIDATES if name in eligible)
        print("Selected:", selected, flush=True)
        train = dev[dev.date < "2025-09-01"]
        test = dev[dev.date >= "2025-09-01"]
        model = CANDIDATES[selected]().fit(train, train.posted_rate)
        p = model.predict(test)
        holdout = {
            "model": selected,
            "train_n": len(train),
            "train_end": "2025-08-31",
            "test_start": "2025-09-01",
            "test_end": "2025-10-31",
            **metrics(test.posted_rate, p),
        }
        baseline = EquipmentBaseline().fit(train, train.posted_rate)
        holdout["baseline"] = metrics(test.posted_rate, baseline.predict(test))
        holdout["by_month"] = {
            str(month): metrics(
                test.loc[index, "posted_rate"],
                pd.Series(p, index=test.index).loc[index],
            )
            for month, index in test.groupby(test.date.dt.month).groups.items()
        }
        holdout["by_equipment"] = {
            str(eq): metrics(
                test.loc[index, "posted_rate"],
                pd.Series(p, index=test.index).loc[index],
            )
            for eq, index in test.groupby("equipment").groups.items()
        }
        unseen = ~test.pickup.isin(train.pickup) | ~test.delivery.isin(train.delivery)
        if unseen.any():
            holdout["unseen_city"] = metrics(test.loc[unseen, "posted_rate"], p[unseen])
        # December has no market index or quote signal: train and test the identical reduced schema.
        core_kind = "ridge" if selected == "ridge_log" else "robust"
        core = RateModel(kind=core_kind, core=True).fit(train, train.posted_rate)
        core_p = core.predict(
            test.drop(
                columns=[
                    "market_index",
                    "quote_signal",
                    "pickup_lat",
                    "pickup_lon",
                    "delivery_lat",
                    "delivery_lon",
                ]
            )
        )
        holdout["december_schema"] = metrics(test.posted_rate, core_p)
        holdout["core_kind"] = core_kind
        pd.DataFrame(
            {
                "load_id": test.load_id,
                "date": test.date,
                "equipment": test.equipment,
                "distance": test.distance,
                "actual": test.posted_rate,
                "predicted": p,
                "core_predicted": core_p,
            }
        ).to_csv(out / "holdout_predictions.csv", index=False)
        (out / "holdout_metrics.json").write_text(json.dumps(holdout, indent=2))
        print("Holdout:", json.dumps(holdout, indent=2), flush=True)
        bundle = {
            "main": CANDIDATES[selected]().fit(dev, dev.posted_rate),
            "december": RateModel(kind=core_kind, core=True).fit(dev, dev.posted_rate),
            "model_name": selected,
            "python": platform.python_version(),
            "train_rows": len(dev),
        }
        joblib.dump(bundle, out / "freight_model.joblib", compress=3)
    predictions = pd.DataFrame(
        {"load_id": val.load_id, "predicted_rate": bundle["main"].predict(val)}
    )
    completed = template[["load_id"]].merge(
        predictions, on="load_id", how="left", validate="one_to_one", sort=False
    )
    if (
        not np.isfinite(completed.predicted_rate).all()
        or (completed.predicted_rate <= 0).any()
    ):
        raise ValueError("Invalid predictions")
    completed.to_csv(
        ROOT / "validation_predictions.csv", index=False, float_format="%.2f"
    )
    december["predicted_rate"] = bundle["december"].predict(december)
    december.to_csv(
        data_path("december_chart_inputs"),
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.2f",
    )
    print(
        "Wrote",
        len(completed),
        "validation predictions and",
        len(december),
        "December predictions.",
        flush=True,
    )


if __name__ == "__main__":
    main()
