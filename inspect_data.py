"""Reproduce the initial freight-data exploration without training or cleaning data.

Run from the repository root: python inspect_data.py
The original audit is reused from train.py; extra summaries reproduce the
interactive pandas checks used before modeling. Only artifacts/eda is written.
"""

import json
from pathlib import Path
import pandas as pd
from train import ROOT, audit, load_data


def main():
    # This also checks unique IDs, template coverage, positive targets/distance,
    # parseable dates, and development/prediction date separation.
    dev, prediction, template, december = load_data()
    output = ROOT / "artifacts" / "eda"
    output.mkdir(parents=True, exist_ok=True)
    original_audit = audit(dev, prediction)
    (output / "audit.json").write_text(json.dumps(original_audit, indent=2))

    schemas, numeric, categories, months, correlations, coordinates = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for name, frame in [
        ("development", dev),
        ("prediction", prediction),
        ("template", template),
        ("december", december),
    ]:
        for column in frame:
            schemas.append(
                {
                    "dataset": name,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "rows": len(frame),
                    "missing": int(frame[column].isna().sum()),
                    "missing_pct": float(100 * frame[column].isna().mean()),
                    "distinct_nonmissing": int(frame[column].nunique()),
                }
            )
        stats = frame.describe(
            percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99], include="number"
        )
        stats = stats.T.rename_axis("column").reset_index()
        stats.insert(0, "dataset", name)
        numeric.append(stats)
        # Do not interpret the empty template or December's generated predictions
        # as labeled training data. Target analysis below is development-only.
        if name not in ["development", "prediction"]:
            continue
        for column in ["pickup", "delivery", "equipment"]:
            counts = (
                frame[column]
                .value_counts(dropna=False)
                .rename_axis("value")
                .reset_index(name="rows")
            )
            counts.insert(0, "column", column)
            counts.insert(0, "dataset", name)
            categories.append(counts)
        work = frame.copy()
        if name == "development":
            work["rate_per_mile"] = work.posted_rate / work.distance
        work["month"] = work.date.dt.strftime("%Y-%m")
        for month, group in work.groupby("month"):
            row = {"dataset": name, "month": month, "rows": len(group)}
            for column in [
                "distance",
                "weight",
                "market_index",
                "quote_signal",
                "posted_rate",
                "rate_per_mile",
            ]:
                if column in group:
                    for statistic in ["mean", "median", "std"]:
                        row[f"{column}_{statistic}"] = float(
                            getattr(group[column], statistic)()
                        )
            if name == "development":
                row["quote_vs_rate_per_mile_correlation"] = float(
                    group.quote_signal.corr(group.rate_per_mile)
                )
            months.append(row)
        corr = (
            work.select_dtypes(include="number")
            .corr()
            .rename_axis("feature")
            .reset_index()
        )
        corr.insert(0, "dataset", name)
        correlations.append(corr)
        for endpoint in ["pickup", "delivery"]:
            latitude, longitude = f"{endpoint}_lat", f"{endpoint}_lon"
            counts = frame.groupby(endpoint)[[latitude, longitude]].nunique()
            counts = counts.rename(
                columns={
                    latitude: "distinct_latitudes",
                    longitude: "distinct_longitudes",
                }
            )
            counts = counts.rename_axis("city").reset_index()
            counts.insert(0, "endpoint", endpoint)
            counts.insert(0, "dataset", name)
            coordinates.append(counts)

    pd.DataFrame(schemas).to_csv(output / "schema.csv", index=False)
    pd.concat(numeric, ignore_index=True).to_csv(
        output / "numeric_summary.csv", index=False
    )
    pd.concat(categories, ignore_index=True).to_csv(
        output / "category_counts.csv", index=False
    )
    pd.DataFrame(months).to_csv(output / "monthly_summary.csv", index=False)
    pd.concat(correlations, ignore_index=True).to_csv(
        output / "numeric_correlations.csv", index=False
    )
    pd.concat(coordinates, ignore_index=True).to_csv(
        output / "coordinate_consistency.csv", index=False
    )

    lines = ["Freight data exploration - no fitting or input modification", ""]
    for name in ["development", "prediction"]:
        item = original_audit[name]
        lines += [
            f"{name}: {item['rows']:,} rows; {item['date_min']} to {item['date_max']}",
            f"  Missing cells by column: {item['missing']}",
            f"  Non-positive weights: {item['negative_weight']}",
            f"  Duplicate full rows excluding load_id: {item['duplicates_excluding_id']}",
        ]
    lines += ["", "Prediction-only categories:"]
    for column in ["pickup", "delivery", "equipment"]:
        lines.append(f"  {column}: {original_audit[f'unseen_{column}']}")
    lines += [
        f"Loads with either city unseen: {original_audit['unseen_either_city_rows']:,}",
        f"Loads with unseen directional lane: {original_audit['unseen_lane_rows']:,}",
        "",
        "Development target quantiles:",
        str(original_audit["target_quantiles"]),
        "",
        "Monthly quote_signal vs rate-per-mile correlations:",
        str(original_audit["quote_correlations_by_month"]),
        "",
        "Example invalid development weights (at most five rows):",
        dev.loc[dev.weight <= 0, ["load_id", "weight"]].head(5).to_string(index=False),
        "",
        "Input contracts passed: unique IDs, complete template coverage, valid dates,",
        "positive distances and labeled rates, and future-only prediction period.",
        "Missing-weight and market-index values are reported, not filled here.",
        "Correlation is a diagnostic of association; it does not establish causation or prove leakage.",
        "Full-development target summaries reproduce the original exploratory work; they are not a blind evaluation.",
    ]
    text = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(text)
    print(text)
    print(f"Wrote reproducible summaries to {output}")


if __name__ == "__main__":
    main()
