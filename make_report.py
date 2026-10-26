"""Generate holdout diagnostic graphs from saved actual and predicted rates."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
out = ROOT / "artifacts"
p = pd.read_csv(out / "holdout_predictions.csv")

# Publish clear diagnostics with no clipped tails or unlabeled quantities.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), dpi=160)
axes[0].scatter(
    p.actual, p.predicted, s=3, alpha=0.25, color="#064A56", rasterized=True
)
limit = max(p.actual.max(), p.predicted.max())
axes[0].plot([0, limit], [0, limit], color="#A34B31", lw=1)
axes[0].set(
    xlabel="Actual posted rate ($)",
    ylabel="Predicted rate ($)",
    title="All holdout loads, including the tail",
)
err = np.abs(p.actual - p.predicted)
axes[1].hist(err, bins=np.linspace(0, err.quantile(0.99), 40), color="#064A56")
axes[1].set(
    xlabel="Absolute error ($)",
    ylabel="Loads (count)",
    title="Error distribution: bottom 99%",
)
axes[1].text(
    0.97,
    0.95,
    "Largest 1% omitted here;\nall loads included in metrics.",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    fontsize=8,
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
fig.suptitle(
    "Source: January-August training; September-October 2025 holdout", fontsize=10
)
fig.tight_layout()
fig.savefig(out / "holdout_diagnostics.png")
plt.close(fig)
