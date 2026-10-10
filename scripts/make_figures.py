"""Figure for the volatility benchmark: a reproduction of Fig. 7.

A grid of subplots compares the realized volatility of the S&P 500 (actual,
black, repeated in every panel) with the out-of-sample forecast of one model
per panel, drawn as plain line plots. The actual series sits at a lower
z-order so it never hides the model lines. QR1 joins the figure automatically
once its column exists in ``data/oos_classical.csv``; until then the classical
baselines carry it.

Input contract:
    data/oos_classical.csv    date, RV, y_log, HARX, ARMAX, AR3, RC[, QR1]
                              (model prediction columns are in log space)

Output:
    <out_dir>/fig7_forecast.png  plus a .alt.txt sidecar

Usage:
    .venv/bin/python scripts/make_figures.py --out-dir figures
    from scripts.make_figures import fig_forecast     # from the repo root
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paper Fig. 7 accent colour (sampled from the paper's PNG).
QR1_ORANGE = "#E06040"
BLACK = "#000000"

# Kept short on purpose so the plot stays readable. QR1 is the headline model
# and is included as soon as its column is present.
FORECAST_MODELS = ["QR1", "HARX", "ARMAX", "AR3", "RC"]
FORECAST_COLORS = {
    "QR1": QR1_ORANGE,
    "HARX": "#0072B2",
    "ARMAX": "#009E73",
    "AR3": "#CC79A7",
    "RC": "#E69F00",
}
FORECAST_STYLES = {"QR1": "-", "HARX": "-", "ARMAX": "--", "AR3": "-.", "RC": ":"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


def _write_alt(png_path: pathlib.Path, text: str) -> pathlib.Path:
    alt_path = png_path.with_suffix(".alt.txt")
    alt_path.write_text(text.strip() + "\n")
    return alt_path


def _skip(msg: str):
    warnings.warn(msg)
    return None


def load_oos(oos_path: str = "data/oos_classical.csv") -> pd.DataFrame:
    return (pd.read_csv(oos_path, parse_dates=["date"])
            .sort_values("date").reset_index(drop=True))


def _grid_shape(n: int) -> tuple[int, int]:
    """Rows x cols for ``n`` panels: one row up to 3, 2x2 for 4, then 2x3."""
    if n <= 1:
        return 1, 1
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    return int(np.ceil(n / 3)), 3


def fig_forecast(oos_path: str = "data/oos_classical.csv",
                 out_dir: str = "figures",
                 models: list[str] | None = None,
                 max_models: int = 6) -> pathlib.Path | None:
    """Paper Fig. 7: one subplot per model, each showing model vs actual.

    Predictions in ``oos_classical.csv`` are in log space and are exponentiated
    to levels to match the actual ``RV`` column. Every panel repeats the actual
    series in black at the lowest z-order, so the model line always draws on
    top of it; panels share the x and y axes for easy comparison.
    """
    try:
        df = load_oos(oos_path)
    except FileNotFoundError:
        return _skip(f"no OOS forecasts at {oos_path}, skipping Fig. 7")

    wanted = models or FORECAST_MODELS
    have = [m for m in wanted if m in df.columns][:max_models]
    missing = [m for m in wanted if m not in df.columns]
    if missing:
        warnings.warn(f"Fig. 7: models not in {oos_path}, skipping: {missing}")
    if not have:
        return _skip("no plottable model columns in the OOS file, skipping Fig. 7")

    actual = (df["RV"].to_numpy() if "RV" in df.columns
              else np.exp(df["y_log"].to_numpy()))
    # In-range 5-year ticks only, so no year label spills into the next panel.
    d0, d1 = df["date"].min(), df["date"].max()
    xticks = [pd.Timestamp(year=y, month=1, day=1)
              for y in range(d0.year, d1.year + 1) if y % 5 == 0]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "fig7_forecast.png"

    nrows, ncols = _grid_shape(len(have))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)

    alt_rows = []
    for i, m in enumerate(have):
        ax = axes.flat[i]
        pred = np.exp(df[m].to_numpy())
        # Actual first and at the lowest z-order, so the model line draws over it.
        ax.plot(df["date"], actual, color=BLACK, linewidth=1.5, alpha=0.8,
                label="Actual", zorder=1)
        ax.plot(df["date"], pred, color=FORECAST_COLORS.get(m, "#555555"),
                linestyle=FORECAST_STYLES.get(m, "-"),
                linewidth=1.8 if m == "QR1" else 1.5, alpha=0.95, label=m,
                zorder=3 if m == "QR1" else 2)
        ax.margins(x=0.01, y=0.05)
        ax.grid(False)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        mse = float(np.mean((actual - pred) ** 2))
        ax.set_title(f"{m}  (MSE {mse:.4f})", fontsize=12)
        ax.legend(frameon=False, loc="upper right", fontsize=10)
        alt_rows.append(f"{m} level MSE {mse:.5f}.")

    # Hide any unused panels in the last row.
    for j in range(len(have), nrows * ncols):
        axes.flat[j].set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Date")
    for ax in axes[:, 0]:
        ax.set_ylabel("Realized Volatility")

    _write_alt(png,
               "Grid of line charts reproducing Fig. 7: each panel compares the realized "
               "volatility of the S&P 500 (black, shown in every panel) with the out-of-sample "
               "forecast of one model (Aug 1997-Dec 2017). "
               f"Models: {', '.join(have)}. " + " ".join(alt_rows))
    fig.savefig(png)
    plt.close(fig)
    return png


def build_all(out_dir: str = "figures",
              oos_path: str = "data/oos_classical.csv") -> dict[str, pathlib.Path | None]:
    """Build the Fig. 7 reproduction; returns {name: png path or None}."""
    p = fig_forecast(oos_path, out_dir)
    print(f"  fig7_forecast -> {p if p is not None else '(skipped: inputs missing)'}")
    return {"fig7_forecast": p}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--oos-path", default="data/oos_classical.csv")
    args = ap.parse_args()
    build_all(args.out_dir, args.oos_path)
