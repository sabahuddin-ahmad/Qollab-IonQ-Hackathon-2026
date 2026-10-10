"""Shared forecast harness for the volatility benchmark.

Interface contract (agreed Phase A so model tracks never block each other):

- data module  -> yields one Fold per forecast date:
                 Fold(X_window, y_window, x_next, y_next, date)
- every model  -> fit(X, y) / predict(X); exposes n_params (trained weights
                 only; frozen reservoir/circuit weights do not count)
- metrics      -> score(y_true, y_pred, fit_seconds, n_params)

Scoring comes in two protocols. "corrected" compares each forecast with the
actual of the same month. "paper" reproduces the authors' released code, whose
rolling loop predicts row ``end`` but evaluates against test rows starting at
``end+1`` (predictions misaligned one month late); it compares pred[:-1] with
actual[1:] over 244 points. Metric definitions follow the authors' code, not
their printed formulas: MSE on log realized volatility, QLIKE as the Patton
loss summed over the out-of-sample period on absolute log volatility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

EXOG = ["MKT", "SMB", "HML", "STR", "DP", "EP", "INF", "IP", "TB", "DEF"]
WIN = 570  # fixed rolling window; first windows expand (pre-1951 lags unavailable)
OOS_START = "1997-08"
OOS_END = "2017-12"

PAPER = {  # Li et al. Table 2, S=1 : (MSE, QLIKE)
    "HAR": (0.1476, 2.0431), "HARX": (0.1508, 2.2436),
    "AR1": (0.1304, 1.7279), "AR3": (0.1178, 1.5893),
    "ARMAX": (0.1145, 1.6196), "RC": (0.1441, 2.1011),
    "RCX": (0.1089, 1.6480), "QR1": (0.1051, 1.4427),
    "QR2": (0.1038, 1.4004),
}


def lags(df: pd.DataFrame, cols: list[str], k: int) -> pd.DataFrame:
    return pd.concat(
        {f"{c}_L{i}": df[c].shift(i) for c in cols for i in range(1, k + 1)},
        axis=1,
    )


def load_panel(csv_path: str = "data/rv_monthly.csv",
               exog_lags: int = 1) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    """Monthly panel with target, levels, and all model feature sets.

    Returns (full, feature_sets, xl_cols). ``full`` always carries 'y' (log RV)
    and 'RV' (levels); rows with incomplete lag history are dropped, so the
    first rolling windows expand until WIN observations are available.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    y = df["RV_log"]
    XL = lags(df, EXOG, exog_lags)
    feat = pd.DataFrame(index=df.index)
    feat["y_L1"] = y.shift(1)
    feat["y_L2"] = y.shift(2)
    feat["y_L3"] = y.shift(3)
    feat["y_Q"] = y.shift(1).rolling(3).mean()
    feat["y_A"] = y.shift(1).rolling(12).mean()
    feat = pd.concat([feat, XL], axis=1)
    full = pd.concat({"y": y, "RV": df["RV"]}, axis=1).join(feat).dropna()
    feature_sets = {
        "AR1": ["y_L1"],
        "AR3": ["y_L1", "y_L2", "y_L3"],
        "ARMAX": ["y_L1", "y_L2", "y_L3"] + list(XL.columns),
        "HAR": ["y_L1", "y_Q", "y_A"],
        "HARX": ["y_L1", "y_Q", "y_A"] + list(XL.columns),
    }
    return full, feature_sets, list(XL.columns)


@dataclass
class Fold:
    X_window: pd.DataFrame  # training features, rows expand until WIN available
    y_window: np.ndarray    # training target (log RV)
    x_next: pd.Series       # feature row for the forecast month
    y_next: float           # realized target (log RV)
    date: pd.Timestamp


def iter_folds(full: pd.DataFrame, win: int = WIN,
               oos_start: str = OOS_START,
               oos_end: str = OOS_END) -> tuple[list[Fold], pd.DatetimeIndex]:
    oos_idx = full.loc[oos_start:oos_end].index
    if len(oos_idx) != 245:
        raise ValueError(f"OOS len {len(oos_idx)} != 245")
    pos = {t: full.index.get_loc(t) for t in oos_idx}
    folds = []
    for t in oos_idx:
        j = pos[t]
        tr = full.iloc[max(0, j - win):j]
        folds.append(Fold(X_window=tr, y_window=tr["y"].to_numpy(),
                          x_next=full.loc[t], y_next=float(full.loc[t, "y"]),
                          date=t))
    return folds, oos_idx


class OLSModel:
    """OLS with intercept; n_params counts trained weights only."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.n_params = len(cols) + 1
        self.fit_seconds = 0.0
        self._beta: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OLSModel":
        t0 = time.perf_counter()
        X1 = np.column_stack([np.ones(len(X)), X[self.cols].to_numpy()])
        self._beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        self.fit_seconds += time.perf_counter() - t0
        return self

    def predict(self, x) -> float:
        x1 = np.concatenate([[1.0], np.asarray(x[self.cols], dtype=float)])
        return float(x1 @ self._beta)


def mse_log(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((y_true - pred) ** 2))


def patton_qlike_sum(y_true_log: np.ndarray, pred_log: np.ndarray) -> float:
    r = np.abs(y_true_log) / np.abs(pred_log)
    return float(np.sum(r - np.log(r) - 1))


def hit_rate(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((np.diff(pred) > 0) == (np.diff(y_true) > 0)))


def score_both(y_true: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Metrics under both protocols.

    corrected: pred[t] vs actual[t] over all 245 OOS months.
    paper:     pred[t] vs actual[t+1] over the first 244 (authors' offset).
    """
    out = {}
    for protocol, yy, pp in (("corrected", y_true, pred),
                             ("paper", y_true[1:], pred[:-1])):
        out[protocol] = {
            "n": len(yy),
            "mse": mse_log(yy, pp),
            "qlike": patton_qlike_sum(yy, pp),
            "hit_rate": hit_rate(yy, pp),
        }
    return out


def dm_test(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano on squared log-errors, authors' version: plain sample
    variance of the loss differential with a t p-value (no HAC correction)."""
    from scipy.stats import t

    d = e1**2 - e2**2
    stat = float(d.mean() / np.sqrt(np.var(d, ddof=1) / len(d)))
    p = float(2 * t.cdf(-abs(stat), df=len(d) - 1))
    return stat, p
