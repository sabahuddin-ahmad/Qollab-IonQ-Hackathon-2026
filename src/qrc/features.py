"""Scaled QR1 input features (docs/qrc_spec.md, section 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["RV", "MKT", "DP", "IP", "RV_q", "STR", "DEF"]  # one per input qubit, in order

# feature -> (panel column, low, high) for min-max scaling
_MINMAX = {
    "RV": ("RV_log", -1.0, 0.0),
    "MKT": ("MKT", -1.0, 1.0),
    "DP": ("DP", 0.0, 1.0),
    "STR": ("STR", -1.0, 1.0),
    "DEF": ("DEF", 0.0, 1.0),
}


@dataclass(frozen=True)
class TargetScale:
    """Affine map between log RV and the [-1, 0] space the readout is fitted in."""

    lo: float
    hi: float

    def scale(self, y):
        return (np.asarray(y, dtype=float) - self.lo) / (self.hi - self.lo) - 1.0

    def unscale(self, s):
        return (np.asarray(s, dtype=float) + 1.0) * (self.hi - self.lo) + self.lo


def qr1_inputs(panel: pd.DataFrame, fit_end: str | None = None) -> tuple[pd.DataFrame, TargetScale]:
    """Seven scaled features per month, indexed like ``panel`` (the raw monthly panel by date).

    ``fit_end=None`` takes scaling extremes over the whole sample, as the authors do
    (looks ahead for STR, DEF and DP). Pass a date such as "1997-07" for the clean protocol.
    Rotation angles are ``pi * value``.
    """
    ref = panel.loc[:fit_end] if fit_end is not None else panel
    out = pd.DataFrame(index=panel.index)
    for name, (col, low, high) in _MINMAX.items():
        lo, hi = ref[col].min(), ref[col].max()
        out[name] = (panel[col] - lo) / (hi - lo) * (high - low) + low
    out["IP"] = panel["IP"] / 100.0
    rv_q = panel["RV_log"].shift(1).rolling(3).mean()  # mean of log RV over months m-3..m-1
    out["RV_q"] = rv_q / rv_q.loc[ref.index].abs().max()
    target = TargetScale(float(ref["RV_log"].min()), float(ref["RV_log"].max()))
    return out[FEATURES], target
