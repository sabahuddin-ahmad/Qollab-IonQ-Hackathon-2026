"""Figures for the Qollab run, drawn with matplotlib. main.py falls back to text if these fail.

Colours are a palette checked for colour-vision deficiency; every series also has a label or a
distinct marker, so nothing depends on colour alone. matplotlib is imported inside the functions
so that importing this module never fails where matplotlib is missing.
"""

from __future__ import annotations

import numpy as np

from .reference import K_LAGS, N_HIDDEN, N_INPUT
from .textplot import FEATURES

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a897f", "#e4e3de"
BLUE, ORANGE, AQUA, PALE = "#2a78d6", "#eb6834", "#1baf7a", "#cde2fb"


def _pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def _clean(ax, grid: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def circuit_figure(on_ionq: bool, steps: int, mid_circuit: bool = False):
    """Schematic of the forecast circuit: one column of RY rotations and one evolution block per month."""
    plt = _pyplot()
    from matplotlib.patches import FancyBboxPatch

    groups = [[f"t-{k}" for _ in range(N_INPUT)] for k in range(K_LAGS, 0, -1)] if on_ionq else [["in"] * N_INPUT]
    n_wires = sum(len(g) for g in groups) + N_HIDDEN
    mem = list(range(n_wires - N_HIDDEN, n_wires))
    input_rows = [list(range(i * N_INPUT, (i + 1) * N_INPUT)) for i in range(len(groups))]
    fig, ax = plt.subplots(figsize=(11, 0.3 * n_wires + 2.1))
    fig.subplots_adjust(left=0.09, right=0.98, top=1 - 1.15 / fig.get_figheight(), bottom=0.03)
    layout = "24 qubits, fresh input qubits for each month (no reset)" if on_ionq else "10 qubits, input qubits reset between months"
    fig.text(0.02, 1 - 0.3 / fig.get_figheight(), f"The forecast circuit: {layout}", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.62 / fig.get_figheight(), "Each month: RY loads seven market features, then that month's inputs and the three memory qubits evolve together.\n"
             f"An evolve block is {steps} Trotter steps: RZ on every qubit and RXX on all 45 pairs ({45 * steps} two-qubit gates). M = measurement.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)
    width = 3.0
    end = K_LAGS * width + 1.3
    for w in range(n_wires):
        last = w // N_INPUT  # an earlier month's input qubit is finished after its own block
        stop = end if (not on_ionq or w >= mem[0] or last == K_LAGS - 1) else last * width + (2.56 if mid_circuit else 2.2)
        ax.plot([-0.1, stop], [w, w], color=MUTED, lw=1, zorder=1)
        label = (f"{FEATURES[w % N_INPUT]}  {groups[w // N_INPUT][0]}" if w < mem[0] else f"memory {w - mem[0]}")
        ax.text(-0.25, w, label, ha="right", va="center", fontsize=8)

    def box(x, rows, w, label, fill, edge, size=7.5):
        ax.add_patch(FancyBboxPatch((x, min(rows) - 0.36), w, max(rows) - min(rows) + 0.72, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fill, ec=edge, lw=1.2, zorder=3))
        ax.text(x + w / 2, (min(rows) + max(rows)) / 2, label, ha="center", va="center", fontsize=size, color=INK, zorder=4)

    for k in range(K_LAGS):
        x0 = k * width
        rows = input_rows[k] if on_ionq else input_rows[0]
        if k > 0 and not on_ionq:
            for r in rows:
                box(x0 - 0.55, [r], 0.42, "|0>", SURFACE, MUTED, 6.5)
        for r in rows:
            box(x0 + 0.1, [r], 0.62, "RY", SURFACE, BLUE)
        name = f"evolve\nmonth t-{K_LAGS - k}"
        if on_ionq and rows[-1] + 1 != mem[0]:  # the block acts on this month's inputs and on the memory qubits
            box(x0 + 0.95, rows, 1.25, name, PALE, BLUE, 8.5)
            box(x0 + 0.95, mem, 1.25, "", PALE, BLUE)
            ax.plot([x0 + 1.575] * 2, [rows[-1] + 0.36, mem[0] - 0.36], color=BLUE, lw=3, zorder=2)
        else:
            box(x0 + 0.95, rows + mem, 1.25, name, PALE, BLUE, 8.5)
        if on_ionq and mid_circuit and k < K_LAGS - 1:
            for r in rows:
                box(x0 + 2.35, [r], 0.42, "M", INK, INK, 6.5)
                ax.text(x0 + 2.56, r, "M", ha="center", va="center", fontsize=6.5, color=SURFACE, zorder=5)
    for r in (input_rows[-1] if on_ionq else input_rows[0]) + mem:
        box(end - 0.75, [r], 0.42, "", INK, INK)
        ax.text(end - 0.54, r, "M", ha="center", va="center", fontsize=6.5, color=SURFACE, zorder=5)
    ax.set_xlim(-0.3, end + 0.1)
    ax.set_ylim(n_wires - 0.4, -0.6)
    ax.axis("off")
    return fig


def z_figure(month: str, labels: list[str], ideal, measured, shots: int, model=None):
    """<Z> on each output qubit: this run against the ideal values (and the local noise model on IonQ)."""
    plt = _pyplot()
    ideal, measured = np.asarray(ideal), np.asarray(measured)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.3))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.2)
    kept = float(measured @ ideal / (ideal @ ideal))
    fig.text(0.02, 0.96, f"Quantum part: what the ten output qubits returned ({month})", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.89, f"Measured <Z> per qubit at {shots} shots. Signal kept relative to ideal: {kept:.2f} (1.00 = no loss). "
             "Bars on this run: shot noise, 2 standard errors.", fontsize=9, color=INK2, va="top")
    _clean(ax, "y")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(6.5, color=GRID, lw=1)
    ax.plot(x, ideal, ls="none", marker="_", ms=22, mew=2, color=MUTED, label="Ideal, no noise")
    if model is not None:
        ax.plot(x - 0.14, np.asarray(model), ls="none", marker="o", ms=8, color=BLUE, mec=SURFACE, mew=2, label="Our local noise model")
    se = np.sqrt(np.clip(1 - measured**2, 0, None) / shots)
    ax.errorbar(x + 0.14, measured, yerr=2 * se, ls="none", color=ORANGE, elinewidth=2, capsize=3, capthick=2, zorder=2)
    ax.plot(x + 0.14, measured, ls="none", marker="D", ms=7, color=ORANGE, mec=SURFACE, mew=2, label="This run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("<Z>")
    ax.legend(frameon=False, loc="upper center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    return fig


def forecast_figure(month: str, actual: float, matching: tuple[str, float], mismatched: tuple[str, float], classical: dict[str, float],
                    hybrid: float | None = None):
    """Forecast error for one month: the quantum reservoir with each readout, the hybrid, and the classical models."""
    plt = _pyplot()
    rows = [(f"QR1, readout {matching[0]}", matching[1] - actual, BLUE, "o"), (f"QR1, readout {mismatched[0]}", mismatched[1] - actual, ORANGE, "D")]
    if hybrid is not None:
        rows.append(("Hybrid: HARX + QR1 values", hybrid - actual, AQUA, "s"))
    rows += [(f"{name}, classical", value - actual, MUTED, "o") for name, value in classical.items()]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    fig.subplots_adjust(left=0.36, right=0.95, top=0.76, bottom=0.13)
    fig.text(0.02, 0.965, f"Forecast for {month}: the readout has to match the device", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.9, "Forecast minus realized log volatility. The two QR1 rows use the same measured values with two sets of ten classical\n"
             "weights; only the first set was trained for this backend. The hybrid adds this run's measured values to HARX's inputs.\n"
             "One month shows the pipeline runs, not how accurate it is.", fontsize=9, color=INK2, va="top", linespacing=1.35)
    _clean(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=INK2, lw=1)
    span = max(abs(e) for _, e, _, _ in rows)
    for i, (label, err, colour, marker) in enumerate(rows):
        ax.plot([0, err], [i, i], color=colour, lw=2, solid_capstyle="round")
        ax.plot(err, i, color=colour, marker=marker, ms=8.5, mec=SURFACE, mew=2)
        ax.text(err + (0.035 * span if err >= 0 else -0.035 * span), i, f"{err:+.2f}", va="center", ha="left" if err >= 0 else "right", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(min(-0.15 * span, min(e for _, e, _, _ in rows) - 0.25 * span), max(e for _, e, _, _ in rows) + 0.25 * span)
    ax.set_xlabel("Forecast error (0 = exactly right)")
    return fig


def findings_figure(f: dict):
    """The three measured findings side by side, from the numbers stored in the demo data."""
    plt = _pyplot()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.1, 0.75, 1.3]})
    fig.subplots_adjust(left=0.12, right=0.985, top=0.7, bottom=0.22, wspace=0.7)
    fig.text(0.02, 0.96, "What we found: forecast error (MSE) on 245 monthly forecasts, lower is better", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.89, "Computed locally, not in this run. Quantum reservoir values are means over 20 random reservoirs (3 for the noise panel).",
             fontsize=9, color=INK2, va="top")

    def dots(ax, rows, title, lo, hi):
        _clean(ax, "x")
        ax.spines["left"].set_visible(False)
        for i, (label, value, colour) in enumerate(rows):
            ax.plot(value, i, "o", ms=9, color=colour, mec=SURFACE, mew=2)
            ax.text(value + 0.03 * (hi - lo), i, f"{value:.4f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], color=INK)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("MSE")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=10)

    dots(a, [("HARX, linear", f["harx_mse"], MUTED), ("ARMAX, linear", f["armax_mse"], MUTED), ("QR1, quantum\nreservoir", f["qr1_mse"], BLUE),
             ("Classical\nreservoir", f["rcx_mse"], MUTED)], "1. Like for like: third of four,\nahead of the classical reservoir", 0.098, 0.131)
    dots(b, [("Entanglement\non", f["gates_on_mse"], BLUE), ("Entanglement\noff", f["gates_off_mse"], ORANGE)],
         f"2. The quantum part matters\n(p = {f['gates_p_value']:.2f})", 0.105, 0.128)
    _clean(c, "y")
    bars = [("No noise", f["no_noise_mse"], MUTED), ("Noisy, readout\ntrained\nwithout noise", f["noise_1pct_unmatched_mse"], ORANGE),
            ("Noisy,\nnoise-matched\nreadout", f["noise_1pct_matched_mse"], BLUE)]
    for i, (label, value, colour) in enumerate(bars):
        c.bar(i, value, width=0.45, color=colour)
        c.text(i, value + 0.012, f"{value:.3f}", ha="center", fontsize=9, color=INK2)
    c.set_xticks(range(3))
    c.set_xticklabels([b_[0] for b_ in bars], fontsize=8, color=INK)
    c.set_xlim(-0.6, 2.6)
    c.set_ylim(0, max(v for _, v, _ in bars) * 1.15)
    c.set_ylabel("MSE")
    c.set_title("3. Device noise (1% gate error) breaks it;\nretraining ten weights repairs it", loc="left", fontsize=9.5, color=INK, pad=10)
    return fig


def history_figure(history: dict, month: str | None = None, this_run: dict[str, float] | None = None):
    """Realized against forecast over the whole test period, one panel per model, on a shared scale.

    ``this_run`` maps a model name to the forecast this run produced for ``month``; it is marked on that panel.
    """
    plt = _pyplot()
    dates = np.array(history["dates"], dtype="datetime64[M]").astype("datetime64[D]")
    realized = np.asarray(history["realized"])
    models = history["models"]
    colours = [BLUE, MUTED, AQUA]
    fig, axes = plt.subplots(len(models), 1, figsize=(9.5, 2.35 * len(models) + 1.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=1 - 1.4 / fig.get_figheight(), bottom=0.06, hspace=0.3)
    fig.text(0.02, 1 - 0.3 / fig.get_figheight(), "Actual against predicted: 245 one-month-ahead forecasts, Aug 1997 to Dec 2017",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.62 / fig.get_figheight(), "Log realized volatility of the S&P 500. Forecasts were computed locally in exact simulation, not in this run."
             + (f"\nThe diamond marks this run's forecast for {month}." if this_run else ""), fontsize=9, color=INK2, va="top", linespacing=1.35)
    for ax, colour, (name, m) in zip(np.atleast_1d(axes), colours, models.items()):
        _clean(ax, "y")
        ax.plot(dates, realized, color=INK, lw=1.1, label="Realized")
        ax.plot(dates, m["forecast"], color=colour, lw=1.7, ls=(0, (4, 1.2)), label="Forecast")
        if this_run and name in this_run and month in history["dates"]:
            ax.plot(dates[history["dates"].index(month)], this_run[name], marker="D", ms=8, color=ORANGE, mec=SURFACE, mew=1.5, ls="none", label="This run")
        ax.set_title(f"{name}   (MSE {m['mse']:.4f})", loc="left", fontsize=9.5, color=INK, pad=6)
        ax.set_ylabel("log volatility")
        ax.set_ylim(-4.7, -0.8)
        ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncols=3)
    return fig
