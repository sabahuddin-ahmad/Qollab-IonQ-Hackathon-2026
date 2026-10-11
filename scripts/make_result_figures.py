"""Result figures for the project page, from files committed in data/ and demo/.

  benchmark.png          where QR1 ranks among the classical models
  controls.png           entangling gates off against on, alone and with HARX (needs scripts/controls_qrc.py)
  complexity.png         what HARX and QR1 each need for one forecast (needs scripts/bench_complexity.py)
  scaling.png            how both models grow with the number of input features (formulas)
  scaling_qubits.png     QR1's cost against qubits, months of memory and shots (needs scripts/bench_scaling.py)
  actual_vs_predicted.png  realized against forecast, month by month, for QR1 and HARX
  actual_vs_predicted_noisy.png  the same for QR1 under gate noise and finite shots (needs scripts/noisy_forecasts.py)
  shots_noise.png        matched training: error against gate noise, one line per shot count
  transfer.png           readout trained under the same noise against one trained without noise
  ionq_calibration.png   IonQ simulator output against our local noise model, per qubit

Writes docs/figures/*.png and docs/figures/alt_text.md. Colours are the first three slots of a
palette checked for colour-vision deficiency, plus one single-hue ramp; series are also
labelled directly, so nothing depends on colour alone.

Usage: uv run --with matplotlib python scripts/make_result_figures.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a897f", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # fewer shots -> more shots / exact
DOT = dict(marker="o", ms=8.5, mec=SURFACE, mew=2)
ALT: dict[str, str] = {}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def new_figure(title: str, subtitle: str, height: float = 4.8):
    fig, ax = plt.subplots(figsize=(8.6, height))
    fig.subplots_adjust(left=0.1, right=0.86, top=1 - 1.05 / height, bottom=0.62 / height)
    fig.text(0.02, 1 - 0.3 / height, title, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.62 / height, subtitle, fontsize=9, color=INK2, va="top", linespacing=1.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return fig, ax


def save(fig, name: str, alt: str) -> None:
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    ALT[name] = alt


def benchmark() -> None:
    classical = pd.read_csv(ROOT / "data/oos_classical.csv")
    qr1 = pd.read_csv(ROOT / "data/oos_qr1.csv")
    weights = pd.read_csv(ROOT / "data/efficiency.csv").set_index("model")["n_params"]
    y = classical["y_log"].to_numpy()
    mse = lambda p: float(np.mean((y - p) ** 2))
    seeds = np.array([mse(qr1[c].to_numpy()) for c in qr1.columns if c.startswith("QR1_s")])
    kinds = {"HARX": "linear", "ARMAX": "linear", "HAR": "linear", "AR3": "linear", "AR1": "linear",
             "RC": "classical reservoir", "RCX": "classical reservoir"}
    rows = [(m, mse(classical[m].to_numpy()), f"{kinds[m]}, {int(weights[m])} weights") for m in kinds]
    rows.append(("QR1", float(seeds.mean()), "quantum reservoir, 10 weights"))
    rows.append(("Persistence", float(np.mean((y[1:] - y[:-1]) ** 2)), "last month's value, 0 weights"))
    rows.sort(key=lambda r: -r[1])

    fig, ax = new_figure("In exact simulation QR1 ranks third, ahead of both classical reservoirs",
                         "Out-of-sample error (MSE on log realized volatility), 245 months, Aug 1997 to Dec 2017. Lower is better.\n"
                         "QR1: mean and full range over 20 random coupling seeds, exact simulation on a local machine.", height=5.0)
    fig.subplots_adjust(left=0.36, right=0.93)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=1)
    for i, (name, value, _) in enumerate(rows):
        if name == "QR1":
            ax.plot([seeds.min(), seeds.max()], [i, i], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(value, i, color=BLUE, **DOT)
        else:
            ax.plot(value, i, color=MUTED, **DOT)
        ax.text(value + 0.0012 if name != "QR1" else seeds.max() + 0.0012, i, f"{value:.4f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{name}  ({desc})" for name, _, desc in rows], color=INK)
    ax.get_yticklabels()[[r[0] for r in rows].index("QR1")].set_fontweight("bold")
    ax.set_xlim(0.098, 0.15)
    ax.set_xlabel("MSE")
    ax.spines["left"].set_visible(False)
    save(fig, "benchmark.png",
         "Dot plot of forecast error for nine models in exact, noiseless simulation, best at the top. The linear models HARX 0.1030 and ARMAX 0.1070 are lowest. "
         f"QR1, the quantum reservoir, is third at {seeds.mean():.4f}, with a range over 20 seeds from {seeds.min():.4f} to {seeds.max():.4f}. "
         "Below it are HAR 0.1154, the classical reservoir RC 0.1171, AR3 0.1180, the classical reservoir RCX 0.1224, AR1 0.1302 and persistence 0.1433.")


def _gate_grid() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/qr1_noise_shots.csv")
    d["shots"] = d["shots"].astype(str)
    return d[(d.noise == "gate") & (d.trotter_steps == 2) & d.level.isin([0.0, 0.001, 0.004, 0.01, 0.03])]


def shots_noise() -> None:
    g = _gate_grid()
    g = g[g.protocol == "consistent"]
    fig, ax = new_figure("QR1 needs about 1024 shots to stay ahead of the classical reservoirs",
                         "QR1 forecast error against two-qubit gate error, readout trained and tested under the same noise and shot count.\n"
                         "Mean of 3 coupling seeds; bars show the range across seeds. Local Aer simulation, generic depolarizing noise.")
    ax.axvspan(1.6, 2.0, color="#efeee9", lw=0)
    ax.text(1.8, 0.1408, "IonQ Forte 1 simulator\nbehaves like 1.6 to 2%", ha="center", va="top", fontsize=8, color=INK2, linespacing=1.3)
    for value, label in ((0.1224, "RCX, classical reservoir"), (0.1154, "HAR"), (0.1030, "HARX")):
        ax.axhline(value, color=MUTED, lw=1)
        ax.text(0.55, value + 0.0004, f"{label}  {value:.4f}", fontsize=8, color=MUTED, va="bottom")
    labels = {"256": "256 shots", "1024": "1024 shots", "4096": "4096 shots", "exact": "exact values"}
    for colour, (key, label) in zip(RAMP, labels.items()):
        s = g[g.shots == key].groupby("level")["mse_mean"].agg(["mean", "min", "max"])
        x = s.index.to_numpy() * 100
        ax.vlines(x, s["min"], s["max"], color=colour, lw=2)
        ax.plot(x, s["mean"], color=colour, lw=2, label=label, **DOT)
        ax.text(3.08, s["mean"].iloc[-1], label, va="center", fontsize=9, color=INK2)
    ax.set_xlim(-0.08, 3.05)
    ax.set_ylim(0.1, 0.142)
    ax.set_xticks([0, 0.4, 1, 2, 3])
    ax.set_xticklabels(["0", "0.4%", "1%", "2%", "3%"])
    ax.set_xlabel("Depolarizing error per two-qubit gate")
    ax.set_ylabel("MSE")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, title="Shots per circuit", title_fontsize=8.5, bbox_to_anchor=(0.0, 0.93))
    m = g.groupby(["level", "shots"])["mse_mean"].mean().unstack("shots")
    save(fig, "shots_noise.png",
         "Line chart of QR1 forecast error against gate error from 0 to 3 percent, one line per shot count. "
         f"With exact values the error stays near {m.loc[0.0, 'exact']:.3f} at every noise level. "
         f"At 1024 shots it rises from {m.loc[0.0, '1024']:.4f} to {m.loc[0.03, '1024']:.4f}; at 256 shots from {m.loc[0.0, '256']:.4f} to {m.loc[0.03, '256']:.4f}. "
         "Reference lines mark RCX 0.1224, HAR 0.1154 and HARX 0.1030: the 256-shot line rises above HAR at every noise level and above RCX beyond about 1 percent, "
         "while the 1024-shot line stays below RCX throughout. No line reaches HARX. A shaded band marks 1.6 to 2 percent, where IonQ's simulator sits.")


def transfer() -> None:
    g = _gate_grid()
    g = g[g.shots == "1024"]
    fig, ax = new_figure("A readout trained without noise fails on a noisy device",
                         "QR1 forecast error at 1024 shots against gate error, for two ways of training the 10 readout weights.\n"
                         "Lines: mean of 3 coupling seeds. Small dots: individual seeds. Log scale. Local Aer simulation.")
    ax.axhline(0.1433, color=MUTED, lw=1)
    ax.text(1.45, 0.1433 * 1.04, "persistence (repeat last month)  0.1433", fontsize=8, color=MUTED, va="bottom")
    series = (("transfer", ORANGE, "trained on a noiseless simulation", "noiseless-trained", "D"),
              ("consistent", BLUE, "trained under the same noise", "noise-matched", "o"))
    for protocol, colour, label, short, marker in series:
        s = g[g.protocol == protocol]
        ax.plot(s["level"] * 100, s["mse_mean"], ls="none", marker=marker, ms=4, color=colour, alpha=0.45, mec="none")
        mean = s.groupby("level")["mse_mean"].mean()
        ax.plot(mean.index * 100, mean, color=colour, lw=2, label=f"Readout {label}", **{**DOT, "marker": marker})
        ax.text(3.08, mean.iloc[-1], short, va="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.09, 2.4)
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xlim(-0.08, 3.05)
    ax.set_xticks([0, 0.4, 1, 2, 3])
    ax.set_xticklabels(["0", "0.4%", "1%", "2%", "3%"])
    ax.set_xlabel("Depolarizing error per two-qubit gate")
    ax.set_ylabel("MSE (log scale)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.9))
    m = g.groupby(["level", "protocol"])["mse_mean"].mean().unstack("protocol")
    save(fig, "transfer.png",
         "Line chart on a log scale comparing two readouts at 1024 shots as gate error grows from 0 to 3 percent. "
         f"Trained under the same noise, the error stays between {m['consistent'].min():.3f} and {m['consistent'].max():.3f}. "
         f"Trained on a noiseless simulation, it rises from {m.loc[0.0, 'transfer']:.2f} to {m.loc[0.01, 'transfer']:.2f} at 1 percent and {m.loc[0.03, 'transfer']:.2f} at 3 percent, "
         "far above persistence at 0.1433. Individual seeds vary widely for the noiseless-trained readout.")


def ionq_calibration() -> None:
    demo = json.loads((ROOT / "demo/qr1_demo.json").read_text())
    runs = [r for r in json.loads((ROOT / "data/ionq_runs.json").read_text())["runs"]
            if r["shots"] == 500 and "1.6%" in r["readout_weights"]]  # one 500-shot run per month
    runs.sort(key=lambda r: r["forecast_month"])
    level = demo["noise_match_p2"]
    kept = lambda v, ideal: float(v @ ideal / (ideal @ ideal))
    x = np.arange(10)
    fig, axes = plt.subplots(1, len(runs), figsize=(11.8, 5.0), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.2, wspace=0.06)
    fig.text(0.02, 0.965, f"IonQ's simulator output matches a {level:.1%} depolarizing model in all three months",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.905, "Measured \u27e8Z\u27e9 on each output qubit of the forecast circuit: 24 qubits, 270 two-qubit gates, 500 shots on Qollab, one run per month.\n"
             "The noise level was set from the 2017-12 circuit; the other two months were run afterwards without changing it.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)
    summary = []
    for ax, run in zip(axes, runs):
        month = [m for m in demo["months"] if m["date"] == run["forecast_month"]][0]
        ideal, model, z = (np.array(v) for v in (month["ideal_features"], month["noise_matched_features"], run["measured_z"]))
        se = np.sqrt((1 - z**2) / run["shots"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(6.5, color=GRID, lw=1)
        ax.plot(x, ideal, ls="none", marker="_", ms=13, mew=2, color=MUTED, label="Ideal, no noise")
        ax.plot(x - 0.17, model, ls="none", color=BLUE, label=f"Local model, {level:.1%} per gate", **{**DOT, "ms": 7})
        ax.errorbar(x + 0.17, z, yerr=2 * se, ls="none", color=ORANGE, elinewidth=1.6, capsize=2.5, capthick=1.6, zorder=2)
        ax.plot(x + 0.17, z, ls="none", color=ORANGE, label="IonQ Forte 1 simulator (bars: shot noise, 2 s.e.)", **{**DOT, "marker": "D", "ms": 6})
        ax.set_title(f"{run['forecast_month']}   signal kept: IonQ {kept(z, ideal):.2f}, model {kept(model, ideal):.2f}",
                     loc="left", fontsize=9.5, color=INK, pad=8)
        ax.set_xticks(x); ax.set_xticklabels([f"in{q}" for q in range(7)] + [f"m{q}" for q in range(3)], fontsize=8)
        ax.set_xlim(-0.6, 9.6); ax.set_ylim(-1, 1)
        summary.append((run["forecast_month"], kept(z, ideal), kept(model, ideal), float(np.abs(z - model).max()), float((((z - model) / se) ** 2).sum())))
    axes[0].set_ylabel("\u27e8Z\u27e9")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5, ncols=3, bbox_to_anchor=(0.0, -0.1))
    fig.text(0.985, 0.03, "in0 to in6: input qubits, one market feature each.  m0 to m2: memory qubits.", fontsize=8.5, color=INK2, ha="right")
    save(fig, "ionq_calibration.png",
         f"Three panels, one per forecast month, comparing per-qubit values on ten output qubits: ideal, a local depolarizing model at {level:.1%} per gate, "
         "and IonQ's Forte 1 simulator at 500 shots with shot-noise bars. In every month IonQ's values are smaller in magnitude than ideal and lie close to the local model; "
         "the three memory qubits lose the most. " + " ".join(f"{m}: IonQ keeps {a:.2f} of the ideal signal, the model {b:.2f}, largest per-qubit gap {c:.3f}." for m, a, b, c, _ in summary)
         + f" Chi-squared of IonQ against the model, summed over the {10 * len(summary)} values: {sum(v[4] for v in summary):.1f}.")


def actual_vs_predicted() -> None:
    classical = pd.read_csv(ROOT / "data/oos_classical.csv", parse_dates=["date"]).set_index("date")
    qr1 = pd.read_csv(ROOT / "data/oos_qr1.csv", parse_dates=["date"]).set_index("date")
    seeds = qr1[[c for c in qr1.columns if c.startswith("QR1_s")]]
    y = classical["y_log"]
    mse = lambda p: float(np.mean((y.to_numpy() - np.asarray(p)) ** 2))
    panels = ((f"QR1, quantum reservoir, coupling seed 0   (MSE {mse(seeds['QR1_s0']):.4f})", seeds["QR1_s0"], BLUE, "QR1 forecast"),
              (f"HARX, best linear model   (MSE {mse(classical['HARX']):.4f})", classical["HARX"], ORANGE, "HARX forecast"))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.07, hspace=0.22)
    fig.text(0.02, 0.965, "Both models follow the level of volatility and miss the sudden jumps", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.918, "Realized and one-month-ahead forecast log volatility of the S&P 500, 245 out-of-sample months, Aug 1997 to Dec 2017.\n"
             "Blue band: range of the QR1 forecast over 20 coupling seeds. Exact simulation on a local machine.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)
    for ax, (title, forecast, colour, label) in zip(axes, panels):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        if colour == BLUE:
            ax.fill_between(seeds.index, seeds.min(axis=1), seeds.max(axis=1), color=BLUE, alpha=0.18, lw=0, label="QR1 range over 20 seeds")
        ax.plot(y.index, y, color=INK, lw=1.2, label="Realized")
        ax.plot(forecast.index, forecast, color=colour, lw=1.8, ls=(0, (4, 1.2)), label=label)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
        ax.set_ylabel("log realized volatility")
        ax.set_ylim(-4.6, -0.9)
        ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncols=3)
    worst = (y - seeds["QR1_s0"]).abs().idxmax()
    save(fig, "actual_vs_predicted.png",
         "Two stacked time series from 1997 to 2017 sharing one scale. Each shows realized monthly log volatility as a solid line and a one-month-ahead forecast as a dashed line: "
         f"QR1 with coupling seed 0 on top (MSE {mse(seeds['QR1_s0']):.4f}), HARX below (MSE {mse(classical['HARX']):.4f}). "
         "Both forecasts track the slow rises and falls of volatility and are smoother than the realized series. "
         f"Both miss sudden spikes, most visibly in late 2008; the largest QR1 miss is {worst:%B %Y}.")


def actual_vs_predicted_noisy() -> None:
    d = pd.read_csv(ROOT / "data/oos_qr1_noisy.csv", parse_dates=["date"]).set_index("date")
    y = d["y_log"]
    mse = lambda c: float(np.mean((y - d[c]) ** 2))
    panels = (("matched_p0.016_s1024", "1024 shots, readout trained under the same noise", BLUE),
              ("matched_p0.016_s256", "256 shots, readout trained under the same noise", BLUE),
              ("noiseless_trained_p0.016_s1024", "1024 shots, readout trained on a noiseless simulation", ORANGE))
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.865, bottom=0.05, hspace=0.24)
    fig.text(0.02, 0.975, "Under IonQ-level noise the forecast holds up only if the readout is trained on it", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.94, "Realized and one-month-ahead QR1 forecast, 245 months, at 1.6% depolarizing error per two-qubit gate.\n"
             f"Coupling seed 0, one draw of shots per panel. Local Aer simulation, not IonQ's noise model. Without noise: MSE {mse('exact_p0'):.4f}.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)
    for ax, (col, title, colour) in zip(axes, panels):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.plot(y.index, y, color=INK, lw=1.2, label="Realized")
        ax.plot(d.index, d[col], color=colour, lw=1.8, ls=(0, (4, 1.2)), label="QR1 forecast")
        ax.set_title(f"{title}   (MSE {mse(col):.4f})", loc="left", fontsize=9.5, color=INK, pad=6)
        ax.set_ylabel("log realized volatility")
        ax.set_ylim(-5.0, -0.9)
        ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncols=2)
    bias = float((d["noiseless_trained_p0.016_s1024"] - y).mean())
    save(fig, "actual_vs_predicted_noisy.png",
         "Three stacked time series from 1997 to 2017 sharing one scale, each showing realized log volatility as a solid line and a QR1 forecast as a dashed line, "
         "all simulated with 1.6 percent gate error. "
         f"Top: 1024 shots with a noise-matched readout tracks the realized series closely (MSE {mse('matched_p0.016_s1024'):.4f}). "
         f"Middle: 256 shots with a noise-matched readout tracks it with more month-to-month jitter (MSE {mse('matched_p0.016_s256'):.4f}). "
         f"Bottom: 1024 shots with a readout trained on a noiseless simulation has the right shape but sits {abs(bias):.2f} {'below' if bias < 0 else 'above'} the realized series on average "
         f"(MSE {mse('noiseless_trained_p0.016_s1024'):.4f}).")


def complexity() -> None:
    d = pd.read_csv(ROOT / "data/complexity.csv")
    size = d[d.kind == "size"].pivot(index="item", columns="model", values="value")
    items = ["trained weights", "input values per forecast", "fixed random parameters", "qubits on IonQ", "two-qubit gates"]
    notes = {"fixed random parameters": "QR1: the 45 coupling strengths, drawn once and never trained",
             "qubits on IonQ": "QR1: 10 in the model, 24 without reset", "two-qubit gates": "QR1: per forecast circuit"}
    times = d[(d.kind == "time") & (d.item != "fit the 10 readout weights")]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 6.0), gridspec_kw={"width_ratios": [1, 1.05]})
    fig.subplots_adjust(left=0.03, right=0.97, top=0.79, bottom=0.1, wspace=0.42)
    fig.text(0.02, 0.965, "QR1 trains fewer weights than HARX but needs far more machinery to make a forecast",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.918, "Left: what each model consists of; every row has its own scale. Right: wall-clock time for one monthly forecast, log scale.\n"
             "Accuracy for reference (MSE, lower is better): HARX 0.1030, QR1 0.1102 in exact simulation.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.02, 1.3); ax.set_ylim(len(items) - 0.45, -0.75)
    for i, item in enumerate(items):
        top = max(size.loc[item].max(), 1)
        ax.text(0, i - 0.33, item[0].upper() + item[1:], fontsize=9.5, color=INK, va="center")
        if item in notes:
            ax.text(0, i + 0.37, notes[item], fontsize=7.5, color=MUTED, va="center")
        for model, colour, dy in (("HARX", ORANGE, -0.13), ("QR1", BLUE, 0.1)):
            v = size.loc[item, model]
            ax.barh(i + dy, v / top, height=0.17, color=colour, left=0)
            ax.plot([0, 0], [i + dy - 0.085, i + dy + 0.085], color=colour, lw=2)  # a zero is still a visible mark
            ax.text(v / top + 0.03, i + dy, f"{v:.0f}", fontsize=9, color=INK2, va="center")
    ax.barh(-9, 1, color=ORANGE, label="HARX, linear model"); ax.barh(-9, 1, color=BLUE, label="QR1, quantum reservoir")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, ncols=2, bbox_to_anchor=(-0.02, 0.97), handlelength=1.2)

    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.grid(axis="x", color=GRID, lw=1); bx.set_axisbelow(True); bx.tick_params(length=0)
    labels = {"fit and forecast": "HARX: fit and forecast", "one circuit, exact (NumPy)": "QR1: exact simulation, laptop",
              "one circuit, noisy (Aer density matrix)": "QR1: noisy simulation, laptop",
              "one circuit, IonQ simulator, 100 shots": "QR1: IonQ simulator, 100 shots",
              "one circuit, IonQ simulator, 500 shots": "QR1: IonQ simulator, 500 shots"}
    fmt = lambda v: f"{v * 1000:.2g} ms" if v < 0.1 else f"{v:.2f} s" if v < 10 else f"{v:.0f} s"
    for i, row in enumerate(times.itertuples()):
        colour, marker = (ORANGE, "D") if row.model == "HARX" else (BLUE, "o")
        bx.plot(row.value, i, color=colour, **{**DOT, "marker": marker})
        bx.text(row.value * 1.9, i, fmt(row.value), fontsize=9, color=INK2, va="center")
    bx.set_yticks(range(len(times))); bx.set_yticklabels([labels[t] for t in times.item], color=INK)
    bx.set_ylim(len(times) - 0.5, -0.5)
    bx.set_xscale("log"); bx.set_xlim(1e-4, 5e3); bx.minorticks_off()
    bx.set_xticks([1e-3, 1e-1, 1e1, 1e3]); bx.set_xticklabels(["1 ms", "0.1 s", "10 s", "1000 s"])
    bx.set_xlabel("Time for one forecast (log scale)")
    t = times.set_index("item")["value"]
    save(fig, "complexity.png",
         "Two panels comparing HARX and QR1. Left, paired bars with counts: trained weights 14 for HARX and 10 for QR1; input values per forecast 13 and 21; "
         "fixed random parameters 0 and 45; qubits on IonQ 0 and 24; two-qubit gates 0 and 270. "
         f"Right, time for one forecast on a log scale: HARX fit and forecast {fmt(t['fit and forecast'])}; QR1 exact simulation {fmt(t['one circuit, exact (NumPy)'])}; "
         f"QR1 noisy simulation {fmt(t['one circuit, noisy (Aer density matrix)'])}; QR1 on IonQ's simulator {fmt(t['one circuit, IonQ simulator, 100 shots'])} at 100 shots "
         f"and {fmt(t['one circuit, IonQ simulator, 500 shots'])} at 500 shots.")


def scaling() -> None:
    """How each model grows with the number of input features d. Formulas, not measurements:
    QR1 uses one qubit per feature plus 3 memory qubits (n = d + 3), 3 months, 2 Trotter steps."""
    d = np.arange(1, 41)
    n = d + 3
    gates = 3 * 2 * n * (n - 1) / 2
    ours = 7
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 5.0))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.13, wspace=0.3)
    fig.text(0.02, 0.965, "As features are added, QR1's readout stays small but its circuit and its simulation do not",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.905, "Growth with the number of input features, from the model definitions (not measured). QR1: one qubit per feature plus 3 memory\n"
             "qubits, 3 months, 2 Trotter steps, all pairs coupled. Dots mark the models as built: QR1 with 7 features, HARX with 13 inputs.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.set_xlim(0, 41); ax.set_xlabel("Input features")
    a, b, c = axes

    a.set_title("Trained weights: both grow in step", loc="left", fontsize=9.5, color=INK, pad=8)
    a.plot(d, d + 1, color=ORANGE, lw=2, label="HARX")
    a.plot(d, d + 3, color=BLUE, lw=2, ls=(0, (4, 1.2)), label="QR1")
    a.plot(13, 14, color=ORANGE, **{**DOT, "marker": "D"}); a.text(14, 11.6, "HARX: 14", fontsize=8.5, color=INK2)
    a.plot(ours, 10, color=BLUE, **DOT); a.text(1.2, 12.8, "QR1: 10", fontsize=8.5, color=INK2)
    a.set_ylim(0, 46); a.set_ylabel("Weights fitted each month")
    a.legend(frameon=False, loc="upper left", fontsize=8.5)

    b.set_title("Two-qubit gates per circuit: grow with the square", loc="left", fontsize=9.5, color=INK, pad=8)
    b.plot(d, gates, color=BLUE, lw=2)
    b.plot(ours, 270, color=BLUE, **DOT); b.text(ours + 2.2, 150, "QR1: 270 gates", fontsize=8.5, color=INK2, va="center")
    b.text(40, 400, "HARX uses no circuit", fontsize=8.5, color=INK2, ha="right")
    b.axvline(8.5, color=MUTED, lw=1)
    b.text(9.2, 5200, "more than 8 features does not fit\nIonQ's 29-qubit simulator\nwithout reset (3 x features + 3)", fontsize=8, color=INK2, va="top", linespacing=1.3)
    b.set_ylim(0, 5600); b.set_ylabel("RXX gates")

    c.set_title("Memory to simulate it: exponential", loc="left", fontsize=9.5, color=INK, pad=8)
    c.plot(d, 16.0 * 4.0**n, color=BLUE, lw=2, label="QR1, with noise (density matrix)")
    c.plot(d, 16.0 * 2.0**n, color=RAMP[0], lw=2, ls=(0, (4, 1.2)), label="QR1, no noise (state vector)")
    c.plot(d, 8.0 * 570 * (d + 1), color=ORANGE, lw=2, ls=(0, (1, 1.2)), label="HARX (its training table)")
    c.plot(ours, 16.0 * 4.0**10, color=BLUE, **DOT); c.text(ours - 1, 16.0 * 4.0**10 * 6, "16 MB", fontsize=8.5, color=INK2, ha="right", va="center")
    c.axhline(16e9, color=MUTED, lw=1); c.text(40, 16e9 * 1.8, "16 GB laptop", fontsize=8, color=MUTED, ha="right")
    c.set_yscale("log"); c.set_ylim(1e3, 1e25); c.minorticks_off()
    c.set_yticks([1e3, 1e6, 1e9, 1e12, 1e15, 1e18]); c.set_yticklabels(["1 kB", "1 MB", "1 GB", "1 TB", "1 PB", "1 EB"])
    c.set_ylabel("Memory (log scale)")
    c.legend(frameon=False, loc="upper left", fontsize=8)
    limit_sv = int(d[16.0 * 2.0**n <= 16e9].max()); limit_dm = int(d[16.0 * 4.0**n <= 16e9].max())
    save(fig, "scaling.png",
         "Three line charts against the number of input features, from 1 to 40, computed from the model definitions. "
         "First: trained weights grow linearly for both models, features plus 1 for HARX and features plus 3 for QR1. "
         "Second: QR1's two-qubit gates per circuit grow with the square of the qubit count, from 270 at 7 features to about 5,400 at 40; HARX uses none. "
         "A marker shows that more than 8 features does not fit a 29-qubit simulator without reset. "
         f"Third, log scale: memory to simulate one QR1 circuit grows exponentially, passing a 16 GB laptop after {limit_dm} features with noise and {limit_sv} without, "
         "while HARX's training table stays under a megabyte. Each added feature quadruples the first and doubles the second.")


def scaling_qubits() -> None:
    """QR1's cost against its three size knobs: qubits, months of memory, shots."""
    t = pd.read_csv(ROOT / "data/scaling_qubits.csv")
    g = _gate_grid()
    g = g[(g.protocol == "consistent") & g.level.isin([0.0, 0.01])].groupby(["level", "shots"])["mse_mean"].mean().unstack("shots")
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.845, bottom=0.07, wspace=0.26, hspace=0.42)
    fig.text(0.02, 0.975, "QR1's cost grows with qubits, memory length and shots; HARX has none of these knobs",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.94, "QR1 as built: 10 qubits, 3 months of memory, 2 Trotter steps, every pair of qubits coupled. Blue dots mark that design.\n"
             "Top left and bottom left are formulas. Top right and bottom right are measured on a laptop (Aer), not on IonQ.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)
    for ax in axes.flat:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(length=0)
    (a, b), (c, e) = axes

    n = np.arange(4, 31)
    a.set_title("Two-qubit gates per circuit: 3 x qubits x (qubits - 1)", loc="left", fontsize=9.5, color=INK, pad=8)
    a.plot(n, 3 * n * (n - 1), color=BLUE, lw=2)
    a.plot(10, 270, color=BLUE, **DOT); a.text(11, 150, "270 gates", fontsize=8.5, color=INK2, va="center")
    a.text(30, 3 * 30 * 29 + 60, "2,610", fontsize=8.5, color=INK2, ha="right", va="bottom")
    a.set_xlabel("Qubits in the reservoir"); a.set_ylabel("RXX gates"); a.set_ylim(0, 3000); a.set_xlim(3, 31)

    b.set_title("Time to simulate one noisy circuit: about 5x per added qubit", loc="left", fontsize=9.5, color=INK, pad=8)
    fit = np.polyfit(t.qubits[t.qubits >= 9], np.log(t.seconds[t.qubits >= 9]), 1)
    more = np.arange(13, 17)
    b.plot(more, np.exp(np.polyval(fit, more)), color=MUTED, lw=2, ls=(0, (4, 1.2)), label="Extrapolated from 9 to 13 qubits")
    b.plot(t.qubits, t.seconds, color=BLUE, lw=2, label="Measured", **DOT)
    b.text(10.5, 0.53 / 1.3, "10 qubits: 0.53 s", fontsize=8.5, color=INK2, va="top")
    b.text(12.6, 62 * 1.5, "13 qubits: 62 s", fontsize=8.5, color=INK2, ha="right", va="bottom")
    hours = float(np.exp(np.polyval(fit, 16))) / 3600
    b.text(16.3, np.exp(np.polyval(fit, 16)) / 12, f"16 qubits:\nabout {hours:.1f} h", fontsize=8.5, color=INK2, ha="right", va="top", linespacing=1.3)
    b.set_yscale("log"); b.minorticks_off(); b.set_ylim(5e-4, 5e4)
    b.set_yticks([1e-3, 1e-1, 1e1, 1e3]); b.set_yticklabels(["1 ms", "0.1 s", "10 s", "1000 s"])
    b.set_xlabel("Qubits in the reservoir"); b.set_ylabel("Seconds per circuit (log scale)"); b.set_xlim(3.5, 16.5)
    b.legend(frameon=False, loc="upper left", fontsize=8.5)

    k = np.arange(1, 9)
    c.set_title("Qubits needed for more months of memory", loc="left", fontsize=9.5, color=INK, pad=8)
    c.axhline(29, color=MUTED, lw=1); c.text(8, 30.2, "IonQ simulator limit: 29 qubits", fontsize=8, color=MUTED, ha="right")
    c.plot(k, 7 * k + 3, color=BLUE, lw=2, label="Without reset (what IonQ on Qollab accepts)", **DOT)
    c.plot(k, np.full(len(k), 10), color=RAMP[0], lw=2, ls=(0, (4, 1.2)), label="With reset between months", **{**DOT, "marker": "s"})
    c.text(3.25, 21.5, "as built: 24", fontsize=8.5, color=INK2, va="top")
    c.set_xlabel("Months of memory fed into the circuit"); c.set_ylabel("Qubits"); c.set_ylim(0, 64); c.set_xlim(0.5, 8.5)
    c.legend(frameon=False, loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.93))

    e.set_title("Forecast error against shots per circuit", loc="left", fontsize=9.5, color=INK, pad=8)
    x, cols = np.arange(4), ["256", "1024", "4096", "exact"]
    e.axhline(0.1030, color=ORANGE, lw=2, ls=(0, (1, 1.2)), label="HARX (no shots needed)")
    e.plot(x, g.loc[0.0, cols], color=BLUE, lw=2, label="QR1, no gate noise", **DOT)
    e.plot(x, g.loc[0.01, cols], color=RAMP[3], lw=2, ls=(0, (4, 1.2)), label="QR1, 1% gate error", **{**DOT, "marker": "s"})
    e.set_xticks(x); e.set_xticklabels(["256", "1024", "4096", "no shot noise"])
    e.set_xlabel("Shots per circuit (IonQ simulator on Qollab: about 0.3 s per shot)"); e.set_ylabel("MSE"); e.set_ylim(0.1, 0.122); e.set_xlim(-0.3, 3.3)
    e.legend(frameon=False, loc="upper right", fontsize=8.5)
    save(fig, "scaling_qubits.png",
         "Four panels on how the cost of QR1 grows. Top left: two-qubit gates per circuit rise with the square of the qubit count, 270 at 10 qubits and 2,610 at 30. "
         f"Top right, log scale: measured laptop time to simulate one noisy circuit rises about fivefold per added qubit, 0.53 seconds at 10 qubits and 62 seconds at 13, extrapolated to about {hours:.1f} hours at 16. "
         "Bottom left: without reset the circuit needs 7 qubits per month of memory plus 3, so 24 for three months and 31 for four, above the 29-qubit IonQ simulator limit; with reset it stays at 10. "
         f"Bottom right: QR1 forecast error falls as shots increase, from {g.loc[0.0, '256']:.4f} at 256 shots to {g.loc[0.0, 'exact']:.4f} with no shot noise, and never reaches HARX at 0.1030.")


def controls() -> None:
    d = pd.read_csv(ROOT / "data/qr1_controls.csv").set_index("model")
    v = lambda name, col="mse": float(d.loc[name, col])
    x = np.arange(5)
    alone = [v("QR1, entangling gates off"), v("QR1, coupling x0.25, mean of 5 seeds"), v("QR1, coupling x0.5, mean of 5 seeds"),
             v("QR1 as designed, mean of 20 seeds", "mse_seed_mean"), v("QR1, coupling x2, mean of 5 seeds")]
    hyb = [v("HARX + reservoir, entangling gates off (= cos of the 7 inputs)"), v("HARX + reservoir, coupling x0.25, mean of 5 seeds"),
           v("HARX + reservoir, coupling x0.5, mean of 5 seeds"), v("HARX + reservoir as designed, mean of 20 seeds", "mse_seed_mean"),
           v("HARX + reservoir, coupling x2, mean of 5 seeds")]
    fig, ax = new_figure("Entangling gates help the reservoir alone, not once HARX is added",
                         "Forecast error as the reservoir's coupling strength is turned up from zero (gates off) to twice the designed value.\n"
                         "Exact simulation, 245 months. Designed strength: mean of 20 coupling seeds; other strengths: mean of 5.")
    fig.subplots_adjust(right=0.8)
    ax.axhline(v("HARX"), color=MUTED, lw=1)
    ax.text(1.5, v("HARX") + 0.0004, "HARX alone  0.1030", fontsize=8, color=MUTED, ha="center", va="bottom")
    ax.plot(x, alone, color=BLUE, lw=2, label="Reservoir alone", **DOT)
    ax.plot(x, hyb, color=ORANGE, lw=2, ls=(0, (4, 1.2)), label="Reservoir values added to HARX", **{**DOT, "marker": "D"})
    ax.text(4.42, alone[-1], "reservoir\nalone", fontsize=9, color=INK2, va="center", linespacing=1.25)
    ax.text(4.42, hyb[-1], "added\nto HARX", fontsize=9, color=INK2, va="center", linespacing=1.25)
    for xi, yi, dy in ((0, alone[0], 0.0014), (3, alone[3], 0.0014), (0, hyb[0], -0.0016), (3, hyb[3], -0.0016)):
        ax.text(xi, yi + dy, f"{yi:.4f}", fontsize=8.5, color=INK2, ha="center", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(["gates off", "quarter", "half", "as designed", "double"])
    ax.set_xlabel("Coupling strength of the entangling gates")
    ax.set_ylabel("MSE")
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(0.096, 0.137)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, bbox_to_anchor=(0.12, 1.0))
    save(fig, "controls.png",
         "Line chart of forecast error against the coupling strength of the entangling gates, from off to double the designed value, for two models. "
         f"For the reservoir alone the error falls from {alone[0]:.4f} with the gates off to {alone[3]:.4f} at the designed strength, then rises to {alone[4]:.4f} at double strength. "
         f"For the reservoir's values added to HARX the error is lowest with the gates off, {hyb[0]:.4f}, and rises steadily to {hyb[3]:.4f} at the designed strength and {hyb[4]:.4f} at double. "
         "A reference line marks HARX alone at 0.1030.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (benchmark, controls, complexity, scaling, scaling_qubits, actual_vs_predicted, actual_vs_predicted_noisy, shots_noise, transfer, ionq_calibration):
        make()
    text = "# Alt text for the figures\n\nGenerated by `scripts/make_result_figures.py`. Use these as the image descriptions on the project page.\n\n"
    text += "".join(f"**{name}**\n\n{alt}\n\n" for name, alt in ALT.items())
    (OUT / "alt_text.md").write_text(text)
    print("saved", ", ".join(ALT), "and alt_text.md in", OUT)


if __name__ == "__main__":
    main()
