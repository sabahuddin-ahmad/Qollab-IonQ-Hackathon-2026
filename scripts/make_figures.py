"""Result figures for the project page, from files committed in data/ and demo/.

  benchmark.png          where QR1 ranks among the classical models
  shots_noise.png        matched training: error against gate noise, one line per shot count
  transfer.png           readout trained under the same noise against one trained without noise
  ionq_calibration.png   IonQ simulator output against our local noise model, per qubit

Writes docs/figures/*.png and docs/figures/alt_text.md. Colours are the first three slots of a
palette checked for colour-vision deficiency, plus one single-hue ramp; series are also
labelled directly, so nothing depends on colour alone.

Usage: uv run --with matplotlib python scripts/make_figures.py
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
    month = demo["months"][-1]
    run = [r for r in json.loads((ROOT / "data/ionq_runs.json").read_text())["runs"] if r["shots"] == 500][0]
    ideal, model, z = (np.array(v) for v in (month["ideal_features"], month["noise_matched_features"], run["measured_z"]))
    se = np.sqrt((1 - z**2) / run["shots"])
    kept = lambda v: float(v @ ideal / (ideal @ ideal))
    x = np.arange(10)
    fig, ax = new_figure(f"IonQ's simulator output matches a {demo['noise_match_p2']:.1%} depolarizing model",
                         f"Measured ⟨Z⟩ on each qubit for the {month['date']} forecast circuit: 24 qubits, 270 two-qubit gates, {run['shots']} shots on Qollab.\n"
                         f"Signal kept relative to ideal: IonQ {kept(z):.2f}, local model {kept(model):.2f}. One month, one run.")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(6.5, color=GRID, lw=1)
    ax.plot(x, ideal, ls="none", marker="_", ms=22, mew=2, color=MUTED, label="Ideal, no noise")
    ax.plot(x - 0.13, model, ls="none", color=BLUE, label=f"Local model, {demo['noise_match_p2']:.1%} per gate", **DOT)
    ax.errorbar(x + 0.13, z, yerr=2 * se, ls="none", color=ORANGE, elinewidth=2, capsize=3.5, capthick=2, zorder=2)
    ax.plot(x + 0.13, z, ls="none", color=ORANGE, label="IonQ Forte 1 simulator (bars: shot noise, 2 s.e.)", **{**DOT, "marker": "D", "ms": 7})
    ax.set_xticks(x)
    ax.set_xticklabels([f"in {q}" for q in range(7)] + [f"mem {q}" for q in range(3)])
    ax.text(3, -0.93, "input qubits, one market feature each", ha="center", fontsize=8.5, color=INK2)
    ax.text(8, -0.93, "memory qubits", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylabel("⟨Z⟩")
    fig.subplots_adjust(right=0.96)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, bbox_to_anchor=(1.25, -0.03), bbox_transform=ax.transData)
    save(fig, "ionq_calibration.png",
         f"Per-qubit comparison for one circuit on ten output qubits: ideal values, a local depolarizing model at {demo['noise_match_p2']:.1%} per gate, and IonQ's Forte 1 simulator at {run['shots']} shots. "
         f"IonQ's values are smaller in magnitude than ideal on every qubit and lie close to the local model. Overall IonQ keeps {kept(z):.2f} of the ideal signal and the model {kept(model):.2f}; "
         "the three memory qubits lose the most, keeping about 0.4.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (benchmark, shots_noise, transfer, ionq_calibration):
        make()
    text = "# Alt text for the figures\n\nGenerated by `scripts/make_figures.py`. Use these as the image descriptions on the project page.\n\n"
    text += "".join(f"**{name}**\n\n{alt}\n\n" for name, alt in ALT.items())
    (OUT / "alt_text.md").write_text(text)
    print("saved", ", ".join(ALT), "and alt_text.md in", OUT)


if __name__ == "__main__":
    main()
