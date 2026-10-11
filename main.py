"""Qollab entry point: forecast S&P 500 monthly volatility with a 10-qubit quantum reservoir (QR1).

The model is hybrid.
  Quantum part   One circuit per forecast. Seven input qubits each carry one market feature and
                 three qubits act as memory. Three months of features are loaded in turn, with all
                 ten qubits evolving under a fixed random Ising interaction after each month. The
                 circuit is never trained. This is what runs on the backend picked in Select QPU.
  Classical part A linear readout with ten weights turns the ten measured averages into a forecast.
                 The weights were trained on a laptop and are shipped frozen in demo/qr1_demo.json,
                 together with the benchmark table. Nothing is trained in this run.
  Hybrid model   A second readout adds the ten measured values to the 13 inputs of HARX, the best
                 linear model, and forecasts from both together.

Form fields
  shots             shots for the circuit
  month             forecast month: 2017-10, 2017-11 or 2017-12
  all_three_months  forecast all three months in one job (built-in simulator only)
  mid_circuit       on IonQ backends, also measure the earlier months' input qubits mid-circuit
  plots             show figures (circuit, measured values, forecast, findings); text charts if off

Two circuit layouts give the same result. The built-in simulator gets 10 qubits with the input
qubits reset between months. IonQ backends do not accept reset and take one circuit per job, so
they get 24 qubits (fresh input qubits for each month) and readout weights trained under noise
matched to IonQ's simulator.

Qollab calls main() after this file runs; do not call it at the bottom.
See README.md for the results and docs/qrc_spec.md for the model.
"""

import json

import numpy as np

from src.qrc.circuit import qr1_circuit, z_all_from_counts
from src.qrc.plots import circuit_figure, findings_figure, forecast_figure, history_figure, z_figure
from src.qrc.textplot import bar_chart, circuit_diagram, z_chart

QUBIT_LABELS = ["in0 RV", "in1 MKT", "in2 DP", "in3 IP", "in4 RVq", "in5 STR", "in6 DEF", "mem0", "mem1", "mem2"]


def show(make_figure) -> bool:
    """Draw one figure and display it. Returns False, after a one-line note, if figures are unavailable."""
    try:
        import matplotlib.pyplot as plt

        figure = make_figure()
        plt.show()
        plt.close(figure)
        return True
    except Exception as error:  # a plotting problem must never stop the run
        print(f"(figure unavailable, showing text instead: {type(error).__name__}: {error})")
        return False


def is_ionq(backend) -> bool:
    return "ionq" in (str(getattr(backend, "name", "")) + type(backend).__module__).lower()


def to_log_volatility(scaled: float, demo: dict) -> float:
    """The readout works in a scaled [-1, 0] space; map its output back to log realized volatility."""
    lo, hi = demo["target_lo"], demo["target_hi"]
    return (scaled + 1.0) * (hi - lo) + lo


def report_story(demo: dict) -> None:
    f = demo["findings"]
    print("\n=== The story: what we found (245 monthly forecasts, computed locally; lower error is better) ===")
    print("1. The claim. A published paper says this quantum reservoir beats classical volatility forecasts. We rebuilt it.")
    print(f"2. A fair test. Scored correctly and like for like, it is third of four: HARX {f['harx_mse']:.4f}, ARMAX {f['armax_mse']:.4f},")
    print(f"   quantum reservoir {f['qr1_mse']:.4f} (20 random reservoirs, {f['qr1_best_seed']:.4f} to {f['qr1_worst_seed']:.4f}), classical reservoir {f['rcx_mse']:.4f}.")
    print(f"   It beats the classical reservoir by {1 - f['qr1_mse'] / f['rcx_mse']:.0%} with 10 trained weights instead of 21.")
    print(f"3. The quantum part matters. With entanglement switched off the same model scores {f['gates_off_mse']:.4f} against {f['gates_on_mse']:.4f}:")
    print(f"   {1 - f['gates_on_mse'] / f['gates_off_mse']:.0%} better with it (p = {f['gates_p_value']:.2f}).")
    print(f"4. Device noise breaks it, and a classical fix repairs it. At 1% gate error and 1024 shots a readout trained without noise")
    print(f"   scores {f['noise_1pct_unmatched_mse']:.2f}; retraining only the ten classical weights under matched noise gives {f['noise_1pct_matched_mse']:.3f}")
    print(f"   (no noise: {f['no_noise_mse']:.3f}). This run shows that effect live, below.")
    print(f"5. A second hybrid. Adding the reservoir's measured values to HARX's inputs gives {f['hybrid_mse']:.4f}, ahead of HARX on "
          f"{f['hybrid_seeds_beating_harx']} of {f['hybrid_seeds']} reservoirs.")
    print("   The gain is 2% and not statistically significant, and our control attributes it to the input encoding, not to entanglement.")
    print("6. The honest boundary. No version beats a linear model given the same information. Not a quantum advantage; nothing ran on hardware.")


def report_month(m: dict, z_all: np.ndarray, weights: np.ndarray, demo: dict, on_ionq: bool, mid_circuit: bool, level: int, shots: int, plots: bool) -> dict:
    """Print and plot one month's result. Returns this run's forecasts, keyed like the history panels."""
    z = z_all[:10]  # the ten end-of-circuit values the readout uses
    ideal = np.array(m["ideal_features"])
    forecast = to_log_volatility(float(z @ weights), demo)
    actual = m["actual_log_rv"]

    print(f"\n=== Forecast for {m['date']} (log realized volatility) ===")
    print("Quantum part: measured <Z> on the ten output qubits (7 input, then 3 memory)")
    print(f"  this run               : {np.round(z, 3).tolist()}")
    print(f"  ideal (no noise, exact): {np.round(ideal, 3).tolist()}")
    print(f"  signal kept vs ideal   : {float(z @ ideal / (ideal @ ideal)):.2f}   (1.00 = no loss)")
    if on_ionq:
        expected = np.array(m["noise_matched_features"])
        print(f"  our local noise model  : {np.round(expected, 3).tolist()}")
        print(f"  largest gap to it      : {np.abs(z - expected).max():.3f}")
    if mid_circuit:
        print(f"  mid-circuit, month t-2 : {np.round(z_all[10:17], 3).tolist()}")
        print(f"  mid-circuit, month t-3 : {np.round(z_all[17:24], 3).tolist()}   (not used by the forecast)")
    model = np.array(m["noise_matched_features"]) if on_ionq else None
    if not (plots and show(lambda: z_figure(m["date"], QUBIT_LABELS, ideal, z, shots, model))):
        series = {"o": ideal, **({"+": model} if on_ionq else {}), "#": z}
        print("\n  Chart of <Z> per qubit:  # this run   o ideal" + ("   + our local noise model" if on_ionq else "") + "   (# covers the others where they coincide)")
        print(z_chart(QUBIT_LABELS, series))

    # Finding 4, live: the same measured values through both readouts. Only one matches this backend's noise.
    plain = to_log_volatility(float(z @ np.array(m["weights"][str(level)])), demo)
    matched = to_log_volatility(float(z @ np.array(m["noise_matched_weights"][str(level)])), demo)
    fits = ("trained under matched noise", "trained without noise") if on_ionq else ("trained without noise", "trained under device noise")
    good, bad = (matched, plain) if on_ionq else (plain, matched)
    print("\nSame measured values, two readouts (the readout has to match the device):")
    print(f"  readout {fits[0]:28s}: {good:8.4f}   error {good - actual:+.4f}   <- matches this backend, used below")
    print(f"  readout {fits[1]:28s}: {bad:8.4f}   error {bad - actual:+.4f}   <- mismatched")

    # The hybrid model: HARX's own inputs plus this run's ten measured values, in one frozen linear readout.
    hybrid_key = "noise_matched_hybrid_weights" if on_ionq else "hybrid_weights"
    hybrid = float(np.r_[m["harx_inputs"], z] @ np.array(m[hybrid_key][str(level)]))

    print("\nClassical part: frozen readout weights applied to those values")
    print(f"  {'QR1, this run':24s}: {forecast:8.4f}   error {forecast - actual:+.4f}")
    print(f"  {'Hybrid HARX + QR1':24s}: {hybrid:8.4f}   error {hybrid - actual:+.4f}   (HARX's 13 inputs plus the ten measured values)")
    for name, value in m["classical_forecasts"].items():
        print(f"  {name + ', classical':24s}: {value:8.4f}   error {value - actual:+.4f}")
    print(f"  {'realized':24s}: {actual:8.4f}")
    if not (plots and show(lambda: forecast_figure(m["date"], actual, (fits[0], good), (fits[1], bad), m["classical_forecasts"], hybrid))):
        errors = [("QR1, this run", abs(forecast - actual)), ("Hybrid HARX + QR1", abs(hybrid - actual))]
        errors += [(name, abs(value - actual)) for name, value in m["classical_forecasts"].items()]
        print("\n  Chart of absolute forecast error for this month (shorter is better):")
        print(bar_chart(errors, 0.0, max(max(e for _, e in errors), 0.5), fmt="{:.2f}"))
    return {"QR1, quantum reservoir (seed 0)": forecast, "Hybrid: HARX + quantum reservoir (seed 0)": hybrid}


def report_benchmark(demo: dict, plots: bool) -> None:
    print("\n=== Full benchmark: 245 months, Aug 1997 to Dec 2017 (computed locally, not in this run) ===")
    print(f"  {'model':40s} {'inputs':20s} {'MSE':>7s}  trained weights")
    for row in demo["table"]:
        print(f"  {row['model']:40s} {row.get('inputs', ''):20s} {row['mse']:7.4f}  {row['trained_parameters']}")
    print("  Lower is better. Compare models with the same inputs.")
    if not plots:
        print("\n  Chart of benchmark MSE, bars start at 0.09:")
        print(bar_chart([(f"{row['model'][:22]} ({row['inputs'][13:]})", row["mse"]) for row in demo["table"]], 0.09, 0.125))


def main(shots: int = 1024, month: str = "2017-12", all_three_months: bool = False, mid_circuit: bool = False, plots: bool = True):
    demo = json.load(open("demo/qr1_demo.json"))
    J = np.array(demo["couplings"])
    steps = demo["trotter_steps"]
    available = [m["date"] for m in demo["months"]]

    # IonQ backends: no reset, one circuit per job. They get the 24-qubit layout and a single month.
    on_ionq = is_ionq(backend)
    if month not in available:
        print(f"Month {month!r} is not in the demo data; using {available[-1]}. Available: {', '.join(available)}")
        month = available[-1]
    if all_three_months and on_ionq:
        print("IonQ backends take one circuit per job, so this run forecasts one month only.")
    if mid_circuit and not on_ionq:
        print("mid_circuit applies to the 24-qubit layout used on IonQ backends; ignored on this backend.")
    mid_circuit = mid_circuit and on_ionq
    chosen = demo["months"] if (all_three_months and not on_ionq) else [m for m in demo["months"] if m["date"] == month]

    # Build one circuit per forecast month from that month's three months of scaled features.
    finish = "measure_mid" if mid_circuit else "measure"
    circuits = [qr1_circuit(np.array(m["inputs"]), J, r=steps, unrolled=on_ionq, finish=finish) for m in chosen]
    c = circuits[0]
    print("=== Quantum reservoir QR1 ===")
    print(f"Backend : {getattr(backend, 'name', type(backend).__name__)}")
    print(f"Circuit : {c.num_qubits} qubits, {c.count_ops().get('rxx', 0)} two-qubit gates, depth {c.depth()}, "
          f"{'24-qubit layout without reset' if on_ionq else '10-qubit layout, input qubits reset between months'}")
    if mid_circuit:
        print("          14 of the 24 measurements are mid-circuit; no gate follows a measurement on the same qubit")
    report_story(demo)
    plots = plots and show(lambda: findings_figure(demo["findings"]))  # if the first figure fails, use text throughout
    if not (plots and show(lambda: circuit_figure(on_ionq, steps, mid_circuit))):
        print(f"\nCircuit diagram. Each 'evolve' block stands for {steps} Trotter steps: RZ on all ten qubits, RXX on all 45 pairs, RZ again.")
        try:
            print(circuit_diagram(on_ionq, steps, mid_circuit))
        except Exception as error:  # a drawing problem must never stop the run
            print(f"(circuit diagram unavailable: {error})")
    print(f"\nJob     : {len(circuits)} circuit(s), {shots} shots each. Submitting...")

    result = backend.run(circuits if len(circuits) > 1 else circuits[0], shots=shots).result()

    # Frozen weights exist for a few shot counts; use the nearest. IonQ backends get the noise-matched set.
    level = min(demo["shot_levels"], key=lambda s: abs(s - shots))
    weights_key = "noise_matched_weights" if on_ionq else "weights"
    for i, m in enumerate(chosen):
        counts = result.get_counts(i) if len(circuits) > 1 else result.get_counts()
        this_run = report_month(m, z_all_from_counts(counts), np.array(m[weights_key][str(level)]), demo, on_ionq, mid_circuit, level, shots, plots)

    # Actual against predicted over the whole test period (computed locally), with the last month of this run marked.
    history = demo["history"]
    print("\n=== Actual against predicted, 245 months (computed locally, not in this run) ===")
    for name, series in history["models"].items():
        print(f"  {name:44s} MSE {series['mse']:.4f}")
    if plots:
        show(lambda: history_figure(history, chosen[-1]["date"], this_run))

    print(f"\nReadout weights: trained locally on features sampled at {level} shots"
          + (f", under a generic depolarizing noise model ({demo['noise_match_p2']:.1%} per two-qubit gate) chosen to match "
             "IonQ's simulator. It approximates IonQ's noise; it is not their model." if on_ionq else ", without noise."))
    report_benchmark(demo, plots)
    print("\nOne month shows that the pipeline runs. It is not evidence about accuracy; the benchmark above is.")
    print("This effort is supported by Qollab & IonQ.")
