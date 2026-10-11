"""Qollab entry point: forecast S&P 500 monthly volatility with a 10-qubit quantum reservoir (QR1).

The model is hybrid.
  Quantum part   One circuit per forecast. Seven input qubits each carry one market feature and
                 three qubits act as memory. Three months of features are loaded in turn, with all
                 ten qubits evolving under a fixed random Ising interaction after each month. The
                 circuit is never trained. This is what runs on the backend picked in Select QPU.
  Classical part A linear readout with ten weights turns the ten measured averages into a forecast.
                 The weights were trained on a laptop and are shipped frozen in demo/qr1_demo.json,
                 together with the benchmark table. Nothing is trained in this run.

Form fields
  shots             shots for the circuit
  month             forecast month: 2017-10, 2017-11 or 2017-12
  all_three_months  forecast all three months in one job (built-in simulator only)
  mid_circuit       on IonQ backends, also measure the earlier months' input qubits mid-circuit

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


def is_ionq(backend) -> bool:
    return "ionq" in (str(getattr(backend, "name", "")) + type(backend).__module__).lower()


def to_log_volatility(scaled: float, demo: dict) -> float:
    """The readout works in a scaled [-1, 0] space; map its output back to log realized volatility."""
    lo, hi = demo["target_lo"], demo["target_hi"]
    return (scaled + 1.0) * (hi - lo) + lo


def report_month(m: dict, z_all: np.ndarray, weights: np.ndarray, demo: dict, on_ionq: bool, mid_circuit: bool) -> None:
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

    print("Classical part: ten frozen readout weights applied to those values")
    print(f"  {'QR1, this run':24s}: {forecast:8.4f}   error {forecast - actual:+.4f}")
    for name, value in m["classical_forecasts"].items():
        print(f"  {name + ', classical':24s}: {value:8.4f}   error {value - actual:+.4f}")
    print(f"  {'realized':24s}: {actual:8.4f}")


def report_benchmark(demo: dict) -> None:
    print("\n=== Full benchmark: 245 months, Aug 1997 to Dec 2017 (computed locally, not in this run) ===")
    print(f"  {'model':40s} {'inputs':20s} {'MSE':>7s}  trained weights")
    for row in demo["table"]:
        print(f"  {row['model']:40s} {row.get('inputs', ''):20s} {row['mse']:7.4f}  {row['trained_parameters']}")
    print("  Lower is better. Compare models with the same inputs.")


def main(shots: int = 1024, month: str = "2017-12", all_three_months: bool = False, mid_circuit: bool = False):
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
    print(f"Job     : {len(circuits)} circuit(s), {shots} shots each. Submitting...")

    result = backend.run(circuits if len(circuits) > 1 else circuits[0], shots=shots).result()

    # Frozen weights exist for a few shot counts; use the nearest. IonQ backends get the noise-matched set.
    level = min(demo["shot_levels"], key=lambda s: abs(s - shots))
    weights_key = "noise_matched_weights" if on_ionq else "weights"
    for i, m in enumerate(chosen):
        counts = result.get_counts(i) if len(circuits) > 1 else result.get_counts()
        report_month(m, z_all_from_counts(counts), np.array(m[weights_key][str(level)]), demo, on_ionq, mid_circuit)

    print(f"\nReadout weights: trained locally on features sampled at {level} shots"
          + (f", under a generic depolarizing noise model ({demo['noise_match_p2']:.1%} per two-qubit gate) chosen to match "
             "IonQ's simulator. It approximates IonQ's noise; it is not their model." if on_ionq else ", without noise."))
    report_benchmark(demo)
    print("\nOne month shows that the pipeline runs. It is not evidence about accuracy; the benchmark above is.")
    print("This effort is supported by Qollab & IonQ.")
