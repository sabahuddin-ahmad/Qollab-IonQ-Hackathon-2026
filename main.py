"""Qollab entry point: forecast S&P 500 monthly volatility with a 10-qubit quantum reservoir (QR1).

What runs here, on the backend picked in Select QPU: one quantum circuit per forecast month. It
loads three months of market features onto seven input qubits, lets them interact with three
memory qubits, and measures all ten. A frozen linear readout turns the ten measured averages
into a forecast.

What does not run here: training the readout and the benchmark table. Those were computed
locally (NumPy and Qiskit Aer) and are shipped in demo/qr1_demo.json. See docs/qrc_spec.md.

Qollab calls main() after this file runs; do not call it at the bottom.
"""

import json

import numpy as np

from src.qrc.circuit import qr1_circuit, z_from_counts


def is_ionq(backend) -> bool:
    return "ionq" in (str(getattr(backend, "name", "")) + type(backend).__module__).lower()


def main(shots: int = 1024, months: int = 1):
    demo = json.load(open("demo/qr1_demo.json"))
    J = np.array(demo["couplings"])
    r = demo["trotter_steps"]
    chosen = demo["months"][-max(1, min(months, len(demo["months"]))):]

    # IonQ backends do not accept mid-circuit reset, so they get the 24-qubit layout in which each
    # month has its own input qubits. It produces the same outcome distribution as the 10-qubit one.
    unrolled = is_ionq(backend)
    circuits = [qr1_circuit(np.array(m["inputs"]), J, r=r, unrolled=unrolled, finish="measure") for m in chosen]
    c = circuits[0]
    print(f"Backend: {getattr(backend, 'name', type(backend).__name__)}")
    print(f"Circuit: {c.num_qubits} qubits, depth {c.depth()}, {c.count_ops().get('rxx', 0)} two-qubit gates, "
          f"{'no reset (24-qubit layout)' if unrolled else 'input qubits reset between months'}")
    print(f"Submitting {len(circuits)} circuit(s) as one job, {shots} shots each...")

    result = backend.run(circuits if len(circuits) > 1 else circuits[0], shots=shots).result()

    level = min(demo["shot_levels"], key=lambda s: abs(s - shots))
    lo, hi = demo["target_lo"], demo["target_hi"]
    for i, m in enumerate(chosen):
        z = z_from_counts(result.get_counts(i) if len(circuits) > 1 else result.get_counts())
        forecast = (float(z @ np.array(m["weights"][str(level)])) + 1.0) * (hi - lo) + lo
        ideal = np.array(m["ideal_features"])
        print(f"\nForecast month {m['date']}  (log realized volatility)")
        print(f"  measured <Z> per qubit : {np.round(z, 3).tolist()}")
        print(f"  ideal, no shot noise   : {np.round(ideal, 3).tolist()}")
        print(f"  largest difference     : {np.abs(z - ideal).max():.3f}")
        print(f"  QR1 forecast (this run): {forecast:.4f}")
        print(f"  realized value         : {m['actual_log_rv']:.4f}   error {forecast - m['actual_log_rv']:+.4f}")
        for name, value in m["classical_forecasts"].items():
            print(f"  {name:5s} forecast         : {value:.4f}   error {value - m['actual_log_rv']:+.4f}")
    print(f"\nReadout: 10 frozen weights trained locally on features sampled at {level} shots.")
    if unrolled:
        print("Note: this backend applies its own device noise model; the weights were trained without noise.")

    print("\nFull benchmark, 245 months Aug 1997 - Dec 2017, computed locally (not in this run):")
    print(f"  {'model':34s} {'MSE':>7s}  trained weights")
    for row in demo["table"]:
        print(f"  {row['model']:34s} {row['mse']:7.4f}  {row['trained_parameters']}")
    print("\nOne month is a demonstration that the circuit runs, not evidence about accuracy.")
    print("This effort is supported by Qollab & IonQ.")
