# Submission summary

**Quantum Reservoir Computing for Realized Volatility.** Qollab × IonQ Hackathon 2026, theme:
Quantum Impact.

We forecast next month's S&P 500 volatility with a 10-qubit quantum reservoir (QR1, from Li et
al., arXiv:2505.13933), rebuilt from scratch in Qiskit, scored against seven classical models on
245 months, stress-tested under shots and gate noise, and run on IonQ's Forte 1 simulator through
Qollab. The headline is a measured, honest one: the quantum reservoir beats the classical
reservoirs with fewer trained weights, does not beat the best linear model, and only works on a
noisy device if its readout is trained under matching noise.

## Novelty and creativity

- **An independent reproduction that changes the paper's ranking.** We rebuilt the data, the
  baselines and the quantum model without using the authors' code. Their HAR and HARX baselines
  are scored one month late; aligned, HARX is the best model (MSE 0.1030), ahead of QR1.
- **Twenty reservoirs, not one.** The paper reports a single random reservoir. Over 20 coupling
  seeds QR1 averages 0.1102 (range 0.1056 to 0.1133); the paper's 0.1051 is at the favourable
  edge of that range.
- **A noise-matched readout.** A readout trained on a noiseless simulation fails on a noisy
  device (MSE 0.47 at 1% gate error, 1024 shots). Training the same ten weights under matching
  noise restores it (0.113). This is the result we would carry to hardware.
- **Calibrating a local noise model to IonQ's simulator from one circuit.** A generic
  depolarizing model at 1.6% per two-qubit gate reproduces the ten values IonQ's simulator
  returned, within shot noise.
- **Negative results reported.** Engineered dissipation, a continuous-stream reservoir and
  reading the discarded qubits as extra features were each tested; none improved the forecast.

## Technical execution

- **Runs on Qollab as required:** `main.py`, Qiskit only, one circuit and one job per run, on
  the built-in simulator (10 qubits with reset) and on the IonQ Forte 1 simulator (24 qubits,
  270 two-qubit gates, no reset, optional mid-circuit measurement).
- **Three implementations that agree:** an exact NumPy reference, the Qiskit circuit (matches
  the reference to 1e-10) and the 24-qubit no-reset layout (same output distribution). Ten
  acceptance tests cover this.
- **Verified against the original:** given the authors' couplings, our model reproduces their
  saved forecasts to 1.2e-6.
- **A full shots and noise study:** 25 settings, 3 seeds, 4 shot levels, two training protocols,
  three noise types and three circuit depths, about 20,000 simulated noisy circuits.
- **Platform limits found and designed around:** IonQ backends on Qollab reject `reset` and
  accept one circuit per job; the 24-qubit layout and frozen weights are the response.
- **Everything is regenerable:** each table and figure has a script, and the results are
  committed as CSV.

## Impact potential

- **Who it is for:** teams evaluating quantum machine learning for financial time series, who
  need to know what such a model delivers once shots, noise and a fair baseline are included.
- **What it tells them:** on this task a 10-qubit reservoir is statistically indistinguishable
  from the best linear model (Diebold-Mariano p = 0.25) and better than classical reservoirs,
  but it costs minutes per forecast on a simulator against under a millisecond, and its circuit
  grows with the square of the number of features.
- **What transfers:** the noise-matched readout and the one-circuit calibration apply to any
  reservoir or kernel model with a linear readout on a noisy device. With about 1,000 shots, the
  matched readout holds accuracy up to the noise level of IonQ's simulator.
- **What it does not show:** a quantum advantage. Ten qubits are simulated exactly on a laptop
  in milliseconds, and nothing here was run on quantum hardware.

## Documentation quality

- **Specification before code:** `docs/qrc_spec.md` defines the model, the three
  implementations, the noise models, the acceptance checks and the results.
- **Claims carry their conditions:** every figure states where its numbers came from (local
  simulation or IonQ's simulator, how many seeds, formula or measurement), and each has alt text
  in `docs/figures/alt_text.md`. Colours are checked for colour-blind safety and no figure
  relies on colour alone.
- **Disclosures:** `DISCLOSURES.md` covers prior work, libraries and licenses, data sources, AI
  assistance and what ran where.
- **Run instructions:** `main.py` explains in its header what runs on Qollab and what was
  computed locally; `scripts/run_demo_local.py` reproduces a Qollab run on a laptop.

## Results at a glance

| Model | MSE, 245 months | Trained weights |
| --- | --- | --- |
| HARX (linear) | 0.1030 | 14 |
| ARMAX (linear) | 0.1070 | 14 |
| **QR1 (quantum reservoir, mean of 20 seeds)** | **0.1102** | **10** |
| HAR (linear) | 0.1154 | 4 |
| RC (classical reservoir) | 0.1171 | 51 |
| AR3 (linear) | 0.1180 | 4 |
| RCX (classical reservoir) | 0.1224 | 21 |

Exact simulation, lower is better. Under 1.6% gate noise with a matched readout, QR1 scores
about 0.114 at 1024 shots and 0.126 at 256 shots (one seed).

This effort is supported by Qollab & IonQ.
