# Quantum reservoirs for volatility forecasting: where the quantum part pays, and how to keep it paying on a noisy device

**Team Name:** QuanTech

**Team Members:** Sabah Ud Din Ahmad, Ali Karimi, Ali Mahani

**Theme:** Quantum Impact

**Demo:** https://qollab.xyz/u/alikarimi/qrc-vol

**GitHub Repository:** https://github.com/sabahuddin-ahmad/Qollab-IonQ-Hackathon-2026

## What is it?

A hybrid quantum-classical forecaster for S&P 500 volatility. A 10-qubit quantum reservoir turns
market data into ten measured values, and a classical regression with ten weights turns those into
next month's forecast. We benchmark it fairly against seven classical models, measure what the
quantum part contributes, and show how to keep it accurate under device noise on IonQ's Forte 1
simulator.

Built with Qiskit.

## What problem we explore and why it matters?

**The problem (finance).** Volatility forecasts set risk limits, option prices and capital
requirements. They are hard to improve: simple linear models from econometrics are strong. A 2025
paper (Li et al., arXiv:2505.13933) reports that a small quantum reservoir beats them on monthly
S&P 500 volatility. We rebuilt that model from public data to test the claim and to find out what
it takes to run it on a quantum device.

**The classical baseline.** Seven models on 245 out-of-sample months: the HAR family used in
industry (HAR, HARX), autoregressions (AR1, AR3, ARMAX) and two classical reservoirs. We found
that the paper scored two of its baselines one month late, and corrected it.

**What the quantum subroutine brings, measured.**

| Comparison | Result |
| --- | --- |
| Quantum reservoir against the classical reservoir given the same inputs | 10% lower error (0.1102 against 0.1224) with 10 trained weights instead of 21, on all 20 random reservoirs |
| The same model with entanglement switched off | 9% worse (0.1205 against 0.1109, p = 0.04) |
| Under 1% gate noise, our noise-matched readout against the usual one | Error 0.113 against 0.47 (no noise: 0.109) |
| Trained weights as more history is fed in | Fixed at 5, within 2% of an autoregression that needs 13 |

**A second hybrid: the reservoir combined with the best classical model.** Since HARX alone is
slightly ahead of the reservoir alone (0.1030 against 0.1102, not statistically significant), we
fed HARX's 13 inputs and the reservoir's ten measured values into one readout.

| Model | MSE | Note |
| --- | --- | --- |
| HARX alone | 0.1030 | 14 weights |
| HARX + quantum reservoir values | 0.1009 | better than HARX on all 20 random reservoirs; holds under gate noise and at 256 shots |
| HARX, then reservoir fitted to its residual | 0.1021 | better than HARX on all 20 |

The combined model improves on HARX by 1 to 2%, which is not statistically significant on 245
months. Our entanglement-off control shows where the gain comes from: the reservoir's input
encoding, which captures the size of last month's market move, and not its entangling dynamics.
Four richer readouts of the reservoir in this hybrid (qubit correlators, measuring at two times,
extra measurement bases, Hamiltonian encoding) did not beat that control either.

**What it does not bring.** No version beats a linear model given the same information, alone or
combined. This is not a quantum advantage.

## Who is this for?

- **Practitioners evaluating quantum machine learning for financial time series**, who need to
  know what such a model delivers once shots, noise and a fair baseline are included.
- **Researchers working on quantum reservoirs or kernels**, who can reuse the noise-matched
  readout and the entanglement-off control on any model with a classical linear readout.
- **Students**, as a complete worked example of a hybrid model, from data to a run on a
  quantum backend, with every result reproducible.

Useful background: linear regression, the idea of a rolling out-of-sample forecast, and basic
Qiskit (qubits, rotations, measurement). No finance background is needed beyond knowing that
volatility measures how much prices move.

## How the quantum part works?

- **Quantum subroutine.** Ten qubits: seven carry one market feature each (volatility, market
  return and five macroeconomic series), loaded as `RY` rotations, and three act as memory. Three
  months of data are fed in turn. After each month all ten qubits evolve under a fixed, random,
  fully connected Ising interaction (`RXX` and `RZ` gates), and the input qubits are discarded
  before the next month is loaded. At the end all ten qubits are measured. The circuit has 270
  two-qubit gates and is never trained.
- **Classical part.** A ridge regression with ten weights maps the ten measured averages to the
  forecast, refitted every month on a rolling window.
- **On a noisy device.** The circuit stays the same. Only the ten classical weights are retrained,
  on features simulated under noise matched to the device. One calibration circuit sets the noise
  level.
- **Two layouts, same result.** Qollab's built-in simulator runs 10 qubits with a reset between
  months. Its IonQ backends do not accept reset, so they run an equivalent 24-qubit circuit with
  fresh input qubits for each month.

## How do I run it?

**On Qollab**

1. Open the project (demo link above), or build the upload with
   `uv run python scripts/make_qollab_zip.py` and create a project from the zip.
2. Pick a backend in Select QPU: the built-in simulator or the IonQ Forte 1 simulator.
3. Set the fields and run: `shots` (default 1024; 500 takes about 160 seconds on the IonQ
   simulator), `month` (`2017-10`, `2017-11` or `2017-12`), `all_three_months`, `mid_circuit`,
   `plots`.

Each run submits one circuit as one job and shows the findings, the circuit, the measured values,
and the forecast next to the classical models' forecasts. Training and the benchmark were computed
locally and ship as frozen weights, because the platform allows one job per run.

**Locally**

```bash
uv sync
uv run --with pytest python -m pytest tests -q        # 12 tests
uv run python scripts/run_demo_local.py               # the Qollab run, on a laptop
```

`README.md` has the full instructions and `scripts/README.md` lists every experiment.

## How we meet the judging criteria

### Novelty and Creativity

Not a tutorial: a technical extension and an original analytical contribution on top of a
published model.

- **A noise-matched readout (noise mitigation).** A reservoir trained in ideal simulation fails on
  a noisy device. Retraining only its ten classical weights under matched noise restores the
  accuracy, with no change to the circuit.
- **Calibrating a noise model from one circuit.** A generic depolarizing model at 1.6% per
  two-qubit gate, set from a single circuit on IonQ's simulator, also fits two further months run
  afterwards without adjustment (chi-squared 36 on 30 measured values).
- **An entanglement-off control (benchmarking).** Running the same model without its entangling
  gates separates what the quantum dynamics contribute from what the classical input encoding
  contributes. It shows the dynamics help the reservoir alone and add nothing once a linear model
  has the same inputs.
- **A reservoir-plus-HARX hybrid, with controls.** Combining the quantum reservoir with the best
  linear model, testing four richer quantum readouts in that hybrid, and checking each against a
  classical control is, as far as we found, new for volatility forecasting.
- **An independent reproduction that changes a published ranking.** Scored against the right
  month, HARX (0.1030) is ahead of the paper's quantum model; the paper's single reservoir
  (0.1051) is at the favourable edge of our 20-seed range (0.1056 to 0.1133).

### Technical Execution

- **Runs on Qollab.** One circuit and one job per run, on the built-in simulator and on the IonQ
  Forte 1 simulator (24 qubits, 270 two-qubit gates). Seven IonQ simulator runs over three months
  are recorded in `data/ionq_runs.json`.
- **Three implementations that agree.** An exact NumPy reference, the Qiskit circuit (matches the
  reference to 1e-10) and the 24-qubit layout (same output distribution).
- **Verified against the original.** Given the authors' couplings, our model reproduces their
  saved forecasts to 1.2e-6.
- **Tested.** Twelve acceptance tests cover the model, both circuit layouts and the figures.
- **Compared with the classical baseline at every step.** Like for like by inputs, at equal weight
  count, with matched regularization, at training windows from 24 to 570 months, and on 2,013
  daily forecasts. A shots and noise study covers about 20,000 simulated noisy circuits.
- **Known issues.** Nothing ran on quantum hardware; the IonQ results are from the noise-model
  simulator. The local noise model is generic, not IonQ's. In the three demo months QR1's forecast
  was less accurate than every classical model's. Three input features are scaled using the whole
  sample, as in the paper.

### Impact Potential

- **A method others can use now.** The noise-matched readout and one-circuit calibration apply to
  any quantum reservoir or kernel model with a linear readout, on any noisy backend.
- **A test others can use now.** The entanglement-off control is cheap and tells a team whether
  their quantum model's accuracy comes from quantum dynamics or from its input encoding.
- **A fair reference point for the field.** A corrected, reproducible benchmark for quantum
  reservoirs on volatility, with the cost stated: minutes per forecast on a simulator against
  under a millisecond for the linear model, and a gate count that grows with the square of the
  qubit count.
- **A lead for follow-up work.** The reservoir discards the sign of its inputs, which the daily
  leverage effect needs. An encoding or readout that keeps it is the clear next experiment.

### Documentation Quality

- **`README.md`**: findings, how to run on Qollab and locally, results with figures, layout.
- **`docs/qrc_spec.md`**: the model specification, noise models and acceptance checks.
- **`scripts/README.md`**: every experiment, grouped by the finding it supports, with what it costs
  to run.
- **`main.py`**: its header explains what runs on the quantum backend and what was computed
  locally, and its output walks through the findings.
- **Figures** in `docs/figures/`, each stating where its numbers came from, with alt text and a
  colour-blind-safe palette.
- **Demo video.**
- **To replicate:** every table and figure has a script, and every result is committed as CSV.
  **To extend:** change the encoding or readout in `src/qrc/`, rerun `scripts/bench_qrc.py` and
  `scripts/controls_qrc.py`. The hybrid experiments are `scripts/hybrid_qrc.py` and
  `scripts/hybrid_variants.py`.

## Disclosures and credits

- **Prior work.** The model, task and protocol come from Li, Mukhopadhyay, Bayat and Habibnia,
  "Quantum Reservoir Computing for Realized Volatility Forecasting", arXiv:2505.13933. Their
  repository has no license, so we read it and checked our results against it, and copied no code
  or files from it.
- **Libraries.** Qiskit and Qiskit Aer (Apache-2.0), NumPy, pandas, SciPy and scikit-learn
  (BSD-3-Clause), reservoirpy (MIT), matplotlib (BSD-style), requests (Apache-2.0), xlrd (BSD),
  pytest (MIT). No third-party source code or templates are included.
- **Data.** Public series from Yahoo Finance, the Kenneth R. French Data Library, Robert Shiller's
  online data and FRED. The repository holds derived values; each source has its own terms.
- **AI tools.** The team used Claude (Anthropic) as an AI assistant throughout, for planning, code,
  experiments, figures and drafting documentation, under the team's direction and review.
- We confirm we are licensed to use all of the above. Full details are in `DISCLOSURES.md`.

## License

This work is distributed under the MIT License. See [LICENSE](../LICENSE).

## Attribution

This project was created during the Qollab x IonQ Hackathon 2026, and is supported by Qollab & IonQ.
