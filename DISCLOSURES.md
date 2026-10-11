# Disclosures

Project: Quantum Reservoir Computing for Realized Volatility, Qollab × IonQ Hackathon 2026,
Quantum Impact theme. Everything in this repository is released under the MIT license (see
`LICENSE`), subject to the notes on data below.

This effort is supported by Qollab & IonQ.

## Team

Team QuanTech: Sabah Ud Din Ahmad, Ali Karimi and Ali Mahani. This is a joint submission. The team
together designed the study, built and audited the classical benchmarks, wrote the quantum
reservoir and the Qollab entry point, ran the experiments and reviewed every result; the commit
history records who committed what.

## Prior work this project builds on

The project reproduces and extends:

> Qingyu Li, Chiranjib Mukhopadhyay, Abolfazl Bayat, Ali Habibnia, "Quantum Reservoir Computing
> for Realized Volatility Forecasting", arXiv:2505.13933.

- **What we took from the paper:** the forecasting task, the QR1 model design (10 qubits, seven
  input features, three months of memory, transverse-field Ising reservoir, ridge readout), the
  rolling evaluation protocol and the set of classical baselines.
- **What we did not take:** code. The authors' repository
  (`LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting`) has no license,
  so none of its code or files are included here. All code in this repository was written for
  this project.
- **How we used the authors' repository:** we read it to resolve details the paper leaves open,
  compared our data panel column by column against their `Data.CSV`, and checked our QR1
  implementation against their saved forecasts (agreement to 1.2e-6 when given their couplings).
  Those files stayed on our machines and are not redistributed.
- **Where we differ from the paper:** see "Findings from auditing the authors' repository" in
  `README.md` and section 8 of `docs/qrc_spec.md`. In short, we score every model with the
  forecast aligned to the right month, which changes the ranking of the HAR-family baselines.

## Outside code and libraries

No third-party source code is copied into this repository. It depends on these packages:

| Package | Used for | License |
| --- | --- | --- |
| Qiskit (>=2.5.2, <3) | Circuits; the only dependency of the Qollab entry point besides NumPy | Apache-2.0 |
| Qiskit Aer 0.17.2 | Local noisy simulation (not used on Qollab) | Apache-2.0 |
| NumPy | Exact reference simulation, readout | BSD-3-Clause |
| pandas | Data handling | BSD-3-Clause |
| SciPy | Test statistics in the harness | BSD-3-Clause |
| scikit-learn | Ridge regression in the classical reservoir baseline | BSD-3-Clause |
| reservoirpy 0.4.2 | Classical reservoir baselines (RC, RCX) | MIT |
| requests, xlrd | Data download and parsing | Apache-2.0, BSD |
| matplotlib | Figures, in the repository and in the Qollab run | Matplotlib license (BSD-style) |
| pylatexenc | Circuit drawings (development only) | MIT |
| pytest | Tests (development only) | MIT |

## Data

`data/rv_monthly.csv` is rebuilt by `scripts/pull_data.py` from public sources: S&P 500 daily
closes from Yahoo Finance, factor returns from the Kenneth R. French Data Library, dividend,
earnings and CPI series from Robert Shiller's online data, and industrial production, T-bill and
corporate spread series from FRED (Federal Reserve Bank of St. Louis). The file holds derived
monthly values, not the raw downloads. Each source has its own terms of use; the MIT license on
this repository covers our code and does not relicense the underlying data.

`data/spx_daily.csv` holds daily log volatility (Garman-Klass estimate) and log returns of the
S&P 500, 1999 to 2018, derived by `scripts/bench_daily.py` from Yahoo Finance daily prices. The same
note on terms of use applies.

Everything else under `data/` and `demo/` was produced by the scripts in this repository.

## Use of AI tools

The team used Claude (Anthropic) as an AI assistant throughout the project, for planning, writing
and testing code, running experiments, making figures and drafting documentation. The team directed
the work, reviewed the output and is responsible for all content in this repository.

## What ran where

- **On Qollab (Qiskit, IonQ simulator):** `main.py` builds one forecast circuit and runs it as
  one job. It ran on Qollab's built-in simulator and on the IonQ Forte 1 noise-model simulator
  (24 qubits, 270 two-qubit gates; runs recorded in `data/ionq_runs.json`).
- **Locally:** training the readout weights, all benchmark tables, the shots and noise study
  (Qiskit Aer, generic depolarizing and damping noise) and the scaling measurements. The frozen
  weights and tables used by `main.py` are shipped in `demo/qr1_demo.json`.
- **Not done:** nothing was run on IonQ quantum hardware. The local noise model is generic and
  was calibrated to IonQ's simulator on one circuit; it is not IonQ's noise model.

## Timeline

The repository was created on 2026-10-09 and the work was done between then and the submission on
2026-10-11. The commit history gives the dates.
