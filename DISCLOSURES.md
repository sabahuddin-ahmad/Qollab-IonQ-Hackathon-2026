# Disclosures

Project: Quantum Reservoir Computing for Realized Volatility, Qollab × IonQ Hackathon 2026,
Quantum Impact theme. Everything in this repository is released under the MIT license (see
`LICENSE`), subject to the notes on data below.

This effort is supported by Qollab & IonQ.

## Team and who did what

| Person | Role | Main contributions |
| --- | --- | --- |
| Ali Karimi | Lead, ML and quantum ML | Model specification, quantum reservoir code, noise study, Qollab entry point, figures |
| Ali Mahani | Classical reservoir computing and ML | Data pull and validation, classical benchmarks, evaluation harness, audit of the authors' code |
| Sabah Ud Din Ahmad | Quantum computing, Qiskit | Repository, Qollab platform constraints, review of the circuit design |

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
| matplotlib, pylatexenc | Figures (development only) | Matplotlib license (BSD-style), MIT |
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

- **Claude Code (Anthropic)** was used throughout by Ali Karimi: planning, drafting the model
  specification, writing and testing the quantum reservoir code, the noise study, the Qollab
  entry point, the figures and the first drafts of the documentation, including this file. The
  team reviewed the output and directed the work; commits made with it carry a co-author line.
- **Paper2Agent** (`jmiao24/Paper2Agent`) was run locally on the paper and the authors'
  repository to explore them. Its output is derived from the authors' unlicensed code and is not
  part of this submission.
- **Google Gemini** was used in one early discussion about possible extensions of the paper.
- TODO (Ali Mahani, Sabah): list any AI tools you used, or state none.

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

TODO: confirm that all work was done inside the hackathon window, with its start date. The first
commit and the commit history give the dates.
