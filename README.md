# Quantum Reservoir Computing for Realized Volatility

A reproduction of Li et al.'s quantum reservoir computing (QRC) benchmark for S&P 500
monthly realized volatility, built for the Qollab × IonQ Hackathon 2026. The repository
currently contains the full classical benchmark suite from the paper, replicated in
Python against freshly pulled open data. The Qiskit quantum reservoir (QR1/QR2) is the
remaining piece and is tracked below as TBD.

Reference paper: Qingyu Li, Chiranjib Mukhopadhyay, Abolfazl Bayat, Ali Habibnia,
"Quantum Reservoir Computing for Realized Volatility Forecasting",
[arXiv:2505.13933v2](https://arxiv.org/html/2505.13933v2).
The authors' code is at
[LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting](https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting);
several findings below come from auditing that repository, and this project cites it
as the source of the benchmark protocol.

```bibtex
@article{li2025qrcrv,
  title   = {Quantum Reservoir Computing for Realized Volatility Forecasting},
  author  = {Li, Qingyu and Mukhopadhyay, Chiranjib and Bayat, Abolfazl and Habibnia, Ali},
  journal = {arXiv preprint arXiv:2505.13933v2},
  year    = {2025}
}
```

## Setup

Python 3.14 is required to match the Qollab runtime. The environment is managed with
`uv`, and `pyproject.toml` is the single source of truth for dependencies:

```bash
uv sync
.venv/bin/python -c "import qiskit, sklearn, reservoirpy; print('ok')"
```

Qiskit is pinned to `>=2.5.2,<3.0.0` to match Qollab's runtime default, and
`qiskit-aer==0.17.2` matches Qollab's PQCA library pin. The packages `xlrd` and
`reservoirpy` are local-only (neither has a Pyodide build) and must stay out of any
Qollab `requirements.txt`; the Qollab demo will run inference-only with frozen weights.

## Data

`scripts/pull_data.py` rebuilds the paper's monthly panel from open, keyless sources:
S&P 500 daily closes from the Yahoo Finance chart API (monthly log realized volatility
plus quarterly/annual averages), Fama–French monthly factors from the Kenneth French
data library, dividend/earnings/CPI histories from Shiller's `ie_data.xls`, and
industrial production, T-bill, and default-spread series from FRED. The output is
`data/rv_monthly.csv`: 816 months (January 1950–December 2017) by 15 columns, about
173 KB, comfortably inside Qollab's 2 MB per-file cap.

The downloaded panel was validated column-by-column against the authors' `Data.CSV`.
The RV, T-bill, default-spread, and market-factor series are identical up to the
authors' normalization; dividend, earnings, size, value, reversal, and production
series match up to small data-vintage revisions (R² > 0.9999); the inflation series is
related but comes from a different CPI variant (ρ ≈ 0.88), which is immaterial at its
scale. No data difference affects any benchmark conclusion.

## Replicated benchmarks

`scripts/bench_classical.py` implements the AR1/AR3/ARMAX/HAR/HARX OLS benchmarks and
a teacher-forced echo-state baseline under the paper's rolling protocol: a fixed
estimation window, refitting every month, 245 one-step-ahead out-of-sample forecasts
from August 1997 to December 2017. `scripts/bench_rc.py` additionally ports the
authors' exact reservoir code — a 3-step stateless echo-state network reset per sample
with `Ridge(1e-7)` readout — to the installed reservoirpy version. OOS predictions are
saved to `data/oos_classical.csv`, which is the comparison harness the quantum models
will drop into. Metrics follow the authors' code: mean squared error on log realized
volatility, and QLIKE as the Patton loss summed over the out-of-sample period on
absolute log volatility (this differs from the formula printed in the paper and was
verified numerically).

| Model         | This repo (MSE / QLIKE)  | Paper Table 2   | Assessment                                             |
| ------------- | ------------------------ | --------------- | ------------------------------------------------------ |
| AR1           | 0.1302 / 1.7172          | 0.1304 / 1.7279 | Reproduced                                             |
| AR3           | 0.1180 / 1.5892          | 0.1178 / 1.5893 | Reproduced                                             |
| HAR           | 0.1479 / 2.0387          | 0.1476 / 2.0431 | Reproduced via the off-by-one finding below            |
| HARX          | 0.1466 / 2.08            | 0.1508 / 2.24   | Structure replicated; small residual from data vintage |
| ARMAX         | 0.1070 / 1.4460          | 0.1145 / 1.6196 | Close; partly different estimator (OLS vs ARIMA)       |
| RC, 50 units  | 0.141–0.153 across seeds | 0.1441 / 2.1011 | Reproduced (their draw lies in-distribution)           |
| RCX, 20 units | best 0.125 observed      | 0.1089 / 1.6480 | Not reproduced; see finding 5                          |

## Findings from auditing the authors' repository

1. The HAR/HARX rolling loop predicts row `end` but evaluates against test rows starting
   at `end+1`, misaligning every forecast by one month. Reproducing that exact
   misalignment reproduces their HAR numbers to three decimals, so their HAR appears
   worse than AR3 for an implementational reason, not a modeling one.
2. The code runs the classical reservoir with `input_scaling=1`, while the paper text
   states 0.1. The code value is what reproduces Table 2.
3. The classical reservoir is a 3-step stateless network (fresh zero state per sample,
   last state kept), not a continuously teacher-forced one.
4. No random seeds are set anywhere in their code, and the files `dff.csv`,
   `Data_raw.csv`, and the differencing script are absent from the repo, so
   point-exact reproduction of the stochastic models is impossible; distributional
   agreement is the achievable standard. HARX additionally uses an undocumented
   293-observation window with four exogenous inputs.
5. The reported RCX value (0.1089) was not reached in any of 15+ draws across two
   independent implementations (best observed 0.125). It should be treated as an
   optimistic draw. The quantum comparison in this project will therefore use our own
   RCX run under the identical protocol rather than the printed number.

## Quantum reservoir model (TBD)

## Layout

```
scripts/pull_data.py       monthly panel pull (Yahoo / French / Shiller / FRED)
scripts/bench_classical.py AR/HAR-family OLS benchmarks + teacher-forced ESN + DM table
scripts/bench_rc.py        paper-faithful 3-step stateless reservoirpy RC/RCX
data/rv_monthly.csv        monthly panel, 816 rows x 15 columns
data/oos_classical.csv     out-of-sample predictions incl. reservoir baselines
src/qrc/                   quantum reservoir implementation (TBD)
```
