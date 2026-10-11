# Scripts

Everything here runs locally with `uv run python scripts/<name>.py`. Nothing here runs on Qollab;
the Qollab entry point is `main.py` at the top of the repository. Results are written to `data/`
and figures to `docs/figures/`, and both are committed, so nothing needs re-running to read the
results.

"Simulates circuits" means minutes to hours on a laptop. Everything else takes seconds to a few
minutes and uses exact NumPy simulation or saved results.

## Run the demo

| Script | What it does |
| --- | --- |
| `run_demo_local.py` | Runs `main.py` on a laptop the way Qollab does. `--as-ionq` exercises the 24-qubit path on a noiseless local stand-in, so its forecast is off by design: the weights it uses expect IonQ's noise |
| `make_demo.py` | Builds `demo/qr1_demo.json`: inputs, frozen readout weights and the benchmark table that `main.py` reads |
| `make_qollab_zip.py` | Packs the nine files a Qollab run needs into one zip |

## 1. The claim: data and the model

| Script | What it does |
| --- | --- |
| `pull_data.py` | Rebuilds the monthly panel from Yahoo Finance, French, Shiller and FRED |
| `bench_qrc.py` | QR1 in exact simulation over 20 coupling seeds |
| `draw_circuit.py` | Circuit drawings |

## 2. A fair benchmark

| Script | What it does |
| --- | --- |
| `bench_classical.py` | AR, HAR, ARMAX, HARX and a classical reservoir, with the paper's scoring and the corrected one |
| `bench_rc.py` | The paper's classical reservoirs, as in the authors' code |
| `bench_reduced.py` | HARX and ARMAX cut to 10 weights, for a like-for-like size comparison |
| `bench_univariate.py` | Volatility-only models, including a volatility-only quantum reservoir; `--lags` matches lag for lag |
| `bench_windows.py` | The main models at training windows from 24 to 570 months |
| `bench_daily.py` | Daily forecasts (2,013 test days); `--sweeps` and `--corner` vary history and training length |
| `bench_daily_lowdata.py` | Low-data control against same-size classical reservoirs, at 4, 6 and 8 qubits |

## 3. Making it work on a noisy device

| Script | What it does |
| --- | --- |
| `sweep_qrc.py simulate` | **Simulates circuits** (about 20,000, on Qiskit Aer) under gate noise and damping; caches outcome probabilities in `cache/` |
| `sweep_qrc.py analyze` | Scores the cached results at each shot count, with noise-matched and noiseless-trained readouts |
| `noisy_forecasts.py` | Month-by-month forecasts under noise, from the cache |
| `transpile_qr1.py` | Gate counts after transpiling to H, RZ and CX, and the cost of a full run |
| `bench_complexity.py`, `bench_scaling.py` | Size, time per forecast and how simulation time grows with qubits (`bench_scaling.py` **simulates circuits**, small ones) |

The IonQ simulator runs themselves were done on Qollab with `main.py`; their outputs are recorded
in `data/ionq_runs.json`.

## 4. What the quantum part contributes

| Script | What it does |
| --- | --- |
| `controls_qrc.py` | Entangling gates off and at scaled strength, alone and combined with HARX; classical transforms; significance tests |
| `hybrid_qrc.py` | HARX's inputs plus the reservoir's values in one readout; `--noise` re-scores it under noise from the cache |
| `hybrid_variants.py` | Correlators, temporal multiplexing, extra measurement bases and Hamiltonian encoding, each with a gates-off control |
| `stream_qrc.py`, `extra_features_qrc.py` | A continuous-stream reservoir, and discarded qubits read as extra features |

## Figures

| Script | What it does |
| --- | --- |
| `make_result_figures.py` | All result figures in `docs/figures/`, with alt text |
| `make_figures.py` | Reproduction of the paper's forecast grid (Fig. 7) |
