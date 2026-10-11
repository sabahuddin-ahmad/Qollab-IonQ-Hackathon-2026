# Project page text

Draft copy for the Qollab project page. Theme tag: **Quantum Impact**.

## Title

Quantum reservoirs for volatility forecasting: where the quantum part pays, and how to keep it
paying on a noisy device

## One-line summary

A hybrid quantum-classical forecaster for S&P 500 volatility, benchmarked fairly against classical
models and made to work under device noise on IonQ's Forte 1 simulator.

## Short description (about 60 words)

We rebuilt a published 10-qubit quantum reservoir for forecasting stock-market volatility and
tested it against seven classical models. It beats the classical reservoir with half the trained
weights. On a noisy device it fails unless its readout is trained under matching noise; with that
fix, shown on IonQ's simulator, it keeps its accuracy. The best linear model remains slightly ahead.

## The problem

Volatility forecasts set risk limits, option prices and capital requirements. They are also hard to
improve: simple linear models from econometrics are strong baselines. A 2025 paper reported that a
small quantum reservoir beats them on monthly S&P 500 volatility. We set out to check that claim
and to find out what it takes to run such a model on real quantum hardware.

## The approach

A hybrid pipeline with a clear division of labour:

- **Quantum subroutine.** Ten qubits. Seven carry one market feature each (volatility, market
  return and five macroeconomic series), loaded as rotations; three act as memory. Three months of
  data are fed in turn, with all ten qubits evolving under a fixed, random, fully connected Ising
  interaction after each month. The circuit is never trained.
- **Classical part.** A ridge regression with ten weights turns the ten measured averages into a
  forecast, refitted every month on a rolling window.

Data are public: Yahoo Finance, the Kenneth French library, Robert Shiller's data and FRED,
rebuilt by script. Everything is written in Qiskit and released under the MIT license.

## What we found

**1. A fair benchmark.** The published classical baselines were scored one month late. Scored
correctly, and compared only with models given the same inputs, the quantum reservoir ranks third
of four: behind the linear models HARX (0.1030) and ARMAX (0.1070), ahead of the classical
reservoir (0.1224), with an error of 0.1102 averaged over 20 random reservoirs.

**2. Where the quantum subroutine makes a difference.**

- Against the classical reservoir given the same inputs: 10% lower error with 10 trained weights
  instead of 21, on all 20 reservoirs.
- With entanglement switched off, the same model is 9% worse (p = 0.04). The entangling dynamics
  are what carry memory between months.
- Its trained weights stay fixed as more history is fed in: five weights reach within 2% of an
  autoregression that needs 13.

**3. Making it work on a noisy device.** A readout trained in noiseless simulation fails under
device noise: at 1% gate error the forecast error rises from 0.109 to 0.47. Retraining only the ten
classical weights under matched noise brings it back to 0.113. The quantum circuit is unchanged.
On IonQ's Forte 1 simulator, one calibration circuit was enough to set the noise level, which then
matched two further months without adjustment.

**4. What the controls say.** Once a linear model has the same inputs, the entangling dynamics add
nothing further. The useful ingredient in a hybrid with HARX is the input encoding, a classical
transform, and adding it to HARX gives the most accurate model in the study (0.0999). The result
held when we changed the lags, the training window, the data frequency (2,013 daily forecasts) and
the amount of training data.

## Impact

- **For teams evaluating quantum machine learning in finance:** a measured answer to what a small
  quantum reservoir delivers once shots, noise and a fair baseline are included.
- **A method that transfers:** the noise-matched readout and one-circuit calibration apply to any
  quantum model with a classical linear readout.
- **A test that transfers:** switching the entangling gates off separates quantum contribution from
  classical input encoding, at almost no cost.

## Limits

- This is not a quantum advantage. The best linear model is slightly more accurate, and ten qubits
  can be simulated exactly on a laptop.
- Nothing ran on quantum hardware. The IonQ results are from the Forte 1 noise-model simulator.
- Conclusions rest on 245 monthly and 2,013 daily forecasts of one index. Differences of 2 to 3%
  between the best models are not statistically significant.

## Run it

Select a backend and press run. Each run builds one forecast circuit, submits it as a single job
and prints the measured values, the forecast, the realized value and the classical forecasts for
the same month.

- `shots`: shots for the circuit (default 1024; 500 takes about 160 seconds on the IonQ simulator)
- `month`: `2017-10`, `2017-11` or `2017-12`
- `all_three_months`: all three months in one job (built-in simulator only)
- `mid_circuit`: on IonQ backends, also measure the earlier months' qubits mid-circuit
- `plots`: show figures (default on); text charts when off

On the built-in simulator the circuit has 10 qubits with a reset between months. IonQ backends do
not accept reset, so they receive an equivalent 24-qubit circuit and noise-matched weights.
Training and the benchmark tables were computed locally and ship with the project as frozen
weights.

## Links

- Repository: https://github.com/sabahuddin-ahmad/Qollab-IonQ-Hackathon-2026
- Summary by judging criterion: `docs/submission_summary.md`
- Model specification: `docs/qrc_spec.md`
- Disclosures (prior work, libraries, data, AI assistance): `DISCLOSURES.md`

## Team

Ali Karimi, Ali Mahani, Sabah Ud Din Ahmad.

## Reference

Q. Li, C. Mukhopadhyay, A. Bayat, A. Habibnia, "Quantum Reservoir Computing for Realized Volatility
Forecasting", arXiv:2505.13933.

This effort is supported by Qollab & IonQ.
