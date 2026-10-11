# QR1 specification

Status: draft for review by Ali K. Owner of the implementation: Sabah (`src/qrc/circuit.py`), readout by Ali K (`src/qrc/readout.py`).

This describes the quantum reservoir "QR1" of Li et al., *Quantum Reservoir Computing for Realized Volatility Forecasting* (arXiv:2505.13933), precisely enough to implement it three ways: an exact reference, a Qiskit circuit for local Aer studies, and the single-job Qollab demo.

**How far to trust it.** The description was checked by re-implementing it from scratch in NumPy and running it privately on the authors' released data and coupling matrix: it reproduces their saved QR1 forecasts to 1.2e-6 (their GPU code is single precision) and their printed MSE 0.10508. The accuracy table in section 8 is produced by code in this repo (`scripts/bench_qrc.py`), and the acceptance checks in section 7 are `tests/test_qrc.py`. The Trotter and noise tables in sections 6 and 6a still come from a throwaway probe on a single random input and must be reproduced on real months before we quote them.

**Provenance.** Written from the paper and from reading the authors' released code (unlicensed; consulted, nothing copied). Couplings are drawn by us. Record this in `DISCLOSURES.md`.

## 1. Model in one paragraph

Ten qubits: 7 input, 3 hidden. For each forecast month t, the reservoir is fed the three previous months in order (t-3, t-2, t-1). Each month's seven features are written onto the input qubits as single-qubit rotations, the whole register evolves under a fixed random Ising Hamiltonian for time τ = 1, and the input qubits are then discarded so only the 3 hidden qubits carry memory forward. After the third month the ten ⟨Z⟩ expectation values are read out. A ridge regression on those ten numbers predicts log realized volatility for month t. Nothing in the reservoir is trained.

## 2. Features

Seven features per month, one per input qubit, in this order. Source column names are from `data/rv_monthly.csv`.

| Qubit | Feature | Built from | Scaling | Range |
| --- | --- | --- | --- | --- |
| 0 | RV | `RV_log` | min-max to [-1, 0] | [-1, 0] |
| 1 | MKT | `MKT` | min-max to [-1, 1] | [-1, 1] |
| 2 | DP | `DP` | min-max to [0, 1] | [0, 1] |
| 3 | IP | `IP` | divide by 100 (percent to fraction), no min-max | about ±0.06 |
| 4 | RV_q | mean of `RV_log` over months m-3, m-2, m-1 | divide by its largest absolute value | [-1, -0.36] |
| 5 | STR | `STR` | min-max to [-1, 1] | [-1, 1] |
| 6 | DEF | `DEF` | min-max to [0, 1] | [0, 1] |

Notes:

- `RV_q` for month m already looks back three months, so `harness.py`'s `y_Q` column is exactly `RV_q` before scaling. Do not use the panel's `RVq` column: it averages raw volatility and correlates only 0.85 with the authors' feature.
- Rotation angle is `θ = π · x`. IP therefore barely rotates its qubit (|θ| < 0.2 rad); this is what the authors do, so keep it for the paper-faithful model.
- **Scaling constants, two protocols.** *Paper protocol:* extremes taken over the full 1950–2017 sample, as the authors do. This looks ahead for three features, whose extremes fall in the test period: STR (Nov 2000, Jan 2001), DEF (Dec 2008) and DP (Aug 2000). The target's extremes are both in the training period (Feb 1964, Oct 1987), so the target scaling does not look ahead. *Clean protocol:* extremes taken from data before August 1997 only. Implement the scaler with a `fit_end` date so both are one argument apart; report the paper protocol as the headline and the clean protocol as a robustness line.
- Start-up rows: the authors zero-fill undefined `RV_q` and leave the first three feature rows at zero. Our harness drops rows with incomplete lags instead, and the rolling window never reaches them in the out-of-sample period, so no special handling is needed.

## 3. Reservoir

- Hamiltonian: `H = Σ_{i<j} J_ij X_i X_j + Σ_i Z_i` over all 10 qubits (all 45 pairs; field strength fixed at 1).
- Couplings: draw a 10×10 matrix of uniform(0, 1) numbers, symmetrize it (`(M + Mᵀ)/2`), zero the diagonal, divide by its largest eigenvalue. Use `numpy.random.default_rng(seed)`; the seed is the only thing that identifies a reservoir. Commit the seeds, not the matrices.
- Evolution time τ = 1 after each month's encoding. Lags K = 3.
- Qubits 0–6 are inputs, 7–9 hidden. `J` is indexed by these qubit numbers.

## 4. Procedure for one forecast month t

1. Hidden qubits start in |000⟩.
2. For lag k = 3, 2, 1 (oldest first), using the features of month t-k:
   1. Put the input qubits in |0…0⟩ (a reset, except the first time) and apply `RY(π · x_q)` on input qubit q.
   2. Evolve all 10 qubits by `exp(-i H τ)`.
   3. If k > 1: discard the input qubits (partial trace), keeping the hidden state. If k = 1: go to step 3.
3. Measure ⟨Z_q⟩ for q = 0…9. These ten numbers are the feature vector `z(t)`.

**Alignment.** `z(t)` uses months t-3…t-1 only and is paired with the target of month t. There is no one-month offset in the quantum model. In the authors' code the offset exists only in the HAR and HARX loops; their AR, ARMAX, reservoir and QR numbers are correctly aligned.

> For Ali M: `harness.score_both` applies the "paper" offset (`pred[:-1]` vs `actual[1:]`) to every model. To reproduce the paper's table it should apply to HAR and HARX only. Scoring AR1, AR3, ARMAX, RC, RCX or QR1 that way will not match the paper's numbers for them.

## 5. Readout

- Target: `RV_log` scaled to [-1, 0] with the same constants as feature 0. Predictions are mapped back to log RV before scoring.
- Model: ridge regression with penalty 1e-8 and **no intercept**: `W = y Zᵀ (Z Zᵀ + 1e-8·I)⁻¹`, prediction `W · z(t)`. Because there is no intercept, the target must be in the scaled space; fitting on raw log RV gives a different model. In our seed test the result was insensitive to the penalty from 1e-8 to 1e-2.
- Rolling window: the authors use the previous 571 months; `harness.WIN` is 570. Use the harness value so every model shares one window, and note the one-month difference in the write-up.
- Trained parameters: 10. Expose `n_params = 10` to the harness.
- For the Qollab demo, freeze `W` from the training window of the demo month and ship it as a small array.

## 6. Three implementations of the same map

| | Exact reference | Qiskit circuit on local Aer | Qollab demo |
| --- | --- | --- | --- |
| Purpose | Headline table, seed spread, ground truth for the circuit | Shot-noise and noise studies | The one job judges run |
| Evolution | `scipy.linalg.expm(-1j·H)` on the 1024-dim space | Trotterized gates (below) | Same circuit as the Aer version |
| Discarding inputs | Partial trace | `reset` on qubits 0–6 (use the `density_matrix` method so reset is exact) | `reset` if supported, otherwise the 24-qubit version below |
| Readout | Exact ⟨Z⟩ | Exact from the density matrix, or sampled with shots | Counts from one job |
| Cost | About 3 s for all 816 months | Seconds per month | One circuit |

**Trotter step (second order), repeated r times per lag:** `RZ(τ/r)` on every qubit, then `RXX(2·J_ij·τ/r)` on all 45 pairs, then `RZ(τ/r)` on every qubit. This uses Qiskit's conventions `RZ(φ) = exp(-iφZ/2)` and `RXX(θ) = exp(-iθ XX/2)`. The XX terms commute with each other, so their order does not matter. XX is the native IonQ two-qubit interaction (Mølmer–Sørensen), which is the concrete "why trapped ions" point.

Measured Trotter error against the exact evolution (seed-0 couplings, features are in [-1, 1]):

| Order | r | Two-qubit gates per month | Max feature error | MSE, readout trained on these features | MSE, exact-trained weights on these features |
| --- | --- | --- | --- | --- | --- |
| exact | – | – | 0 | 0.1099 | 0.1099 |
| 1 | 2 | 270 | 0.20 | 0.1146 | 0.1179 |
| 1 | 4 | 540 | 0.083 | 0.1107 | 0.1112 |
| 2 | 2 | 270 | 0.041 | 0.1095 | 0.1110 |
| 2 | 4 | 540 | 0.009 | 0.1098 | 0.1100 |
| 2 | 8 | 1080 | 0.002 | 0.1099 | 0.1099 |

The right r depends on noise, because more steps cut Trotter error but add gate error. With depolarizing error 0.004 on every RXX (one random input, Aer density-matrix method), the total feature error against the exact noiseless answer was:

| r | RXX gates per month | Total feature error at 0.004 two-qubit error |
| --- | --- | --- |
| 1 | 135 | 0.26 |
| 2 | 270 | 0.15 |
| 4 | 540 | 0.24 |
| 8 | 1080 | 0.40 |

Use **second order with r = 4 for noiseless runs** and **r = 2 for anything noisy, including the Qollab demo**. Confirm the noisy optimum on real months before relying on it. Train the frozen demo weights on features from the *same* circuit and noise setting the demo uses, so there is no mismatch between training and the demo.

The Qiskit circuit built this way agrees with a NumPy implementation of the same Trotter formula to 1.6e-14, so the angle and ordering conventions above are checked.

## 6a. Noise in the local simulator

Simulate with `AerSimulator(method="density_matrix")` and end the circuit with `save_probabilities()`. That returns the exact distribution over the 1,024 outcomes with noise included, treats `reset` exactly, and makes shots free: cache the probabilities per month and noise setting, then sample shots from them in NumPy.

Two ways to add noise, answering different questions:

- **Gate-error model (hardware realism).** A `NoiseModel` with `depolarizing_error(p2, 2)` on every `rxx` and `depolarizing_error(p2/10, 1)` on every `ry`; `rz` stays noiseless. Sweep p2 over 0, 0.001, 0.004, 0.01, 0.03. Transpile with basis gates `rxx, rz, ry, reset` and `optimization_level=0`, otherwise RXX is decomposed and the noise attaches to the wrong gates. Measured shrinkage of the ⟨Z⟩ features: 0.95, 0.82, 0.62, 0.26 of ideal at the four non-zero levels; about 1.2 s per circuit at r = 4.
- **Explicit channel after each Trotter step (dissipation rate γ).** Append `error.to_instruction()` on every qubit with strength `1 − exp(−2γτ/r)`. Use amplitude damping or depolarizing. Z-dephasing commutes with the Z readout and barely moves the features even at γ = 1, so a sweep over it is uninformative.

Rules for the study:

- Always report noise together with shots. A retrained readout absorbs a uniform shrinkage of the features, so with exact expectation values noise looks nearly harmless; the cost appears as more shots needed for the same precision.
- Report two readout protocols: trained on noisy features (headline; what one would do on hardware) and ideal-trained weights applied to noisy features (how badly a simulator-trained model transfers).
- This is generic, uniform, memoryless noise. It is not IonQ's device noise model: no crosstalk, no angle-dependent gate error, no idling error during resets. Say so wherever the results appear.

**If Qollab's simulator does not accept mid-circuit `reset`:** give every lag its own fresh input qubits and never reuse them. That is 7×3 + 3 = 24 qubits, measured only at the end (read ⟨Z⟩ on the three hidden qubits and the seven lag-1 input qubits). It is mathematically identical to the reset version. Sabah to confirm the simulator's qubit limit.

## 6b. Universal-gate cost (H/RZ/CX) and full-run Big-O

`scripts/transpile_qr1.py` recompiles the exact demo circuit
(`src/qrc/circuit.py:qr1_circuit`) with
`qiskit.compiler.transpile(basis_gates=["h", "rz", "cx", "reset", "measure"],
optimization_level=0)` so the Hadamard cost is explicit:

```bash
.venv/bin/python scripts/transpile_qr1.py [--r 2] [--shots 1024] [--opt 0]
```

Decomposition at `opt=0` (verified in the script output):

- `1x RXX -> 4x H + 2x CX + 1x RZ`
- `1x RY -> 2x H + 3x RZ`
- `1x RZ -> 1x RZ`

Per forecast month (`N=10` qubits, `K=3` lags, `r` Trotter steps, `N_input=7`):
native `RXX = K*r*N(N-1)/2 = 135*r` (270 at `r=2`, 540 at `r=4`),
transpiled `CX = 2*RXX`, `H = 4*RXX + 2*K*N_input`, `RZ = RXX + 2*K*r*N + 3*K*N_input`
(at `r=2`: `540 CX, 1122 H, 453 RZ`; depth `85 -> 365` on the 10-qubit layout,
`81 -> 353` on the 24-qubit unrolled layout). `opt=1` cancels `H-H` pairs
(`162 H`, same `CX`/`RZ`) but keeps the same order.

Notations: `S` = shots per circuit, `P = 816` panel rows (`804` in the harness),
`WIN = 570` rolling window, `F = 245` folds, `d = 10` readout features,
`M_train = 803` unique window rows, `M_test = 245` OOS months (from
`src/harness.py:iter_folds`; the script asserts these). Full unique job is
`804` circuits because test months reappear in later training windows.

| Phase | Quantum | Classical |
| --- | --- | --- |
| Train | `O(M_train*S*K*r*N^2)` gate-shots (`CX~4.4e8, H~9.2e8, RZ~3.7e8` at `S=1024, r=2`) | `O(P)` scaling + `O(M_train*2^N*N)` counts-to-`<Z>` + `O(F*WIN*d^2+F*d^3)` ridge fits |
| Test | `O(M_test*S*K*r*N^2)` gate-shots (`CX~1.4e8, H~2.8e8, RZ~1.1e8` at `S=1024, r=2`) | `O(M_test*2^N*N)` counts-to-`<Z>` + `O(M_test*d)` predict + `O(M_test)` score |

The IonQ native two-qubit gate is `RXX` (Molmer-Sorensen); the `H/CX` form is
for cost accounting only and must not be submitted to the backend.

## 7. Acceptance checks before anything is reported

1. Exact reference, alignment: the feature vector for month t does not change when any data from month t onward is altered.
2. Circuit against exact: on 20 months, second-order r = 4 features agree with the exact reference to within 0.01, using the density-matrix method and no shots.
3. Reset against 24-qubit unrolled: identical features on a few months (statevector, small r).
4. Plausibility: exact QR1 on our panel lands near MSE 0.11 (section 8). A value far from that means a scaling, ordering or alignment bug.

## 8. What to expect

**This repo's numbers** (`scripts/bench_qrc.py` and `scripts/bench_classical.py`, same harness, correctly aligned, out-of-sample MSE on log RV over the 245 test months):

| Model | MSE | Trained parameters |
| --- | --- | --- |
| HARX | 0.1030 | 14 |
| ARMAX | 0.1070 | 14 |
| **QR1, exact, mean of 20 coupling seeds** | **0.1102** (sd 0.0020, range 0.1056–0.1133) | 10 |
| HAR | 0.1154 | 4 |
| RC (classical reservoir) | 0.1171 | 51 |
| AR3 | 0.1180 | 4 |
| RCX (classical reservoir with macro features) | 0.1224 | 21 |
| AR1 | 0.1302 | 2 |
| Persistence | 0.1433 | 0 |

QR1 with clean (pre-August-1997) feature scaling: 0.1110 mean, sd 0.0019. Averaging the 20 seeds' forecasts gives 0.1083.

For reference, from the private check on the authors' own data: their QR1 with their coupling draw is 0.1051 (reproduced); 20 of our coupling seeds on their data give 0.1105 mean, sd 0.0020, best 0.1058.

Consequences for how we present the work:

- None of our seeds reached the authors' 0.1051; their draw sits about 2.7 standard deviations better than a typical one, and their feature set was chosen by forward selection for that draw. Report QR1 as a **mean and spread over seeds**, never a single best seed.
- QR1 ranks third. It beat HARX in 0 of 20 seeds and ARMAX in 1 of 20, so it does **not** match the best classical models; both are linear regressions on the same macro features.
- QR1 beat both classical reservoirs, HAR and AR3 in 20 of 20 seeds. The like-for-like comparison is reservoir against reservoir: same idea (fixed random dynamics, linear readout), and the quantum one reaches about 10% lower error than RCX with 10 trained weights instead of 21. That is the defensible "where the quantum part improves" claim.
- There is no parameter-count or accuracy advantage over HARX. The roadmap's framing needs adjusting to say this.

## 9. Decisions for Ali K

- Headline protocol for feature scaling: paper (full-sample) or clean (pre-1997). Recommendation: paper as headline for comparability, clean as a stated robustness line.
- Number of coupling seeds for the headline row. Recommendation: 20.
- QR2 is a stretch goal. It is the same scheme with a different coupling draw, the feature set RV, MKT, STR, RV_q, EP, INF, DEF, and two readout times (after τ/2 and τ), giving 20 features. Note our panel's INF is a different CPI variant from the authors'.
