# Submission summary

**Quantum Reservoir Computing for Realized Volatility: making it work on a noisy device, and
measuring what it buys you.** Qollab × IonQ Hackathon 2026, theme: Quantum Impact.

## The story in brief

1. **The claim we tested.** Li et al. (arXiv:2505.13933) report that a 10-qubit quantum
   reservoir, QR1, forecasts S&P 500 monthly volatility better than classical models. We rebuilt
   the data, the baselines and the model from scratch in Qiskit.
2. **What a fair test shows.** With every forecast scored against the right month and 20 random
   reservoirs instead of one, QR1 ranks third of eight: behind two linear models, ahead of both
   classical reservoirs, with the fewest trained weights of the top five.
3. **What happens on a noisy device.** We ran QR1 on IonQ's Forte 1 simulator through Qollab. A
   readout trained on a noiseless simulation fails under device noise. Training the same ten
   weights under matching noise restores the accuracy, and one calibration circuit is enough to
   set the noise level.
4. **Where the value comes from.** Switching the entangling gates off shows that the quantum
   dynamics improve the reservoir on its own, and add nothing once a linear model has the same
   inputs. The useful ingredient in a hybrid is the input encoding, a classical transform.
5. **What a team can take away.** A recipe for running reservoir models on noisy hardware, a
   control experiment that separates quantum contribution from classical transform, and measured
   cost and scaling. Not a quantum advantage: nothing here ran on hardware, and ten qubits are
   simulated exactly on a laptop in milliseconds.

## Novelty and creativity

- **A noise-matched readout.** At 1% gate error and 1024 shots, a readout trained without noise
  scores an MSE of 0.47; trained under the same noise it scores 0.113, against 0.109 with no
  noise at all. The quantum circuit is unchanged; only the ten classical weights are retrained.
- **Calibrating a noise model from one circuit.** A generic depolarizing model at 1.6% per
  two-qubit gate reproduces the ten values IonQ's simulator returned, within shot noise, and a
  second run confirmed it. On IonQ's simulator the forecast error for the demo month fell from
  +1.15 (noiseless-trained weights) to +0.48 (matched weights).
- **A gates-off control.** Running the same model with the entangling gates removed isolates what
  the quantum dynamics contribute. Alone, they cut the error by 9% (0.1205 to 0.1109, p = 0.04).
  Added to the best linear model, they contribute nothing: the gates-off version scores 0.0999
  and the full reservoir 0.1009.
- **An independent reproduction that changes the paper's ranking.** The authors' HAR and HARX
  baselines are scored one month late. Aligned, HARX is the best model (0.1030), ahead of QR1.
  The paper's single reservoir (0.1051) sits at the favourable edge of our 20-seed range
  (0.1056 to 0.1133, mean 0.1102).
- **Negative results reported.** Engineered dissipation, a continuous-stream reservoir, extra
  readout features, qubit correlators, temporal multiplexing, extra measurement bases and
  Hamiltonian encoding were each tested; none beat its control.

## Technical execution

- **Runs on Qollab as required:** `main.py`, Qiskit only, one circuit and one job per run, on
  the built-in simulator (10 qubits with reset) and on the IonQ Forte 1 simulator (24 qubits,
  270 two-qubit gates, no reset, optional mid-circuit measurement). Three IonQ simulator runs
  are recorded in `data/ionq_runs.json`.
- **Three implementations that agree:** an exact NumPy reference, the Qiskit circuit (matches
  the reference to 1e-10) and the 24-qubit no-reset layout (same output distribution). Ten
  acceptance tests cover this.
- **Verified against the original:** given the authors' couplings, our model reproduces their
  saved forecasts to 1.2e-6.
- **A full shots and noise study:** 25 settings across 3 coupling seeds, 4 shot levels, two
  training protocols, three noise types and three circuit depths, about 20,000 simulated noisy
  circuits.
- **Platform limits found by testing and designed around:** IonQ backends on Qollab reject
  `reset`, accept mid-circuit measurement when no gate follows on the measured qubit, and take
  one circuit per job. The 24-qubit layout and frozen weights are the response.
- **Everything is regenerable:** each table and figure has a script, and the results are
  committed as CSV.

## Impact potential

- **Who it is for:** teams evaluating quantum machine learning for financial time series, who
  need to know what such a model delivers once shots, noise and a fair baseline are included.
- **A method that transfers.** The noise-matched readout and the one-circuit calibration apply
  to any reservoir or kernel model with a linear readout on a noisy device. With about 1,000
  shots, the matched readout holds accuracy up to the noise level of IonQ's simulator.
- **A test that transfers.** The gates-off control is cheap and tells a team whether their
  quantum model's accuracy comes from quantum dynamics or from its classical input encoding. It
  agrees with recent theory that reset-based reservoirs draw their nonlinearity from the
  encoding (arXiv:2607.12035).
- **An honest account of the cost.** QR1 is statistically indistinguishable from the best linear
  model (Diebold-Mariano p = 0.25) and better than classical reservoirs, at minutes per forecast
  on a simulator against under a millisecond. Its gate count grows with the square of the qubit
  count.
- **A by-product for the classical side.** The reservoir's encoding pointed to a missing term in
  HARX: the size of last month's market move. Adding it gives the best model we measured
  (0.0999 against 0.1030; not statistically significant on 245 months).

## Documentation quality

- **Specification before code:** `docs/qrc_spec.md` defines the model, the three
  implementations, the noise models, the acceptance checks and the results.
- **Claims carry their conditions:** every figure states where its numbers came from (local
  simulation or IonQ's simulator, how many seeds, formula or measurement), and each has alt text
  in `docs/figures/alt_text.md`. Colours are checked for colour-blind safety and no figure
  relies on colour alone.
- **Disclosures:** `DISCLOSURES.md` covers prior work, libraries and licenses, data sources, AI
  assistance and what ran where.
- **Run instructions:** `README.md` explains how to run on Qollab; `main.py` states in its
  header what runs there and what was computed locally; `scripts/run_demo_local.py` reproduces
  a Qollab run on a laptop.

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

Exact simulation, lower is better.

| QR1 under noise, 1024 shots | Readout trained under the same noise | Readout trained without noise |
| --- | --- | --- |
| No gate error | 0.113 | 0.163 |
| 1% per two-qubit gate | 0.113 | 0.47 |
| 3% per two-qubit gate | 0.117 | 1.14 |

Local simulation with a generic depolarizing model, mean of 3 coupling seeds.

## Limits

- Nothing was run on quantum hardware. The local noise model is generic and was matched to
  IonQ's simulator on one circuit for one month.
- All conclusions rest on 245 test months of one series. Differences of 2 to 3% between the best
  models are not statistically significant.
- The control experiments were chosen after seeing earlier results, so they explain what we
  observed and are not independent tests.
- We tested one reservoir design at ten qubits. This does not show that no quantum model could
  help on this task.

This effort is supported by Qollab & IonQ.
