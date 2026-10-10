# IonQ Hackathon Roadmap

Oct 9, 2026 · @GFX

We rebuild a published quantum reservoir computing result for S&P 500 volatility in Qiskit from open data, correct its classical baselines, and test it under finite shots and noise — entered under the Quantum Impact theme. Submission deadline: Oct 11, 2026, 19:00 ET on Qollab.

## Summary

- **Direction (decided Fri Oct 10):** we enter as the team that rebuilt a published quantum-advantage claim from open data, corrected its baselines, and then ran the two experiments the paper could not — finite shots and noise — on a Qiskit implementation judges can run in minutes. We do not claim "quantum beats classical"; our own reproduction shows the gap closes once HARX is scored correctly (0.1043 vs QR2's 0.1038).
- **What we build:** a paper-faithful QR1 in Qiskit — 10 qubits (7 input + 3 hidden), three sequential lag injections with the input register reset between lags, Trotterized fully-connected transverse-field Ising evolution, ⟨Z⟩ readout, ridge regression — plus a corrected classical benchmark table. QR2 (ensemble) is a stretch goal.
- **Already done (Ali M, Thu night):** open-data monthly panel (816 months, 1950–2017, verified against the authors' Data.CSV), AR/HAR/HARX/ARMAX/ESN baselines, 245 out-of-sample forecasts, and the audit of the authors' code. The quantum model is the critical path.
- **Experiments, in priority order:** (1) corrected-baseline table with QR1; (2) shot-noise study with the ridge parameter re-tuned per shot level; (3) dissipation-rate sweep in the Aer noise model, testing the non-monotonic optimum Sannia et al. predict; (4) stretch: QR2, retraining-time line, gate-depth comparison.
- **Who does what:** Ali K — QRC spec, readout/CV, shots analysis, project page, video, submission. Ali M — corrected table, figures, harness and demo entry point, judge-mode test. Sabah — Qollab capability spike, Qiskit circuit, all simulator runs, Qollab publishing.
- **Gates (ET):** Fri 08:00 architecture locked (reset support known); Fri 14:00 QR1 forecasts exist; Fri 20:00 minimum viable submission; Sat 02:00 shots done; Sat 08:00 noise done; Sat 12:00 code freeze; Sat 17:00 submitted.
- **Deadline confirmed:** Sat Oct 11, 19:00 ET. About 39 hours remain from Fri 04:00 ET.

## Hard rules and deadlines

Build window: Oct 9, 19:00 ET to **Oct 11, 19:00 ET** ([terms](https://qollab.xyz/programs/hackathon/terms)); confirmed by Ali K on Oct 10. We target submission by Sat 17:00 ET to keep a two-hour buffer.

Non-negotiable requirements from the rules:

- Qiskit is mandatory.
- The project must run on Qollab's free simulators. Judges run it themselves on a free local simulator; results from IonQ hardware add zero points. So the demo must run end-to-end in minutes on a laptop-class simulator.
- Published and runnable on Qollab.xyz under MIT license before the deadline. Any external repo the project relies on must also be MIT.
- Qollab project page explaining what it does, the problem, how to run it, and enough detail to replicate.
- 2–3 minute demo video at a public link that opens without sign-in.
- All work done inside the build window; templates, forks, AI use, and existing code allowed only if disclosed — judging covers only what we add.
- Attribution line on the project: "This effort is supported by Qollab & IonQ".
- Project stays on Qollab for 12 months; winners must post on social media once.

## Win strategy

We enter under the Quantum Impact theme, which explicitly asks for "hybrid quantum-classical solutions to real-world problems… with a classical baseline and measurable gains" — exactly the QRC-vs-classical benchmark we discussed. The "beyond-tutorial bar" is met by a technical extension: rigorous benchmarking plus a noise study.

Our pitch in one sentence: a published paper says a 10-qubit reservoir beats classical volatility models; we rebuilt it from open data in Qiskit, corrected the baselines, and then asked what happens when you add the two things real hardware has and the paper left out — finite shots and noise. We win on rigor and novelty, not on a quantum-advantage claim we could not defend.

| Rubric criterion | How we score it |
| --- | --- |
| Novelty & Creativity | First open Qiskit implementation of the paper's reservoir; a corrected benchmark; shot-noise and dissipation-rate studies the paper does not contain |
| Technical Execution | The paper's saved AR/HAR/HARX/QR numbers reproduced to float precision before we change anything; rolling protocol; seeded, cached, reproducible pipeline; demo under 5 minutes |
| Impact Potential | A reusable, honest benchmarking protocol for quantum-ML-in-finance claims; a ridge-only readout means millisecond refits |
| Documentation Quality | Project page with architecture figure, both-protocol results table, replication steps, audit findings and full disclosures |

Two judging realities shape everything: judges run the code themselves on a free simulator (so it must just work, fast), and documentation is worth as much as the model (so we reserve the last \~6 hours for it).

**What is ours vs the paper's.** We disclose arXiv:2505.13933 and the first author's repo (github.com/LeeQY1996, unlicensed, consulted only) as the starting point; judging covers only what we add. Ours: (1) the first MIT-licensed, documented, Qiskit implementation of the model; (2) a corrected benchmark — the authors' HAR/HARX loop scores each forecast one month late, their RCX result was not reproduced in 15+ draws, their MCS "QLIKE" cell uses squared errors, and nothing is seeded; we report paper-protocol and corrected scores side by side; (3) a shot-noise study (the paper used exact, infinite-shot expectation values); (4) a dissipation-rate sweep (the paper's evolution is purely unitary). Present (2) as implementation findings from reproducing released code, with the authors credited — careful, not hostile.

## Locked decisions

These close the open questions from the Oct 9 team call. Push back tonight or they stand.

1. **Direction: corrected benchmark + shots + noise.** No "quantum wins" claims anywhere in the write-up or video. The headline table shows QR1 beside the corrected classical baselines; the two noise studies are the novelty.
2. **Data: the team's open-data panel** (`data/rv_monthly.csv`, 816 months 1950–2017, rebuilt by `scripts/pull_data.py` from Yahoo, Ken French, Shiller and FRED; verified against the authors' Data.CSV — RV exact to 9e-16, other features correlate 0.9997–1.0, INF is a different CPI variant). Target = log realized volatility. The authors' files are consulted for validation only and never copied or committed.
3. **FinTSB and Qlib are protocol citations only** (FinTSB is GPL; Kaggle data is contractually off-limits).
4. **Model: paper-faithful QR1.** 10 qubits = 7 input + 3 hidden; per sample, three lag injections (k = 3): encode the lag's features into the input register with RY rotations, evolve the full register under H = Σ J\_ij X\_i X\_j + Σ h\_i Z\_i for time τ via a 1–2 step Trotter product (RXX/RZZ layers + RZ), reset the input register, repeat; after the third injection measure ⟨Z\_i⟩ on all qubits (optionally ⟨Z\_iZ\_j⟩) as features; ridge regression readout refit per rolling step. Couplings and fields drawn once from a seeded RNG and frozen. If Qollab's simulator does not support `reset`, fallback = 3 input + 3 hidden qubits unrolled over three lags (12 qubits). Our earlier sliding-window variant is a secondary ablation only.
5. **Validation: the paper's rolling protocol** — fixed estimation window, monthly refit, 245 one-step forecasts Aug 1997–Dec 2017 — reported under BOTH the authors' scoring (one-month offset) and corrected alignment. Readout ridge parameter chosen by cross-validation inside the training window only; re-tuned at every shot level.
6. **Workflow:** GitHub is the workshop (`sabahuddin-ahmad/Qollab-IonQ-Hackathon-2026`, MIT), branches per person, PRs to main; Qollab is the showroom. Final merged code uploaded to the Qollab project with a working `main.py` demo before the deadline. Heavy results (ESN, LSTM, full sweeps) are precomputed CSVs committed to the repo; the demo recomputes only the fast path.
7. **Disclosure file from day one.** `DISCLOSURES.md`: AI assistance (Claude, Gemini), libraries, the dataset reconstruction, the authors' repo as consulted reference, every paper we follow.

## Technical design

**Pipeline:** `data/rv_monthly.csv` (open-data panel) → log realized volatility, three lags per sample → paper-faithful QR1 circuit → measured features, cached to CSV → ridge readout refit per rolling step → 245 one-step forecasts → shared metrics. Classical baselines write to the same forecast file (`data/oos_classical.csv`), so every model is scored by identical code.

**Quantum reservoir (Ali K designs, Sabah implements):**

- Registers: 7 input qubits + 3 hidden qubits (paper's partition). Stateless per sample: fresh |0⟩ state for every forecast window.
- Per lag ℓ = 1..3: encode the lag's feature vector (univariate: lagged log-RV, replicated or scaled across input qubits per the spec) as RY(θ) with θ = π·σ(x); evolve the full 10-qubit register under the fully-connected transverse-field Ising Hamiltonian for time τ via a 1–2 step Trotter product (all-to-all RXX or RZZ layer + RZ layer); reset the 7 input qubits.
- Readout after lag 3: ⟨Z\_i⟩ for all 10 qubits (optionally the 45 ⟨Z\_iZ\_j⟩ correlators); estimated exactly (statevector/density matrix) or from shots.
- Readout model: ridge regression on the feature vector, refit at every rolling step; ridge parameter by cross-validation within the training window.
- Backends: Aer statevector (exact), Aer with shots, Aer density-matrix or trajectory with a noise model (depolarizing/dephasing per gate and idle); the same circuit submits to Qollab's simulators unchanged.
- Cost control: features are the expensive part and are computed once per (sample, backend, shot level, noise level) and cached to CSV; ridge fits are milliseconds, so re-tuning and refitting are free.

**Classical baselines (Ali M, done):** AR1, AR3, HAR, HARX, ARMAX and the paper-faithful 3-step stateless echo-state network (RC/RCX) under the paper's rolling protocol, in `scripts/bench_classical.py` and `scripts/bench_rc.py`. Still to add: the corrected-alignment scoring column. An LSTM row is optional and ships as a precomputed CSV if at all.

**Metrics (shared module):** MSE and QLIKE on log realized volatility as the authors' code defines them (QLIKE is the Patton loss summed over the out-of-sample period, which differs from the formula printed in the paper), hit rate, pairwise Diebold-Mariano (the authors' version has no HAC correction — say so), Model Confidence Set on squared errors; plus trained-parameter count and refit wall-time per model.

**The experiments, in priority order:**

1. **Corrected benchmark table (minimum viable submission).** QR1 (exact expectations) beside AR1, AR3, HAR, HARX, ARMAX, ESN under the paper's protocol and under corrected scoring; MSE, QLIKE, hit rate, Diebold-Mariano vs each baseline, MCS on squared errors. One table, both columns, no spin.
2. **Shot-noise study.** The same QR1 features estimated from 256, 1024, 4096 shots and exactly; ridge parameter re-tuned by cross-validation at each level; MSE/QLIKE vs shots. Answers the paper's biggest gap (it used infinite-shot expectation values and a near-zero ridge penalty).
3. **Dissipation-rate sweep.** Depolarizing/dephasing strength γ swept across \~5 levels in the Aer noise model (plus the IonQ noise model on Qollab as one point); MSE vs γ with the Sannia et al. non-monotonic prediction stated up front. The honest question is whether hardware-level noise sits below, at, or above the optimum.
4. **Stretch:** QR2 ensemble; retraining wall-time of the ridge readout vs the LSTM (the Impact line); transpiled two-qubit-gate count of the all-to-all Ising step on IonQ's topology vs a heavy-hex map (the "why trapped ions" table). Skip multi-step S=5 and hardware runs — one paragraph each as future work.

**Reference papers** (cite in DISCLOSURES.md and on the project page; the design follows these, no code copied):

- [QRC for Realized Volatility Forecasting, arXiv:2505.13933](https://arxiv.org/abs/2505.13933) — fully-connected TFI reservoir on financial data vs GARCH/ML baselines; argues trapped-ion all-to-all connectivity is the natural platform — our "why IonQ" citation.
- [Gate-Based QRC for Multivariate Time Series on NISQ Hardware, arXiv:2510.13634](https://arxiv.org/abs/2510.13634) — Trotterized TFI circuit on real hardware; noisy device beat the noiseless simulator on one task — precedent for the noise-as-resource experiment.
- [Chen, Nurdin & Yamamoto, PRApplied 14, 024065 (2020), arXiv:2001.09498](https://arxiv.org/abs/2001.09498) — foundational gate-based dissipative QRC on NISQ devices.
- [Suzuki et al., Natural quantum reservoir computing, Sci. Rep. 12 (2022)](https://link.springer.com/10.1038/s41598-022-05061-w) — hardware noise as the dissipation/nonlinearity source.
- Fujii & Nakajima, PRApplied 8, 024030 (2017) — original Ising-dynamics QRC.
- [Otieno et al., QRC for stock price forecasting, arXiv:2602.13094](https://arxiv.org/pdf/2602.13094) — ≤6-qubit QRC on stock data, portable to trapped ions.

## Roles

| Who | Owns | Deliverables, in order |
| --- | --- | --- |
| Ali Karimi (lead, ML + QML) | Direction, QRC design, readout, analysis, submission | `docs/qrc_spec.md` by Fri 08:00 ET (qubit partition, encoding map, J/h/τ and Trotter choices, readout observables, fallback if no reset); `src/qrc/readout.py` (ridge + in-window CV) plugged into the harness; shot-noise analysis and table; `DISCLOSURES.md`; project page text and video script; records the video; submits on Qollab |
| Ali Mahani (classical RC + ML) | Benchmark harness, figures, demo entry point | `aligned` scoring option in `bench_classical.py` and the corrected table (both protocols) by Fri 12:00; `scripts/make_figures.py` producing the headline table and the two experiment figures from cached CSVs; `main.py` demo config that runs QR1 + linear baselines on ≤5 windows in under 5 minutes; retraining-time measurement (ridge vs LSTM); judge-mode test from a fresh clone on Sat |
| Sabah (quantum computing, Qiskit) | Qollab platform, circuit, every simulator run | Qollab capability spike by Fri 06:00 ET: does `reset` work, does the Aer noise model and shots work, which packages `requirements.txt` may list, file-size and runtime limits, draft project published; `src/qrc/circuit.py` per the spec with feature caching; QR1 exact run for all 245 windows by Fri 14:00; shots sweep Fri night; noise sweep Sat morning; Qollab upload and final judge-mode run on the platform |

Interface contract to agree tonight so the three tracks never block each other: the data module hands every model `(X_window, y_next)` arrays per fold; every model exposes `fit(X, y)` / `predict(X)`; the metrics module consumes `(y_true, y_pred, fit_seconds, n_params)`. Once those signatures are in the repo, each person works entirely on their own branch.

## 48-hour timeline

All times ET, written Fri Oct 10 at about 04:00 ET with \~39 hours left. Each phase ends with a gate; if a gate slips more than \~3 hours, invoke the cut list. Data, baselines, metrics and forecast CSVs already exist in the repo, so the plan starts at the quantum model.

**Phase A — Architecture lock (Fri 04:00–12:00).** Sabah: Qollab capability spike first (reset, noise model, shots, allowed packages, limits), results pinned in Discord by 06:00; then start `circuit.py`. Ali K: `docs/qrc_spec.md` by 08:00, choosing the reset or unrolled architecture from Sabah's findings; then `readout.py`. Ali M: `aligned` scoring in `bench_classical.py`, corrected table for the classical models, and the `forecast(model) -> csv` interface the quantum code plugs into. *Gate 08:00: architecture locked. Gate 12:00: QR1 produces a feature vector and a forecast for one window on Aer.*

**Phase B — QR1 end to end (Fri 12:00–20:00).** Sabah: exact-expectation run over all 245 windows, features cached to CSV (target 14:00); then wire shots and noise-model options. Ali K: ridge readout with in-window CV on the cached features; QR1 row into the corrected table; sanity-check against the authors' QR1 numbers (0.1051 MSE under their protocol) and record the deviation honestly. Ali M: `main.py` demo config (QR1 + AR/HAR on ≤5 windows, < 5 min), figure script v1, `DISCLOSURES.md` skeleton with Ali K. *Gate 20:00: minimum viable submission — repo tagged `mvp`, demo runs from a fresh clone, corrected table complete with a QR1 row.*

**Phase C — Shots (Fri 20:00 – Sat 02:00).** Sabah: shot-level feature runs (256, 1024, 4096) for all windows, run in the background and cached. Ali K: ridge re-tuning per shot level, MSE/QLIKE-vs-shots table. Ali M: shots figure; retraining-time measurement (ridge refit vs one LSTM retrain) as a one-line Impact table; start the project page results section. Sleep in shifts — someone must be awake to babysit the sweep. *Gate 02:00: shots figure exists.*

**Phase D — Noise sweep and freeze (Sat 02:00–12:00).** Sabah: noise-model runs at \~5 γ levels plus the IonQ noise model on Qollab as one point; cached. Ali K: noise analysis with the Sannia prediction stated; writes the project page and video script; `DISCLOSURES.md` complete. Ali M: noise figure, final headline table, README replication steps; if slack, QR2 ensemble or the gate-depth table. *Gate 12:00: code freeze — only docs and packaging after this.*

**Phase E — Package and submit (Sat 12:00–19:00).** Sabah: upload the merged repo to the Qollab project, run it there exactly as a judge would, fix packaging only. Ali M: judge-mode test from a fresh clone on a second machine; verify the MIT license and attribution line are visible. Ali K: record the 2–3 minute video (public link, tested in an incognito window), finalize the project page, publish, submit. *Target: submitted by 17:00 with a two-hour buffer; hard deadline 19:00.*

## Risks and cut list

| Risk | Mitigation |
| --- | --- |
| Qollab's simulator rejects mid-circuit `reset` | Decided at the Fri 08:00 gate from Sabah's spike; fallback is 3 input + 3 hidden qubits unrolled over three lags (12 qubits) |
| Qollab runtime (Pyodide) cannot import torch or reservoirpy | Demo recomputes only QR1 + linear baselines; ESN/LSTM results ship as precomputed CSVs with the scripts that made them |
| Simulator time explodes (245 windows × shot levels × noise levels) | Compute features once per setting and cache; cap at 4 shot levels and 5 noise levels; run sweeps in the background overnight; density-matrix noise at 10 qubits is cheap |
| Qollab upload or run quirks eat the last hours | Draft project published in Phase A; code freeze 7 hours before the deadline; judge-mode test from a fresh clone |
| QR1 only ties the corrected baselines | That is the finding and it is publishable; the novelty lives in the shots and noise studies, not in a win |
| Looks like an attack on the authors | Credit the released code, call them implementation findings, show both scoring columns, cite the repo |
| Licensing | No code from the authors' unlicensed repo; our implementation follows the paper's description; everything else is MIT or Apache |

Cut order when time runs short: (1) gate-depth table, (2) retraining-time line, (3) QR2 ensemble, (4) noise sweep down to 3 levels, (5) shots down to 2 levels. Never cut: QR1 on the exact simulator, the corrected table, the Fri 20:00 `mvp` tag, the Qollab run test, the video, the project page.

## Submission checklist

- [ ] Deadline confirmed Sat Oct 11, 19:00 ET; internal target Sat 17:00 ET (done)
- [ ] Qollab project created by Ali K (team captain) and linked to the GitHub repo URL
- [ ] Code merged to main, MIT LICENSE file, runs via `main.py` on Qollab's free simulator
- [ ] `demo` config finishes in under \~5 minutes on a laptop
- [ ] DISCLOSURES.md: AI assistance, libraries, FinTSB protocol citation, any borrowed circuit ideas
- [ ] Project page: problem, method (with architecture figure), results tables/plots, how to run, how to extend
- [ ] Attribution on the page: "This effort is supported by Qollab & IonQ"
- [ ] Demo video 2–3 min, public link, opens without sign-in, link tested in an incognito window
- [ ] Project published on Qollab (not draft) before the deadline
- [ ] Post-submit: screenshot the confirmation

Sources: [hackathon terms](https://qollab.xyz/programs/hackathon/terms) · [Qollab docs](https://qollab.xyz/learn/docs) · [FinTSB](https://github.com/TongjiFinLab/FinTSB) · team call transcript, Oct 9, 2026.
