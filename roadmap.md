# IonQ Hackathon Roadmap

Oct 9, 2026 · @GFX

We rebuild a published quantum reservoir computing result for S&P 500 volatility in Qiskit from open data, correct its classical baselines, and test it under finite shots and noise — entered under the Quantum Impact theme. Submission deadline: Oct 11, 2026, 19:00 ET on Qollab.

## Summary

- **Direction:** we enter as the team that rebuilt a published quantum-advantage claim from open data, corrected its baselines, and then ran the two experiments the paper could not — finite shots and noise — on a Qiskit implementation whose demo is a single job on Qollab's IonQ simulator, with every sweep simulated locally. We do not claim the reservoir is more accurate than the best classical model: once HARX is scored correctly the gap closes (0.1043 vs QR2's 0.1038). The improvement we claim is efficiency.
- **What we build:** a paper-faithful QR1 in Qiskit — 10 qubits (7 input + 3 hidden), three sequential lag injections with the input register reset between lags, Trotterized fully-connected transverse-field Ising evolution, ⟨Z⟩ readout, ridge regression — plus a corrected classical benchmark table. QR2 (ensemble) is a stretch goal.
- **Already done (Ali M):** open-data monthly panel (816 months, 1950–2017, verified against the authors' Data.CSV), AR/HAR/HARX/ARMAX/ESN baselines, 245 out-of-sample forecasts, and the audit of the authors' code. The quantum model is the critical path.
- **Experiments, in priority order:** (1) corrected-baseline table with QR1; (2) efficiency comparison — trained parameters and refit time vs LSTM, ESN and HARX; (3) shot-noise study with the ridge parameter re-tuned per shot level; (4) dissipation-rate sweep in the Aer noise model; (5) stretch: QR2, gate-depth comparison.
- **Who does what:** Ali K — QRC spec, readout, shots analysis, disclosures, written criterion summary, project page, video, submission. Ali M — corrected table, efficiency comparison, accessible figures, demo entry point, judge-mode tests. Sabah — #hackathon-help questions and Qollab capability spike, Qiskit circuit, all simulator runs, Qollab publishing.
- **Gates, ET (Calgary time in brackets):** Sat 14:00 (12:00) architecture locked; Sat 17:00 (15:00) QR1 forecasts one month; Sun 00:00 (Sat 22:00) minimum viable submission; Sun 10:00 (08:00) shots and noise figures; Sun 12:00 (10:00) code freeze; Sun 13:00 (11:00) final run on Qollab done; Sun 17:00 (15:00) submitted.
- **Deadline:** Sunday Oct 11, 19:00 ET = 17:00 in Calgary. An earlier version of this doc had every weekday one day early; the dates were right.

## Hard rules and deadlines

Build window: Friday Oct 9, 19:00 ET to **Sunday Oct 11, 19:00 ET (17:00 in Calgary)**, the same moment for every node ([terms](https://qollab.xyz/programs/hackathon/terms); organizer participant plan). We target submission by Sunday 17:00 ET to keep a two-hour buffer. Partial projects are still eligible, so the minimum viable submission is a real safety net.

Non-negotiable requirements from the rules:

- Built with Qiskit.
- Runs without any errors on the quantum simulator available through Qollab (IonQ's simulator) — an explicit requirement for the Quantum Impact theme. Judges run it themselves; hardware results add zero points. The simulator queue is shared, so debug locally and do the final platform run early.
- Published and runnable on Qollab.xyz under an MIT license before the deadline. Any external repo the project relies on must also be MIT.
- Qollab project page: what it does, the problem, how to run it, enough detail to replicate or extend.
- A written summary covering each judging criterion: novelty, technical execution, impact, documentation.
- A short demo video at a public link (terms say 2–3 minutes; not scored on production quality).
- A code repository link and the theme tag "Quantum Impact" (the tag is required for theme awards).
- Disclosure of every external library, template and piece of outside code, with confirmation that we are licensed to use them. AI tools are allowed if disclosed.
- Follows industry best practices on UI accessibility.
- Built during the build window; nothing started before kickoff.
- Attribution line: "This effort is supported by Qollab & IonQ."
- Before the deadline: open the published project from a logged-out browser or a teammate's account and run it start to finish.
- Project stays live on Qollab for 12 months; winners post once on social media.
- Teams must be at the same node and in the same category; mixed student/professional teams are not yet supported.

## Win strategy

We enter under Quantum Impact and tag the project that way. The theme asks for a strong classical baseline and then "where a quantum subroutine improves solution quality, efficiency, or accuracy." Our honest answer is efficiency: provided our QR1 reproduces the authors' 0.105 MSE, it matches the best corrected classical baseline on accuracy and beats the LSTM and the classical reservoir, while training only a ridge readout of about ten weights that refits in milliseconds. The beyond-tutorial bar names noise studies and benchmarking as qualifying technical extensions, which is exactly our plan. Placement awards (1st–3rd) do not require a theme.

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

1. **Direction: corrected benchmark + shots + noise.** No "quantum wins" claims anywhere in the write-up or video; the improvement we do claim is efficiency — accuracy on par with the best corrected baseline from a readout with about ten trained weights. The headline table shows QR1 beside the corrected classical baselines; the two noise studies are the novelty.
2. **Data: the team's open-data panel** (`data/rv_monthly.csv`, 816 months 1950–2017, rebuilt by `scripts/pull_data.py` from Yahoo, Ken French, Shiller and FRED; verified against the authors' Data.CSV — RV exact to 9e-16, other features correlate 0.9997–1.0, INF is a different CPI variant). Target = log realized volatility. The authors' files are consulted for validation only and never copied or committed.
3. **FinTSB and Qlib are protocol citations only** (FinTSB is GPL; Kaggle data is contractually off-limits).
4. **Model: paper-faithful QR1.** 10 qubits = 7 input + 3 hidden; per sample, three lag injections (k = 3): encode the lag's features into the input register with RY rotations, evolve the full register under H = Σ J\_ij X\_i X\_j + Σ h\_i Z\_i for time τ via a 1–2 step Trotter product (RXX/RZZ layers + RZ), reset the input register, repeat; after the third injection measure ⟨Z\_i⟩ on all qubits (optionally ⟨Z\_iZ\_j⟩) as features; ridge regression readout refit per rolling step. Couplings and fields drawn once from a seeded RNG and frozen. If Qollab's simulator does not support `reset`, fallback = 3 input + 3 hidden qubits unrolled over three lags (12 qubits). Our earlier sliding-window variant is a secondary ablation only.
5. **Validation: the paper's rolling protocol** — fixed estimation window, monthly refit, 245 one-step forecasts Aug 1997–Dec 2017 — reported under BOTH the authors' scoring (one-month offset) and corrected alignment. Readout ridge parameter chosen by cross-validation inside the training window only; re-tuned at every shot level.
6. **Workflow:** GitHub is the workshop (`sabahuddin-ahmad/Qollab-IonQ-Hackathon-2026`, MIT), branches per person, PRs to main; Qollab is the showroom. Final merged code uploaded to the Qollab project with a working `main.py` demo before the deadline. Heavy results (ESN, LSTM, full sweeps) are precomputed CSVs committed to the repo; the Qollab demo is inference-only: one circuit, one job, frozen readout weights.
7. **Disclosure file from day one.** `DISCLOSURES.md`: AI assistance (Claude, Gemini), libraries, the dataset reconstruction, the authors' repo as consulted reference, every paper we follow.

## Technical design

**Pipeline:** `data/rv_monthly.csv` (open-data panel) → log realized volatility, three lags per sample → paper-faithful QR1 circuit → measured features, cached to CSV → ridge readout refit per rolling step → 245 one-step forecasts → shared metrics. Classical baselines write to the same forecast file (`data/oos_classical.csv`), so every model is scored by identical code.

**Where each thing runs (corrected after Sabah's note, Sat Oct 10).** There are two simulators and they do different jobs.

- **Local, on our own machines with `qiskit-aer`:** every experiment. That is the exact QR1 features for all 816 monthly samples (one small circuit per month, reused across the 245 rolling refits), the shot levels, and the noise sweep. These are ordinary Python loops over thousands of 10-qubit circuits and none of them is an IonQ job. Outputs are cached CSVs plus the frozen readout weights, committed to the repo with the scripts that made them.
- **On Qollab's IonQ simulator:** one job per run, no loops. The published demo submits a single circuit (one forecast month), reads the measured features, applies the frozen ridge weights shipped with the project, and prints the forecast beside the classical baselines and the precomputed tables. No training and no sweep happens on Qollab. If support confirms that one job may carry a batch of circuits, the demo forecasts a handful of months in that one job; otherwise it stays at one.

The write-up must label which numbers came from local Aer simulation and which from the Qollab IonQ simulator. The noise sweep uses Aer's generic depolarizing and dephasing channels, not IonQ's device noise model, and we say so.

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

1. **Corrected benchmark table (minimum viable submission).** QR1 (exact expectations, simulated locally) beside AR1, AR3, HAR, HARX, ARMAX, ESN under the paper's protocol and under corrected scoring; MSE, QLIKE, hit rate, Diebold-Mariano vs each baseline, MCS on squared errors. One table, both columns, no spin.
2. **Efficiency comparison (core, about one hour).** For QR1, HARX, the classical reservoir and the LSTM: number of trained parameters, wall-time of one monthly refit, and MSE. This is our answer to the theme's "where does the quantum subroutine improve" question, so it is not optional.
3. **Shot-noise study (local, Aer).** The same QR1 features estimated from 256, 1024, 4096 shots and exactly; ridge parameter re-tuned by cross-validation at each level; MSE/QLIKE vs shots. Answers the paper's biggest gap (it used infinite-shot expectation values and a near-zero ridge penalty).
4. **Dissipation-rate sweep (local, Aer).** Depolarizing/dephasing strength γ swept across ~5 levels in an Aer noise model; MSE vs γ with the Sannia et al. non-monotonic prediction stated up front. This is a generic-noise study, not IonQ's device model. If Qollab offers a noisy IonQ simulator, the single demo circuit run there is one qualitative comparison point, not part of the sweep.
5. **Stretch:** QR2 ensemble; transpiled two-qubit-gate count of the all-to-all Ising step on IonQ's topology vs a heavy-hex map. Skip multi-step S=5 and hardware runs — one paragraph each as future work.

**Reference papers** (cite in DISCLOSURES.md and on the project page; the design follows these, no code copied):

- [QRC for Realized Volatility Forecasting, arXiv:2505.13933](https://arxiv.org/abs/2505.13933) — fully-connected TFI reservoir on financial data vs GARCH/ML baselines; argues trapped-ion all-to-all connectivity is the natural platform — our "why IonQ" citation.
- [Gate-Based QRC for Multivariate Time Series on NISQ Hardware, arXiv:2510.13634](https://arxiv.org/abs/2510.13634) — Trotterized TFI circuit on real hardware; noisy device beat the noiseless simulator on one task — precedent for the noise-as-resource experiment.
- [Chen, Nurdin & Yamamoto, PRApplied 14, 024065 (2020), arXiv:2001.09498](https://arxiv.org/abs/2001.09498) — foundational gate-based dissipative QRC on NISQ devices.
- [Suzuki et al., Natural quantum reservoir computing, Sci. Rep. 12 (2022)](https://link.springer.com/10.1038/s41598-022-05061-w) — hardware noise as the dissipation/nonlinearity source.
- Fujii & Nakajima, PRApplied 8, 024030 (2017) — original Ising-dynamics QRC.
- [Otieno et al., QRC for stock price forecasting, arXiv:2602.13094](https://arxiv.org/pdf/2602.13094) — ≤6-qubit QRC on stock data, portable to trapped ions.

## Roles

| Who | Owns | Deliverables, in order (times ET, Calgary in brackets) |
| --- | --- | --- |
| Ali Karimi (lead, ML + QML) | Direction, QRC design, readout, analysis, submission | `docs/qrc_spec.md` by Sat 14:00 (12:00): qubit partition, encoding map, J/h/τ and Trotter choices, readout observables, fallback if no reset; `src/qrc/readout.py` (ridge + in-window CV) and the export of frozen readout weights for the Qollab demo; shot-noise and noise analysis; `DISCLOSURES.md` including the statement that we are licensed to use every dependency; the written summary with one section per judging criterion; project page; video; publishes and submits on Qollab |
| Ali Mahani (classical RC + ML) | Benchmark harness, figures, demo entry point | `aligned` scoring option in `bench_classical.py` and the both-protocol table by Sat 14:00 (12:00); efficiency comparison (trained parameters, refit time, MSE); `scripts/make_figures.py` with colourblind-safe palettes and alt text for every figure; the Qollab `main.py`: build one circuit, submit one job, apply frozen weights, print the forecast with the baselines and precomputed tables; fresh-clone test and logged-out browser test on Sunday |
| Sabah (quantum computing, Qiskit) | Qollab platform, circuit, simulations | In #hackathon-help: confirm one job per run, and ask whether one job may carry several circuits, whether mid-circuit `reset`, shots and a noisy simulator are available, and which packages and file sizes are allowed; publish a draft project; `src/qrc/circuit.py` per the spec with feature caching; **local Aer runs** (plain loops on our machines, not IonQ jobs): exact QR1 features for all 816 months by Sat 21:00 (19:00), then the shot levels and the noise sweep left running Saturday night; on Qollab: the single-job demo, error-free, by Sun 13:00 (11:00); book one mentor slot to sanity-check the framing |

Interface contract to agree tonight so the three tracks never block each other: the data module hands every model `(X_window, y_next)` arrays per fold; every model exposes `fit(X, y)` / `predict(X)`; the metrics module consumes `(y_true, y_pred, fit_seconds, n_params)`. Once those signatures are in the repo, each person works entirely on their own branch.

## 48-hour timeline

Times are ET with Calgary (MDT) in brackets; ET is two hours ahead. Re-timed Saturday Oct 10 at 03:00 ET (01:00 in Calgary) with 40 hours left and a night's sleep still to take. Each phase ends with a gate; if a gate slips more than about 3 hours, invoke the cut list. Data, baselines, metrics and forecast CSVs already exist in the repo, so the plan starts at the quantum model.

**Phase A — Architecture lock (to Sat 14:00 ET [12:00]).** Re-timed Saturday 11:30 ET: Ali M's both-protocol benchmarks are already on his branch; the spec and the circuit have not started, so this gate moves two hours. Sabah: confirm in #hackathon-help what one job may contain (one circuit or a batch), and whether mid-circuit `reset`, shots and a noisy simulator are available; pin the answers in Discord; start `circuit.py`. Ali K: `docs/qrc_spec.md`, choosing the reset or unrolled architecture from what Sabah has learned; then `readout.py`. Ali M: open the PR for the both-protocol table and define the `forecast(model) -> csv` interface the quantum code plugs into. *Gate Sat 14:00 ET [12:00]: architecture locked and written down.*

**Phase B — QR1 end to end (Sat 14:00 ET to midnight [12:00–22:00]).** Sabah: one-month forecast on local Aer by 17:00 ET [15:00], then exact-expectation features for all 816 months, cached to CSV, by 21:00 ET [19:00] (a local loop, seconds per circuit); in parallel, the same single circuit submitted once as a job on Qollab. Ali K: ridge readout with in-window CV on the cached features; QR1 row into the corrected table; frozen weights exported for the demo; sanity-check against the authors' QR1 (0.1051 MSE under their protocol) and record the deviation honestly. Ali M: efficiency comparison; Qollab `main.py` (one circuit, one job, frozen weights); figure script v1; `DISCLOSURES.md` skeleton with Ali K. *Gate Sun 00:00 ET [Sat 22:00]: minimum viable submission — repo tagged `mvp`, the single-job demo runs on Qollab, corrected table complete with a QR1 row.*

**Phase C — Local sweeps overnight (Sun 00:00 to 10:00 ET [Sat 22:00 to Sun 08:00]).** Sabah: start the shot-level runs (256, 1024, 4096) and the noise-model runs (~5 γ levels) on a local machine with Aer before sleeping, cached to CSV; these never touch Qollab or IonQ. Confirm in the morning that they finished. Ali K: Sunday morning, ridge re-tuning per shot level and the noise analysis with the Sannia prediction stated; start the written criterion summary. Ali M: Sunday morning, shots and noise figures with accessible palettes and alt text. Everyone sleeps; nobody babysits a sweep that can be rerun. *Gate Sun 10:00 ET [08:00]: both figures exist.*

**Phase D — Freeze and platform run (Sun 10:00–13:00 ET [08:00–11:00]).** Code freeze at Sun 12:00 ET [10:00]: only docs and packaging after this. Sabah: upload the merged repo to the Qollab project and run the single-job demo on Qollab's IonQ simulator exactly as a judge would, with no errors, before the shared queue gets busy. Ali M: final headline table, README replication steps (including how to rerun the local sweeps), fresh-clone test on a second machine. Ali K: `DISCLOSURES.md` complete, project page draft. *Gate Sun 13:00 ET [11:00]: an error-free run on Qollab is recorded.*

**Phase E — Package and submit (Sun 13:00–19:00 ET \[11:00–17:00\]).** Ali K: record the demo video once (public link), finish the project page and the written criterion summary, set the Quantum Impact tag and the repository link, publish, submit. Ali M: open the published project from a logged-out browser and run it start to finish; verify the MIT license, attribution line and alt text are visible. Sabah: fix packaging only; stand by for platform issues in #hackathon-help. *Target: submitted by Sun 17:00 ET \[15:00\]; hard deadline Sun 19:00 ET \[17:00\].*

## Risks and cut list

| Risk | Mitigation |
| --- | --- |
| Qollab's IonQ simulator allows one job per run and no loops | The demo is one circuit and one job with frozen readout weights; all training, shots and noise sweeps run locally on Aer and ship as cached CSVs; ask whether a job may carry several circuits |
| Qollab's simulator rejects mid-circuit `reset` | Asked in #hackathon-help and tested hands-on before the Sat 14:00 ET gate; fallback is 3 input + 3 hidden qubits unrolled over three lags (12 qubits) |
| The demo errors on Qollab's simulator, which the theme forbids | Draft project published in Phase A; the single-job demo runs on Qollab at the Saturday-midnight ET minimum-viable gate, not for the first time on Sunday |
| The shared simulator queue is slow near the deadline | Final platform run by Sun 13:00 ET; one job only; all debugging local on Aer |
| Qollab runtime (Pyodide) cannot import torch, reservoirpy or scikit-learn | The demo needs only Qiskit and NumPy: frozen weights are a small array applied with a dot product; everything heavier is precomputed |
| Local simulation time explodes (816 circuits × shot levels × noise levels) | Features computed once per setting and cached; cap at 4 shot levels and 5 noise levels; 10-qubit circuits take milliseconds to seconds each |
| A judge thinks the results came from IonQ when they came from Aer | Every table and figure is labelled with its simulator; the project page has a short "what runs where" section |
| QR1 only ties the corrected baselines | Expected; the claimed improvement is efficiency, and the novelty lives in the shots and noise studies |
| Looks like an attack on the authors | Credit the released code, call them implementation findings, show both scoring columns, cite the repo |
| Licensing statement | No code from the authors' unlicensed repo; our implementation follows the paper's description; every dependency is MIT, BSD or Apache and is listed with its licence |

Cut order when time runs short: (1) gate-depth table, (2) QR2 ensemble, (3) noise sweep down to 3 levels, (4) shots down to 2 levels. Never cut: QR1 on the exact simulator, the corrected table, the efficiency comparison, the Saturday-midnight ET `mvp` tag, the error-free Qollab run, the written criterion summary, the video, the project page.

## Submission checklist

- [x] Deadline confirmed: Sunday Oct 11, 19:00 ET (17:00 in Calgary); internal target Sunday 17:00 ET (15:00)
- [ ] Team registration checked: all three at the same node and in the same category
- [ ] Qollab project created by Ali K with the theme tag "Quantum Impact" and the code repository link
- [ ] Code merged to main, MIT LICENSE file, runs via `main.py` with no errors on the simulator available through Qollab
- [ ] `demo` config finishes in under \~5 minutes; final platform run (one circuit, one job, frozen weights) recorded by Sunday 13:00 ET
- [ ] DISCLOSURES.md: AI assistance, every library with its licence, the statement that we are licensed to use them, the authors' repo as a consulted-only reference, papers followed
- [ ] Project page: problem, method (with architecture figure), results tables/plots, how to run, how to extend
- [ ] Written summary with one short section per judging criterion: novelty, technical execution, impact, documentation
- [ ] Accessibility: colourblind-safe palettes, alt text on every figure, readable contrast, no information carried by colour alone
- [ ] Attribution on the page: "This effort is supported by Qollab & IonQ."
- [ ] Demo video, public link, opens without sign-in
- [ ] Project published on Qollab (not draft), then opened from a logged-out browser or a teammate's account and run start to finish
- [ ] Post-submit: screenshot the confirmation

Sources: [hackathon terms](https://qollab.xyz/programs/hackathon/terms) · [Qollab docs](https://qollab.xyz/learn/docs) · [FinTSB](https://github.com/TongjiFinLab/FinTSB) · team call transcript, Oct 9, 2026 · organizer participant plan, shared Oct 10.
