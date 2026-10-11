# IonQ Hackathon Roadmap

> **Note.** This is the team's planning document from the first two days of the hackathon. It is kept as a
> record and is not up to date. For what was built and found, see `README.md` and
> `docs/submission_summary.md`.

Oct 9, 2026 · @GFX

We rebuild a published quantum reservoir computing result for S&P 500 volatility in Qiskit from open data, correct its classical baselines, and test it under finite shots and noise — entered under the Quantum Impact theme. Submission deadline: Oct 11, 2026, 19:00 ET on Qollab.

## Summary

- **Direction:** we enter as the team that rebuilt a published quantum-advantage claim from open data, corrected its baselines, and then ran the two experiments the paper could not — finite shots and noise — on a Qiskit implementation whose demo is a single job on Qollab's IonQ simulator, with every sweep simulated locally. We do not claim the reservoir matches the best classical model: in our runs QR1 ranks third, behind HARX and ARMAX. The improvement we claim is like-for-like: it beats both classical reservoirs in every seed, with fewer trained weights.
- **What we build:** a paper-faithful QR1 in Qiskit — 10 qubits (7 input + 3 hidden), three sequential lag injections with the input register reset between lags, Trotterized fully-connected transverse-field Ising evolution, ⟨Z⟩ readout, ridge regression — plus a corrected classical benchmark table. QR2 (ensemble) is a stretch goal.
- **Already done (Ali M):** open-data monthly panel (816 months, 1950–2017, verified against the authors' Data.CSV), AR/HAR/HARX/ARMAX/ESN baselines, 245 out-of-sample forecasts, and the audit of the authors' code. The quantum model is the critical path.
- **Experiments, in priority order:** (1) corrected-baseline table with QR1; (2) efficiency comparison — trained parameters and refit time vs LSTM, ESN and HARX; (3) shot-noise study with the ridge parameter re-tuned per shot level; (4) dissipation-rate sweep in the Aer noise model; (5) stretch: QR2, gate-depth comparison.
- **Who does what:** Ali K — QRC spec, readout, shots analysis, disclosures, written criterion summary, project page, video, submission. Ali M — corrected table, efficiency comparison, accessible figures, demo entry point, judge-mode tests. Sabah — #hackathon-help questions and Qollab capability spike, Qiskit circuit, all simulator runs, Qollab publishing.
- **Gates, ET (Calgary time in brackets):** Sat 14:00 (12:00) architecture locked; Sat 17:00 (15:00) QR1 forecasts one month; Sun 00:00 (Sat 22:00) minimum viable submission; Sun 10:00 (08:00) shots and noise figures; Sun 12:00 (10:00) code freeze; Sun 13:00 (11:00) final run on Qollab done; Sun 17:00 (15:00) submitted.
- **Deadline:** Sunday Oct 11, 19:00 ET = 17:00 in Calgary. An earlier version of this doc had every weekday one day early; the dates were right.

## Status, Saturday 15:00 ET (13:00 in Calgary)

QR1 is built and measured, earlier than planned. The numbers changed the claim we can make, so read this before the sections below.

- **Built:** `docs/qrc_spec.md` (PR #2) and a working QR1 in `src/qrc/` (PR #3): exact NumPy reference, the Qiskit circuit in 10-qubit reset and 24-qubit no-reset forms, local noise models, the readout, and 10 passing acceptance tests. The spec was checked by reproducing the authors' saved QR1 forecasts to 1.2e-6.
- **Result on our panel, corrected scoring:** QR1 averages MSE 0.1102 over 20 coupling seeds (sd 0.0020). That is third, behind HARX (0.1030) and ARMAX (0.1070), and ahead of HAR (0.1154), the classical reservoirs RC (0.1171) and RCX (0.1224), and AR3 (0.1180). QR1 beat HARX in 0 of 20 seeds and both classical reservoirs in 20 of 20.
- **The claim we can defend:** reservoir against reservoir. Same idea, fixed random dynamics with a linear readout, and the quantum one is about 10% more accurate than RCX with 10 trained weights instead of 21. We do not claim to match or beat the best classical model, and we have no efficiency advantage over HARX.
- **A new finding about the paper:** the authors' QR1 figure of 0.1051 was not reached by any of 20 random coupling draws on their own data (mean 0.1105). Their result is about 2.7 standard deviations better than a typical draw, with features selected for that draw. We report a mean and spread over seeds.
- **Corrections to this roadmap:** the inputs are seven different features, one per qubit (RV, MKT, DP, IP, RV_q, STR, DEF), not lagged RV alone; the readout is ⟨Z⟩ on the 10 qubits with no intercept; the fallback without `reset` is the 24-qubit circuit, not a 12-qubit one. The spec is the source of truth for the model.
- **Still unknown:** nothing has run on Qollab yet. Whether its simulator accepts `reset`, 24 qubits, or several circuits in one job decides the demo. No shots or noise sweep has been run; the machinery exists and is tested.
- **Next, by person:** Sabah — take over PR #3, run one `finish="measure"` circuit as a single job on Qollab today, then the noise × shots grid locally. Ali M — apply the one-month "paper" offset in `score_both` to HAR and HARX only, add the QR1 rows to the table and to `data/efficiency.csv`, build `main.py` on `qr1_circuit` and `z_from_counts` with frozen weights. Ali K — review PRs #2 and #3, then the noise × shots analysis and the write-up.

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

We enter under Quantum Impact and tag the project that way. The theme asks for a strong classical baseline and then "where a quantum subroutine improves solution quality, efficiency, or accuracy." Our honest answer is a like-for-like one: against classical reservoir computers, which share its design of fixed random dynamics and a linear readout, QR1 is more accurate in every one of 20 seeds while training 10 weights instead of 21 or 51. It does not beat the linear models HARX and ARMAX, and we say so. The beyond-tutorial bar names noise studies and benchmarking as qualifying technical extensions, which is exactly our plan. Placement awards (1st–3rd) do not require a theme.

Our pitch in one sentence: a published paper says a 10-qubit reservoir beats classical volatility models; we rebuilt it from open data in Qiskit, corrected the baselines, and then asked what happens when you add the two things real hardware has and the paper left out — finite shots and noise. We win on rigor and novelty, not on a quantum-advantage claim we could not defend.

| Rubric criterion | How we score it |
| --- | --- |
| Novelty & Creativity | First open Qiskit implementation of the paper's reservoir; a corrected benchmark; shot-noise and dissipation-rate studies the paper does not contain |
| Technical Execution | The paper's saved AR/HAR/HARX/QR numbers reproduced to float precision before we change anything; rolling protocol; seeded, cached, reproducible pipeline; demo under 5 minutes |
| Impact Potential | A reusable, honest benchmarking protocol for quantum-ML-in-finance claims; a ridge-only readout means millisecond refits |
| Documentation Quality | Project page with architecture figure, both-protocol results table, replication steps, audit findings and full disclosures |

Two judging realities shape everything: judges run the code themselves on a free simulator (so it must just work, fast), and documentation is worth as much as the model (so we reserve the last \~6 hours for it).

**What is ours vs the paper's.** We disclose arXiv:2505.13933 and the first author's repo (github.com/LeeQY1996, unlicensed, consulted only) as the starting point; judging covers only what we add. Ours: (1) the first MIT-licensed, documented, Qiskit implementation of the model; (2) a corrected benchmark — the authors' HAR/HARX loop scores each forecast one month late, their RCX result was not reproduced in 15+ draws, their MCS "QLIKE" cell uses squared errors, nothing is seeded, and their QR1 figure of 0.1051 was not reached by any of 20 random coupling draws on their own data (mean 0.1105); we report paper-protocol and corrected scores side by side; (3) a shot-noise study (the paper used exact, infinite-shot expectation values); (4) a dissipation-rate sweep (the paper's evolution is purely unitary). Present (2) as implementation findings from reproducing released code, with the authors credited — careful, not hostile.

## Locked decisions

These close the open questions from the Oct 9 team call. Push back tonight or they stand.

1. **Direction: corrected benchmark + shots + noise.** No "quantum wins" claims anywhere in the write-up or video; the improvement we do claim is reservoir against reservoir — QR1 beats both classical reservoirs in every seed with fewer trained weights, and ranks behind HARX and ARMAX. The headline table shows QR1 beside the corrected classical baselines; the two noise studies are the novelty.
2. **Data: the team's open-data panel** (`data/rv_monthly.csv`, 816 months 1950–2017, rebuilt by `scripts/pull_data.py` from Yahoo, Ken French, Shiller and FRED; verified against the authors' Data.CSV — RV exact to 9e-16, other features correlate 0.9997–1.0, INF is a different CPI variant). Target = log realized volatility. The authors' files are consulted for validation only and never copied or committed.
3. **FinTSB and Qlib are protocol citations only** (FinTSB is GPL; Kaggle data is contractually off-limits).
4. **Model: paper-faithful QR1.** 10 qubits = 7 input + 3 hidden; per sample, three lag injections (k = 3): encode the lag's features into the input register with RY rotations, evolve the full register under H = Σ J\_ij X\_i X\_j + Σ h\_i Z\_i for time τ via a 1–2 step Trotter product (RXX/RZZ layers + RZ), reset the input register, repeat; after the third injection measure ⟨Z\_i⟩ on all qubits (optionally ⟨Z\_iZ\_j⟩) as features; ridge regression readout refit per rolling step. Couplings and fields drawn once from a seeded RNG and frozen. If Qollab's simulator does not support `reset`, fallback = a 24-qubit circuit with fresh input qubits for each lag (identical result). Our earlier sliding-window variant is a secondary ablation only.
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

- Registers: 7 input + 3 hidden qubits; a fresh |000⟩ hidden state for every forecast month.
- Inputs: seven scaled features per month, one per input qubit (RV, MKT, DP, IP, RV_q, STR, DEF), written as RY(π·x). The scaling of each feature is in the spec.
- Dynamics: H = Σ J_ij X_i X_j + Σ Z_i with random couplings fixed by a seed; evolve for τ = 1 after each of the three months, discarding (resetting) the input qubits between months.
- Readout features: ⟨Z⟩ on all 10 qubits after the third month.
- Readout model: ridge regression without an intercept, in the scaled target space, refit at every rolling step; 10 trained weights.
- Three implementations: exact NumPy reference (about 4 s per seed for all months), Qiskit circuit with second-order Trotter steps (4 per month noiseless, 2 when noisy) on local Aer, and the same circuit as the single Qollab job. Without `reset`, a 24-qubit circuit gives the identical result.
- Source of truth: `docs/qrc_spec.md`; code in `src/qrc/`; checks in `tests/test_qrc.py`.

**Classical baselines (Ali M, done):** AR1, AR3, HAR, HARX, ARMAX and the paper-faithful 3-step stateless echo-state network (RC/RCX) under the paper's rolling protocol, in `scripts/bench_classical.py` and `scripts/bench_rc.py`. Still to add: the corrected-alignment scoring column. An LSTM row is optional and ships as a precomputed CSV if at all.

**Metrics (shared module):** MSE and QLIKE on log realized volatility as the authors' code defines them (QLIKE is the Patton loss summed over the out-of-sample period, which differs from the formula printed in the paper), hit rate, pairwise Diebold-Mariano (the authors' version has no HAC correction — say so), Model Confidence Set on squared errors; plus trained-parameter count and refit wall-time per model.

**The experiments, in priority order:**

1. **Corrected benchmark table (minimum viable submission).** QR1 (exact expectations, simulated locally) beside AR1, AR3, HAR, HARX, ARMAX, ESN under the paper's protocol and under corrected scoring; MSE, QLIKE, hit rate, Diebold-Mariano vs each baseline, MCS on squared errors. One table, both columns, no spin.
2. **Efficiency comparison (core, about one hour).** For QR1, HARX, the classical reservoir and the LSTM: number of trained parameters, wall-time of one monthly refit, and MSE. This is our answer to the theme's "where does the quantum subroutine improve" question, so it is not optional. The comparison that carries the claim is QR1 against the classical reservoirs RC and RCX: same design, and QR1 is more accurate in every seed with fewer trained weights.
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

**Phase A — Architecture lock (done Saturday early afternoon).** The spec is PR #2 and a working QR1 is PR #3. Left over from this phase: Sabah's answers from #hackathon-help on what one Qollab job may contain (one circuit or a batch) and whether mid-circuit `reset` and 24 qubits are accepted; Ali M's fix so the one-month "paper" offset in `score_both` applies to HAR and HARX only. *Gate met about 50 minutes late.*

**Phase B — QR1 end to end (Saturday, to midnight ET [22:00]).** Already done: exact features for all months and seeds, the readout, and the QR1 row (`scripts/bench_qrc.py`, `data/oos_qr1.csv`). Still to do — Sabah: submit one `finish="measure"` circuit as a single job on Qollab and decode it with `z_from_counts`; confirm the spec's Trotter and noise tables on real months. Ali K: review PRs #2 and #3; export frozen readout weights for the demo month, trained on features from the same circuit the demo runs. Ali M: QR1 rows in the both-protocol table and in `data/efficiency.csv`; Qollab `main.py` (one circuit, one job, frozen weights); figure script v1; `DISCLOSURES.md` skeleton with Ali K. *Gate Sun 00:00 ET [Sat 22:00]: minimum viable submission — repo tagged `mvp`, the single-job demo runs on Qollab, corrected table complete with a QR1 row.*

**Phase C — Local sweeps overnight (Sun 00:00 to 10:00 ET [Sat 22:00 to Sun 08:00]).** Sabah: cache exact outcome probabilities for every month at each noise level (shots at 256, 1024 and 4096 are then sampled from the cache for free); start the noise-model runs (~5 γ levels) on a local machine with Aer before sleeping, cached to CSV; these never touch Qollab or IonQ. Confirm in the morning that they finished. Ali K: Sunday morning, ridge re-tuning per shot level and the noise analysis with the Sannia prediction stated; start the written criterion summary. Ali M: Sunday morning, shots and noise figures with accessible palettes and alt text. Everyone sleeps; nobody babysits a sweep that can be rerun. *Gate Sun 10:00 ET [08:00]: both figures exist.*

**Phase D — Freeze and platform run (Sun 10:00–13:00 ET [08:00–11:00]).** Code freeze at Sun 12:00 ET [10:00]: only docs and packaging after this. Sabah: upload the merged repo to the Qollab project and run the single-job demo on Qollab's IonQ simulator exactly as a judge would, with no errors, before the shared queue gets busy. Ali M: final headline table, README replication steps (including how to rerun the local sweeps), fresh-clone test on a second machine. Ali K: `DISCLOSURES.md` complete, project page draft. *Gate Sun 13:00 ET [11:00]: an error-free run on Qollab is recorded.*

**Phase E — Package and submit (Sun 13:00–19:00 ET \[11:00–17:00\]).** Ali K: record the demo video once (public link), finish the project page and the written criterion summary, set the Quantum Impact tag and the repository link, publish, submit. Ali M: open the published project from a logged-out browser and run it start to finish; verify the MIT license, attribution line and alt text are visible. Sabah: fix packaging only; stand by for platform issues in #hackathon-help. *Target: submitted by Sun 17:00 ET \[15:00\]; hard deadline Sun 19:00 ET \[17:00\].*

## Risks and cut list

| Risk | Mitigation |
| --- | --- |
| Qollab's IonQ simulator allows one job per run and no loops | The demo is one circuit and one job with frozen readout weights; all training, shots and noise sweeps run locally on Aer and ship as cached CSVs; ask whether a job may carry several circuits |
| Qollab's simulator rejects mid-circuit `reset` | Asked in #hackathon-help and tested hands-on before the Sat 14:00 ET gate; fallback is the 24-qubit circuit with fresh input qubits for each lag, already built and tested |
| The demo errors on Qollab's simulator, which the theme forbids | Draft project published in Phase A; the single-job demo runs on Qollab at the Saturday-midnight ET minimum-viable gate, not for the first time on Sunday |
| The shared simulator queue is slow near the deadline | Final platform run by Sun 13:00 ET; one job only; all debugging local on Aer |
| Qollab runtime (Pyodide) cannot import torch, reservoirpy or scikit-learn | The demo needs only Qiskit and NumPy: frozen weights are a small array applied with a dot product; everything heavier is precomputed |
| Local simulation time explodes (816 circuits × shot levels × noise levels) | Features computed once per setting and cached; cap at 4 shot levels and 5 noise levels; 10-qubit circuits take milliseconds to seconds each |
| A judge thinks the results came from IonQ when they came from Aer | Every table and figure is labelled with its simulator; the project page has a short "what runs where" section |
| QR1 ranks behind HARX and ARMAX | Measured and stated plainly; the claim is reservoir against reservoir, and the novelty lives in the corrected benchmark and the shots and noise studies |
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
