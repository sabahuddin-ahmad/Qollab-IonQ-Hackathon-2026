# Video script

Target length about 2 minutes 45 seconds. The narration is about 390 words, 2 minutes 35 seconds at
150 words per minute, which leaves a little room for pauses. Each
block gives the time, what is on screen and the narration.

Almost everything on screen comes from one Qollab run on the IonQ Forte 1 simulator: the run now
shows the findings, the circuit, the measured values, the forecast and the actual-against-predicted
plot in that order, so the video can follow the output from top to bottom. Two figures come from
`docs/figures/` instead and are marked.

| Time | On screen | Narration |
| --- | --- | --- |
| 0:00 to 0:15 | Title card: project title, team QuanTech, "Quantum Impact". | Volatility forecasts drive risk limits and option prices. A recent paper claims a ten-qubit quantum reservoir forecasts stock-market volatility better than classical models. We rebuilt it to find out what it really delivers, and what it takes to run it on a quantum device. |
| 0:15 to 0:30 | Qollab: select the IonQ Forte 1 simulator, set 500 shots, press run. Cut to the output starting. | Everything you will see comes from one run on Qollab, on IonQ's Forte 1 simulator. It submits a single circuit and reports what we found. |
| 0:30 to 0:50 | Run output: the circuit figure. Point to the input qubits, the memory qubits, the evolve blocks. | The model is hybrid. Seven qubits each carry one market feature, and three act as memory. Three months of data are loaded in turn, with the qubits interacting after each month. The circuit is never trained. A classical regression with ten weights turns the measurements into a forecast. |
| 0:50 to 1:20 | Run output: the "What we found" figure. Point to panel 1, then panel 2. | First, a fair benchmark. The published baselines were scored against the wrong month. Scored correctly, and compared like for like, the quantum reservoir is third of four: behind two linear models, and ahead of the classical reservoir, with ten percent lower error and half the trained weights. And the quantum part is doing real work: switch the entanglement off, and the model gets nine percent worse. |
| 1:20 to 1:50 | Same figure, panel 3. Then the run's forecast figure: point to the two QR1 rows. | Now the real obstacle. On a noisy device, a model trained in ideal simulation fails: its error rises fourfold. Our fix is to retrain only the ten classical weights under matched noise. The circuit stays the same. You can see it live in this run: the same measured values give an error of point four with the matched readout, and more than one with the unmatched one. |
| 1:50 to 2:05 | `docs/figures/ionq_calibration.png`. | We set the noise level from a single circuit on IonQ's simulator. It then matched two more months without any adjustment. |
| 2:05 to 2:25 | Run output: the forecast figure, hybrid row. Then the actual-against-predicted figure, scrolling over the three panels. | We also built a second hybrid, feeding the reservoir's measured values into the best linear model alongside its own inputs. Over twenty years of monthly forecasts it edges that model on all twenty random reservoirs, by two percent, which is too small to be statistically significant. Our controls trace the gain to how the inputs are encoded, not to entanglement. |
| 2:25 to 2:45 | Text card with three lines: "Better than the classical reservoir, with half the weights", "Works under device noise with a matched readout", "Not a quantum advantage: a linear model with the same inputs is as good". Then the attribution line. | So what is the impact? A quantum reservoir that beats its classical counterpart, a recipe that makes such models work on noisy hardware, and a simple test for whether the quantum part matters. It is not a quantum advantage, and we say so. Everything is open and reproducible. This effort is supported by Qollab and IonQ. |

## Recording notes

- **Record the Qollab run first.** A 500-shot run on the IonQ simulator takes about 160 seconds.
  Record it in full, then cut from "Submitting..." to the result, or speed that part up.
- **Scroll the output slowly** and pause on each figure for a few seconds; the narration is timed
  to five stops: circuit, findings, forecast, forecast again for the hybrid row, actual against
  predicted. The findings figure appears before the circuit in the output, so the video visits
  them out of order; that is fine, or swap the 0:30 and 0:50 blocks to follow the page exactly.
- **The numbers you say should match your recording.** The script says "point four" and "more than
  one" for the two readouts, which is what the 2017-11 run gave (+0.40 and +1.08). If your
  recorded run shows different values, read those instead.
- **Use month 2017-11 or 2017-12.** Both show the matched-readout effect clearly.
- The rules say production quality is not scored. A screen recording with voice-over is enough.
- **If it must be shorter,** drop the 1:50 to 2:05 block (the calibration figure) and trim the
  opening.
