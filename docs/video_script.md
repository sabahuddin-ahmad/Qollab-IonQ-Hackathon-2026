# Video script

Target length 2 minutes 40 seconds, about 390 spoken words. Each block gives the time, what is on
screen and the narration. Figures are in `docs/figures/`.

| Time | On screen | Narration |
| --- | --- | --- |
| 0:00 to 0:15 | Title card: project name, team, "Quantum Impact". | Volatility forecasts drive risk limits and option prices. A recent paper claims a ten-qubit quantum reservoir forecasts stock-market volatility better than classical models. We rebuilt it to find out what it really delivers, and what it takes to run it on a quantum device. |
| 0:15 to 0:40 | `qr1_circuit.png`. Point to the seven input qubits, the three memory qubits, then the measurements. | The model is hybrid. Seven qubits each carry one market feature, and three act as memory. We feed in three months of data, letting the qubits interact after each month, and measure all ten. The circuit is never trained. A classical regression with only ten weights turns the measurements into a forecast. |
| 0:40 to 1:05 | `benchmark.png`. Highlight the top group, then the QR1 row. | First, a fair benchmark. The published baselines were scored against the wrong month. Scored correctly, and compared like for like, the quantum reservoir is third of four: behind two linear models, and ahead of the classical reservoir, with ten percent lower error and half the trained weights. |
| 1:05 to 1:25 | `controls.png`. Trace the blue line from "gates off" to "as designed". | Is the quantum part doing anything? We switched the entangling gates off. The model got nine percent worse. The entanglement is what carries memory from month to month. |
| 1:25 to 1:55 | `transfer.png`, then `ionq_calibration.png`. | Now the real obstacle. On a noisy device, a model trained in ideal simulation fails: its error rises fourfold. Our fix is to retrain only the ten classical weights under matched noise. The quantum circuit stays the same, and the accuracy comes back. On IonQ's Forte 1 simulator, a single calibration circuit set the noise level, and it then matched two more months without adjustment. |
| 1:55 to 2:15 | Screen recording of the Qollab run on the IonQ simulator: select backend, run, output scrolling to the forecast line. | Here it is running on Qollab: one circuit, one job, twenty-four qubits and two hundred and seventy two-qubit gates on the IonQ simulator, producing a forecast next to the classical ones. |
| 2:15 to 2:40 | Text card with three lines: "Better than the classical reservoir, with half the weights", "Works under device noise with a matched readout", "Not a quantum advantage: the best linear model is still ahead". Then the attribution line. | So what is the impact? A quantum reservoir that beats its classical counterpart, a recipe that makes such models work on noisy hardware, and a simple test for whether the quantum part matters. It is not a quantum advantage: the best linear model is still slightly ahead, and we say so. Everything is open and reproducible. This effort is supported by Qollab and IonQ. |

## Recording notes

- Record the Qollab segment first. A 500-shot run on the IonQ simulator takes about 160 seconds, so
  record it in full and speed it up, or cut from "Running..." to the output.
- Read the narration at an even pace; the table is timed at about 150 words per minute.
- The rules say production quality is not scored. A screen recording with voice-over is enough.
- If the video must be under 2 minutes, drop the 1:05 to 1:25 block and shorten the opening.
