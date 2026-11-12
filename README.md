# QuRescue: quantum dispatch for disaster-relief drones

**Problem:** In a flood, earthquake or frontline emergency, a few drones face many requests for help (injured people, families on rooftops, villages without water, gas leaks, unknown areas). **Who does each drone help first?** The objectives compete: *most critical first* vs. *most people helped* vs. battery and distance, with limited drones, limited range, limited time, and limited qubits.

**Approach:** A multi-objective **QUBO** solved with **QAOA in Qiskit**, plus **warm-start re-planning** when the situation changes (water rises, aftershock, new casualty).

## Quick start (about 5 minutes)

```bash
pip install -r requirements.txt
streamlit run app.py            # live demo
python benchmark.py             # experiments -> results/*.png + summary.json (about 2.5 min)
python run_on_ibm.py --mode border --fake   # noisy IBM-device simulation (no account needed)
python run_on_ibm.py --mode border          # real IBM hardware (see setup at top of file)
```
`tutorial.ipynb` walks through the whole method step by step.

## Demo script (2 minutes)
1. Pick **Flood relief**, slider at 0.6, press **Solve**. Point out: *14 qubits, 18 pairs pruned by domain rules, QAOA = optimal, greedy is not.*
2. Press **Water rising**. The urgencies change, and the convergence chart shows **warm start (red)** settling immediately while **cold start (grey)** wanders.
3. Drag the slider to **0** and press Solve: the drones go to the villages (406 people). At **1**, they go to the injured man and the rooftop family (61 people, much higher urgency). *That is the triage dilemma, made explicit.*
4. Switch to **Earthquake** and **Border post**: same engine, different disaster.

## Design decisions (judges will ask)
| Decision | Why |
|---|---|
| Variables only for feasible drone-request pairs | Payload and range rules cut 32 naive variables to 14 qubits: domain knowledge = fewer qubits |
| "At most one" constraints as pairwise penalties | No slack qubits needed (a strict capacity inequality would add qubits) |
| Penalty P = 2 x max benefit (+0.1) | Breaking a rule always costs more than any reward gained |
| Benefit = alpha·urgency + (1-alpha)·log(people) - lambda·distance | Makes the competing objectives explicit and tunable by the operator |
| Hamiltonian normalised to max coefficient 1 | Keeps QAOA angles in the same range across scenarios, which makes warm start work |
| Warm start = parameter transfer | Re-planning reuses the previous optimal angles, so the optimiser starts near a good region |
| Hardware: optimise angles on simulator, sample once on QPU | Uses seconds of quantum time instead of minutes |

## Results (local Qiskit statevector simulator, p = 2, 8 change events per mode)
| Mode | Qubits | Warm start finds optimal plan | Cold start | Greedy | Warm vs cold final energy (median) |
|---|---|---|---|---|---|
| Flood | 14 | **100%** | 88% (got stuck in 3/8 runs) | 0% | warm 2.8% better |
| Earthquake | 10 | **100%** | 75% | 50% | warm 9.6% better |
| Border | 6 | **100%** | 100% | 100% | warm 8.2% better |

- Warm start reaches 95% of the best energy in **about 1–17 evaluations vs. about 100–160 for cold start**.
- Trade-off (flood): slider 0 gives **406 people / urgency 2.3**, slider 1 gives **61 people / urgency 3.2**. See `results/tradeoff_*.png`.
- Noisy IBM Torino simulation (p = 1): the 6-qubit border plan still finds the optimal dispatch. At 10 qubits, noise brings P(optimal) close to random guessing. **Hardware noise is why small, pruned problems matter today.**

## Honest limitations
- At these sizes, brute force is instant: **we do not claim quantum advantage.** We show *how* to formulate drone dispatch for quantum hardware and that warm starting makes re-planning faster and more reliable.
- QAOA puts only a small probability on the exact optimum (about 0.1–6%). We sample 4,096 times and keep the best feasible plan, which is standard QAOA practice.
- Scenarios are synthetic. Real deployment would need GIS maps, live reports and weather data.

## Future work
Decompose large disasters into regional sub-problems; a warm-start mixer (Egger et al. 2021); multi-sortie scheduling; real maps of flood-prone districts; error mitigation on hardware.

## Ethics
QuRescue optimises **rescue, medical evacuation, supply and survey** missions. It performs no targeting.

## Structure
```
qurescue/scenario.py  disaster modes, drones, requests, events
qurescue/qubo.py      multi-objective QUBO + Ising conversion
qurescue/solvers.py   brute force, greedy, QAOA (cold / warm)
app.py                Streamlit demo
benchmark.py          experiments and plots
run_on_ibm.py         real IBM hardware / noisy fake backend
tutorial.ipynb        step-by-step tutorial
```
