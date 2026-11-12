"""Run a QuRescue plan on real IBM quantum hardware (or a noisy fake backend).

Strategy (saves your free quantum minutes): optimise the QAOA angles on the local
simulator, then send ONE sampling job to the QPU with those angles and compare
the hardware result to the noiseless simulator and the true optimum.

  # 1. one-time setup (token + instance CRN from quantum.cloud.ibm.com):
  python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='YOUR_API_KEY', instance='YOUR_CRN', overwrite=True)"

  # 2. run on real hardware:
  python run_on_ibm.py --mode border
  # fallback without account: noisy simulation of a real IBM device
  python run_on_ibm.py --mode border --fake

Start with --mode border (6 qubits) - small circuits survive hardware noise best.
"""
import argparse
import json
from collections import Counter

from qiskit.transpiler import generate_preset_pass_manager

from qurescue.scenario import make_scenario
from qurescue.qubo import build_qubo, bits_to_x
from qurescue.solvers import brute_force, qaoa, _build, assignment


def analyse(model, counts, opt_cost, shots):
    best_x, best_c = None, float("inf")
    hits = 0
    feasible = 0
    for b, v in counts.items():
        x = bits_to_x(b)
        c = model.cost(x)
        if model.feasible(x):
            feasible += v
            if c < best_c:
                best_x, best_c = x, c
        if abs(c - opt_cost) < 1e-9:
            hits += v
    return {"best_cost": best_c, "found_optimum": abs(best_c - opt_cost) < 1e-9,
            "p_optimal": hits / shots, "p_feasible": feasible / shots, "x": best_x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="border", choices=["flood", "earthquake", "border"])
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--fake", action="store_true", help="noisy local simulation of an IBM device")
    a = ap.parse_args()

    sc = make_scenario(a.mode)
    model = build_qubo(sc, alpha=a.alpha)
    opt = brute_force(model)
    sim = qaoa(model, reps=a.reps, seed=1, optimum_cost=opt["cost"], shots=a.shots)
    print(f"{a.mode}: {model.n} qubits | optimum cost {opt['cost']:.3f} | "
          f"simulator P(optimal) = {sim['p_opt']:.3f}")

    _, ansatz = _build(model, a.reps)
    circ = ansatz.assign_parameters(sim["params"])
    circ.measure_all()

    if a.fake:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        from qiskit_aer import AerSimulator
        backend = AerSimulator.from_backend(FakeTorino())
        label = "FakeTorino (noisy simulation of IBM Torino)"
        isa = generate_preset_pass_manager(backend=backend, optimization_level=3).run(circ)
        counts = backend.run(isa, shots=a.shots).result().get_counts()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=model.n)
        label = backend.name
        print(f"Submitting to {label} ...")
        isa = generate_preset_pass_manager(backend=backend, optimization_level=3).run(circ)
        job = SamplerV2(mode=backend).run([isa], shots=a.shots)
        print("Job id:", job.job_id(), "(you can watch it at quantum.cloud.ibm.com)")
        counts = job.result()[0].data.meas.get_counts()

    hw = analyse(model, Counter(counts), opt["cost"], a.shots)
    random_p = 1 / 2 ** model.n
    print(f"\nBackend: {label}   transpiled depth: {isa.depth()}")
    print(f"  P(optimal)  hardware {hw['p_optimal']:.4f} | simulator {sim['p_opt']:.4f} | "
          f"random guessing {random_p:.4f}")
    print(f"  P(feasible) hardware {hw['p_feasible']:.3f}")
    print(f"  Best plan found on hardware is optimal: {hw['found_optimum']}")
    req = {r.id: r for r in sc.requests}
    for r, d in assignment(model, hw["x"]):
        print(f"   drone #{d} ({sc.drones[d].kind}) -> {req[r].label}")
    with open(f"results/hardware_{a.mode}{'_fake' if a.fake else ''}.json", "w") as f:
        json.dump({"backend": label, "qubits": model.n, "depth": isa.depth(),
                   "p_opt_hw": hw["p_optimal"], "p_opt_sim": sim["p_opt"], "p_random": random_p,
                   "p_feasible_hw": hw["p_feasible"], "found_optimum": bool(hw["found_optimum"])},
                  f, indent=2)


if __name__ == "__main__":
    main()
