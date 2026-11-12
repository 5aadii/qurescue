"""Solvers: brute force (ground truth), greedy (classical baseline), QAOA cold / warm start."""
import itertools
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from .qubo import QuboModel, to_ising, bits_to_x


def brute_force(model: QuboModel):
    n = model.n
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    costs = X @ model.c + np.einsum("bi,ij,bj->b", X, model.Q, X)
    k = int(np.argmin(costs))
    return {"x": X[k].astype(int), "cost": float(costs[k])}


def greedy(model: QuboModel):
    x = np.zeros(model.n, dtype=int)
    used_r, used_d = set(), set()
    for k in np.argsort(-model.benefit):
        r, d = model.pairs[k]
        if model.benefit[k] > 0 and r not in used_r and d not in used_d:
            x[k] = 1
            used_r.add(r)
            used_d.add(d)
    return {"x": x, "cost": model.cost(x)}


def _build(model, reps):
    H, scale, const = to_ising(model)
    ansatz = QAOAAnsatz(cost_operator=H, reps=reps)
    ansatz = transpile(ansatz, basis_gates=["rz", "rx", "rzz", "h", "cx", "sx", "x"],
                       optimization_level=1)
    return H, ansatz


def qaoa(model: QuboModel, reps=2, init_params=None, seed=0, maxiter=200, shots=4096,
         optimum_cost=None, rhobeg=None):
    """Run QAOA on a local statevector simulator.
    init_params=None -> cold start (random angles). Otherwise -> warm start
    (parameter transfer: reuse optimised angles from the previous, similar problem)."""
    H, ansatz = _build(model, reps)
    estimator = StatevectorEstimator()
    history = []

    def energy(theta):
        e = float(estimator.run([(ansatz, H, theta)]).result()[0].data.evs)
        history.append(e)
        return e

    rng = np.random.default_rng(seed)
    if init_params is None:
        x0 = rng.uniform(0, np.pi, ansatz.num_parameters)
        mode = "cold"
    else:
        x0 = np.asarray(init_params, dtype=float)
        mode = "warm"

    t0 = time.time()
    res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": rhobeg or 0.4})
    runtime = time.time() - t0

    meas = ansatz.measure_all(inplace=False)
    counts = StatevectorSampler(seed=seed).run([(meas, res.x)], shots=shots).result()[0].data.meas.get_counts()

    # Pick the best feasible sampled assignment
    best_x, best_c = None, np.inf
    for b, _ in sorted(counts.items(), key=lambda t: -t[1]):
        x = bits_to_x(b)
        if model.feasible(x):
            c = model.cost(x)
            if c < best_c:
                best_x, best_c = x, c

    p_opt = None
    if optimum_cost is not None:
        hits = sum(v for b, v in counts.items()
                   if abs(model.cost(bits_to_x(b)) - optimum_cost) < 1e-9)
        p_opt = hits / shots

    return {"mode": mode, "x": best_x, "cost": float(best_c), "params": res.x,
            "history": history, "nfev": res.nfev, "runtime": runtime,
            "p_opt": p_opt, "counts": counts, "qubits": model.n,
            "depth": ansatz.depth()}


def evals_to_reach(history, target, tol=0.05):
    """Number of optimiser evaluations until energy is within tol of target (lower is better)."""
    span = max(abs(target), 1e-9)
    for i, e in enumerate(history):
        if e <= target + tol * span:
            return i + 1
    return len(history)


def assignment(model: QuboModel, x):
    return [model.pairs[k] for k in range(model.n) if x[k]]
