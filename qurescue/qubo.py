"""Multi-objective QUBO for drone-to-request assignment, and its Ising (qubit) form."""
from dataclasses import dataclass
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .scenario import COMPAT, dist


@dataclass
class QuboModel:
    pairs: list            # list of (request_id, drone_id), one per qubit
    benefit: np.ndarray    # benefit of each pair
    Q: np.ndarray          # upper-triangular quadratic coefficients
    c: np.ndarray          # linear coefficients
    penalty: float
    pruned: int            # how many naive pairs were removed by domain rules

    @property
    def n(self):
        return len(self.pairs)

    def cost(self, x):
        x = np.asarray(x, dtype=float)
        return float(self.c @ x + x @ self.Q @ x)

    def feasible(self, x):
        x = np.asarray(x)
        used_d, used_r = set(), set()
        for k, (r, d) in enumerate(self.pairs):
            if x[k]:
                if r in used_r or d in used_d:
                    return False
                used_r.add(r)
                used_d.add(d)
        return True


def build_qubo(sc, alpha=0.5, energy_weight=0.3, max_qubits=14):
    """
    alpha: 1.0 = 'most critical first' (urgency), 0.0 = 'most people helped'.
    Variables x_(r,d)=1 if drone d serves request r, created only when the payload
    matches and the drone can make the round trip (domain pruning saves qubits).
    Constraints (at most one request per drone, at most one drone per request)
    are pairwise penalties: no slack qubits needed.
    """
    max_people = max(r.people for r in sc.requests) or 1
    naive = len(sc.requests) * len(sc.drones)

    cand = []
    for r in sc.requests:
        for d in sc.drones:
            if r.kind not in COMPAT[d.kind]:
                continue
            dd = dist(r.pos, d.pos)
            if 2 * dd > d.range_km:
                continue
            people_n = np.log1p(r.people) / np.log1p(max_people)
            reward = alpha * r.urgency + (1 - alpha) * people_n
            b = reward - energy_weight * (dd / 12.0)
            cand.append(((r.id, d.id), b))

    cand.sort(key=lambda t: -t[1])
    cand = cand[:max_qubits]
    pairs = [p for p, _ in cand]
    benefit = np.array([b for _, b in cand])
    n = len(pairs)

    P = 2.0 * float(np.max(np.abs(benefit))) + 0.1
    c = -benefit.copy()
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            (ri, di), (rj, dj) = pairs[i], pairs[j]
            if ri == rj or di == dj:
                Q[i, j] += P
    return QuboModel(pairs, benefit, Q, c, P, naive - n)


def to_ising(model: QuboModel):
    """x_i = (1 - z_i)/2  ->  H = sum h_i Z_i + sum J_ij Z_i Z_j + const.
    Returns a SparsePauliOp normalised to max |coeff| = 1, plus the scale and offset."""
    n = model.n
    h = np.zeros(n)
    J = {}
    const = 0.0
    for i in range(n):
        const += model.c[i] / 2
        h[i] += -model.c[i] / 2
    for i in range(n):
        for j in range(i + 1, n):
            q = model.Q[i, j]
            if q == 0:
                continue
            const += q / 4
            h[i] += -q / 4
            h[j] += -q / 4
            J[(i, j)] = J.get((i, j), 0) + q / 4
    terms = [("Z", [i], h[i]) for i in range(n) if abs(h[i]) > 1e-12]
    terms += [("ZZ", [i, j], v) for (i, j), v in J.items()]
    scale = max(abs(t[2]) for t in terms)
    op = SparsePauliOp.from_sparse_list([(p, q, v / scale) for p, q, v in terms], num_qubits=n)
    return op, scale, const


def bits_to_x(bitstring: str):
    """Qiskit bitstrings are little-endian: qubit 0 is the rightmost character."""
    return np.array([int(b) for b in bitstring[::-1]])
