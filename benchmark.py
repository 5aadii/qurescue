"""Runs the QuRescue experiments and saves plots + a results summary into results/.

1. Trade-off sweep (Pareto): 'most people helped' <-> 'most critical first'
2. Warm-start vs cold-start QAOA after the situation changes (event re-solve)
3. Quantum vs classical baselines (brute force = ground truth, greedy)

Usage:  python benchmark.py            (about 1-3 minutes on a laptop)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qurescue.scenario import make_scenario, escalate, dist
from qurescue.qubo import build_qubo
from qurescue.solvers import brute_force, greedy, qaoa, evals_to_reach, assignment

os.makedirs("results", exist_ok=True)
REPS = 2


def plan_metrics(sc, model, x):
    req = {r.id: r for r in sc.requests}
    dr = {d.id: d for d in sc.drones}
    urg = people = km = 0.0
    for r, d in assignment(model, x):
        urg += req[r].urgency
        people += req[r].people
        km += 2 * dist(req[r].pos, dr[d].pos)
    return urg, people, km


def tradeoff(mode="flood"):
    sc = make_scenario(mode)
    alphas = np.linspace(0, 1, 11)
    rows = []
    for a in alphas:
        m = build_qubo(sc, alpha=a)
        bf = brute_force(m)
        q = qaoa(m, reps=REPS, seed=1, optimum_cost=bf["cost"])
        u, p, km = plan_metrics(sc, m, q["x"])
        rows.append({"alpha": float(a), "urgency_served": u, "people_served": p, "km": km,
                     "qaoa_matches_optimum": abs(q["cost"] - bf["cost"]) < 1e-9})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc_ = ax.scatter([r["people_served"] for r in rows], [r["urgency_served"] for r in rows],
                     c=[r["alpha"] for r in rows], cmap="coolwarm", s=90, edgecolor="k", zorder=3)
    ax.plot([r["people_served"] for r in rows], [r["urgency_served"] for r in rows], "k--", alpha=.4)
    plt.colorbar(sc_, label="slider: 0 = most people  ->  1 = most critical")
    ax.set_xlabel("People reached")
    ax.set_ylabel("Total urgency served")
    ax.set_title(f"Trade-off curve found by QAOA ({mode})")
    ax.grid(alpha=.3)
    fig.tight_layout()
    if mode == "flood":
        fig.savefig(f"results/tradeoff_{mode}.png", dpi=150)
    plt.close(fig)
    return rows


def warm_vs_cold(mode="flood", n_events=8):
    sc = make_scenario(mode)
    m0 = build_qubo(sc, alpha=0.6)
    base = qaoa(m0, reps=REPS, seed=0)
    out = []
    example = None
    for e in range(n_events):
        sc2 = escalate(sc, seed=100 + e)
        m = build_qubo(sc2, alpha=0.6)
        bf = brute_force(m)
        warm = qaoa(m, reps=REPS, init_params=base["params"], optimum_cost=bf["cost"])
        cold = qaoa(m, reps=REPS, seed=200 + e, optimum_cost=bf["cost"])
        target = min(min(warm["history"]), min(cold["history"]))
        out.append({
            "event": e,
            "qubits": m.n,
            "warm_final_energy": float(min(warm["history"])),
            "cold_final_energy": float(min(cold["history"])),
            "warm_total_evals": warm["nfev"], "cold_total_evals": cold["nfev"],
            "warm_evals_to_95pct": evals_to_reach(warm["history"], target),
            "cold_evals_to_95pct": evals_to_reach(cold["history"], target),
            "warm_found_opt": abs(warm["cost"] - bf["cost"]) < 1e-9,
            "cold_found_opt": abs(cold["cost"] - bf["cost"]) < 1e-9,
            "warm_p_opt": warm["p_opt"], "cold_p_opt": cold["p_opt"],
            "greedy_found_opt": abs(greedy(m)["cost"] - bf["cost"]) < 1e-9,
        })
        if example is None:
            example = (warm["history"], cold["history"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(example[1], label="cold start (random angles)", color="#888")
    ax.plot(example[0], label="warm start (angles from previous plan)", color="#d62728")
    ax.set_xlabel("Optimiser evaluations (quantum circuit runs)")
    ax.set_ylabel("Expected cost <H>")
    ax.set_title("Re-planning after the situation changes")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    if mode == "flood":
        fig.savefig(f"results/warm_vs_cold_{mode}.png", dpi=150)
    plt.close(fig)

    return out


if __name__ == "__main__":
    summary = {}
    for mode in ["flood", "earthquake", "border"]:
        print(f"== {mode} ==")
        t = tradeoff(mode)
        wc = warm_vs_cold(mode)
        summary[mode] = {
            "tradeoff": t,
            "warm_vs_cold": wc,
            "mean_warm_evals": float(np.mean([r["warm_evals_to_95pct"] for r in wc])),
            "mean_cold_evals": float(np.mean([r["cold_evals_to_95pct"] for r in wc])),
            "qaoa_opt_rate_warm": float(np.mean([r["warm_found_opt"] for r in wc])),
            "qaoa_opt_rate_cold": float(np.mean([r["cold_found_opt"] for r in wc])),
            "cold_stuck_events": int(sum(r["cold_final_energy"] > 0.8 * r["warm_final_energy"] for r in wc)),
            "greedy_opt_rate": float(np.mean([r["greedy_found_opt"] for r in wc])),
            "mean_p_opt_warm": float(np.mean([r["warm_p_opt"] for r in wc])),
            "mean_p_opt_cold": float(np.mean([r["cold_p_opt"] for r in wc])),
            "median_energy_gap_pct": float(np.median([
                100 * (r["warm_final_energy"] - r["cold_final_energy"]) / abs(r["cold_final_energy"])
                for r in wc])),
        }
        s = summary[mode]
        print(f"  warm evals {s['mean_warm_evals']:.0f} vs cold {s['mean_cold_evals']:.0f} | "
              f"QAOA optimal: warm {s['qaoa_opt_rate_warm']:.0%} cold {s['qaoa_opt_rate_cold']:.0%} | "
              f"greedy optimal {s['greedy_opt_rate']:.0%} | P(opt) warm {s['mean_p_opt_warm']:.3f} "
              f"cold {s['mean_p_opt_cold']:.3f} | median warm-vs-cold final energy: "
              f"{s["median_energy_gap_pct"]:+.1f}% (negative = warm better)")
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print("Saved plots and results/summary.json")
