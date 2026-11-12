"""QuRescue live demo.  Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import streamlit as st

from qurescue.scenario import make_scenario, escalate, REQUEST_STYLE, dist
from qurescue.qubo import build_qubo
from qurescue.solvers import brute_force, greedy, qaoa, assignment

st.set_page_config(page_title="QuRescue", layout="wide")
DRONE_MARK = {"medical": ("P", "#d62728"), "cargo": ("s", "#1f77b4"), "sensor": ("^", "#9467bd")}

# ---------------- sidebar ----------------
st.sidebar.title("QuRescue")
st.sidebar.caption("Quantum dispatch for disaster-relief drones")
mode = st.sidebar.selectbox("Scenario", ["flood", "earthquake", "border"],
                            format_func=lambda m: {"flood": "Flood relief",
                                                   "earthquake": "Earthquake response",
                                                   "border": "Border post (casualty evac)"}[m])
alpha = st.sidebar.slider("Priority:  most people helped  <->  most critical first",
                          0.0, 1.0, 0.6, 0.1)
reps = st.sidebar.select_slider("QAOA layers (p)", [1, 2, 3], value=2)

ss = st.session_state
if ss.get("mode") != mode:
    ss.mode = mode
    ss.sc = make_scenario(mode)
    ss.params = None
    ss.events = 0
    ss.last = None

sc = ss.sc
c1, c2 = st.sidebar.columns(2)
solve_clicked = c1.button("Solve", type="primary", use_container_width=True)
event_clicked = c2.button(f"{sc.event_name}", use_container_width=True)
if st.sidebar.button("Reset scenario", use_container_width=True):
    ss.sc = make_scenario(mode); ss.params = None; ss.events = 0; ss.last = None
    st.rerun()

# ---------------- solve ----------------
if event_clicked:
    ss.events += 1
    ss.sc = escalate(ss.sc, seed=100 + ss.events)
    sc = ss.sc

if solve_clicked or event_clicked:
    model = build_qubo(sc, alpha=alpha)
    bf = brute_force(model)
    with st.spinner(f"Running QAOA on {model.n} qubits..."):
        if event_clicked and ss.params is not None and len(ss.params) == 2 * reps:
            warm = qaoa(model, reps=reps, init_params=ss.params, optimum_cost=bf["cost"])
            cold = qaoa(model, reps=reps, seed=ss.events, optimum_cost=bf["cost"])
            main, compare = warm, cold
        else:
            main, compare = qaoa(model, reps=reps, seed=1, optimum_cost=bf["cost"]), None
    ss.params = main["params"]
    ss.last = dict(model=model, bf=bf, gr=greedy(model), main=main, compare=compare,
                   alpha=alpha, event=event_clicked)

# ---------------- layout ----------------
st.title("QuRescue: who does each drone help first?")
st.caption(sc.title)
left, right = st.columns([3, 2])

def draw_map(sc, model=None, x=None):
    fig, ax = plt.subplots(figsize=(7, 6.3))
    for (hx, hy, r) in sc.hazard_zones:
        ax.add_patch(Circle((hx, hy), r, color="#6baed6" if sc.mode == "flood" else "#bdbdbd",
                            alpha=.35, lw=0))
    ax.scatter(*sc.base, marker="*", s=400, c="gold", edgecolor="k", zorder=5)
    ax.annotate("Relief base", sc.base, xytext=(5, -15), textcoords="offset points", fontsize=8)
    req = {r.id: r for r in sc.requests}
    dr = {d.id: d for d in sc.drones}
    if model is not None and x is not None:
        for r, d in assignment(model, x):
            p, q = dr[d].pos, req[r].pos
            ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", lw=2,
                        color=DRONE_MARK[dr[d].kind][1], alpha=.85))
    for r in sc.requests:
        name, col = REQUEST_STYLE[r.kind]
        ax.scatter(*r.pos, s=80 + 6 * np.sqrt(r.people) * 10, c=col, alpha=.85, edgecolor="k", zorder=4)
        ax.annotate(f"{r.label}\nurgency {r.urgency:.2f} · {r.people} ppl", r.pos,
                    xytext=(7, 5), textcoords="offset points", fontsize=7)
    for d in sc.drones:
        m, col = DRONE_MARK[d.kind]
        ax.scatter(*d.pos, marker=m, s=140, c=col, edgecolor="k", zorder=6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in REQUEST_STYLE.values()]
    handles += [plt.Line2D([], [], marker=m, ls="", color=c, label=f"{k} drone")
                for k, (m, c) in DRONE_MARK.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)
    ax.set_xlim(0, 11); ax.set_ylim(0, 11); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return fig

L = ss.last
with left:
    if L:
        st.pyplot(draw_map(sc, L["model"], L["main"]["x"]))
    else:
        st.pyplot(draw_map(sc))
        st.info("Press **Solve** to plan, then press the event button to see warm-start re-planning.")

with right:
    if L:
        model, main, bf, gr = L["model"], L["main"], L["bf"], L["gr"]
        req = {r.id: r for r in sc.requests}
        dr = {d.id: d for d in sc.drones}
        plan = assignment(model, main["x"])
        people = sum(req[r].people for r, _ in plan)
        urg = sum(req[r].urgency for r, _ in plan)
        km = sum(2 * dist(req[r].pos, dr[d].pos) for r, d in plan)

        m1, m2, m3 = st.columns(3)
        m1.metric("Qubits", model.n, f"-{model.pruned} pruned by domain rules", delta_color="off")
        m2.metric("People reached", people)
        m3.metric("Urgency served", f"{urg:.2f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Flight km", f"{km:.1f}")
        m5.metric("QAOA = optimal?", "Yes" if abs(main["cost"] - bf["cost"]) < 1e-9 else "No")
        m6.metric("Greedy = optimal?", "Yes" if abs(gr["cost"] - bf["cost"]) < 1e-9 else "No")

        st.subheader("Dispatch plan")
        rows = [{"Drone": f"#{d} ({dr[d].kind})", "Goes to": req[r].label,
                 "Type": REQUEST_STYLE[req[r].kind][0], "Urgency": req[r].urgency,
                 "People": req[r].people} for r, d in plan]
        waiting = [req[r].label for r in req if r not in {p[0] for p in plan}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if waiting:
            st.caption("Waiting for next sortie: " + "; ".join(waiting))

        st.subheader("QAOA convergence")
        fig, ax = plt.subplots(figsize=(5, 2.8))
        if L["compare"] is not None:
            ax.plot(L["compare"]["history"], color="#888", label=f"cold start ({L['compare']['nfev']} evals)")
            ax.plot(main["history"], color="#d62728", label=f"warm start ({main['nfev']} evals)")
            ax.legend(fontsize=8)
            st.caption(f"After **{sc.event_name.lower()}**: warm start reuses the previous plan's "
                       f"QAOA angles instead of starting from random ones.")
        else:
            ax.plot(main["history"], color="#d62728")
        ax.set_xlabel("circuit evaluations", fontsize=8); ax.set_ylabel("<H>", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"Circuit depth {main['depth']} · {main['runtime']:.1f}s on local Qiskit "
                   f"statevector simulator · p = {reps}")
