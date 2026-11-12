"""QuRescue live demo.  Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qurescue.scenario import make_scenario, escalate, REQUEST_STYLE, dist
from qurescue.qubo import build_qubo, bits_to_x
from qurescue.solvers import brute_force, greedy, qaoa, assignment

st.set_page_config(page_title="QuRescue · Quantum dispatch for relief drones",
                   page_icon="🛟", layout="wide")

ORANGE, BLUE, GREY, CARD = "#f28c28", "#5aa9e6", "#9fb0c3", "#17283f"
DRONE_STYLE = {"medical": ("cross", "#ff6b6b", "Medical drone"),
               "cargo": ("square", BLUE, "Cargo drone"),
               "sensor": ("triangle-up", "#b48ef0", "Sensor drone")}
REQ_COLORS = {"medical": "#ff6b6b", "stranded": ORANGE, "supply": BLUE,
              "gas_leak": "#b48ef0", "survey": "#5cc98b"}
MODES = {"flood": "🌊 Flood relief", "earthquake": "🏔️ Earthquake response",
         "border": "🪖 Border post (casualty evac)"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #17283f; border-radius: 12px; padding: 14px 18px;}
div[data-testid="stMetricLabel"] p {color: #9fb0c3;}
.hero h1 {font-size: 2.4rem; margin-bottom: 0;}
.hero p {color: #9fb0c3; font-size: 1.05rem; margin-top: .3rem;}
.pill {display:inline-block; background:#17283f; color:#eef2f6; border-radius:999px;
       padding:4px 12px; margin-right:6px; font-size:.85rem;}
.ok {color:#5cc98b; font-weight:600;} .bad {color:#ff6b6b; font-weight:600;}
</style>""", unsafe_allow_html=True)


# ---------------- cached solvers ----------------
@st.cache_data(show_spinner=False)
def scenario_at(mode, events):
    sc = make_scenario(mode)
    for e in range(1, events + 1):
        sc = escalate(sc, seed=100 + e)
    return sc


@st.cache_data(show_spinner=False)
def solve(mode, events, alpha, reps, init, seed):
    sc = scenario_at(mode, events)
    model = build_qubo(sc, alpha=alpha)
    bf = brute_force(model)
    if init is None:
        res = qaoa(model, reps=reps, seed=seed, maxiter=150, optimum_cost=bf["cost"])
    else:
        res = qaoa(model, reps=reps, init_params=np.array(init), maxiter=120, rhobeg=0.15,
                   optimum_cost=bf["cost"])
    return model, bf, greedy(model), res


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### 🛟 QuRescue")
    st.caption("Quantum dispatch for disaster-relief drones")
    mode = st.selectbox("Scenario", list(MODES), format_func=MODES.get)
    alpha = st.slider("Priority", 0.0, 1.0, 0.6, 0.1,
                      help="0 = help the most people · 1 = most critical first")
    st.caption("0 = most people helped · 1 = most critical first")
    reps = st.select_slider("QAOA layers (p)", [1, 2, 3], value=2,
                            help="More layers = more expressive circuit, but deeper and slower")
    compare = st.toggle("Compare with cold start on events", value=True)

ss = st.session_state
if ss.get("mode") != mode:
    ss.update(mode=mode, events=0, params=None, last=None)

sc = scenario_at(mode, ss.events)
with st.sidebar:
    c1, c2 = st.columns(2)
    solve_clicked = c1.button("Solve", type="primary", use_container_width=True)
    event_clicked = c2.button(sc.event_name, use_container_width=True)
    if st.button("Reset scenario", use_container_width=True):
        ss.update(events=0, params=None, last=None)
        st.rerun()
    st.divider()
    st.caption("Built with Qiskit · [GitHub](https://github.com/5aadii/qurescue)")

if event_clicked:
    ss.events += 1
    sc = scenario_at(mode, ss.events)

if solve_clicked or event_clicked:
    warm = event_clicked and ss.params is not None and len(ss.params) == 2 * reps
    with st.spinner("Running QAOA…"):
        model, bf, gr, main = solve(mode, ss.events, alpha, reps,
                                    tuple(ss.params) if warm else None, 1)
        cold = solve(mode, ss.events, alpha, reps, None, ss.events)[3] if (warm and compare) else None
    ss.params = list(main["params"])
    ss.last = dict(model=model, bf=bf, gr=gr, main=main, cold=cold, warm=warm,
                   alpha=alpha, events=ss.events)

# ---------------- header ----------------
st.markdown(f"""<div class="hero"><h1>Who should each drone help first?</h1>
<p>{sc.title}. QuRescue turns the choice into a QUBO and solves it with QAOA in Qiskit.</p></div>""",
            unsafe_allow_html=True)
st.markdown(f'<span class="pill">{len(sc.drones)} drones</span>'
            f'<span class="pill">{len(sc.requests)} requests</span>'
            f'<span class="pill">events so far: {ss.events}</span>', unsafe_allow_html=True)
st.write("")


def draw_map(sc, model=None, x=None):
    fig = go.Figure()
    for hx, hy, r in sc.hazard_zones:
        fig.add_shape(type="circle", x0=hx - r, y0=hy - r, x1=hx + r, y1=hy + r,
                      fillcolor="rgba(90,169,230,0.18)" if sc.mode == "flood" else "rgba(159,176,195,0.15)",
                      line_width=0, layer="below")
    req = {r.id: r for r in sc.requests}
    dr = {d.id: d for d in sc.drones}
    if model is not None and x is not None:
        for r, d in assignment(model, x):
            p, q = dr[d].pos, req[r].pos
            fig.add_annotation(x=q[0], y=q[1], ax=p[0], ay=p[1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3,
                               arrowsize=1.2, arrowwidth=2.5, arrowcolor=DRONE_STYLE[dr[d].kind][1],
                               opacity=0.9)
    for kind, color in REQ_COLORS.items():
        rs = [r for r in sc.requests if r.kind == kind]
        if not rs:
            continue
        fig.add_trace(go.Scatter(
            x=[r.pos[0] for r in rs], y=[r.pos[1] for r in rs], mode="markers+text",
            name=REQUEST_STYLE[kind][0], text=[r.label for r in rs],
            textposition=["bottom center" if (kind == "supply" and r.pos[1] < 5) else "top center" for r in rs],
            textfont=dict(size=11, color="#eef2f6"),
            marker=dict(size=[14 + 2.2 * np.sqrt(r.people) for r in rs], color=color,
                        line=dict(color="#0f1b2d", width=2)),
            customdata=[[r.urgency, r.people] for r in rs],
            hovertemplate="<b>%{text}</b><br>urgency %{customdata[0]:.2f}<br>%{customdata[1]} people<extra></extra>"))
    for kind, (sym, color, name) in DRONE_STYLE.items():
        ds = [d for d in sc.drones if d.kind == kind]
        if ds:
            fig.add_trace(go.Scatter(x=[d.pos[0] for d in ds], y=[d.pos[1] for d in ds],
                                     mode="markers", name=name,
                                     marker=dict(symbol=sym, size=16, color=color,
                                                 line=dict(color="#eef2f6", width=1.5)),
                                     hovertemplate=f"{name}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[sc.base[0]], y=[sc.base[1]], mode="markers", name="Relief base",
                             marker=dict(symbol="star", size=22, color="#ffd166",
                                         line=dict(color="#0f1b2d", width=1)),
                             hovertemplate="Relief base<extra></extra>"))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor=CARD, plot_bgcolor=CARD,
                      xaxis=dict(visible=False, range=[0, 11]),
                      yaxis=dict(visible=False, range=[0, 11], scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.02, font=dict(color="#c9d4e0", size=11)),
                      font=dict(color="#eef2f6"))
    return fig


L = ss.last
tab_plan, tab_quantum, tab_about = st.tabs(["🗺️ Dispatch plan", "⚛️ Under the hood", "ℹ️ How it works"])

with tab_plan:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(draw_map(sc, L["model"], L["main"]["x"]) if L else draw_map(sc),
                        use_container_width=True, config={"displayModeBar": False})
    with right:
        if not L:
            st.info("Press **Solve** to plan the first sortie. Then press "
                    f"**{sc.event_name}** to see QAOA re-plan from its previous answer.")
            st.markdown("**Requests waiting**")
            st.dataframe(pd.DataFrame([{"Request": r.label, "Type": REQUEST_STYLE[r.kind][0],
                                        "People": r.people, "Urgency": r.urgency}
                                       for r in sorted(sc.requests, key=lambda r: -r.urgency)]),
                         hide_index=True, use_container_width=True)
        else:
            model, main, bf, gr = L["model"], L["main"], L["bf"], L["gr"]
            req = {r.id: r for r in sc.requests}
            dr = {d.id: d for d in sc.drones}
            plan = assignment(model, main["x"])
            people = sum(req[r].people for r, _ in plan)
            urg = sum(req[r].urgency for r, _ in plan)
            km = sum(2 * dist(req[r].pos, dr[d].pos) for r, d in plan)
            q_ok = abs(main["cost"] - bf["cost"]) < 1e-9
            g_ok = abs(gr["cost"] - bf["cost"]) < 1e-9

            m1, m2, m3 = st.columns(3)
            m1.metric("People reached", people)
            m2.metric("Urgency served", f"{urg:.2f}")
            m3.metric("Flight km", f"{km:.0f}")
            st.markdown(
                f"QAOA plan vs exact optimum: <span class='{'ok' if q_ok else 'bad'}'>"
                f"{'✓ optimal' if q_ok else '✗ not optimal'}</span> &nbsp;·&nbsp; "
                f"Greedy: <span class='{'ok' if g_ok else 'bad'}'>"
                f"{'✓ optimal' if g_ok else '✗ misses it'}</span>", unsafe_allow_html=True)

            st.markdown("**Dispatch plan**")
            rows = [{"Drone": f"{DRONE_STYLE[dr[d].kind][2]} #{d}", "Goes to": req[r].label,
                     "People": req[r].people, "Urgency": req[r].urgency} for r, d in plan]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"People": st.column_config.NumberColumn(width="small"),
                                        "Urgency": st.column_config.NumberColumn(width="small", format="%.2f")})
            waiting = [req[r].label for r in req if r not in {p[0] for p in plan}]
            if waiting:
                st.caption("Next sortie: " + " · ".join(waiting))

            fig = go.Figure()
            if L["cold"] is not None:
                fig.add_trace(go.Scatter(y=L["cold"]["history"], name=f"cold start ({L['cold']['nfev']} runs)",
                                         line=dict(color=GREY, width=2)))
            fig.add_trace(go.Scatter(y=main["history"],
                                     name=f"{'warm' if L['warm'] else 'QAOA'} ({main['nfev']} runs)",
                                     line=dict(color=ORANGE, width=3)))
            fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0),
                              title=dict(text="QAOA convergence", font=dict(size=14)),
                              paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(color="#c9d4e0"),
                              xaxis_title="circuit evaluations", yaxis_title="⟨H⟩",
                              legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if L["warm"]:
                st.caption(f"After **{sc.event_name.lower()}**, QAOA started from the previous plan's angles.")

with tab_quantum:
    if not L:
        st.info("Solve a scenario first.")
    else:
        model, main, bf = L["model"], L["main"], L["bf"]
        a, b, c, d = st.columns(4)
        a.metric("Qubits", model.n)
        b.metric("Pairs pruned", model.pruned, help="Drone–request pairs removed by payload and range rules")
        c.metric("Circuit depth", main["depth"])
        d.metric("Penalty P", f"{model.penalty:.2f}", help="2 × largest benefit + 0.1")
        st.markdown("**What the quantum circuit returned:** the most frequent measured plans out of "
                    "4,096 shots, ranked. ★ orange = the exact optimal plan (added if it ranked lower); "
                    "blue = valid plan; grey = breaks a rule.")
        counts = main["counts"]
        shots = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda t: -t[1])
        top = ranked[:12]
        opt_rank = next((i for i, (bs, _) in enumerate(ranked)
                         if abs(model.cost(bits_to_x(bs)) - bf["cost"]) < 1e-9), None)
        if opt_rank is not None and opt_rank >= 12:
            top = top[:11] + [ranked[opt_rank]]
        labels, colors, probs, hover = [], [], [], []
        for i, (bstr, v) in enumerate(top):
            xx = bits_to_x(bstr)
            optimal = abs(model.cost(xx) - bf["cost"]) < 1e-9
            rank = ranked.index((bstr, v)) + 1
            labels.append(f"★ #{rank} {bstr}" if optimal else f"#{rank} {bstr}")
            probs.append(v / shots)
            colors.append(ORANGE if optimal else (BLUE if model.feasible(xx) else "#4a5562"))
            hover.append(f"cost {model.cost(xx):.3f}" + (" · optimal plan" if optimal else
                         "" if model.feasible(xx) else " · breaks a rule"))
        fig = go.Figure(go.Bar(x=probs, y=labels, orientation="h", marker_color=colors,
                               customdata=hover,
                               hovertemplate="%{y}<br>p = %{x:.4f}<br>%{customdata}<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=CARD,
                          plot_bgcolor=CARD, font=dict(color="#c9d4e0", family="monospace"),
                          yaxis=dict(autorange="reversed", type="category"), xaxis_title="probability")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ratio = main["p_opt"] * 2 ** model.n
        st.caption(f"P(exact optimum) this run: {main['p_opt']:.4f}, about {ratio:.0f}× a random guess "
                   f"({1 / 2 ** model.n:.5f}). Low-depth QAOA spreads probability across many plans, "
                   "so we sample and keep the best plan that obeys every rule.")
        st.markdown("**Each qubit is one possible pair**")
        req = {r.id: r for r in sc.requests}
        drs = {d.id: d for d in sc.drones}
        st.dataframe(pd.DataFrame([{"Qubit": k, "Drone": f"#{d} ({drs[d].kind})", "Request": req[r].label,
                                    "Benefit": round(float(model.benefit[k]), 3),
                                    "In plan": "✓" if main["x"][k] else ""}
                                   for k, (r, d) in enumerate(model.pairs)]),
                     hide_index=True, use_container_width=True)

with tab_about:
    st.markdown("""
#### How QuRescue works
1. **Scenario:** drones (medical, cargo, sensor) and requests for help, each with urgency and people.
2. **Pruning:** a qubit only exists for pairs that make sense (right payload, battery can make the round trip).
3. **QUBO:** benefit = α·urgency + (1−α)·log(people) − λ·distance; clashing pairs (same drone or same request) get a penalty.
4. **QAOA in Qiskit:** the QUBO becomes an Ising Hamiltonian; QAOA (COBYLA optimiser) searches for its lowest energy.
5. **Sample and check:** 4,096 shots, keep the best plan that obeys every rule, compare with brute force and greedy.
6. **Warm start:** when the situation changes, QAOA starts from the last plan's angles instead of random ones.

We don't claim a quantum speedup: brute force is instant at this size. The point is a clean way to put rescue dispatch
on quantum hardware, and a re-planning trick that makes QAOA faster and more reliable.

QuRescue plans rescue, medical, supply and survey flights only. It does no targeting.

[Code, report and tutorial on GitHub](https://github.com/5aadii/qurescue)
""")
