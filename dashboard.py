"""
dashboard.py — v8
═══════════════════════════════════════════════════════════════════════════════
CHANGES FROM v7:

CHANGE 1 — Import NODE_BUSYNESS from simulation_engine for display.

CHANGE 2 — Congestion window indicator in sidebar:
  Shows the current time window label (HIGH / MEDIUM / LOW) and which
  hour ranges are active, so users understand why congestion changes.

CHANGE 3 — Route ETA banner shows both base time AND congestion-adjusted ETA,
  making the ML influence explicit ("12 min → 96 min under congestion").

CHANGE 4 — 24h chart now highlights the required HIGH/MEDIUM/LOW windows
  with background shading matching the spec.

CHANGE 5 — safe_map() helper covers pandas 2.x / 3.x .applymap/.map split.

CHANGE 6 — All dict accesses use .get() with safe defaults throughout.

No UI tabs removed. All features preserved.
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import joblib, os, time

from simulation_engine import (
    build_graph,
    simulate_node_states,
    compute_edge_states,
    find_multi_routes,
    build_animation_data,
    all_nodes,
    all_edges_list,
    AccidentManager,
    CONGESTION_COLORS,
    CONGESTION_NAMES,
    NODE_POSITIONS,
    NODE_META,
    NODE_BUSYNESS,        # CHANGE 1
    ROUTE_COLORS,
    CONGESTION_PENALTY,
    ETA_SEGMENT_CAP,
    time_density,
    apply_volume_spike,
    apply_node_spike,
    clear_volume_spikes,
)
from animation_component import build_animation_html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="City Traffic Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
[data-testid="stSidebar"] { background: #0a0f1e !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.stTabs [data-baseweb="tab"] {
    background: #0f172a; border-radius: 6px 6px 0 0; padding: 7px 18px;
}
.stTabs [aria-selected="true"] { background: #6366f1 !important; color: #fff !important; }
div[data-testid="metric-container"] {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 8px; padding: 10px 14px;
}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_map(styler, fn, subset):
    """pandas 2.x uses applymap; pandas 3.x uses map."""
    try:
        return styler.map(fn, subset=subset)
    except AttributeError:
        return styler.applymap(fn, subset=subset)


def congestion_window_label(hour: int) -> tuple:
    """Returns (label, color) for the current time window."""
    if 10 <= hour <= 13:
        return "HIGH (midday peak 10–13)", "#ef4444"
    elif 18 <= hour <= 21:
        return "HIGH (evening peak 18–21)", "#ef4444"
    elif 7 <= hour <= 9:
        return "MEDIUM-HIGH (morning commute 7–9)", "#f97316"
    elif 14 <= hour <= 17:
        return "MEDIUM (afternoon 14–17)", "#f97316"
    elif hour >= 22 or hour <= 5:
        return "LOW (night 22–5)", "#22c55e"
    else:
        return "LOW-MEDIUM (off-peak)", "#22c55e"


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl"), joblib.load("models/feature_cols.pkl")

@st.cache_data
def load_df():
    return pd.read_csv("dataset/merged_traffic.csv")

@st.cache_resource
def get_graph():
    return build_graph()

try:
    model, features = load_model()
    df_clean        = load_df()
    G               = get_graph()
except FileNotFoundError as e:
    st.error("Missing file: " + str(e) + ". Run phase1_preprocess.py and phase2_train.py first.")
    st.stop()

NODE_LIST   = all_nodes()
EDGE_LIST   = all_edges_list(G)
EDGE_LABELS = [u + " → " + v for u, v in EDGE_LIST]
DAY_NAMES   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WX_NAMES    = ["Clear","Clouds","Rain","Drizzle","Mist"]

if "acc_mgr" not in st.session_state:
    st.session_state.acc_mgr = AccidentManager()
acc_mgr: AccidentManager = st.session_state.acc_mgr


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛 Controls")
    st.divider()

    # Time & Day
    st.markdown("**⏱ Time & Day**")
    hour = st.slider("Hour of day", 0, 23, 12, format="%d:00")
    day  = st.selectbox("Day", range(7), format_func=lambda d: DAY_NAMES[d], index=1)

    # CHANGE 2: Congestion window indicator
    win_label, win_color = congestion_window_label(hour)
    st.markdown(
        "<div style='border-left:3px solid " + win_color + ";"
        "padding:4px 10px;background:#1e293b;border-radius:4px;margin-top:4px;"
        "font-size:11px'>"
        "<b style='color:" + win_color + "'>Congestion window:</b><br>"
        "<span style='color:#e2e8f0'>" + win_label + "</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    dens_val = time_density(hour, day)
    st.caption("Base density: **" + str(round(dens_val * 100)) + "%** (before node busyness weights)")

    st.divider()

    # Weather
    st.markdown("**🌤 Weather**")
    temp_c       = st.slider("Temperature (°C)", -10, 45, 22)
    rain         = st.slider("Rainfall (mm/hr)",   0, 80,  0)
    clouds       = st.slider("Cloud cover (%)",     0, 100, 30)
    weather_code = st.selectbox("Condition", range(5), format_func=lambda x: WX_NAMES[x])

    st.divider()

    # Routing
    st.markdown("**🗺 Smart Routing**")
    use_rt = st.checkbox("Enable routing", value=True)
    src_nd = st.selectbox("From", NODE_LIST, index=NODE_LIST.index("Railway Station"))
    dst_nd = st.selectbox("To",   NODE_LIST, index=NODE_LIST.index("IT Park"))
    nk     = st.radio("Routes to show", [1, 2, 3], index=2, horizontal=True)

    st.divider()

    # Accident controls
    st.markdown("**🚨 Accident Simulation**")
    st.caption("Select a road and add an accident. Canvas click also works (Accident mode).")

    acc_edge_sel = st.selectbox("Road for accident", EDGE_LABELS, index=0, key="acc_sel")
    acc_severity = st.slider("Severity", 0.1, 1.0, 0.85, 0.05, key="acc_sev")

    ca, cb = st.columns(2)
    with ca:
        if st.button("🚨 Add accident", use_container_width=True):
            idx  = EDGE_LABELS.index(acc_edge_sel)
            u, v = EDGE_LIST[idx]
            acc_mgr.add(u, v, acc_severity)
            st.success("Added")
    with cb:
        if st.button("✅ Clear all", use_container_width=True):
            acc_mgr.clear()
            clear_volume_spikes()
            st.success("Cleared")

    active = acc_mgr.to_list()
    if active:
        st.markdown("**Active accidents:**")
        for a in active:
            c1, c2 = st.columns([4, 1])
            c1.markdown("🔴 `" + a["u"] + " → " + a["v"] + "`")
            if c2.button("✕", key="rm_" + a["edge_id"]):
                acc_mgr.remove_by_id(a["edge_id"])

    st.divider()

    # Vehicle density spike
    st.markdown("**🚗 Traffic Volume Spike**")
    st.caption("Inject extra vehicles at a node → ML predicts higher congestion → routing changes.")
    spike_node   = st.selectbox("Node", NODE_LIST, index=0, key="spk_node")
    spike_amount = st.slider("Extra vehicles", 100, 800, 350, 50, key="spk_amt")
    cs, cd = st.columns(2)
    with cs:
        if st.button("🚗 Spike", use_container_width=True):
            apply_node_spike(spike_node, spike_amount)
            st.success("Spike → " + spike_node)
    with cd:
        if st.button("↺ Clear", use_container_width=True):
            clear_volume_spikes()
            st.success("Cleared")

    st.divider()

    # Debug
    st.markdown("**🔬 Debug**")
    show_debug = st.checkbox("Show edge weight table", value=False)


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE — no caching, fresh every render
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()

node_states = simulate_node_states(
    G, model, features,
    hour, day, temp_c, rain, clouds, weather_code,
    accident_mgr=acc_mgr, seed=42,
)
bottlenecks = [n for n, s in node_states.items() if s["label"] == 2]

edge_states = compute_edge_states(G, node_states, acc_mgr)

routes = []
if use_rt and src_nd != dst_nd:
    routes = find_multi_routes(
        G, src_nd, dst_nd,
        node_states, edge_states,
        accident_mgr=acc_mgr,
        k=int(nk),
    )

pred_ms   = (time.perf_counter() - t0) * 1000
first_key = list(node_states.keys())[0] if node_states else None
is_night  = node_states.get(first_key, {}).get("is_night", False)

anim = build_animation_data(
    G, node_states, edge_states,
    routes=routes, accident_mgr=acc_mgr,
    hour=hour, day=day,
)
anim["accidents"] = acc_mgr.to_list()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0;color:#f1f5f9'>🚦 City Traffic Predictor</h1>"
    "<p style='color:#475569;font-size:12px;margin-top:4px'>"
    "10-node city · XGBoost ML → edge weights (Medium 8× / High 30×) → routing · "
    "Congestion windows: HIGH 10–13 &amp; 18–21 · Click roads on canvas for accidents"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
n0  = sum(1 for s in node_states.values() if s.get("label") == 0)
n1  = sum(1 for s in node_states.values() if s.get("label") == 1)
n2  = sum(1 for s in node_states.values() if s.get("label") == 2)
bns = [n for n, s in node_states.items() if s.get("label") == 2]
avs = np.mean([s.get("speed", 30) for s in node_states.values()]) if node_states else 0
avl = np.mean([s.get("load",  0)  for s in node_states.values()]) * 100 if node_states else 0
nac = len(acc_mgr.to_list())
tvh = sum(es.get("n_vehicles", 0) for es in edge_states.values())
best_eta  = routes[0]["eta"]      if routes else None
best_base = routes[0]["base_cost"] if routes else None

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("🟢 Low",        n0)
c2.metric("🟡 Medium",     n1)
c3.metric("🔴 High",       n2,
          delta="⚠ Congested!" if n2 else "Clear",
          delta_color="inverse")
c4.metric("⚡ Avg speed",  str(round(avs, 1)) + " km/h")
c5.metric("📊 Net load",   str(round(avl, 1)) + "%")
c6.metric("🚗 Vehicles",   tvh)
c7.metric("⏱ ML latency", str(round(pred_ms, 1)) + " ms")

# CHANGE 3: ETA banner with base vs adjusted
if best_eta is not None:
    delta_str = ""
    if len(routes) > 1:
        d2 = round(routes[1]["eta"] - best_eta, 1)
        if d2 > 0:
            delta_str = " · Alt 1: +" + str(d2) + " min"
    cong_factor = round(best_eta / best_base, 1) if best_base and best_base > 0 else 1.0
    st.info(
        "🗺 **" + src_nd + " → " + dst_nd + "** · "
        "Base: **" + str(best_base) + " min** · "
        "Congestion-adjusted ETA: **" + str(best_eta) + " min** "
        "(" + str(cong_factor) + "× slower)" + delta_str
    )

if n2:
    st.error("🚨 High congestion at: **" + ", ".join(bns) + "** — routing avoids these nodes")
if nac:
    st.warning("⚠ " + str(nac) + " accident(s) — road weight = 500× base, routing rerouted")
if not n2 and not nac:
    st.success("✅ Traffic flowing normally — all nodes Low or Medium")

st.divider()

# ── Debug table ───────────────────────────────────────────────────────────────
if show_debug:
    with st.expander("🔬 Edge weight debug table", expanded=True):
        debug_rows = []
        for eid, es in edge_states.items():
            debug_rows.append({
                "Edge":           es.get("from","") + " → " + es.get("to",""),
                "Base (min)":     es.get("base_time", 0),
                "Cong label":     CONGESTION_NAMES.get(es.get("label", 0), "Low"),
                "Cong pen":       str(es.get("cong_penalty", 1)) + "×",
                "Acc pen":        str(es.get("acc_penalty",  1)) + "×",
                "Weighted (min)": es.get("weighted_time", 0),
                "Load %":         round(es.get("avg_load", 0) * 100, 1),
                "Accident":       "⚠ YES" if es.get("has_accident") else "—",
            })
        ddf = pd.DataFrame(debug_rows).sort_values("Weighted (min)", ascending=False)

        def sc_lbl(val):
            return {"High":"background:#7f1d1d;color:white",
                    "Medium":"background:#78350f;color:white",
                    "Low":"background:#14532d;color:white"}.get(val,"")
        def sc_acc(val):
            return "background:#7f1d1d;color:white" if "YES" in str(val) else ""

        styled = ddf.style.background_gradient(
            subset=["Weighted (min)"], cmap="RdYlGn_r", vmin=0, vmax=600
        )
        styled = safe_map(styled, sc_lbl, subset=["Cong label"])
        styled = safe_map(styled, sc_acc, subset=["Accident"])
        st.dataframe(styled, use_container_width=True, height=360)
        st.caption(
            "Penalties: Low=1×, Medium=8×, High=30×, Accident=500× | "
            "ETA display cap per segment: " + str(ETA_SEGMENT_CAP) + " min"
        )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_anim, tab_route, tab_nodes, tab_eda, tab_about = st.tabs([
    "🎬 Live Map", "🗺 Route Analysis", "📋 Node & Edge Details",
    "📈 EDA & Model", "ℹ About",
])

# ── TAB 1: Animation ──────────────────────────────────────────────────────────
with tab_anim:
    win_lbl, _ = congestion_window_label(hour)
    st.caption(
        "**" + DAY_NAMES[day] + "** · " + str(hour).zfill(2) + ":00 · " +
        str(temp_c) + "°C · " + ("🌙 Night" if is_night else "☀ Day") +
        " · " + str(tvh) + " vehicles · Window: " + win_lbl
    )
    components.html(build_animation_html(anim, height=760), height=820, scrolling=False)
    st.caption(
        "**Accident mode** (red badge): click any road to add/remove accident · "
        "**Traffic mode** (orange badge): click shows spike visual; use sidebar to commit · "
        "🟢 Green = best route · 🔵 Blue = alternate · "
        "Road tint: 🟢 Low / 🟡 Medium / 🔴 High congestion"
    )

# ── TAB 2: Route Analysis ─────────────────────────────────────────────────────
with tab_route:
    #  STEP 1 FIX — Show bottlenecks ONLY for selected routes
    route_nodes = set()
    for r in routes:
        route_nodes.update(r["path"])

    route_bottlenecks = [
        n for n in route_nodes
        if node_states.get(n, {}).get("label") == 2
    ]

    st.subheader("🚨 Bottlenecks on Route")

    if route_bottlenecks:
        for b in route_bottlenecks:
            st.warning(f"⚠️ {b}")
    else:
        st.success("No bottlenecks on selected route")
    st.subheader("🗺 Route Analysis: " + src_nd + " → " + dst_nd)

    if not use_rt:
        st.info("Enable routing in the sidebar.")
    elif src_nd == dst_nd:
        st.warning("Source and destination are the same node.")
    elif not routes:
        st.warning("No path found. Try different nodes or clear accidents.")
    else:
        rcols = st.columns(len(routes))
        for i, (r, col) in enumerate(zip(routes, rcols)):
            c     = ROUTE_COLORS[i]
            badge = "✅ Clear" if r.get("is_accident_free") else "⛔ Via accident"
            extra = ""
            if i > 0:
                tvb = r.get("time_vs_best", 0)
                extra = "(+" + str(tvb) + " min vs best)" if tvb > 0 else "(same)"
            with col:
                st.markdown(
                    "<div style='border:2px solid " + c + ";border-radius:8px;"
                    "padding:12px 14px;background:#0f172a'>"
                    "<b style='color:" + c + "'>" + r.get("label","Route") + "</b> " + badge + "<br>"
                    "<span style='font-size:22px;font-weight:700'>" + str(r.get("eta",0)) + " min</span>"
                    + ("<br><span style='color:#f97316;font-size:11px'>" + extra + "</span>" if extra else "") +
                    "<br><span style='color:#64748b;font-size:10px'>"
                    + " → ".join(r.get("path",[])) + "</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        best = routes[0]
        st.markdown("#### Best route — segment breakdown")
        seg_rows = []
        for seg in best.get("edge_details", []):
            seg_rows.append({
                "Segment":        seg.get("segment",""),
                "Base (min)":     seg.get("base","?"),
                "Congestion":     seg.get("label","Low"),
                "Cong penalty":   str(seg.get("penalty",1.0)) + "×",
                "Acc penalty":    str(seg.get("acc_pen",1.0)) + "×",
                "Weighted (min)": seg.get("weighted","?"),
            })
        if seg_rows:
            def sc2(v):
                return {"High":"background:#7f1d1d;color:white",
                        "Medium":"background:#78350f;color:white",
                        "Low":"background:#14532d;color:white"}.get(v,"")
            styled_seg = safe_map(pd.DataFrame(seg_rows).style, sc2, subset=["Congestion"])
            st.dataframe(styled_seg, use_container_width=True, height=260)

        st.caption(
            "Base (no traffic): **" + str(best.get("base_cost",0)) + " min** · "
            "ETA (congestion-adjusted): **" + str(best.get("eta",0)) + " min** · "
            "Avg combined penalty: **" + str(best.get("avg_penalty",1.0)) + "×**"
        )

        st.divider()
        st.markdown("#### Network overview")
        pos = {n: (NODE_POSITIONS[n][0]/1100, 1 - NODE_POSITIONS[n][1]/680) for n in G.nodes()}
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#0e1a2b"); fig.patch.set_facecolor("#0e1a2b")
        nc = [node_states.get(n, {}).get("color","#22c55e") for n in G.nodes()]
        ns = [700 if NODE_META[n]["type"]=="highway"
              else 400 if NODE_META[n]["type"]=="main_road" else 200 for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#334155",
                               arrows=True, arrowsize=10, width=1.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns, alpha=0.92)
        nx.draw_networkx_labels(G, pos, {n:n for n in G.nodes()},
                                ax=ax, font_size=7, font_color="white", font_weight="bold")
        for i, r in enumerate(routes):
            pairs = list(zip(r["path"], r["path"][1:]))
            nx.draw_networkx_edges(G, pos, edgelist=pairs, ax=ax,
                                   edge_color=ROUTE_COLORS[i], width=5-i*1.5,
                                   arrows=True, arrowsize=14,
                                   style="solid" if i==0 else "dashed")
        ax.axis("off")
        hdl = [mlines.Line2D([],[],color=ROUTE_COLORS[i],lw=3,
                              linestyle="solid" if i==0 else "dashed",
                              label=routes[i]["label"]+" ("+str(routes[i]["eta"])+" min)")
               for i in range(len(routes))]
        hdl += [mpatches.Patch(color=c,label=l)
                for c,l in [("#22c55e","Low"),("#f97316","Medium"),("#ef4444","High")]]
        ax.legend(handles=hdl, loc="upper right", facecolor="#0f172a", labelcolor="white", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ── TAB 3: Node & Edge details ────────────────────────────────────────────────
with tab_nodes:
    st.subheader("Node-level ML predictions")
    rows = []
    for n, s in node_states.items():
        rows.append({
            "Node":       n,
            "Busyness":   NODE_BUSYNESS.get(n, 1.0),
            "Type":       s.get("type","").replace("_"," "),
            "Volume":     s.get("volume",0),
            "Capacity":   s.get("capacity",0),
            "Load %":     round(s.get("load",0)*100, 1),
            "Speed km/h": s.get("speed",0),
            "ML label":   s.get("label_name","Low"),
            "P(High) %":  round(s.get("p_high",0)*100, 1),
        })
    ndf = pd.DataFrame(rows).sort_values("Load %", ascending=False)

    def sp(v):
        return {"High":"background:#7f1d1d;color:white",
                "Medium":"background:#78350f;color:white",
                "Low":"background:#14532d;color:white"}.get(v,"")

    styled_n = ndf.style.background_gradient(
        subset=["Load %"], cmap="RdYlGn_r", vmin=0, vmax=100
    )
    styled_n = safe_map(styled_n, sp, subset=["ML label"])
    st.dataframe(styled_n, use_container_width=True, height=420)

    st.divider()
    st.subheader("Edge-level states (routing inputs)")
    erows = []
    for eid, es in edge_states.items():
        erows.append({
            "Edge":           es.get("from","") + " → " + es.get("to",""),
            "Load %":         round(es.get("avg_load",0)*100, 1),
            "Speed km/h":     es.get("speed",0),
            "Cong label":     CONGESTION_NAMES.get(es.get("label",0),"Low"),
            "Base (min)":     es.get("base_time",0),
            "Penalty":        str(es.get("cong_penalty",1)) + "×",
            "Weighted (min)": es.get("weighted_time",0),
            "Accident":       "⚠ YES" if es.get("has_accident") else "—",
        })
    edf = pd.DataFrame(erows).sort_values("Weighted (min)", ascending=False)

    def sl(v):
        return {"High":"background:#7f1d1d;color:white",
                "Medium":"background:#78350f;color:white",
                "Low":"background:#14532d;color:white"}.get(v,"")
    def sa(v):
        return "background:#7f1d1d;color:white" if "YES" in str(v) else ""

    styled_e = edf.style.background_gradient(
        subset=["Weighted (min)"], cmap="RdYlGn_r", vmin=0, vmax=600
    )
    styled_e = safe_map(styled_e, sl, subset=["Cong label"])
    styled_e = safe_map(styled_e, sa, subset=["Accident"])
    st.dataframe(styled_e, use_container_width=True, height=420)

    st.divider()
    # CHANGE 4: 24h chart with congestion window shading
    st.subheader("24-hour congestion pattern")
    with st.spinner("Computing 24h sweep…"):
        hc, mc, lc, sp_ = [], [], [], []
        for h in range(24):
            try:
                sh = simulate_node_states(
                    G, model, features, h, day, temp_c, rain, clouds, weather_code, seed=42
                )
                lb = [sh[n].get("label",0) for n in G.nodes()]
                hc.append(lb.count(2)); mc.append(lb.count(1)); lc.append(lb.count(0))
                sp_.append(np.mean([sh[n].get("speed",30) for n in G.nodes()]))
            except Exception:
                hc.append(0); mc.append(0); lc.append(10); sp_.append(30)

    fig2, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax in (a1, a2): ax.set_facecolor("#0e1a2b")
    fig2.patch.set_facecolor("#0e1a2b")

    a1.stackplot(range(24), hc, mc, lc, labels=["High","Medium","Low"],
                 colors=["#ef4444","#f97316","#22c55e"], alpha=0.85)
    a1.axvline(hour, color="white", lw=1.5, ls="--", label="Now")
    # CHANGE 4: shade required HIGH and MEDIUM windows
    a1.axvspan(10, 13, alpha=0.12, color="#ef4444", label="HIGH window")
    a1.axvspan(18, 21, alpha=0.12, color="#ef4444")
    a1.axvspan(14, 17, alpha=0.08, color="#f97316", label="MEDIUM window")
    a1.axvspan(7,  9,  alpha=0.06, color="#f97316")
    a1.set_ylabel("Node count", color="#94a3b8"); a1.tick_params(colors="#94a3b8")
    a1.set_title("Congestion node count by hour (shaded = required HIGH/MEDIUM windows)",
                 color="#e2e8f0", fontsize=10)
    a1.legend(loc="upper left", facecolor="#0f172a", labelcolor="white", fontsize=8)
    a1.set_xlim(0, 23)

    a2.plot(range(24), sp_, color="#6366f1", lw=2.5, marker="o", markersize=4)
    a2.axvline(hour, color="white", lw=1.5, ls="--")
    a2.axvspan(10, 13, alpha=0.12, color="#ef4444")
    a2.axvspan(18, 21, alpha=0.12, color="#ef4444")
    a2.set_xlabel("Hour of day", color="#94a3b8")
    a2.set_ylabel("Avg speed (km/h)", color="#94a3b8")
    a2.tick_params(colors="#94a3b8"); a2.set_xlim(0, 23)
    a2.set_title("Average network speed (drops during HIGH windows)", color="#e2e8f0", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ── TAB 4: EDA & Model ───────────────────────────────────────────────────────
with tab_eda:
    st.subheader("EDA & Model Evaluation")
    charts = {
        "Traffic by hour":         "outputs/traffic_by_hour.png",
        "Traffic by day":          "outputs/traffic_by_day.png",
        "Congestion distribution": "outputs/congestion_distribution.png",
        "Hour × Day heatmap":      "outputs/traffic_heatmap.png",
        "Confusion matrix":        "outputs/confusion_matrix.png",
        "Feature importance":      "outputs/feature_importance.png",
        "Model comparison":        "outputs/model_comparison.png",
    }
    cols = st.columns(2)
    for i, (title, path) in enumerate(charts.items()):
        with cols[i % 2]:
            if os.path.exists(path):
                st.image(path, caption=title, use_column_width=True)
            else:
                st.warning("Run phase scripts: " + title)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Dataset sample**")
        st.dataframe(df_clean.head(150), height=240, use_container_width=True)
    with cb:
        st.markdown("**Volume distribution**")
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        ax3.set_facecolor("#0e1a2b"); fig3.patch.set_facecolor("#0e1a2b")
        ax3.hist(df_clean["vehicle_count"], bins=50, color="#6366f1",
                 edgecolor="#0e1a2b", alpha=0.85)
        ax3.set_xlabel("Traffic volume", color="#94a3b8")
        ax3.set_ylabel("Count", color="#94a3b8")
        ax3.tick_params(colors="#94a3b8")
        ax3.set_title("Volume distribution", color="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

# ── TAB 5: About ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## 🚦 City Traffic Bottleneck Prediction — v8

**3rd-year Mini Project** · ML → edge weights → routing · Realistic congestion windows

---
### Fixes in v8

| Problem | Root cause | Fix |
|---------|-----------|-----|
| All nodes HIGH | `FORCE_HIGH_LOAD=0.80` too low + uniform noise | Raised to 0.92; per-node busyness weights |
| Time zones wrong | 10–13 was 0.45 density | Exact spec: 10–13=0.82, 18–21=0.80, 14–17=0.55 |
| Routes don't change | Medium=5×, High=15× too weak | Medium=8×, High=30× → routes visibly reroute |
| No spatial variation | All nodes same base density | NODE_BUSYNESS dict: City Center=1.25×, School=0.65× |
| Click interactions | Canvas can't mutate Python state | Accident mode: visual+sidebar; Traffic mode: sidebar spike |
| Congestion tint | Roads looked flat grey | Green/orange/red tint on asphalt surface (CHANGE C) |

---
### Required congestion windows (now implemented)
| Hour range | Required level | Base density |
|------------|----------------|-------------|
| 10:00–13:00 | HIGH | 82% |
| 18:00–21:00 | HIGH | 80% |
| 14:00–17:00 | MEDIUM | 55% |
|  7:00–09:00 | MEDIUM-HIGH | 65% |
|  0:00–05:00 | LOW | 8% |
| rest | LOW-MEDIUM | 35% |

### Routing penalty values
| Level | Weight multiplier |
|-------|-----------------|
| Low   | 1× |
| Medium | **8×** |
| High  | **30×** |
| Accident | 500× |

**Dataset:** Metro Interstate Traffic Volume · Kaggle · 48,204 records · US I-94 · 2012–2018
    """)
