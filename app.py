"""
Supply Chain Visualizer & Control Dashboard.

This Streamlit application serves as the primary front-end for the Supply Chain 
Optimization environment. It provides real-time topology configuration, interactive 
3D WebGL visualizations, disruption toggles, and dual-mode agent evaluation
(Autonomous LLM vs. Pathfinding Heuristics).
"""

import streamlit as st

# Handle Meta OpenEnv Hackathon standard Status Check natively
if st.query_params.get("status") == "true":
    st.json({"status": "ready", "message": "Supply Chain API is live"})
    st.stop()
import streamlit.components.v1 as components
import json
import time
import os
import re
from openai import OpenAI
from supply_chain_env import SupplyChainEnv

def call_llm_agent(env, step, start_hub, dest_hub):
    """
    Interfaces with the LiteLLM proxy to deduce the optimal next routing action.
    
    Dynamically constructs a localized text representation of the agent's immediate 
    surroundings and validates the returned decision trajectory.
    
    Args:
        env (SupplyChainEnv): The active simulation environment.
        step (int): The current progression tick.
        start_hub (str): Designated origin identifier.
        dest_hub (str): Designated target identifier.
        
    Returns:
        int: The securely parsed index of the selected route or action.
    """
    api_key = os.environ.get("API_KEY", "EMPTY_KEY")
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    curr_node = env.idx_to_node[env.current_idx]
    N = env.num_nodes
    dest_idx = env.node_to_idx.get(dest_hub, N - 1)
    
    valid_neighbors = []
    connections = env.adjacency_list.get(curr_node, {})
    for neighbor in connections.keys():
        n_idx = env.node_to_idx[neighbor]
        if neighbor in env.deleted_nodes:
            continue
        status_str = "Clear" if env.node_states.get(neighbor, 0) == 0 else "Disrupted"
        valid_neighbors.append(f"Node {n_idx} ({status_str})")

    system_prompt = (
        "You are an Autonomous Supply Chain Agent. Output ONLY the integer ID of your next action.\n"
        f"Goal: Reach Node {dest_idx}."
    )
    user_prompt = f"Step: {step}. Current Node: {env.current_idx}. Neighbors: {', '.join(valid_neighbors)}."
    
    try:
        if not client.api_key or client.api_key == "EMPTY_KEY":
            return env.current_idx

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        return env.current_idx
    except Exception as exc:
        print(f"[DEBUG] Model inference failed: {exc}", flush=True)
        return env.current_idx


# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Global Sketch Theme CSS (Forced Light Mode Overrides)
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Architects+Daughter&family=Comic+Neue:wght@400;700&family=Caveat:wght@400;500;600;700&display=swap');

    /* ── Global Notebook Background & Force Dark Text ── */
    .stApp {
        background-color: #fdf6e3 !important;
        background-image:
            linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px) !important;
        background-size: 24px 24px !important;
    }

    /* Override dark-mode default text making things invisible */
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp li, 
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #3d3929 !important;
    }

    /* ── Typography (Protect icons but style text) ── */
    p, li, label, .stSelectbox, .stTextInput, .stSlider {
        font-family: 'Comic Neue', cursive !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Caveat', cursive !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #fef9ef !important;
        border-right: 3px dashed #c8b99a !important;
    }

    /* ── Expander Wrappers ── */
    [data-testid="stExpander"] {
        border: 2px solid #3d3929 !important;
        border-radius: 6px !important;
        background-color: #fffcf5 !important;
        margin-bottom: 15px;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] details {
        background-color: #fffcf5 !important;
    }

    /* ── Input Boxes (Fix unreadable dark backgrounds) ── */
    .stTextInput > div > div > input, 
    [data-baseweb="select"] > div,
    [data-baseweb="base-input"],
    .stNumberInput > div > div > input {
        background-color: #fffcf5 !important;
        border: 2px solid #3d3929 !important;
        color: #3d3929 !important;
        -webkit-text-fill-color: #3d3929 !important;
    }

    /* ── Styled Toggles (Black sketch border & visibility fixes) ── */
    div[data-testid="stToggle"] {
        border: 2.5px solid #3d3929 !important;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        background-color: #fffcf5 !important;
        box-shadow: 2px 2px 0px #c8b99a !important;
    }

    /* ── Buttons ── */
    div.stButton > button[kind="primary"] {
        font-family: 'Caveat', cursive !important;
        font-size: 1.4rem !important; font-weight: 700 !important;
        background-color: #fbbf24 !important; 
        color: #3d3929 !important;
        border: 2.5px solid #3d3929 !important; border-radius: 6px !important;
        box-shadow: 3px 3px 0px #3d3929 !important;
        padding: 10px 0 !important;
        transition: transform 0.1s, box-shadow 0.1s;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translate(-1px, -1px);
        box-shadow: 5px 5px 0px #3d3929 !important;
    }
    div.stButton > button:not([kind="primary"]) {
        font-family: 'Comic Neue', cursive !important;
        border: 2px solid #3d3929 !important;
        background-color: #fffcf5 !important;
        color: #3d3929 !important;
    }

    /* ── Hero ── */
    .hero-title {
        font-family: 'Caveat', cursive;
        font-size: 3.4rem; font-weight: 700; color: #3d3929;
        margin-bottom: 0; line-height: 1.1;
    }
    .hero-sub {
        font-family: 'Architects Daughter', cursive;
        font-size: 1.1rem; color: #8a7e6b !important; margin-top: 2px;
    }
    .hero-line {
        border: none; border-top: 2.5px dashed #c8b99a;
        margin: 18px 0 24px;
    }

    /* ── Sketch Cards ── */
    .sketch-row { display: flex; gap: 18px; flex-wrap: wrap; margin: 0 0 18px; }
    .sketch-card {
        flex: 1; min-width: 140px;
        padding: 18px 20px; border-radius: 6px;
        background-color: #fffcf5 !important;
        border: 2.5px solid #3d3929;
        box-shadow: 4px 4px 0px #c8b99a;
        position: relative;
    }
    .card-label {
        font-family: 'Architects Daughter', cursive;
        font-size: 0.78rem; color: #8a7e6b !important;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .card-value {
        font-family: 'Caveat', cursive;
        font-size: 2.4rem; font-weight: 700; color: #3d3929;
        line-height: 1.1; margin-top: 2px;
    }
    .card-value.safe { color: #16a34a !important; }
    .card-value.warn { color: #d97706 !important; }
    .card-value.danger { color: #dc2626 !important; }
    .card-value.info { color: #2563eb !important; }

    /* ── Section Headers ── */
    .section-title {
        font-family: 'Caveat', cursive;
        font-size: 1.8rem; font-weight: 700; color: #3d3929;
        border-bottom: 2.5px solid #3d3929;
        padding-bottom: 6px; margin: 12px 0 16px;
        position: relative;
    }
    .section-title::after {
        content: ''; position: absolute; bottom: -4px; left: 0;
        width: 40px; height: 3px; background-color: #fbbf24 !important;
    }

    /* ── Log Entries ── */
    .log-entry {
        font-family: 'Architects Daughter', cursive;
        font-size: 0.92rem; line-height: 1.6;
        padding: 10px 14px; margin-bottom: 8px;
        border-radius: 4px; border: 2px solid #3d3929;
        background-color: #fffcf5; color: #3d3929;
    }
    .log-move   { border-left: 5px solid #16a34a; }
    .log-crisis { border-left: 5px solid #dc2626; background-color: #fef2f2 !important; }
    .log-delete { border-left: 5px solid #d97706; background-color: #fffbeb !important; }
    .log-block  { border-left: 5px solid #94a3b8; background-color: #f8fafc !important; }
    .log-goal   { border-left: 5px solid #2563eb; background-color: #eff6ff !important; font-weight: 700; }
    .log-timeout { border-left: 5px solid #d97706; background-color: #fffbeb !important; }

    /* ── Progress Bar & Chart ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #fbbf24, #f59e0b) !important;
        border-radius: 4px;
    }
    .stLineChart { border: 2px solid #3d3929; border-radius: 6px; padding: 4px; background-color: #fffcf5 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Session State Defaults
# ──────────────────────────────────────────────────────────────
if "nodes" not in st.session_state:
    st.session_state.nodes = ["Mumbai", "Surat", "Ahmedabad", "Jaipur", "Delhi", "Pune", "Hyderabad"]
if "edges" not in st.session_state:
    st.session_state.edges = [
        ("Mumbai", "Surat", 10), ("Mumbai", "Pune", 5),
        ("Surat", "Ahmedabad", 12), ("Surat", "Jaipur", 20),
        ("Ahmedabad", "Jaipur", 15), ("Ahmedabad", "Delhi", 25),
        ("Jaipur", "Delhi", 10), ("Pune", "Hyderabad", 15),
        ("Hyderabad", "Delhi", 30), ("Mumbai", "Ahmedabad", 18)
    ]
if "disruptions" not in st.session_state:
    st.session_state.disruptions = {n: 0 for n in st.session_state.nodes}
if "deleted_nodes" not in st.session_state:
    st.session_state.deleted_nodes = []
if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0
if "run_history" not in st.session_state:
    st.session_state.run_history = []

# ──────────────────────────────────────────────────────────────
# Build Environment
# ──────────────────────────────────────────────────────────────
env_adj = {node: {} for node in st.session_state.nodes}
for f, t, c in st.session_state.edges:
    if f in env_adj:
        env_adj[f][t] = c
env = SupplyChainEnv(adjacency_list=env_adj, disruption_states=st.session_state.disruptions)
env.deleted_nodes = list(st.session_state.deleted_nodes)

# ──────────────────────────────────────────────────────────────
# Hero Header
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">✏️ Supply Chain Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Meta OpenEnv · Phase 2 Compliant · Sketch Edition 3D</p>', unsafe_allow_html=True)
st.markdown('<hr class="hero-line"/>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Dashboard Stat Cards
# ──────────────────────────────────────────────────────────────
active_nodes = [n for n in st.session_state.nodes if n not in st.session_state.deleted_nodes]
active_edges = [e for e in st.session_state.edges if e[0] not in st.session_state.deleted_nodes and e[1] not in st.session_state.deleted_nodes]
crisis_count = sum(1 for n in active_nodes if st.session_state.disruptions.get(n, 0) == 1)
score_val = st.session_state.last_score
score_class = "safe" if score_val > 0.3 else ("warn" if score_val > 0 else "danger")

st.markdown(f"""
<div class="sketch-row">
    <div class="sketch-card">
        <div class="card-label">Active Hubs</div>
        <div class="card-value info">{len(active_nodes)}</div>
    </div>
    <div class="sketch-card">
        <div class="card-label">Live Routes</div>
        <div class="card-value info">{len(active_edges)}</div>
    </div>
    <div class="sketch-card">
        <div class="card-label">Crises ⚡</div>
        <div class="card-value {'danger' if crisis_count > 0 else 'safe'}">{crisis_count}</div>
    </div>
    <div class="sketch-card">
        <div class="card-label">Last Score</div>
        <div class="card-value {score_class}">{score_val:.3f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar — Mission Control
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title" style="font-size:1.4rem; padding-top:0; border:none; margin-bottom:5px;">📍 Route Config</div>', unsafe_allow_html=True)
    
    start_hub = st.selectbox("Origin Hub", st.session_state.nodes, index=0)
    end_idx = len(st.session_state.nodes) - 1 if st.session_state.nodes else 0
    dest_hub = st.selectbox("Destination Hub", st.session_state.nodes, index=end_idx)
    if start_hub == dest_hub:
        st.error("Origin and Destination must differ!")

    st.markdown('<hr class="hero-line"/>', unsafe_allow_html=True)
    intel_mode = st.radio("Intelligence Mode", options=["Autonomous (Llama-3)", "Heuristic (Dijkstra)"])

    st.markdown('<hr class="hero-line"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.4rem; padding-top:0; border:none; margin-bottom:5px;">🏗️ Network Ops</div>', unsafe_allow_html=True)

    with st.expander("Add Transit Hub"):
        new_node = st.text_input("Hub Name", placeholder="e.g. Bangalore")
        if st.button("Add Hub", use_container_width=True) and new_node:
            if new_node not in st.session_state.nodes:
                st.session_state.nodes.append(new_node)
                st.session_state.disruptions[new_node] = 0
                st.rerun()

    with st.expander("Establish Route"):
        rc1, rc2 = st.columns(2)
        with rc1:
            from_n = st.selectbox("From", st.session_state.nodes, key="f_node")
        with rc2:
            to_n = st.selectbox("To", st.session_state.nodes, key="t_node")
        cost_val = st.slider("Travel Cost", 1, 50, 10)
        if st.button("Create Route", use_container_width=True) and from_n != to_n:
            if from_n != to_n:
                st.session_state.edges.append((from_n, to_n, cost_val))
                st.rerun()

    with st.expander("Emergency Hub Deletion"):
        deletable = [n for n in st.session_state.nodes
                     if n != st.session_state.nodes[0]
                     and n != st.session_state.nodes[-1]
                     and n not in st.session_state.deleted_nodes] if len(st.session_state.nodes) > 2 else []
        del_target = st.selectbox("Target Hub", deletable) if deletable else None
        if st.button("Execute Deletion", type="primary", use_container_width=True) and del_target:
            node_idx = st.session_state.nodes.index(del_target)
            N = len(st.session_state.nodes)
            env.step(N + node_idx)
            st.session_state.deleted_nodes = list(env.deleted_nodes)
            st.rerun()

    st.markdown('<hr class="hero-line"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.4rem; padding-top:0; border:none; margin-bottom:5px;">🔥 Disruptions</div>', unsafe_allow_html=True)
    for node in st.session_state.nodes:
        if node not in st.session_state.deleted_nodes:
            is_on = bool(st.session_state.disruptions.get(node, 0))
            toggled = st.toggle(f"**Crisis at {node}**", value=is_on, key=f"disrupt_{node}")
            st.session_state.disruptions[node] = 1 if toggled else 0

    st.markdown('<hr class="hero-line"/>', unsafe_allow_html=True)
    run_simulation = st.button("✏️ DEPLOY AGENT", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────
# 3D Force Graph Builder (Error-Proof & Pinned Versions)
# ──────────────────────────────────────────────────────────────
def build_3d_graph_html(nodes_list, edges_list, disruptions, deleted, start, dest):
    active = [n for n in nodes_list if n not in deleted]
    if not active:
        return "<p style='font-family:Architects Daughter; color:#3d3929;'>No active nodes.</p>"

    graph_nodes = []
    for n in active:
        if n == start: 
            color = "#4ade80" # Sketch green
            val = 25
        elif n == dest: 
            color = "#60a5fa" # Sketch blue
            val = 25
        elif disruptions.get(n, 0) == 1: 
            color = "#f87171" # Sketch red
            val = 20
        else: 
            color = "#fbbf24" # Sketch yellow/gold
            val = 14
            
        graph_nodes.append({
            "id": n,
            "name": n,
            "color": color,
            "val": val
        })

    graph_edges = []
    for f, t, c in edges_list:
        if f in active and t in active:
            graph_edges.append({
                "source": f,
                "target": t,
                "cost": c,
                "color": "#3d3929" # Ink color for links
            })

    graph_data = json.dumps({
        "nodes": graph_nodes,
        "links": graph_edges
    })

    return f"""
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@700&display=swap" rel="stylesheet">
        <style> 
            body {{ margin: 0; padding: 0; background-color: transparent; overflow: hidden; }} 
            #error-msg {{ color: #dc2626; padding: 20px; font-family: sans-serif; font-weight: bold; background: #fee2e2; border: 2px solid #dc2626; }}
        </style>
        <!-- Stable pinned versions of ThreeJS and 3d-force-graph -->
        <script src="https://unpkg.com/three@0.147.0/build/three.min.js"></script>
        <script src="https://unpkg.com/three-spritetext@1.6.5/dist/three-spritetext.min.js"></script>
        <script src="https://unpkg.com/3d-force-graph@1.72.3/dist/3d-force-graph.min.js"></script>
    </head>
    <body style="background: transparent;">
        <div id="3d-graph"></div>
        <script>
            try {{
                const gData = {graph_data};
                const elem = document.getElementById('3d-graph');
                
                const Graph = ForceGraph3D()(elem)
                    .graphData(gData)
                    .nodeColor(node => node.color)
                    .nodeVal(node => node.val)
                    .nodeResolution(32) // Smooth spheres
                    
                    // Always-visible text labels
                    .nodeThreeObjectExtend(true)
                    .nodeThreeObject(node => {{
                        if (typeof SpriteText !== 'undefined') {{
                            const sprite = new SpriteText(node.name);
                            sprite.color = '#3d3929'; 
                            sprite.fontFace = "'Caveat', cursive";
                            sprite.fontWeight = 'bold';
                            sprite.textHeight = 16;
                            sprite.center.y = -0.7; // Float slightly above node
                            return sprite;
                        }}
                        return null;
                    }})
                    
                    .linkDirectionalParticles(link => 3)
                    .linkDirectionalParticleWidth(3.5)
                    .linkDirectionalParticleColor(() => '#3d3929') // Dark ink particles
                    .linkDirectionalParticleSpeed(0.005)
                    .linkDirectionalArrowLength(5)
                    .linkDirectionalArrowColor(() => '#3d3929')
                    .linkDirectionalArrowRelPos(1)
                    .linkColor(link => link.color)
                    .backgroundColor('rgba(0,0,0,0)'); // Transparent background to show notebook grid!
                    
                // Spread the network out significantly
                Graph.d3Force('charge').strength(-400);
                Graph.d3Force('link').distance(120);
                    
                // Slow continuous rotation (zoomed out to fit)
                let angle = 0;
                setInterval(() => {{
                    Graph.cameraPosition({{
                        x: 600 * Math.sin(angle),
                        z: 600 * Math.cos(angle)
                    }});
                    angle += 0.0010;
                }}, 20);
            }} catch (err) {{
                document.getElementById('3d-graph').innerHTML = "<div id='error-msg'>JS Error loading the 3D Graph:<br/>" + err.message + "</div>";
                console.error(err);
            }}
        </script>
    </body>
    </html>
    """

# ──────────────────────────────────────────────────────────────
# Main Content — Graph + Logs
# ──────────────────────────────────────────────────────────────
col_graph, col_logs = st.columns([3, 2])

with col_graph:
    st.markdown('<div class="section-title">🗺️ Interactive 3D Topology</div>', unsafe_allow_html=True)
    graph_html = build_3d_graph_html(
        st.session_state.nodes, st.session_state.edges,
        st.session_state.disruptions, st.session_state.deleted_nodes,
        start_hub, dest_hub,
    )
    
    # Render inside a sketch container
    st.markdown("""
        <div style="background: #fdf6e3; border: 2.5px solid #3d3929; border-radius: 8px; box-shadow: 4px 4px 0px #c8b99a; padding: 0px; overflow: hidden; margin-bottom: 10px;">
    """, unsafe_allow_html=True)
    components.html(graph_html, height=530, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-family:Architects Daughter; color:#8a7e6b; text-align:center;'>✨ Drag to Rotate | Scroll to Zoom</p>", unsafe_allow_html=True)

with col_logs:
    st.markdown('<div class="section-title">📋 Agent Log</div>', unsafe_allow_html=True)

    if run_simulation:
        if start_hub == dest_hub:
            st.error("Cannot deploy — Origin and Destination must differ.")
        else:
            obs, info = env.reset(options={"start_node": start_hub, "destination_node": dest_hub})
            total_reward = 0.0
            terminated = False
            step_logs = []
            progress = st.progress(0, text="Sketching the route...")

            for step in range(1, 16):
                time.sleep(0.3)
                progress.progress(step / 15, text=f"Step {step}/15")

                curr_node = env.idx_to_node[env.current_idx]
                
                # Autonomous Routing Heuristic (Dijkstra)
                import heapq
                def get_smart_action():
                    adj = env.adjacency_list
                    dest = env.idx_to_node[env.destination_idx]
                    
                    pq = [(0, curr_node, [])]
                    visited = set()
                    while pq:
                        cost, node, path = heapq.heappop(pq)
                        if node == dest: return path
                        if node in visited: continue
                        visited.add(node)
                        for neighbor, weight in adj.get(node, {}).items():
                            if neighbor in env.deleted_nodes: continue
                            # Strongly penalize disrupted nodes to force routing around them
                            penalty = 1000 if env.node_states.get(neighbor, 0) == 1 else 0
                            heapq.heappush(pq, (cost + weight + penalty, neighbor, path + [neighbor]))
                    return None

                if intel_mode == "Autonomous (Llama-3)":
                    action = call_llm_agent(env, step, start_hub, dest_hub)
                else:
                    best_path = get_smart_action()
                    if best_path and len(best_path) > 0:
                        action = env.node_to_idx[best_path[0]]
                    else:
                        action = env.action_space.sample() # Fallback

                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                new_node = step_info['current_node']
                N = env.num_nodes

                if action >= N:
                    act_node = env.idx_to_node.get(action - N, "?")
                    cls = "log-delete"
                    txt = f"<b>Step {step}</b> — 🗑️ Deleted <b>{act_node}</b> (r={reward:.3f})"
                elif new_node == curr_node and action != env.current_idx:
                    cls = "log-block"
                    txt = f"<b>Step {step}</b> — 🚫 Blocked to {env.idx_to_node.get(action, '?')} (r={reward:.3f})"
                else:
                    is_crisis = st.session_state.disruptions.get(new_node, 0) == 1
                    cls = "log-crisis" if is_crisis else "log-move"
                    icon = "⚡" if is_crisis else "→"
                    txt = f"<b>Step {step}</b> — {curr_node} {icon} <b>{new_node}</b> (r={reward:.3f})"

                step_logs.append((cls, txt))

                if terminated:
                    step_logs.append(("log-goal", f"<b>DELIVERED</b> in {step} steps! Total: {total_reward:.3f}"))
                    break

            progress.progress(1.0, text="Done!" if terminated else "Timed out.")

            if not terminated:
                step_logs.append(("log-timeout", f"<b>TIMEOUT</b> — 15 steps used. Total: {total_reward:.3f}"))

            for cls, txt in step_logs:
                st.markdown(f'<div class="log-entry {cls}">{txt}</div>', unsafe_allow_html=True)

            raw_score = total_reward / 10.0 if total_reward > 1 else total_reward # Basic Normalization
            clamped = max(0.001, min(0.999, raw_score))
            st.session_state.last_score = clamped
            st.session_state.run_history.append({"score": clamped, "terminated": terminated})
            st.rerun()

    elif st.session_state.run_history:
        latest = st.session_state.run_history[-1]
        status = "Delivered" if latest["terminated"] else "Timed Out"
        st.markdown(f'<div class="log-entry log-goal"><b>Last Run</b> — Score: {latest["score"]:.3f} — {status}</div>', unsafe_allow_html=True)

        if len(st.session_state.run_history) > 1:
            st.markdown('<div class="section-title" style="margin-top:18px;">📈 Score History</div>', unsafe_allow_html=True)
            scores = [r["score"] for r in st.session_state.run_history]
            st.line_chart(scores, use_container_width=True, color="#d97706")
    else:
        st.markdown(
            '<div class="log-entry log-block">Configure the network, then press <b>✏️ DEPLOY AGENT</b>.</div>',
            unsafe_allow_html=True,
        )