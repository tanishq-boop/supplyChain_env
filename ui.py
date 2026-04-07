import streamlit as st
import numpy as np
from supply_chain_env import SupplyChainEnv
from streamlit_agraph import agraph, Node, Edge, Config

if "nodes" not in st.session_state:
    st.session_state.nodes = ["Mumbai", "Surat", "Ahmedabad", "Jaipur", "Delhi"]

if "edges" not in st.session_state:
    st.session_state.edges = [
        ("Mumbai", "Surat", 10),
        ("Surat", "Ahmedabad", 15),
        ("Ahmedabad", "Jaipur", 20),
        ("Jaipur", "Delhi", 25)
    ]

if "disruptions" not in st.session_state:
    st.session_state.disruptions = {node: 0 for node in st.session_state.nodes}

if "deleted_nodes" not in st.session_state:
    st.session_state.deleted_nodes = []

if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0

st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")

env_adj = {node: {} for node in st.session_state.nodes}
for f, t, c in st.session_state.edges:
    env_adj[f][t] = c
    
env = SupplyChainEnv(adjacency_list=env_adj, disruption_states=st.session_state.disruptions)
env.deleted_nodes = list(st.session_state.deleted_nodes)

st.title("Two-Layer Supply Chain Optimization 🚚")
st.markdown("Configure your network, trigger disruptions, delete hubs dynamically, and watch the RL agent navigate.")

score_col, node_count_col, edge_count_col = st.columns(3)
with score_col:
    st.metric("Last Mission Reward", f"{st.session_state.last_score:.2f}")
with node_count_col:
    st.metric("Total Hubs (Active)", len([n for n in st.session_state.nodes if n not in st.session_state.deleted_nodes]))
with edge_count_col:
    active_edges = [e for e in st.session_state.edges if e[0] not in st.session_state.deleted_nodes and e[1] not in st.session_state.deleted_nodes]
    st.metric("Active Routes", len(active_edges))

st.divider()

with st.sidebar:
    st.header("⚙️ Network Settings")
    
    with st.expander("➕ Add New Transit Hub", expanded=False):
        new_node = st.text_input("Hub Name")
        if st.button("Add Hub") and new_node:
            if new_node not in st.session_state.nodes:
                st.session_state.nodes.append(new_node)
                st.session_state.disruptions[new_node] = 0
                st.rerun()

    with st.expander("🔗 Add Route Connection", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            from_n = st.selectbox("From", st.session_state.nodes, key="f_node")
        with c2:
            to_n = st.selectbox("To", st.session_state.nodes, key="t_node")
        cost_val = st.number_input("Travel Cost", min_value=1, value=10)
        if st.button("Add Route") and from_n != to_n:
            st.session_state.edges.append((from_n, to_n, cost_val))
            st.rerun()
            
    with st.expander("🗑️ Remove Transit Hub", expanded=False):
        deletable_nodes = []
        if len(st.session_state.nodes) > 2:
            start_node = st.session_state.nodes[0]
            end_node = st.session_state.nodes[-1]
            deletable_nodes = [n for n in st.session_state.nodes if n != start_node and n != end_node and n not in st.session_state.deleted_nodes]
            
        del_target = st.selectbox("Select Hub to Delete", deletable_nodes) if deletable_nodes else None
        
        if st.button("Delete Hub") and del_target:
            node_idx = st.session_state.nodes.index(del_target)
            N = len(st.session_state.nodes)
            env.step(N + node_idx)
            st.session_state.deleted_nodes = list(env.deleted_nodes)
            st.rerun()

    st.divider()

    st.subheader("🔥 Live Disruptions")
    for node in st.session_state.nodes:
        if node not in st.session_state.deleted_nodes:
            is_on = bool(st.session_state.disruptions.get(node, 0))
            check = st.checkbox(f"Crisis at {node}", value=is_on)
            st.session_state.disruptions[node] = 1 if check else 0

    st.divider()
    
    run_simulation = st.button("🚀 RUN RL AGENT", type="primary", use_container_width=True)

col_graph, col_logs = st.columns([2, 1])

with col_graph:
    st.subheader("Interactive Logistics Map")
    st.caption("Nodes are draggable. Red nodes indicate active disruptions.")
    
    visual_nodes = []
    for node in st.session_state.nodes:
        if node in env.deleted_nodes:
            continue
        is_hit = st.session_state.disruptions.get(node, 0) == 1
        visual_nodes.append(
            Node(id=node, 
                 label=node, 
                 size=25, 
                 color="#FF4B4B" if is_hit else "#00CC96")
        )

    visual_edges = []
    for frm, to, cst in st.session_state.edges:
        if frm in env.deleted_nodes or to in env.deleted_nodes:
            continue
        visual_edges.append(
            Edge(source=frm, target=to, label=f"Cost: {cst}", color="#A0AEC0")
        )

    graph_config = Config(
        width="100%", height=500, directed=True, physics=True, 
        nodeHighlightBehavior=True, highlightColor="#F7A046"
    )

    if visual_nodes:
        agraph(nodes=visual_nodes, edges=visual_edges, config=graph_config)

if run_simulation:
    with col_logs:
        st.subheader("🤖 Simulation Logs")
        
        env.preserve_deletions = True
        obs, info = env.reset()
        env.preserve_deletions = False
        
        total_reward = 0.0
        terminated = False
        
        progress_bar = st.progress(0)
        
        for step in range(1, 16):
            progress_bar.progress(step / 15)
            
            curr_node = env.idx_to_node[env.current_idx]
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            new_node = step_info['current_node']
            
            N = env.num_nodes
            if action >= N:
                act_node = env.idx_to_node.get(action - N, "Unknown")
                st.write(f"🗑️ **Step {step}**: Agent randomly attempted to Delete {act_node}")
            else:
                if new_node == curr_node and action != env.current_idx:
                    st.write(f"🚫 **Step {step}**: Blocked (No path to {env.idx_to_node.get(action, 'Unknown')})")
                else:
                    icon = "⚠️" if st.session_state.disruptions.get(new_node, 0) else "🚛"
                    st.write(f"{icon} **Step {step}**: {curr_node} ➔ {new_node}")

            if terminated:
                st.success(f"🏁 Goal Reached in {step} steps!")
                st.balloons()
                break
        
        if not terminated:
            st.warning("⏳ Timeout: Goal not reached.")
            
        st.session_state.last_score = total_reward
        if st.button("Update Scoreboard"):
            st.rerun()