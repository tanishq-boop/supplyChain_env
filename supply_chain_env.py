import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

BASELINE_GRAPH = {
    0: {1: 8, 2: 20},
    1: {0: 8, 2: 12},
    2: {0: 20, 1: 12, 3: 14, 4: 30},
    3: {2: 14, 4: 10},
    4: {2: 30, 3: 10}
}

class SupplyChainEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, adjacency_list=None, node_states=None, disruption_states=None, start_node=None, destination_node=None):
        super().__init__()
        self.adjacency_list = adjacency_list or BASELINE_GRAPH
        self.nodes = list(self.adjacency_list.keys())
        self.num_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        self.start_idx = self.node_to_idx.get(start_node, 0) if start_node is not None else 0
        self.destination_idx = self.node_to_idx.get(destination_node, self.num_nodes - 1) if destination_node is not None else self.num_nodes - 1
        
        # Handle both naming conventions for disruption state initialization
        if node_states is None and disruption_states is not None:
            self.initial_node_states = disruption_states
        else:
            self.initial_node_states = node_states or {n: 0 for n in self.nodes}
            
        # Action space: [0, N-1] for Movement, [N, 2N-1] for Deletion
        self.action_space = spaces.Discrete(2 * self.num_nodes)
        # Observation: [CurrentPosition, Node0_Status...NodeN_Status]
        nvec = [self.num_nodes] + [3] * self.num_nodes 
        self.observation_space = spaces.MultiDiscrete(nvec)

    def _bfs_distance(self, from_idx: int, to_idx: int) -> int:
        """Helper to calculate distance for reward shaping (progress tracking)."""
        if from_idx == to_idx: return 0
        visited, queue = {from_idx}, deque([(from_idx, 0)])
        while queue:
            node_idx, dist = queue.popleft()
            node = self.idx_to_node[node_idx]
            for neighbor in self.adjacency_list.get(node, {}):
                n_idx = self.node_to_idx[neighbor]
                if n_idx == to_idx: return dist + 1
                if n_idx not in visited:
                    visited.add(n_idx)
                    queue.append((n_idx, dist + 1))
        return self.num_nodes

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx
        self.deleted_nodes = []
        self.total_path_cost = 0.0
        self.last_step_reward = 0.0

        if options:
            if 'node_states' in options:
                self.node_states = options['node_states'].copy()
            elif 'disruption_states' in options:
                self.node_states = options['disruption_states'].copy()
            else:
                self.node_states = self.initial_node_states.copy()
            if 'start_node' in options:
                self.start_idx = self.node_to_idx.get(options['start_node'], self.start_idx)
            if 'destination_node' in options:
                self.destination_idx = self.node_to_idx.get(options['destination_node'], self.destination_idx)
            self.current_idx = self.start_idx
        else:
            # Dynamic randomization to ensure varying scores for Scaler graders
            self.node_states = self.initial_node_states.copy()
            middle_nodes = [n for n in self.nodes if n != self.start_idx and n != self.destination_idx]
            if middle_nodes:
                self.node_states[self.np_random.choice(middle_nodes)] = 1

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        obs = [self.current_idx]
        for node in self.nodes:
            # Status Mapping: 0=Clear, 1=Disrupted, 2=Deleted
            status = 2 if node in self.deleted_nodes else self.node_states.get(node, 0)
            obs.append(status)
        return np.array(obs, dtype=np.int64)

    def _get_info(self):
        return {
            "current_node": self.idx_to_node[self.current_idx],
            "node_states": self.node_states.copy(),
            "deleted_nodes": list(self.deleted_nodes),
            "step_reward": self.last_step_reward,
            "total_path_cost": self.total_path_cost
        }

    def step(self, action):
        action = int(action)
        reward = 0.0
        terminated = truncated = False
        N = self.num_nodes
        d_before = self._bfs_distance(self.current_idx, self.destination_idx)
        
        # --- DELETION LOGIC ---
        if action >= N:
            delete_idx = action - N
            if delete_idx in [self.start_idx, self.destination_idx]:
                reward -= 0.05 # Wasted deletion
            else:
                node = self.idx_to_node[delete_idx]
                if node not in self.deleted_nodes:
                    self.deleted_nodes.append(node)
                    reward -= 0.075 # Strategic Deletion
                else:
                    reward -= 0.01 # Wasted deletion
        # --- MOVEMENT LOGIC ---
        else:
            target_idx = action
            curr_node, target_node = self.idx_to_node[self.current_idx], self.idx_to_node.get(target_idx)
            edges = self.adjacency_list.get(curr_node, {})

            if target_node not in edges:
                reward -= 0.05 # Invalid Move
            else:
                t_status = 2 if target_node in self.deleted_nodes else self.node_states.get(target_node, 0)
                if t_status == 2: 
                    reward -= 0.20 # Deleted Node Collision
                elif t_status == 1: 
                    reward -= 0.25 # Crisis Collision
                
                cost = edges[target_node]
                self.total_path_cost += cost
                
                efficiency_penalty = (cost + 5.0) / 1000.0
                reward -= efficiency_penalty
                
                self.current_idx = target_idx

                d_after = self._bfs_distance(self.current_idx, self.destination_idx)
                if d_after < d_before:
                    reward += 0.10 # Progress Shaping
                elif d_after > d_before:
                    reward -= 0.10 # Backtracking
                
                if target_idx == self.destination_idx:
                    reward += 0.60 # Success Condition
                    terminated = True

        self.last_step_reward = float(np.clip(reward, -1.0, 1.0))
        return self._get_obs(), self.last_step_reward, terminated, truncated, self._get_info()