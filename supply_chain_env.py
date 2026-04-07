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

    def __init__(self, adjacency_list=None, node_states=None, disruption_states=None):
        super().__init__()

        self.adjacency_list = adjacency_list or BASELINE_GRAPH
        self.nodes = list(self.adjacency_list.keys())
        self.num_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        self.start_idx = 0
        self.destination_idx = self.num_nodes - 1

        self.deleted_nodes = []

        if node_states is None and disruption_states is not None:
            self.node_states = disruption_states
        else:
            self.node_states = node_states or {n: 0 for n in self.nodes}
            
        self.action_space = spaces.Discrete(2 * self.num_nodes)
        
        nvec = [self.num_nodes] + [3] * self.num_nodes
        self.observation_space = spaces.MultiDiscrete(nvec)

        self.current_idx = self.start_idx

    def _bfs_distance(self, from_idx: int, to_idx: int) -> int:
        if from_idx == to_idx:
            return 0
        visited = {from_idx}
        queue = deque([(from_idx, 0)])
        while queue:
            node_idx, dist = queue.popleft()
            node = self.idx_to_node[node_idx]
            for neighbor in self.adjacency_list.get(node, {}):
                neighbor_idx = self.node_to_idx[neighbor]
                if neighbor_idx == to_idx:
                    return dist + 1
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append((neighbor_idx, dist + 1))
        return self.num_nodes

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx

        # If not explicitly preserved by the ui state runner
        if not hasattr(self, 'preserve_deletions') or not self.preserve_deletions:
            self.deleted_nodes = []

        if options:
            if 'node_states' in options:
                self.node_states = options['node_states']
            elif 'disruption_states' in options:
                self.node_states = options['disruption_states']
            if 'start_node' in options:
                self.current_idx = self.node_to_idx.get(options['start_node'], 0)

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        obs = [self.current_idx]
        for node in self.nodes:
            status = 2 if node in self.deleted_nodes else self.node_states.get(node, 0)
            obs.append(status)
        return np.array(obs, dtype=np.int64)

    def _get_info(self):
        return {
            "current_node": self.idx_to_node[self.current_idx],
            "destination_node": self.idx_to_node[self.destination_idx],
            "node_states": self.node_states.copy(),
            "deleted_nodes": list(self.deleted_nodes)
        }

    def step(self, action):
        action = int(action)
        reward = -1.0
        terminated = False
        truncated = False
        
        N = self.num_nodes
        
        if action >= N:
            delete_idx = action - N
            if delete_idx == 0 or delete_idx == self.destination_idx:
                reward -= 50
            else:
                delete_node = self.idx_to_node[delete_idx]
                if delete_node not in self.deleted_nodes:
                    self.deleted_nodes.append(delete_node)
                    self.node_states[delete_node] = 2
                    reward -= 5
                else:
                    reward -= 10
        else:
            target_idx = action
            current_node = self.idx_to_node[self.current_idx]
            target_node = self.idx_to_node.get(target_idx, "Unknown")
            
            connections = self.adjacency_list.get(current_node, {})

            if target_node not in connections:
                reward -= 50
            else:
                t_status = 2 if target_node in self.deleted_nodes else self.node_states.get(target_node, 0)
                if t_status == 2:
                    reward -= 200
                elif t_status == 1:
                    reward -= 100
                    
                dist_before = self._bfs_distance(self.current_idx, self.destination_idx)
                
                travel_cost = connections[target_node]
                reward -= travel_cost
                self.current_idx = target_idx

                dist_after = self._bfs_distance(self.current_idx, self.destination_idx)
                if dist_after < dist_before:
                    reward += 15
                elif dist_after > dist_before:
                    reward -= 5

                if target_idx == self.destination_idx:
                    reward += 500
                    terminated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()