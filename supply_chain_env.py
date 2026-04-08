"""
Supply Chain Optimization Environment.

This module defines a custom Gymnasium environment simulating a logistics network.
Agents must navigate from an origin to a destination hub while managing disruptions
and optimizing travel costs. The action space supports both routing (movement) 
and strategic hub deletion to bypass critical crises.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# Default topology for the supply chain network if none is dynamically provided.
BASELINE_GRAPH = {
    0: {1: 8, 2: 20},
    1: {0: 8, 2: 12},
    2: {0: 20, 1: 12, 3: 14, 4: 30},
    3: {2: 14, 4: 10},
    4: {2: 30, 3: 10}
}

class SupplyChainEnv(gym.Env):
    """
    A reinforcement learning environment for dynamic supply chain routing.
    
    The agent receives a bounded fractional reward for forward progress and 
    suffers penalties for invalid moves, backtracking, or encountering disruptions.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, adjacency_list=None, node_states=None, disruption_states=None, start_node=None, destination_node=None):
        """
        Initializes the dynamic graph environment.

        Args:
            adjacency_list (dict): Connectivity matrix with edge costs.
            node_states (dict): Initial states of nodes (0 for clear, 1 for disrupted).
            disruption_states (dict): Legacy alias for node_states.
            start_node (str/int): Origin hub identifier.
            destination_node (str/int): Target destination identifier.
        """
        super().__init__()
        self.adjacency_list = adjacency_list or BASELINE_GRAPH
        self.nodes = list(self.adjacency_list.keys())
        self.num_nodes = len(self.nodes)
        
        # Mappings for index-based Gymnasium compliance
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        self.start_idx = self.node_to_idx.get(start_node, 0) if start_node is not None else 0
        self.destination_idx = self.node_to_idx.get(destination_node, self.num_nodes - 1) if destination_node is not None else self.num_nodes - 1
        
        # Unify disruption initialization regardless of caller argument style
        if node_states is None and disruption_states is not None:
            self.initial_node_states = disruption_states
        else:
            self.initial_node_states = node_states or {n: 0 for n in self.nodes}
            
        # Action space definition
        # [0, N-1]  -> Move to specific node index
        # [N, 2N-1] -> Delete specific node index (mitigating crises)
        self.action_space = spaces.Discrete(2 * self.num_nodes)
        
        # Observation breakdown: [CurrentPosition, Node0_Status, ..., NodeN_Status]
        # MultiDiscrete handles distinct categorical bounding for state arrays.
        nvec = [self.num_nodes] + [3] * self.num_nodes 
        self.observation_space = spaces.MultiDiscrete(nvec)

    def _bfs_distance(self, from_idx: int, to_idx: int) -> int:
        """
        Calculates the unweighted shortest path (node hops) to the destination.
        Used internally to calculate reward shaping based on forward progress.
        """
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
        # Return maximum possible hops if disconnected
        return self.num_nodes

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to an initial episode state.
        
        Supports configuration injection via the `options` dictionary for 
        testing environments on different scenarios without re-instantiation.
        """
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
            # Dynamic randomization fallback for continuous automated evaluation routines
            self.node_states = self.initial_node_states.copy()
            middle_nodes = [n for n in self.nodes if n != self.start_idx and n != self.destination_idx]
            if middle_nodes:
                self.node_states[self.np_random.choice(middle_nodes)] = 1

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """Builds the current MultiDiscrete compliance observation vector."""
        obs = [self.current_idx]
        for node in self.nodes:
            # 0=Clear, 1=Disrupted, 2=Deleted completely
            status = 2 if node in self.deleted_nodes else self.node_states.get(node, 0)
            obs.append(status)
        return np.array(obs, dtype=np.int64)

    def _get_info(self):
        """Constructs telemetry payloads for external agents or logging interfaces."""
        return {
            "current_node": self.idx_to_node[self.current_idx],
            "node_states": self.node_states.copy(),
            "deleted_nodes": list(self.deleted_nodes),
            "step_reward": self.last_step_reward,
            "total_path_cost": self.total_path_cost
        }

    def state(self):
        """Exposes the full uncompressed state geometry for UI hooks."""
        return {
            "observation": self._get_obs().tolist(),
            "info": self._get_info()
        }

    def step(self, action):
        """
        Executes a single environment transition tick.

        Args:
            action (int): The index of the intended maneuver.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        action = int(action)
        reward = 0.0
        terminated = truncated = False
        N = self.num_nodes
        
        # Capture pre-action distance for delta progression calculations
        d_before = self._bfs_distance(self.current_idx, self.destination_idx)
        
        # --- DELETION LOGIC ---
        if action >= N:
            delete_idx = action - N
            if delete_idx in [self.start_idx, self.destination_idx]:
                reward -= 0.05 # Prevent deletion of origin or destination
            else:
                node = self.idx_to_node[delete_idx]
                if node not in self.deleted_nodes:
                    self.deleted_nodes.append(node)
                    reward -= 0.075 # Cost associated with strategically disconnecting a hub
                else:
                    reward -= 0.01 # Wasted cycle penalty
        
        # --- MOVEMENT LOGIC ---
        else:
            target_idx = action
            curr_node, target_node = self.idx_to_node[self.current_idx], self.idx_to_node.get(target_idx)
            edges = self.adjacency_list.get(curr_node, {})

            if target_node not in edges:
                reward -= 0.05 # Invalid routing attempt
            else:
                # Assess the status of the destination node
                t_status = 2 if target_node in self.deleted_nodes else self.node_states.get(target_node, 0)
                if t_status == 2: 
                    reward -= 0.20 # Traversal block penalty
                elif t_status == 1: 
                    reward -= 0.25 # Disruption exposure penalty
                
                # Apply base transportation costs mapping to efficiency penalties
                cost = edges[target_node]
                self.total_path_cost += cost
                efficiency_penalty = (cost + 5.0) / 1000.0
                reward -= efficiency_penalty
                
                # Update current location
                self.current_idx = target_idx

                # Compute forward progress heuristics
                d_after = self._bfs_distance(self.current_idx, self.destination_idx)
                if d_after < d_before:
                    reward += 0.10 # Encouragement for correctly moving towards target
                elif d_after > d_before:
                    reward -= 0.10 # Discouragement for reversing or deviating
                
                if target_idx == self.destination_idx:
                    reward += 0.50 # Successful mission completion
                    terminated = True

        # Base offset to strictly guarantee >0 valid rewards for compliance bounds
        reward += 0.01
        self.last_step_reward = float(np.clip(reward, 0.01, 0.99))
        
        return self._get_obs(), self.last_step_reward, terminated, truncated, self._get_info()