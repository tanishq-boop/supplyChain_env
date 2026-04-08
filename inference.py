"""
Supply Chain AI Agent Inference Logic.

This module executes automated agent evaluations using OpenAI's wrapper for the LLM API. 
It initializes tasks dynamically, formats observation state strings into prompts, 
parses natural language intent into integer actions, and logs structured telemetry 
for validation graders.
"""

import os
import re
from typing import List
from openai import OpenAI
from supply_chain_env import SupplyChainEnv

# Validated endpoint configurations for LiteLLM proxy injections
API_KEY = os.environ.get("API_KEY") 
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"
ENV_NAME = "supply_chain"

MAX_STEPS = 15
TEMPERATURE = 0.0
MAX_TOKENS = 10

TASKS = [
    ("task_1_easy", {"start_node": 0, "destination_node": 4, "node_states": {}}),
    ("task_2_medium", {"start_node": 0, "destination_node": 4, "node_states": {2: 1}}),
    ("task_3_hard", {"start_node": 0, "destination_node": 4, "node_states": {1: 1, 2: 1}}),
]

def get_model_action(client: OpenAI, step: int, obs: list, env: SupplyChainEnv, start_node: int, destination_node: int) -> int:
    """
    Prompts the LLM agent to select an optimal move based on local graph observations.

    Args:
        client: The initialized OpenAI client instance.
        step: Current environment step counter.
        obs: Observation vector describing positions and node statuses.
        env: SupplyChainEnv instance providing topology context.
        start_node: Origin hub configuration.
        destination_node: Target hub configuration.

    Returns:
        int: The selected graph action or deletion maneuver. 
    """
    current_idx = int(obs[0])
    status_array = obs[1:]
    
    N = len(env.adjacency_list)
    current_node = env.idx_to_node[current_idx]
    
    valid_neighbors = []
    connections = env.adjacency_list.get(current_node, {})
    for neighbor in connections.keys():
        n_idx = env.node_to_idx[neighbor]
        if neighbor in env.deleted_nodes:
            continue
        status_str = "Clear" if status_array[n_idx] == 0 else "Disrupted"
        valid_neighbors.append(f"Node {n_idx} ({status_str})")

    system_prompt = (
        "You are an Autonomous Supply Chain Agent. Output ONLY the integer ID of your next action.\n"
        f"Goal: Reach Node {env.node_to_idx.get(destination_node, N-1)}."
    )
    user_prompt = f"Step: {step}. Current Node: {current_idx}. Neighbors: {', '.join(valid_neighbors)}."
    
    try:
        if not client.api_key or client.api_key == "EMPTY_KEY":
            return current_idx

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Robustly extract the integer choice irrespective of conversational artifacts
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        return current_idx
    except Exception as exc:
        print(f"[DEBUG] Model inference failed: {exc}", flush=True)
        return current_idx

def main() -> None:
    """Entrypoint for automated sequential task evaluation workflow."""
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    baseline_graph = {
        0: {1: 10, 2: 15},
        1: {0: 10, 2: 12, 3: 20},
        2: {0: 15, 1: 12, 3: 15, 4: 25},
        3: {1: 20, 2: 15, 4: 10},
        4: {2: 25, 3: 10}
    }
    
    for task_id, options in TASKS:
        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.001 # Initialize with non-zero floor
        
        node_states = options.get("node_states", {})
        env = SupplyChainEnv(adjacency_list=baseline_graph, node_states=node_states)

        # RULE: One [START] per task in the loop
        print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

        try:
            obs, info = env.reset(options=options)
            
            for step in range(1, MAX_STEPS + 1):
                action = get_model_action(client, step, obs.tolist(), env, options["start_node"], options["destination_node"])
                obs, reward, terminated, truncated, info = env.step(action)

                rewards.append(float(reward))
                steps_taken = step
                
                done_str = "true" if (terminated or truncated) else "false"
                print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error=null", flush=True)

                if terminated or truncated:
                    success = bool(terminated)
                    break

            total_raw_reward = sum(rewards)
            raw_score = total_raw_reward / 100.0 
            score = max(0.001, min(0.999, raw_score))

        except Exception as e:
            print(f"[DEBUG] {task_id} error: {e}", flush=True)
            score = 0.001
            
        finally:
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            success_str = "true" if success else "false"
            print(f"[END] task={task_id} success={success_str} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    main()