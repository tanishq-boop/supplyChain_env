import os
from typing import List
from openai import OpenAI
from supply_chain_env import SupplyChainEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
TASK_NAME = "supply_chain_routing"

MAX_STEPS = 15
TEMPERATURE = 0.1
MAX_TOKENS = 10

def log_start(task_id: str, model: str) -> None:
    print(f"[START] task_id={task_id} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool) -> None:
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val}", flush=True)

def log_end(success: bool, steps: int, score: float) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f}", flush=True)

def get_model_action(client: OpenAI, step: int, obs: list, env: SupplyChainEnv) -> int:
    current_idx = int(obs[0])
    status_array = obs[1:]
    
    N = len(env.adjacency_list)
    current_node = env.idx_to_node[current_idx]
    destination_idx = N - 1
    
    connections = env.adjacency_list.get(current_node, {})
    deleted_nodes = env.deleted_nodes
    
    valid_neighbors = []
    for neighbor in connections.keys():
        n_idx = env.node_to_idx[neighbor]
        
        # Discovery Constraint: Only append neighbors NOT in deleted_nodes
        if neighbor in deleted_nodes:
            continue
            
        status = status_array[n_idx]
        status_str = "Clear"
        if status == 1:
            status_str = "Disrupted"
            
        valid_neighbors.append({"Node ID": n_idx, "Status": status_str})
    
    system_prompt = (
        "You are an autonomous supply chain routing AI.\n"
        f"You are at Node {current_idx}. Target is {destination_idx}. "
        f"To Move, pick 0 to {N-1}. To Delete a dangerous node, pick {N} to {2*N-1}.\n"
        "Do not output any other text or reasoning."
    )
    
    user_prompt = f"Step: {step}\nAvailable Active Neighbors:\n"
    for n in valid_neighbors:
        user_prompt += f"- Node ID: {n['Node ID']} (Status: {n['Status']})\n"
        
    user_prompt += "\nOutput your chosen next action integer."
    
    try:
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
        action = int(text)
        return action
    except Exception as exc:
        print(f"[DEBUG] Model inference failed: {exc}", flush=True)
        return current_idx

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    baseline_graph = {
        0: {1: 10, 2: 15},
        1: {0: 10, 2: 12, 3: 20},
        2: {0: 15, 1: 12, 3: 15, 4: 25},
        3: {1: 20, 2: 15, 4: 10},
        4: {2: 25, 3: 10}
    }
    baseline_node_states = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    
    env = SupplyChainEnv(adjacency_list=baseline_graph, node_states=baseline_node_states)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_id=TASK_NAME, model=MODEL_NAME)

    try:
        obs, info = env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            action = get_model_action(client, step, obs.tolist(), env)
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(float(reward))
            steps_taken = step

            log_step(step=step, action=str(action), reward=reward, done=terminated)

            if terminated or truncated:
                if terminated:
                    success = True
                break

        score = sum(rewards)

    finally:
        log_end(success=success, steps=steps_taken, score=score)

if __name__ == "__main__":
    main()