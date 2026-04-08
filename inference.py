import os
import argparse
from typing import List
from openai import OpenAI
from supply_chain_env import SupplyChainEnv

API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY_KEY") 
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
TASK_NAME = "supply_chain_routing"

MAX_STEPS = 15
TEMPERATURE = 0.1
MAX_TOKENS = 10

def log_start(task_id: str, model: str) -> None:
    print(f"[START] task_id={task_id} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, info: dict, done: bool) -> None:
    done_val = str(done).lower()
    step_rew = info.get("step_reward", 0.0)
    total_cost = info.get("total_path_cost", 0.0)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} step_reward={step_rew:.2f} total_path_cost={total_cost:.2f} done={done_val}", flush=True)

def log_end(success: bool, steps: int, score: float) -> None:
    final_score = max(0.01, min(0.99, score))
    print(f"[END] success={str(success).lower()} steps={steps} score={final_score:.2f}", flush=True)

def get_model_action(client: OpenAI, step: int, obs: list, env: SupplyChainEnv, start_node: int, destination_node: int) -> int:
    current_idx = int(obs[0])
    status_array = obs[1:]
    
    N = len(env.adjacency_list)
    current_node = env.idx_to_node[current_idx]
    destination_idx = env.node_to_idx.get(destination_node, N - 1)
    
    start_name = env.idx_to_node.get(env.node_to_idx.get(start_node, 0), str(start_node))
    destination_name = env.idx_to_node.get(destination_idx, str(destination_node))
    
    connections = env.adjacency_list.get(current_node, {})
    deleted_nodes = env.deleted_nodes
    
    valid_neighbors = []
    for neighbor in connections.keys():
        n_idx = env.node_to_idx[neighbor]
        if neighbor in deleted_nodes:
            continue
            
        status = status_array[n_idx]
        status_str = "Clear" if status == 0 else "Disrupted"
        valid_neighbors.append({"Node ID": n_idx, "Status": status_str})
    
    system_prompt = (
        "You are an Autonomous Supply Chain Agent. You are evaluated on Safety and Efficiency. Every move consumes resources. Every Crisis node encountered is a critical failure. You have the authority to Delete high-risk nodes if the cost of the detour exceeds the cost of deletion (75 points). Act as a rational economic actor.\n"
        f"MISSION BRIEF: Your current starting hub is {start_name}. You must navigate the network to reach the destination hub: {destination_name}.\n"
        f"You are at Node {current_idx}. Target is {destination_idx}. "
        f"To Move, pick 0 to {N-1}. To Delete a dangerous node, pick {N} to {2*N-1}.\n"
        "Do not output any other text or reasoning."
    )
    
    user_prompt = f"Step: {step}\nAvailable Active Neighbors:\n"
    for n in valid_neighbors:
        user_prompt += f"- Node ID: {n['Node ID']} (Status: {n['Status']})\n"
    user_prompt += "\nOutput your chosen next action integer."
    
    try:
        if client.api_key == "EMPTY_KEY":
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
        return int(text)
    except Exception as exc:
        print(f"[DEBUG] Model inference failed: {exc}", flush=True)
        return current_idx

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_node", type=int, default=0, help="Start node index/name")
    parser.add_argument("--destination_node", type=int, default=4, help="Destination node index/name")
    args = parser.parse_args()

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
        obs, info = env.reset(options={"start_node": args.start_node, "destination_node": args.destination_node})
        
        for step in range(1, MAX_STEPS + 1):
            action = get_model_action(client, step, obs.tolist(), env, args.start_node, args.destination_node)
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(float(reward))
            steps_taken = step
            log_step(step=step, action=str(action), reward=reward, info=info, done=terminated)

            if terminated or truncated:
                if terminated:
                    success = True
                break

        total_raw_reward = sum(rewards)
        score = total_raw_reward / 100.0 

    finally:
        log_end(success=success, steps=steps_taken, score=score)

if __name__ == "__main__":
    main()