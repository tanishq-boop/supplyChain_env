import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from supply_chain_env import SupplyChainEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
TASK_NAME = "supply_chain_routing"
BENCHMARK = "openenv_round1"

MAX_STEPS = 15
TEMPERATURE = 0.1
MAX_TOKENS = 10

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous supply chain routing AI. 
    You must navigate a network of 5 cities:
    0: Mumbai (Origin)
    1: Surat
    2: Ahmedabad
    3: Jaipur
    4: Delhi (Destination)
    
    Rules:
    1. Your goal is to reach Node 4 (Delhi).
    2. You will be provided your 'Current Node' and an array of 'Disruptions' (0=Clear, 1=Disrupted).
    3. Moving into a disrupted node carries a massive -100 penalty. Avoid them at all costs.
    4. You must reply with EXACTLY ONE INTEGER (0, 1, 2, 3, or 4) representing the node you want to travel to next. Do not output any other text or reasoning.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, step: int, obs: list) -> int:
    current_node = obs[0]
    disruptions = obs[1:]
    
    user_prompt = f"Step: {step}\nCurrent Node: {current_node}\nDisruptions array (Index matches City ID): {disruptions}\nOutput your next node as a single integer."
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strictly extract the direct integer returned by the LLM for action alignment
        action = int(text)
        return action
    except Exception as exc:
        # Fallback safety action in case of broken inference or non-integer hallucination
        print(f"[DEBUG] Model request failed or failed to parse integer: {exc}", flush=True)
        return 0

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    baseline_graph = {
        0: {1: 10},
        1: {0: 10, 2: 12, 3: 20},
        2: {1: 12, 3: 15, 4: 25},
        3: {1: 20, 2: 15, 4: 10},
        4: {2: 25, 3: 10}
    }
    baseline_disruptions = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    
    env = SupplyChainEnv(adjacency_list=baseline_graph, disruption_states=baseline_disruptions)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, info = env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            action = get_model_action(client, step, obs.tolist())
            obs, reward, terminated, truncated, info = env.step(action)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(action), reward=reward, done=terminated, error=error)

            if terminated or truncated:
                if terminated:
                    success = True
                break

        score = sum(rewards)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()