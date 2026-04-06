import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from supply_chain_env import SupplyChainEnv

app = FastAPI(title="Supply Chain Optimizer API")

baseline_graph = {
    0: {1: 10},
    1: {0: 10, 2: 12, 3: 20},
    2: {1: 12, 3: 15, 4: 25},
    3: {1: 20, 2: 15, 4: 10},
    4: {2: 25, 3: 10}
}
baseline_disruptions = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}

env = SupplyChainEnv(adjacency_list=baseline_graph, disruption_states=baseline_disruptions)

class StepPayload(BaseModel):
    action: int

@app.post("/reset")
def reset_env():
    obs, info = env.reset()
    return {"observation": obs.tolist()}

@app.post("/step")
def step_env(payload: StepPayload):
    obs, reward, terminated, truncated, info = env.step(payload.action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated)
    }

@app.get("/health")
def health():
    return {"status": "ready"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()