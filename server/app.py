import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from supply_chain_env import SupplyChainEnv

app = FastAPI(title="Supply Chain Optimizer API")

# Ensure these match your supply_chain_env.py logic
baseline_graph = {
    0: {1: 10},
    1: {0: 10, 2: 12, 3: 20},
    2: {1: 12, 3: 15, 4: 25},
    3: {1: 20, 2: 15, 4: 10},
    4: {2: 25, 3: 10}
}
# Using 0=Clear, 1=Disrupted, 2=Deleted logic
baseline_states = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}

env = SupplyChainEnv(adjacency_list=baseline_graph, node_states=baseline_states)

class ResetRequest(BaseModel):
    start_node: Optional[int] = None
    destination_node: Optional[int] = None

class StepPayload(BaseModel):
    action: int

@app.get("/")
def home():
    return {"status": "ready", "message": "Supply Chain API is live"}

@app.post("/reset")
def reset_env(request: ResetRequest):
    options = {}
    if request.start_node is not None:
        options["start_node"] = request.start_node
    if request.destination_node is not None:
        options["destination_node"] = request.destination_node
    
    obs, info = env.reset(options=options if options else None)
    return {"observation": obs.tolist(), "status": "reset_success"}

@app.get("/reset")
def reset_env_get():
    obs, info = env.reset()
    return {"observation": obs.tolist(), "status": "reset_success"}

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
    """
    This function is now explicitly callable by the OpenEnv 
    multi-mode deployment validator.
    """
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()