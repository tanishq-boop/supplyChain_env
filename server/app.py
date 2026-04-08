import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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
    options: Optional[Dict[str, Any]] = None

class ActionRequest(BaseModel):
    action: int

class ObservationResponse(BaseModel):
    observation: List[int]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    observation: List[int]
    info: Dict[str, Any]

@app.get("/")
def home():
    return {"status": "ready", "message": "Supply Chain API is live"}

@app.post("/reset", response_model=ObservationResponse)
def reset_env(request: ResetRequest):
    obs, info = env.reset(options=request.options)
    return ObservationResponse(
        observation=obs.tolist(),
        reward=0.0,
        terminated=False,
        truncated=False,
        info=info
    )

@app.post("/step", response_model=ObservationResponse)
def step_env(request: ActionRequest):
    obs, reward, terminated, truncated, info = env.step(request.action)
    return ObservationResponse(
        observation=obs.tolist(),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info
    )

@app.get("/state", response_model=StateResponse)
def state_env():
    state_data = env.state()
    return StateResponse(
        observation=state_data["observation"],
        info=state_data["info"]
    )

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