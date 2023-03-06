from typing import Dict, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import simulator.prediction as prediction
from simulator.dataset_generation import get_ns3_sim_result

app = FastAPI()

# solve the problem of HTTP requests getting blocked because of cross-origin
# policies in web browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_nn(api_params: Dict[str, float]):
    throughput, delay = prediction.predict(torch.tensor(list(api_params.values())))
    return throughput, delay, ""


class Ns3Parameter(BaseModel):
    name: str
    unit: Optional[str]
    value: float


class InputParams(BaseModel):
    params: list[Ns3Parameter]


@app.post("/predict/{method}")
def predict(api_params: InputParams, method: str):
    param_dict = {param.name: param.value for param in api_params.params}

    throughput, delay, _ = run_nn(param_dict) if method == "nn" else get_ns3_sim_result(param_dict)
    return {"input": api_params.params, "output": [Ns3Parameter(name="throughput", value=throughput, unit="MBit/s"),
                                                   Ns3Parameter(name="delay", value=delay, unit="ms")]}
