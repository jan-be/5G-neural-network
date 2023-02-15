from typing import Dict

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


class InputParams(BaseModel):
    params: Dict[str, float]


@app.post("/predict/{method}")
def predict(api_params: InputParams, method: str):
    throughput, delay, _ = run_nn(api_params.params) if method == "nn" else get_ns3_sim_result(api_params.params)
    return {"input": api_params.params, "output": {"throughput [MBit/s]": throughput, "delay [ms]": delay}}
