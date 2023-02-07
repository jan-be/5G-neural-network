from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dataset_generation import get_ns3_sim_result
import prediction
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def run_nn(params):
    hm = prediction.predict(torch.tensor(list(params.values())))
    return hm[0], hm[1], ""


class MyParams(BaseModel):
    params: Dict[str, float]


@app.post("/predict/{method}")
def predict(myParams: MyParams, method: str):
    throughput, delay, _ = run_nn(myParams.params) if method == "nn" else get_ns3_sim_result(myParams.params)
    return {"input": myParams.params, "output": {"throughput": throughput, "delay": delay}}
