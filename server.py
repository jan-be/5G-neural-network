from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dataset_generation import get_ns3_sim_result

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
    return "hmm", "hmm", "hmm"


class MyParams(BaseModel):
    params: Dict[str, str]

@app.post("/predict/{method}")
def predict(myParams: MyParams, method: str):
    throughput, delay, _ = run_nn(myParams.params) if method == "nn" else get_ns3_sim_result(myParams.params)
    return {"input": myParams.params, "output": {"throughput": throughput, "delay": delay}}
