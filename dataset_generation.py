import multiprocessing
import random
import shlex
import subprocess as sp
from abc import ABC
from dataclasses import dataclass
import re


class Distribution(ABC):
    def get_random_value(self) -> float:
        pass


@dataclass
class LinearDistribution(Distribution):
    min: float
    max: float
    isFloat: bool

    def get_random_value(self) -> float:
        return random.uniform(self.min, self.max) if self.isFloat else random.randrange(int(self.min),
                                                                                        int(self.max) + 1)


@dataclass
class LogDistribution:
    mu: float
    sigma: float

    def get_random_value(self) -> float:
        return random.lognormvariate(self.mu, self.sigma)


@dataclass
class Ns3Parameter:
    name: str
    distribution: any


ns3_parameters = [
    Ns3Parameter("numRowsGnb", LinearDistribution(1, 8, False)),
    Ns3Parameter("numColumnsGnb", LinearDistribution(1, 8, False)),
    Ns3Parameter("numRowsUe", LinearDistribution(1, 8, False)),
    Ns3Parameter("numColumnsUe", LinearDistribution(1, 8, False)),
    Ns3Parameter("gnbTxPower", LinearDistribution(1, 50, True)),
    Ns3Parameter("ueTxPower", LinearDistribution(1, 40, True)),
    Ns3Parameter("centralFrequency", LinearDistribution(0.7e9, 3.5e9, True)),
    Ns3Parameter("bandwidth", LinearDistribution(1e6, 1e8, True)),
    Ns3Parameter("gnbUeDistance", LogDistribution(5, 2))
]


def generate_random_parameters():
    param_dict = {}

    for params in ns3_parameters:
        param_dict[params.name] = params.distribution.get_random_value()

    return param_dict


def stringify_params(params: dict[str, float]):
    ns3_input_str = ""

    for key, value in params.items():
        ns3_input_str += f" --{key}={value}"

    return ns3_input_str


def get_ns3_sim_result(params):
    process = sp.Popen(shlex.split(f'ns3 run "cttc-nr-mimo-demo --useFixedRi {stringify_params(params)}"'),
                       stdout=sp.PIPE,
                       stderr=sp.PIPE, shell=False)

    raw_ns3_out = ""
    for line in process.stdout.readlines():
        raw_ns3_out += line.decode("utf-8")

    throughput = re.findall(r'Mean flow throughput: ([-+]?(?:\d*\.*\d+))', raw_ns3_out)
    delay = re.findall(r'Mean flow delay: ([-+]?(?:\d*\.*\d+))', raw_ns3_out)

    if len(throughput) != 1 or len(delay) != 1:
        raise ValueError("something strange is happening")

    return throughput[0], delay[0], raw_ns3_out


def run_sim_to_csv(process_name):
    print('hello', process_name)

    params = generate_random_parameters()

    throughput, delay, raw_ns3_out = get_ns3_sim_result(params)

    input_csv_params_str = ", ".join(map(str, list(params.values())))

    print(f"{input_csv_params_str}, {throughput}, {delay}")

    f = open("out.csv", "a")
    f.write(f"{input_csv_params_str}, {throughput}, {delay}\r\n")
    f.close()

    print(raw_ns3_out)


# https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes

if __name__ == '__main__':
    # use all available cores, otherwise specify the number you want as an argument
    pool = multiprocessing.Pool()
    for i in range(10_000_000):
        pool.apply_async(run_sim_to_csv, args=(i,))
    pool.close()
    pool.join()
