import joblib
import torch

import simulator.training as training

# the parameters used in the model that returned the best result
model_params = {"n_input": 9, "n_hidden": 30, "n_hidden2": 11, "n_out": 2, "dropout": 0}
model = training.get_model(**model_params)

model.load_state_dict(torch.load("./best_model/checkpoint.pt")[0])
scaler = joblib.load("./best_model/scaler.txt")
model.eval()


def to_scaled(params: torch.Tensor) -> torch.Tensor:
    """
    :param params: 9x1 dimension
    :return: 9x1 dimension
    """
    return torch.tensor(scaler.transform(torch.cat((params, torch.tensor([0, 0])), 0).reshape(1, -1))[:, :9]).float()


def from_scaled(params: torch.Tensor) -> torch.Tensor:
    """
    :param params: 2x1 dimension
    :return: 2x1 dimension
    """

    remerged = (torch.cat((torch.ones([1, 9]), params), 1))

    return torch.tensor(scaler.inverse_transform(remerged.detach().numpy())[0, 9:])


def predict(params: torch.Tensor) -> (float, float):
    """
    :param params: 9x1 dimension
    :return: throughput, delay
    """
    result = from_scaled(model(to_scaled(params)))

    return result[0].item(), result[1].item()
