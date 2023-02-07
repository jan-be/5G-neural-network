import joblib
import torch
import torch.nn as nn

n_input, n_hidden, n_hidden2, dropout, n_out = 9, 30, 11, 0, 2


def predict(params: torch.Tensor):
    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.Dropout(p=dropout),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_hidden2),
                          nn.ReLU(),
                          nn.Linear(n_hidden2, n_out),
                          nn.Sigmoid())

    model.load_state_dict(torch.load("best_model/checkpoint.pt")[0])
    scaler = joblib.load("best_model/scaler.txt")

    model.eval()

    scaled_in = torch.tensor(scaler.transform(torch.cat((params, torch.tensor([0, 0])), 0).reshape(1, -1))[:, :9]).float()

    result_01scaled = model(scaled_in)

    remerged = (torch.cat((scaled_in, result_01scaled), 1))

    result_notscaled = torch.tensor(scaler.inverse_transform(remerged.detach().numpy())[:, 9:])

    return result_notscaled[0, 0].item(), result_notscaled[0, 1].item()
