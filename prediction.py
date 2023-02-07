import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn import preprocessing
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
import os
from hyperopt.pyll import scope
import joblib
from ray.air.checkpoint import Checkpoint

def do_training(config):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    device

    #based on https://towardsdatascience.com/build-a-simple-neural-network-using-pytorch-38c55158028d
    n_input = 9
    n_out = 2
    num_workers = 0
    n_hidden = 30
    n_hidden2 = 11
    batch_size = 10
    learning_rate = .005
    momentum = .95
    dropout = config["dropout"]
    # n_hidden = 15
    # n_hidden2 = 10
    # batch_size = 10
    # learning_rate = 0.001
    # momentum = .99
    # num_workers = 0
    # dropout = 0.2

    full_pd = pd.read_csv(f"{os.environ['TUNE_ORIG_WORKING_DIR']}/Dataset_237k.csv")
    full_tensor = torch.tensor(full_pd.to_numpy()).float()
    full_pd.shape

    scaler = preprocessing.MinMaxScaler()

    full_scaled = torch.tensor(scaler.fit_transform(full_tensor)).float()
    full_scaled[:10, :]

    train_n = {"data": full_scaled[:80000, :9].to(device), "target": full_scaled[:80000, 9:].to(device)}
    test_n = {"data": full_scaled[:-20000, :9].to(device), "target": full_scaled[:-20000, 9:].to(device)}

    test_n["data"].requires_grad=False
    test_n["target"].requires_grad=False

    train, validate, test = torch.utils.data.random_split(full_scaled, [200000, 27485, 10000])
    #train, validate, test = torch.utils.data.random_split(full_scaled, [80000, 14765, 10000])
    train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # model = nn.Sequential(nn.Linear(9, 300),
    #                       nn.ReLU(),
    #                       nn.Linear(300, 2),
    #                       nn.Sigmoid())
    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.Dropout(p=dropout),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_hidden2),
                          nn.ReLU(),
                          nn.Linear(n_hidden2, n_out),
                          nn.Sigmoid())
    model.to(device)
    print(model)

    loss_function = nn.HuberLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # torch.multiprocessing.set_start_method('spawn')
    import time

    tim = time.time()

    losses = []
    test_losses = []
    for i in range(200):
        for j, item in enumerate(train_loader):
            train_x, train_y = item[:, :9], item[:, 9:]

            train_x = train_x.to(device)
            train_y = train_y.to(device)

            pred_y = model(train_x)
            loss = loss_function(pred_y, train_y)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()

            optimizer.step()

        permutation = torch.randperm(test_n["data"].size(0))
        samples = test_n["data"][permutation[:1000]]

        test_pred_y = model(samples)
        test_loss = loss_function(test_pred_y, test_n["target"][permutation[:1000]]).item()
        test_losses.append(test_loss)

        os.makedirs("my_model", exist_ok=True)
        torch.save((model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        joblib.dump(scaler, "my_model/scaler.txt")
        checkpoint = Checkpoint.from_directory("my_model")

        tune.report(mean_accuracy=test_loss, checkpoint=checkpoint)

    print(time.time() - tim)

    print(test_losses)

space = {
    "n_hidden": scope.int(hp.quniform("n_hidden", 5, 100, q=1)),
    "n_hidden2": scope.int(hp.quniform("n_hidden2", 1, 15, q=1)),
    "batch_size": scope.int(hp.quniform("batch_size", 8, 128, q=1)),
    "learning_rate": hp.loguniform("learning_rate", -10, -1),
    "momentum": hp.uniform("momentum", 0.1, 1.0),
    "dropout": hp.uniform("dropout", 0, 1),
}

hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")

search_space = {
    "dropout": tune.grid_search([0,.01,.1,.2,.3,.5,.6,.7,8,.9,.99]),
}

tuner = tune.Tuner(do_training, param_space=search_space)


# tuner = tune.Tuner(
#     do_training,
#     tune_config=tune.TuneConfig(
#         num_samples=14,
#         search_alg=hyperopt_search,
#     ),
# )
results = tuner.fit()
