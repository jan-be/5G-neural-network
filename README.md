# Artificial Neural Networks for Prediction in 5G Networks

## How to run within Container
```shell
docker compose build
docker compose up
```

Open http://localhost:8080

## Run the Dataset Generation
- same as running within container
- `docker exec -it 2022ws_artificial-neural-networks-5g-nn-backend-1 bash` to get into the container shell
- `poetry run python dataset_generation.py`
- grab the generated csv file from the directory

## Run the NN Model Training
- same as running within container
- `docker exec -it 2022ws_artificial-neural-networks-5g-nn-backend-1 bash` to get into the container shell
- `poetry run python training.py`
- ray saves the results to `~/ray_results`, where you can grab the `my_model` directory of the model which performed best.
- you can also run `poetry run tensorboard --logdir ~/ray_results ` to spawn another webserver to look at the visualization of the results at http://localhost:6006