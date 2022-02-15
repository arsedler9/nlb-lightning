from ray import tune
from train_sae import train


# Wrap `train` in a function that unpacks `config`
def tune_train(config):
    return train(**config)


# Define the hyperparameters to pass into `train`
config = dict(
    run_tag="baseline",
    dataset_name=tune.grid_search(
        [
            "mc_maze",
            "mc_rtt",
            "area2_bump",
            "dmfc_rsg",
            "mc_maze_small",
            "mc_maze_medium",
            "mc_maze_large",
        ]
    ),
    bin_width=tune.grid_search([20, 5]),
    phase=tune.grid_search(["val", "test"]),
    verbose=False,
    log_every_n_epochs=20,
)
# Start all training runs in parallel
tune.run(
    tune_train,
    config=config,
    resources_per_trial={
        "cpu": 2,
        "gpu": 0.3,
    },
    num_samples=1,
    log_to_file=False,
)
