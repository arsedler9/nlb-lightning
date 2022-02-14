import os

import dotenv
import pytorch_lightning as pl
import torch

from nlb_lightning.callbacks import (
    EvaluationCallback,
    RasterPlotCallback,
    TrajectoryPlotCallback,
)
from nlb_lightning.datamodules import NLBDataModule
from nlb_lightning.models import SequentialAutoencoder

# Load the NLB environment variables
dotenv.load_dotenv(override=True)
RUNS_HOME = os.environ["RUNS_HOME"]

RUN_TAG = "test_run"

pl.seed_everything(0)
# Create the datamodule
datamodule = NLBDataModule(
    dataset_name="area2_bump",
    phase="val",
    bin_width=20,
    batch_size=256,
    num_workers=4,
)
# Infer the data dimensionality
datamodule.setup()
n_heldin = datamodule.train_data[0].shape[2]
n_heldout = datamodule.train_data[2].shape[2]
# Build the model
model = SequentialAutoencoder(
    input_size=n_heldin,
    hidden_size=100,
    output_size=n_heldin + n_heldout,
    learning_rate=1e-3,
    weight_decay=1e-4,
    dropout=0.05,
)
# Create the callbacks
callbacks = [
    RasterPlotCallback(log_every_n_epochs=5),
    TrajectoryPlotCallback(log_every_n_epochs=5),
    EvaluationCallback(log_every_n_epochs=5),
]
# Create the trainer
trainer = pl.Trainer(
    default_root_dir=os.path.join(RUNS_HOME, RUN_TAG),
    max_epochs=12_200,
    callbacks=callbacks,
    gpus=int(torch.cuda.is_available()),
    log_every_n_steps=1,
)
# Train the model
trainer.fit(model=model, datamodule=datamodule)
