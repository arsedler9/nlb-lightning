import os

import dotenv
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from nlb_lightning.callbacks import (
    EvaluationCallback,
    RasterPlotCallback,
    TrajectoryPlotCallback,
)
from nlb_lightning.datamodules import NLBDataModule
from nlb_lightning.models import SequentialAutoencoder
from nlb_lightning.submission import make_submission

# Load the NLB environment variables
dotenv.load_dotenv(override=True)
RUNS_HOME = os.environ["RUNS_HOME"]


def train(
    run_tag="test_run",
    dataset_name="mc_maze_large",
    phase="val",
    bin_width=20,
    batch_size=256,
    hidden_size=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    dropout=0.05,
    seed=0,
    verbose=True,
    log_every_n_epochs=5,
):
    """An example training function for the sequential
    autoencoder that works across all datasets, phases,
    and bin widths. At the end of training, writes
    predictions to HDF5 file for submission. Can be run
    individually or in parallel across datasets.

    Parameters
    ----------
    run_tag : str, optional
        The tag that identifies a given experiment,
        by default "test_run"
    dataset_name : str, optional
        One of the data tags specified by the NLB organizers,
        by default "mc_maze_large"
    phase : str, optional
        The phase of the competition - either "val" or "test",
        by default "val"
    bin_width : int, optional
        The width of data bins, by default 20
    batch_size : int, optional
        The number of samples to process in each batch,
        by default 256
    hidden_size : int, optional
        The hidden dimensionality of the network, which
        determines the dimensionality of both the encoders
        and decoders, by default 100
    learning_rate : _type_, optional
        The learning rate to use for optimization,
        by default 1e-3
    weight_decay : _type_, optional
         The weight decay to regularize optimization,
         by default 1e-4
    dropout : float, optional
        The ratio of neurons to drop in dropout layers,
        by default 0.05
    seed : int, optional
        The random seed for model initialization and training,
        by default 0
    verbose : bool, optional
        Whether to allow verbose trainer output, by default True
    log_every_n_epochs : int, optional
        Frequency of logging by callbacks, by default 5
    """
    pl.seed_everything(seed)
    # Create the datamodule
    datamodule = NLBDataModule(
        dataset_name=dataset_name,
        phase=phase,
        bin_width=bin_width,
        batch_size=batch_size,
        num_workers=4,
    )
    # Infer the data dimensionality
    datamodule.setup()
    n_heldin = datamodule.train_data[0].shape[2]
    n_heldout = datamodule.train_data[2].shape[2]
    # Build the model
    model = SequentialAutoencoder(
        input_size=n_heldin,
        hidden_size=hidden_size,
        output_size=n_heldin + n_heldout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
    )
    # Create the callbacks
    callbacks = [ModelCheckpoint(monitor="valid/loss", save_last=True, mode="min")]
    if phase == "val":
        callbacks.extend(
            [
                RasterPlotCallback(log_every_n_epochs=log_every_n_epochs),
                TrajectoryPlotCallback(log_every_n_epochs=log_every_n_epochs),
                EvaluationCallback(log_every_n_epochs=log_every_n_epochs),
            ]
        )
    # Create the trainer
    runtag_dir = os.path.join(RUNS_HOME, run_tag)
    data_tag = os.path.basename(datamodule.save_path)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(runtag_dir, data_tag),
        max_epochs=2_000,
        callbacks=callbacks,
        gpus=int(torch.cuda.is_available()),
        log_every_n_steps=1,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
    )
    # Train the model
    trainer.fit(model=model, datamodule=datamodule)
    # Save model outputs for submission
    save_path = os.path.join(runtag_dir, f"submission-{phase}.h5")
    make_submission(model, trainer, save_path)


if __name__ == "__main__":
    train()
