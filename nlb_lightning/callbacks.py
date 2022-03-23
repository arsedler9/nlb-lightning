import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA

from nlb_tools.evaluation import (
    bits_per_spike,
    eval_psth,
    speed_tp_correlation,
    velocity_decoding,
)

plt.switch_backend("Agg")


def get_tensorboard_summary_writer(writers):
    """Gets the TensorBoard SummaryWriter from a logger
    or logger collection to allow writing of images.

    Parameters
    ----------
    writers : obj or list[obj]
        An object or list of objects to search for the
        SummaryWriter.

    Returns
    -------
    torch.utils.tensorboard.writer.SummaryWriter
        The SummaryWriter object.
    """
    writer_list = writers if isinstance(writers, list) else [writers]
    for writer in writer_list:
        if isinstance(writer, torch.utils.tensorboard.writer.SummaryWriter):
            return writer
    else:
        return None


def fig_to_rgb_array(fig):
    """Converts a matplotlib figure into an array
    that can be logged to tensorboard.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be converted.

    Returns
    -------
    np.array
        The figure as an HxWxC array of pixel values.
    """
    # Convert the figure to a numpy array
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        fig_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = fig_data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def batch_fwd(model, batch):
    """Performs the forward pass for a given model and data batch.

    Parameters
    ----------
    model : pl.LightningModule
        The model to pass data through.
    batch : tuple[torch.Tensor]
        A tuple of batched input tensors.

    Returns
    -------
    tuple[torch.Tensor]
        A tuple of batched output tensors.
    """
    input_data, recon_data, *other_input, behavior = batch
    input_data = input_data.to(model.device)
    other_input = [oi.to(model.device) for oi in other_input]
    return model.forward(input_data, *other_input)


class RasterPlotCallback(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred rates and logs to tensorboard. Heldin/heldout
    and observed/forward distinctions are indicated by
    dividing lines.
    """

    def __init__(self, batch_fwd=batch_fwd, n_samples=2, log_every_n_epochs=20):
        """Initializes the callback.

        Parameters
        ----------
        batch_fwd: func, optional
            A function that takes a model and a batch of data and
            performs the forward pass, returning the model output.
            May be useful if your model requires specialized I/O.
        n_samples : int, optional
            The number of samples to plot, by default 2
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 20
        """
        self.batch_fwd = batch_fwd
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for the TensorBoard SummaryWriter
        writer = get_tensorboard_summary_writer(trainer.logger.experiment)
        if writer is None:
            return
        # Get data samples
        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        input_data, recon_data, *_ = batch
        # Compute data sizes
        _, steps_tot, neur_tot = recon_data.shape
        batch_size, steps_obs, neur_in = input_data.shape
        # Compute model output
        rates, *_ = self.batch_fwd(pl_module, batch)
        # Convert data to numpy arrays
        recon_data = recon_data.detach().cpu().numpy()
        rates = rates.detach().cpu().numpy()
        # Create subplots
        fig, axes = plt.subplots(
            self.n_samples, 2, sharex=True, sharey=True, figsize=(10, 10)
        )
        for i, ax_row in enumerate(axes):
            for ax, array in zip(ax_row, [recon_data, rates]):
                ax.imshow(array[i].T)
                ax.vlines(steps_obs, 0, neur_tot, color="coral")
                ax.hlines(neur_in, 0, steps_tot, color="coral")
                ax.set_xlim(0, steps_tot)
                ax.set_ylim(0, neur_tot)
        plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        writer.add_image("raster_plot", im, trainer.global_step, dataformats="HWC")


class TrajectoryPlotCallback(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, batch_fwd=batch_fwd, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        batch_fwd: func, optional
            A function that takes a model and a batch of data and
            performs the forward pass, returning the model output.
            May be useful if your model requires specialized I/O.
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.batch_fwd = batch_fwd
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for the TensorBoard SummaryWriter
        writer = get_tensorboard_summary_writer(trainer.logger.experiment)
        if writer is None:
            return
        # Get the validation dataset
        val_dataloader = trainer.datamodule.val_dataloader()
        input_data, recon_data, *_ = trainer.datamodule.valid_data
        # Pass data through the model
        latents = [self.batch_fwd(pl_module, batch)[1] for batch in val_dataloader]
        latents = torch.cat(latents).detach().cpu().numpy()
        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = latents.shape
        if n_lats > 3:
            latents_flat = latents.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            latents = pca.fit_transform(latents_flat)
            latents = latents.reshape(n_samp, n_step, 3)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        else:
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in latents:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        writer.add_image("trajectory_plot", im, trainer.global_step, dataformats="HWC")


class EvaluationCallback(pl.Callback):
    """Computes and logs all evaluation metrics for the Neural Latents
    Benchmark to tensorboard. These include `co_bps`, `fp_bps`,
    `behavior_r2`, `psth_r2`, and `tp_corr`.
    """

    def __init__(
        self, batch_fwd=batch_fwd, log_every_n_epochs=20, decoding_cv_sweep=False
    ):
        """Initializes the callback.

        Parameters
        ----------
        batch_fwd: func, optional
            A function that takes a model and a batch of data and
            performs the forward pass, returning the model output.
            May be useful if your model requires specialized I/O.
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        decoding_cv_sweep : bool, optional
            Whether to run a cross-validated hyperparameter sweep to
            find optimal regularization values, by default False
        """
        self.batch_fwd = batch_fwd
        self.log_every_n_epochs = log_every_n_epochs
        self.decoding_cv_sweep = decoding_cv_sweep

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get entire validation dataset from dataloader
        input_data, recon_data, *_, behavior = trainer.datamodule.valid_data
        recon_data = recon_data.detach().cpu().numpy()
        behavior = behavior.detach().cpu().numpy()
        # Get model predictions for the entire validation dataset
        val_dataloader = trainer.datamodule.val_dataloader()
        # Pass the data through the model
        rates = [self.batch_fwd(pl_module, batch)[0] for batch in val_dataloader]
        rates = torch.cat(rates).detach().cpu().numpy()
        # Compute co-smoothing bits per spike
        _, n_obs, n_heldin = input_data.shape
        heldout = recon_data[:, :n_obs, n_heldin:]
        rates_heldout = rates[:, :n_obs, n_heldin:]
        co_bps = bits_per_spike(rates_heldout, heldout)
        pl_module.log("nlb/co_bps", max(co_bps, -1.0))
        # Compute forward prediction bits per spike
        forward = recon_data[:, n_obs:]
        rates_forward = rates[:, n_obs:]
        fp_bps = bits_per_spike(rates_forward, forward)
        pl_module.log("nlb/fp_bps", max(fp_bps, -1.0))
        # Get relevant training dataset from datamodule
        *_, train_behavior = trainer.datamodule.train_data
        train_behavior = train_behavior.detach().cpu().numpy()
        # Get model predictions for the training dataset
        train_dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        train_rates = [
            self.batch_fwd(pl_module, batch)[0] for batch in train_dataloader
        ]
        train_rates = torch.cat(train_rates).detach().cpu().numpy()
        # Get firing rates for observed time points
        rates_obs = rates[:, :n_obs]
        train_rates_obs = train_rates[:, :n_obs]
        # Compute behavioral decoding performance
        if "dmfc_rsg" in trainer.datamodule.hparams.dataset_name:
            tp_corr = speed_tp_correlation(heldout, rates_obs, behavior)
            pl_module.log("nlb/tp_corr", tp_corr)
        else:
            with warnings.catch_warnings():
                # Ignore LinAlgWarning from early in training
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                behavior_r2 = velocity_decoding(
                    train_rates_obs,
                    train_behavior,
                    trainer.datamodule.train_decode_mask,
                    rates_obs,
                    behavior,
                    trainer.datamodule.eval_decode_mask,
                    self.decoding_cv_sweep,
                )
            pl_module.log("nlb/behavior_r2", max(behavior_r2, -1.0))
        # Compute PSTH reconstruction performance
        if hasattr(trainer.datamodule, "psth"):
            psth = trainer.datamodule.psth
            cond_idxs = trainer.datamodule.val_cond_idxs
            jitter = trainer.datamodule.eval_jitter
            psth_r2 = eval_psth(psth, rates_obs, cond_idxs, jitter)
            pl_module.log("nlb/psth_r2", max(psth_r2, -1.0))
