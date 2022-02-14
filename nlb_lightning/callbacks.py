import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from nlb_tools.evaluation import (
    bits_per_spike,
    eval_psth,
    speed_tp_correlation,
    velocity_decoding,
)

def fig_to_rgb_array(fig):
    """Converts a matplotlib figure into an RGB array for logging.

    Args:
        fig ([type]): [description]
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


class RasterPlotCallback(pl.Callback):
    def __init__(self, n_samples=2, log_every_n_epochs=20):
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get data samples
        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        heldin, heldin_forward, heldout, heldout_forward, behavior = batch
        # Compute data sizes
        fwd_steps = heldin_forward.shape[1]
        batch_size, n_obs, n_heldin = heldin.shape
        # Compute model output
        batch_out = pl_module.forward(heldin.to(pl_module.device), fwd_steps)
        preds, _ = batch_out
        # Combine input data for raster and convert to numpy arrays
        heldin_full = torch.cat([heldin, heldin_forward], dim=1)
        heldout_full = torch.cat([heldout, heldout_forward], dim=1)
        data = torch.cat([heldin_full, heldout_full], dim=2).detach().cpu().numpy()
        _, total_steps, total_neurons = data.shape
        preds = np.exp(preds.detach().cpu().numpy())
        # Create subplots
        fig, axes = plt.subplots(
            self.n_samples, 2, sharex=True, sharey=True, figsize=(10, 10)
        )
        for i, ax_row in enumerate(axes):
            for ax, array in zip(ax_row, [data, preds]):
                ax.imshow(array[i].T)
                ax.vlines(n_obs, 0, total_neurons, color="coral")
                ax.hlines(n_heldin, 0, total_steps, color="coral")
                ax.set_xlim(0, total_steps)
                ax.set_ylim(0, total_neurons)
        plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.logger.experiment.add_image(
            "raster_plot", im, trainer.global_step, dataformats="HWC"
        )


class TrajectoryPlotCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100):
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get model predictions for the entire validation dataset
        val_dataloader = trainer.datamodule.val_dataloader()
        n_fwd = trainer.datamodule.valid_data[1].shape[1]

        def model_fwd(x):
            return pl_module.forward(x.to(pl_module.device), n_fwd)

        latents = torch.cat([model_fwd(batch[0])[1] for batch in val_dataloader])
        latents = latents.detach().cpu().numpy()
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
            ax.scatter(*traj[0], alpha=0.1, s=10, c="g")
            ax.scatter(*traj[-1], alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.logger.experiment.add_image(
            "trajectory_plot", im, trainer.global_step, dataformats="HWC"
        )


class EvaluationCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=20, decoding_cv_sweep=False):
        self.log_every_n_epochs = log_every_n_epochs
        self.decoding_cv_sweep = decoding_cv_sweep

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get entire validation dataset from dataloader
        (
            heldin,
            heldin_forward,
            heldout,
            heldout_forward,
            behavior,
        ) = trainer.datamodule.valid_data
        heldout = heldout.detach().cpu().numpy()
        behavior = behavior.detach().cpu().numpy()
        # Get model predictions for the entire validation dataset
        val_dataloader = trainer.datamodule.val_dataloader()
        n_fwd = heldin_forward.shape[1]

        def model_fwd(x):
            return pl_module.forward(x.to(pl_module.device), n_fwd)

        preds = torch.cat([model_fwd(batch[0])[0] for batch in val_dataloader])
        preds = torch.exp(preds).detach().cpu().numpy()
        # Compute co-smoothing bits per spike
        _, n_obs, n_heldin = heldin.shape
        preds_heldout = preds[:, :n_obs, n_heldin:]
        co_bps = bits_per_spike(preds_heldout, heldout)
        pl_module.log("nlb/co_bps", co_bps)
        # Compute forward prediction bits per spike
        preds_forward = preds[:, n_obs:]
        forward = torch.cat([heldin_forward, heldout_forward], dim=2)
        fp_bps = bits_per_spike(preds_forward, forward.detach().cpu().numpy())
        pl_module.log("nlb/fp_bps", fp_bps)
        # Get relevant training dataset from datamodule
        *_, train_behavior = trainer.datamodule.train_data
        train_behavior = train_behavior.detach().cpu().numpy()
        # Get model predictions for the training dataset
        train_dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        train_preds = torch.cat([model_fwd(batch[0])[0] for batch in train_dataloader])
        train_preds = torch.exp(train_preds).detach().cpu().numpy()
        # Get firing rates for observed time points
        preds_obs = preds[:, :n_obs]
        train_preds_obs = train_preds[:, :n_obs]
        # Compute behavioral decoding performance
        if "dmfc_rsg" in trainer.datamodule.hparams.dataset_name:
            tp_corr = speed_tp_correlation(heldout, preds_obs, behavior)
            pl_module.log("nlb/tp_corr", tp_corr)
        else:
            behavior_r2 = velocity_decoding(
                train_preds_obs,
                train_behavior,
                trainer.datamodule.train_decode_mask,
                preds_obs,
                behavior,
                trainer.datamodule.eval_decode_mask,
                self.decoding_cv_sweep,
            )
            pl_module.log("nlb/behavior_r2", max(behavior_r2, -10.0))
        # Compute PSTH reconstruction performance
        if hasattr(trainer.datamodule, "psth"):
            psth = trainer.datamodule.psth
            cond_idxs = trainer.datamodule.val_cond_idxs
            jitter = trainer.datamodule.eval_jitter
            psth_r2 = eval_psth(psth, preds_obs, cond_idxs, jitter)
            pl_module.log("nlb/psth_r2", max(psth_r2, -10.0))
