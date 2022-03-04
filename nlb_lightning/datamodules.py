import os

import dotenv
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the NLB environment variables
dotenv.load_dotenv(override=True)
PREP_HOME = os.environ["PREP_HOME"]
TRAIN_INPUT_FILE = os.environ["TRAIN_INPUT_FILE"]
EVAL_INPUT_FILE = os.environ["EVAL_INPUT_FILE"]
EVAL_TARGET_FILE = os.environ["EVAL_TARGET_FILE"]


def to_tensor(array):
    """Converts a loaded numpy array to a tensor
    and ensures correct dtype

    Parameters
    ----------
    array : np.array
        The numpy array to convert.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    return torch.tensor(array, dtype=torch.float)


def assemble_recon_data(heldin, heldin_forward, heldout, heldout_forward):
    """Combines heldin/heldout and observed/forward data
    into a single tensor.

    Parameters
    ----------
    heldin : torch.Tensor
        A BxTxN tensor.
    heldin_forward : torch.Tensor
        A BxT_FWDxN tensor.
    heldout : torch.Tensor
        A BxTxN_OUT tensor.
    heldout_forward : torch.Tensor
        A BxT_FWDxN_OUT tensor.

    Returns
    -------
    torch.tensor
        A Bx(T+T_FWD)x(N+N_OUT) tensor.
    """
    heldin_full = torch.cat([heldin, heldin_forward], dim=1)
    heldout_full = torch.cat([heldout, heldout_forward], dim=1)
    recon_data = torch.cat([heldin_full, heldout_full], dim=2)
    return recon_data


class NLBDataModule(pl.LightningDataModule):
    """Loads from preprocessed HDF5 files created using
    functions in `nlb_tools.make_tensors` and builds PyTorch
    `TensorDataset`s and `DataLoader`s that handle batching
    and shuffling.
    """

    def __init__(
        self,
        dataset_name="mc_maze_large",
        phase="val",
        bin_width=5,
        batch_size=64,
        num_workers=4,
    ):
        """Initializes the datamodule.

        Parameters
        ----------
        dataset_name : str, optional
            One of the data tags specified by the NLB organizers,
            by default "mc_maze_large"
        phase : str, optional
            The phase of the competition - either "val" or "test",
            by default "val"
        bin_width : int, optional
            The width of data bins, by default 5
        batch_size : int, optional
            The number of samples to process in each batch,
            by default 64
        num_workers : int, optional
            The number of subprocesses to use for data loading,
            by default 4
        """
        super().__init__()
        self.save_hyperparameters()
        # Get the save path to the data
        self.save_path = os.path.join(
            PREP_HOME, f"{dataset_name}-{bin_width:02}ms-{phase}"
        )

    def setup(self, stage=None):
        """Loads the data from preprocessed HDF5 files.

        Parameters
        ----------
        stage : str, optional
            Ignored, by default None
        """
        # Load the training data arrays from file
        train_data_path = os.path.join(self.save_path, TRAIN_INPUT_FILE)
        with h5py.File(train_data_path, "r") as h5file:
            # Store the dataset
            heldin = to_tensor(h5file["train_spikes_heldin"][()])
            heldin_fwd = to_tensor(h5file["train_spikes_heldin_forward"][()])
            heldout = to_tensor(h5file["train_spikes_heldout"][()])
            heldout_fwd = to_tensor(h5file["train_spikes_heldout_forward"][()])
            behavior = to_tensor(h5file["train_behavior"][()])
        # Assemble the dataset
        input_data = heldin
        recon_data = assemble_recon_data(heldin, heldin_fwd, heldout, heldout_fwd)
        self.train_data = (input_data, recon_data, behavior)
        self.train_ds = TensorDataset(*self.train_data)
        # Load the evaluation input from file
        eval_data_path = os.path.join(self.save_path, EVAL_INPUT_FILE)
        with h5py.File(eval_data_path, "r") as h5file:
            heldin = to_tensor(h5file["eval_spikes_heldin"][()])
        input_data = heldin
        # Check if evaluation target data is available
        target_data_path = os.path.join(self.save_path, EVAL_TARGET_FILE)
        if os.path.isfile(target_data_path):
            # Load the validation data arrays from file
            with h5py.File(target_data_path, "r") as h5file:
                # Match nlb_tools unique naming for non-5ms bin widths
                groupname = self.hparams.dataset_name
                if self.hparams.bin_width != 5:
                    groupname += f"_{self.hparams.bin_width}"
                h5group = h5file[groupname]
                # Store the dataset
                heldin_fwd = to_tensor(h5group["eval_spikes_heldin_forward"][()])
                heldout = to_tensor(h5group["eval_spikes_heldout"][()])
                heldout_fwd = to_tensor(h5group["eval_spikes_heldout_forward"][()])
                behavior = to_tensor(h5group["eval_behavior"][()])
                # Assemble the dataset
                recon_data = assemble_recon_data(
                    heldin, heldin_fwd, heldout, heldout_fwd
                )
                self.valid_data = (input_data, recon_data, behavior)
                # Store PSTHs and condition labels for evaluation
                if "psth" in h5group:
                    self.psth = h5group["psth"][()]
                    self.val_cond_idxs = h5group["eval_cond_idx"][()]
                    # Store trial jitter if found
                    if "eval_jitter" in h5group:
                        self.eval_jitter = h5group["eval_jitter"][()]
                    else:
                        self.eval_jitter = np.zeros(heldin.shape[0], dtype=int)
                # Store decoding masks if found
                if "eval_decode_mask" in h5group:
                    self.train_decode_mask = h5group["train_decode_mask"][()]
                    self.eval_decode_mask = h5group["eval_decode_mask"][()]
                else:
                    n_train = self.train_data[0].shape[0]
                    n_valid = heldin.shape[0]
                    self.train_decode_mask = np.ones((n_train, 1), dtype=bool)
                    self.eval_decode_mask = np.ones((n_valid, 1), dtype=bool)
        else:
            self.valid_data = (input_data,)
        self.valid_ds = TensorDataset(*self.valid_data)

    def train_dataloader(self, shuffle=True):
        """Returns a dataloader for the training data.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data, by default True

        Returns
        -------
        torch.utils.data.DataLoader
            A dataloader that generates data during training.
        """
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return train_dl

    def val_dataloader(self):
        """Returns a dataloader for the validation data.

        Returns
        -------
        torch.utils.data.DataLoader
            A dataloader that generates data during validation.
        """
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
