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
    return torch.tensor(array, dtype=torch.float)


class NLBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name="mc_maze_large",
        phase="val",
        bin_width=5,
        batch_size=64,
        num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Get the save path to the data
        self.save_path = os.path.join(
            PREP_HOME, f"{dataset_name}-{bin_width:02}ms-{phase}"
        )

    def setup(self, stage=None):
        # Load the training data arrays from file
        train_data_path = os.path.join(self.save_path, TRAIN_INPUT_FILE)
        with h5py.File(train_data_path, "r") as h5file:
            # Store the dataset
            heldin_train = to_tensor(h5file["train_spikes_heldin"][()])
            self.train_data = (
                heldin_train,
                to_tensor(h5file["train_spikes_heldin_forward"][()]),
                to_tensor(h5file["train_spikes_heldout"][()]),
                to_tensor(h5file["train_spikes_heldout_forward"][()]),
                to_tensor(h5file["train_behavior"][()]),
            )
        self.train_ds = TensorDataset(*self.train_data)
        # Check if evaluation data is available
        eval_data_path = os.path.join(self.save_path, EVAL_INPUT_FILE)
        target_data_path = os.path.join(self.save_path, EVAL_TARGET_FILE)
        if os.path.isfile(target_data_path):
            # Load the validation data arrays from file
            with h5py.File(eval_data_path, "r") as h5file:
                heldin = to_tensor(h5file["eval_spikes_heldin"][()])
            with h5py.File(target_data_path, "r") as h5file:
                # Match nlb_tools unique naming for non-5ms bin widths
                groupname = self.hparams.dataset_name
                if self.hparams.bin_width != 5:
                    groupname += f"_{self.hparams.bin_width}"
                h5group = h5file[groupname]
                # Store the dataset
                self.valid_data = (
                    heldin,
                    to_tensor(h5group["eval_spikes_heldin_forward"][()]),
                    to_tensor(h5group["eval_spikes_heldout"][()]),
                    to_tensor(h5group["eval_spikes_heldout_forward"][()]),
                    to_tensor(h5group["eval_behavior"][()]),
                )
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
                    self.train_decode_mask = np.ones(
                        (heldin_train.shape[0], 1), dtype=bool
                    )
                    self.eval_decode_mask = np.ones((heldin.shape[0], 1), dtype=bool)
            self.valid_ds = TensorDataset(*self.valid_data)

    def train_dataloader(self, shuffle=True):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return train_dl

    def val_dataloader(self):
        if not hasattr(self, "valid_ds"):
            return None
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
