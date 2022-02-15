import logging
import os

import dotenv

from nlb_tools.make_tensors import (
    make_eval_input_tensors,
    make_eval_target_tensors,
    make_train_input_tensors,
)
from nlb_tools.nwb_interface import NWBDataset

# Load the NLB environment variables
dotenv.load_dotenv(override=True)
DATA_HOME = os.environ["DATA_HOME"]
PREP_HOME = os.environ["PREP_HOME"]
TRAIN_INPUT_FILE = os.environ["TRAIN_INPUT_FILE"]
EVAL_INPUT_FILE = os.environ["EVAL_INPUT_FILE"]
EVAL_TARGET_FILE = os.environ["EVAL_TARGET_FILE"]

# Specify the dandi subpaths
DANDI_SUBPATH = {
    "mc_maze": "000128/sub-Jenkins",
    "mc_rtt": "000129/sub-Indy",
    "area2_bump": "000127/sub-Han",
    "dmfc_rsg": "000130/sub-Haydn",
    "mc_maze_large": "000138/sub-Jenkins",
    "mc_maze_medium": "000139/sub-Jenkins",
    "mc_maze_small": "000140/sub-Jenkins",
}


def main(
    dataset_name="mc_maze_large",
    phase="val",
    bin_width=5,
):
    """Loads NLB datasets downloaded from DANDI and stored
    at `DATA_HOME`, preprocesses the dataset using `nlb_tools`,
    and stores output at `PREP_HOME`.

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
    """
    assert phase in ["val", "test"]
    # Get paths from constants and make sure they exist
    data_path = os.path.join(DATA_HOME, DANDI_SUBPATH[dataset_name])
    save_path = os.path.join(PREP_HOME, f"{dataset_name}-{bin_width:02}ms-{phase}")
    os.makedirs(save_path, exist_ok=True)
    # Load the data from the NWB file
    dataset = NWBDataset(data_path)
    # Resample to the desired bin width
    dataset.resample(bin_width)
    # Determine training and evaluation splits
    if phase == "val":
        train_trial_split = "train"
        eval_trial_split = "val"
    else:
        train_trial_split = ["train", "val"]
        eval_trial_split = "test"
    # Set common arguments for making inputs
    make_data_args = dict(
        dataset=dataset,
        dataset_name=dataset_name,
        return_dict=True,
    )
    # Create training input tensors
    make_train_input_tensors(
        **make_data_args,
        trial_split=train_trial_split,
        include_behavior=True,
        include_forward_pred=True,
        save_path=os.path.join(save_path, TRAIN_INPUT_FILE),
    )
    # Create input tensors for evaluation
    make_eval_input_tensors(
        **make_data_args,
        trial_split=eval_trial_split,
        save_path=os.path.join(save_path, EVAL_INPUT_FILE),
    )
    if phase == "val":
        # Create target tensors for evaluation
        make_eval_target_tensors(
            **make_data_args,
            train_trial_split=train_trial_split,
            eval_trial_split=eval_trial_split,
            include_psth=True,
            save_path=os.path.join(save_path, EVAL_TARGET_FILE),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Preprocess datasets for all splits and resolutions
    for dataset_name in DANDI_SUBPATH:
        for phase in ["val", "test"]:
            for bin_width in [5, 20]:
                main(dataset_name=dataset_name, phase=phase, bin_width=bin_width)
