"""
demo_to_trainingdata.py

Create training data from demonstrations with support for multiple camera views.
"""
import sys
sys.path.append("..")

import torch
import helper
import pathlib
import json
from pprint import pformat
import numpy as np
from sensorprocessing.sp_helper import load_picturefile_to_tensor


def create_RNN_training_sequence_xy(x_seq, y_seq, sequence_length):
    """Create supervised training data for RNNs such as LSTM from two sequences. In this data, from a string of length sequence_length in x_seq we are predicting the next item in y_seq.
    Returns the results as tensors
    """
    # Prepare training data
    total_length = x_seq.shape[0]
    inputs = []
    targets = []
    for i in range(total_length - sequence_length):
        # Input is a subsequence of length `sequence_length`
        input_seq = x_seq[i:i + sequence_length]
        # Shape: [sequence_length, latent_size]

        # Target is the next vector after the input sequence
        target = y_seq[i + sequence_length]
        # Shape: [output_size]

        # Append to lists
        inputs.append(torch.tensor(input_seq))
        targets.append(torch.tensor(target))

    # Convert lists to tensors for training
    inputs = torch.stack(inputs)   # Shape: [num_samples, sequence_length, latent_size]
    targets = torch.stack(targets) # Shape: [num_samples, latent_size]
    return inputs, targets


