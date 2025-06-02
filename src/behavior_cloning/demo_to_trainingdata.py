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




