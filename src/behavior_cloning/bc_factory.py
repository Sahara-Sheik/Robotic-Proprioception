"""
bc_factory.py

Creating different models for behavior cloning based on the specification in the exp/run
"""

import pathlib
import socket
import torch.nn as nn
import torch.optim as optim
from bc_MLP import bc_MLP
from bc_LSTM import bc_LSTM, bc_LSTM_Residual
from bc_LSTM_MDN import bc_LSTM_MDN, mdn_loss

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"


def create_bc_model(exp, exp_sp, device):
    if exp["controller"] == "bc_MLP":
        model = bc_MLP(exp, exp_sp)
    elif exp["controller"] == "bc_LSTM":
        model = bc_LSTM(exp, exp_sp)
    elif exp["controller"] == "bc_LSTM_Residual":
        model = bc_LSTM_Residual(exp, exp_sp)
    elif exp["controller"] == "bc_LSTM_MDN":
        model = bc_LSTM_MDN(exp, exp_sp)
    else:
        raise Exception(f"Unknown controller specified {exp['controller']}")    
    model.to(device)
    criterion = create_criterion(exp, device)
    optimizer = create_optimizer(exp, model)
    return model, criterion, optimizer


def create_criterion(exp, device):
    if exp["loss"] == "MSELoss":
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        criterion = criterion.to(device)
    elif exp["loss"] == "MDNLoss":
        criterion = mdn_loss 
        # criterion = criterion.to(device)
        # Note that this is a bit different in parameters
    else:
        raise Exception(f"Loss function {exp['loss']} not implemented yet")
    return criterion

def create_optimizer(exp, model):
    if exp["optimizer"] == "Adam":
        lr = exp["optimizer_lr"]
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer {exp['optimizer']} not implemented yet")
    return optimizer

def external_setup(setupname):
    """Create an external directory 'setupname' where the generated exp/runs and results will go. This allows separating a set of experiments both for training and robot running. 

    Under this directory, there will be two directories:
    * 'exprun' - contains the copied necessary expruns from the source code + the programatically generated expruns. 
    * 'result' - contains the training data and the trained models. 
    
    The training data should go into result/demonstration under some directory (eg. touch-apple).
    """
        # host specific directories
    hostname = socket.gethostname()
    print(f"Hostname is {hostname}")
    if hostname == "raven":
        raise Exception("Not configured yet")
    elif hostname == "szenes.local":
        bc_path = pathlib.Path(f"~/Documents/Develop/Data/{setupname}").expanduser()
    elif hostname == "glassy":
        bc_path = pathlib.Path(f"~/Work/_DataExternal/{setupname}").expanduser()
    else:
        bc_path = pathlib.Path(Config()["experiment_external"], setupname)

    exprun_path = pathlib.Path(bc_path, "exprun")
    result_path = pathlib.Path(bc_path, "result")

    print(f"Path for external experiments: {exprun_path}")
    exprun_path.mkdir(exist_ok=True, parents=True)
    print(f"Path for external data: {result_path}")
    result_path.mkdir(exist_ok=True, parents=True)

    Config().set_experiment_path(exprun_path)
    Config().set_experiment_data(result_path)

    # Copy the necessary experiments into the external directory
    Config().copy_experiment("demonstration")
    Config().copy_experiment("sensorprocessing_conv_vae")
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("automate")
    Config().copy_experiment("behavior_cloning")
    Config().copy_experiment("controllers")

    return exprun_path, result_path