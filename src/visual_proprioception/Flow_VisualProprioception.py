#!/usr/bin/env python
# coding: utf-8

# # Visual proprioception flow
# 
# Create the full flow for training models for visual proprioception. This notebook programmatically generates a set of exp/runs that cover all the necessary components for a visual proprioception system (sensor processing,  visual proprioception regressor and verification notebooks).
# 
# Then, it writes the exp/runs into an external directory full separated from the github source, and creates an automation script that runs them. A separate directory for the results is also created. 
# 
# Finally, it runs the necessary notebooks to execute the whole flow using papermill.
# 
# The results directory contain the output of this flow, both in terms of trained models, as well as results (in the verification exp/run).

# In[1]:


import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import copy
import pprint
import pathlib
import yaml
import tqdm
import papermill
import visproprio_helper 
from demonstration.demonstration import list_demos
from demonstration.demopack import import_demopack, group_chooser_sp_vp_standard


# # Setting up the separate directory
# Setting up a separate directory for generated exp/run config files and the results. This cell will create a new directory. 

# In[ ]:


# the most likely changing things
# flow_name = "VisualProprioception_flow_01"
flow_name = "cvpr_simulation_054"
#demonstration_run = "touch-apple"
#demopack_name = "touch-apple"
# demopack_name = "automove-pack-01"
demopack_name = "cvpr-simulation"
# demonstration_cam = "dev2"
demonstration_cam = "dev054"

do_VAE = True
do_VGG = True
do_RESNET = True

# determine these values based on experience
epochs_sp = 10
# epochs_vp = 1000
# epochs_sp = 300 # way too much, at least for VAE
epochs_vp = 1000
image_size = [256, 256] # for vgg... etc

# Use exist_ok not to-re-run previously successfully run models
creation_style = "exist-ok"
# creation_style = "discard-old"


# In[ ]:


exprun_path, result_path = visproprio_helper.external_setup(flow_name, Config()["flows_path"])

demopack_path = pathlib.Path(Config()["demopacks_path"], demopack_name)
selection = import_demopack(demopack_path, group_chooser_sp_vp_standard)
#
# Configuring the training and validation data, based 
# on all the demonstrations of a particular type
#
experiment = "demonstration"
exp = Config().get_experiment(experiment, demopack_name)

sp_training_data = [[demopack_name, demo, demonstration_cam] for demo in selection["sp_training"]]
sp_validation_data = [[demopack_name, demo, demonstration_cam] for demo in selection["sp_validation"]]
vp_training_data = [[demopack_name, demo, demonstration_cam] for demo in selection["vp_training"]]
vp_validation_data = [[demopack_name, demo, demonstration_cam] for demo in selection["vp_validation"]]


# In[3]:


print(selection)


# In[4]:


demos = list_demos(exp)
# print(demos)
print(list_demos(exp, "sp"))
[s for s in demos if s.startswith("sp_training" + "_")]


# In[5]:


def generate_sp_conv_vae(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the conv-vae sensorprocessing with the right training data and parameters. Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation. 
    NOTE: a similar function is in Flow_BehaviorCloning.
    """
    val = {}
    val["latent_size"] = params["latent_size"]
    val["epochs"] = params["epochs"]
    val["save_period"] = 5
    val["training_data"] = params["training_data"]
    val["validation_data"] = params["validation_data"]
    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file 
    v = {}
    v["name"] = "Train_SP_Conv-VAE"
    v["notebook"] = "sensorprocessing/Train_Conv_VAE.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[6]:


def generate_sp_propriotuned_cnn(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned CNN with the right training data and parameters. 
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation. 
    """
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32
    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file 
    v = {}
    v["name"] = "Train_SP_CNN"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_CNN.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[7]:


def generate_vp_train(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training visual proprioception regressor.  
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation. 
    """
    val = copy.copy(params)

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file 
    v = {}
    v["name"] = f"Train_{run_name}"
    v["notebook"] = "visual_proprioception/Train_VisualProprioception.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[8]:


def generate_vp_verify(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the verification of the visual proprioception regressor.  
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation. 
    """
    val = copy.copy(params)

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file 
    v = {}
    v["name"] = f"Verify_{run_name}"
    v["notebook"] = "TODO Verify.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[9]:


def generate_vp_compare(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the verification of the visual proprioception regressor.  
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation. 
    """
    val = copy.copy(params)
    val["name"] = exp_name

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file 
    v = {}
    v["name"] = f"Compare_{run_name}"
    v["notebook"] = "visual_proprioception/Compare_VisualProprioception.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# ### Generate the exp/runs to be run

# In[ ]:


expruns = []
# overall values
latent_sizes = [128, 256] # the possible latent sizes we consider
cnntypes = ["vgg19", "resnet50"] # the CNN architectures we consider




# *******************************************
# generate the sensorprocessing models
# *******************************************
sps = [] # the list of the sensorprocessing models (exp/run)
for latent_size in latent_sizes:

    # generate the vae exprun
    exp_name = "sensorprocessing_conv_vae"
    run_name = f"sp_conv_vae_{latent_size}_0001"
    params = {}
    params["latent_size"] = latent_size
    params["epochs"] = epochs_sp
    params["training_data"] = sp_training_data
    params["validation_data"] = sp_validation_data
    exprun = generate_sp_conv_vae(
        exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name = run_name)
    exprun["latent_size"] = latent_size
    if do_VAE:
        sps.append(exprun)
        expruns.append(exprun)

    # generate the propriotuned expruns
    for cnntype in cnntypes:
        exp_name = "sensorprocessing_propriotuned_cnn"
        run_name = f"sp_{cnntype}_{latent_size}_0001"
        params = {}
        params["image_size"] = image_size
        params["latent_size"] = latent_size
        params["epochs"] = epochs_sp
        params["training_data"] = sp_training_data
        params["validation_data"] = sp_validation_data
        if cnntype == "vgg19":
            if not do_VGG:
                continue
            params["class"] = "VGG19ProprioTunedSensorProcessing"
            params["model"] = "VGG19ProprioTunedRegression"
        elif cnntype == "resnet50":
            if not do_RESNET:
                continue
            params["class"] = "ResNetProprioTunedSensorProcessing"
            params["model"] = "ResNetProprioTunedRegression"
            params["freeze_feature_extractor"] = True
            params["reductor_step_1"] = 512
            params["proprio_step_1"] = 64
            params["proprio_step_2"] = 16
        else:
            raise Exception(f"Unknown cnntype {cnntype}")
        params["loss"] = "MSELoss" # alternative L1Loss
        params["learning_rate"] = 0.001
        # alternative
        exprun = generate_sp_propriotuned_cnn(
            exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name = run_name)
        exprun["latent_size"] = latent_size
        exprun["cnntype"] = cnntype        
        sps.append(exprun)
        expruns.append(exprun)

    # FIXME: add here the ViT models

# *******************************************
# generate the proprioception models
# *******************************************
vpruns = []
vpruns_latent = {128:[], 256:[]}
for spexp, sprun,latent_size in [(a["experiment"],a["run"],a["latent_size"]) for a in sps]:
    print(spexp, sprun, latent_size)
    # *** generate the vp train expruns ***
    exp_name = "visual_proprioception"
    run_name = "vp_" + sprun[3:]
    vpruns.append(run_name)
    vpruns_latent[latent_size].append(run_name)
    params = {}
    params["name"] = run_name
    params["output_size"] = 6
    params["encoding_size"] = latent_size
    params["training_data"] = vp_training_data
    params["validation_data"] = vp_validation_data

    params["regressor_hidden_size_1"] = 64
    params["regressor_hidden_size_1"] = 64
    params["loss"] = "MSE"
    params["epochs"] = epochs_vp
    params["batch_size"] = 64
    # FIXME this is hackish, should not do it this way
    if "vae" in sprun.lower():
        params["sensor_processing"] = "ConvVaeSensorProcessing"
    elif "resnet" in sprun.lower():
        params["sensor_processing"] = "ResNetProprioTunedSensorProcessing"
    elif "vgg19" in sprun.lower():
        params["sensor_processing"] = "VGG19ProprioTunedSensorProcessing"
    else:
        raise Exception(f"Unexpected sprun {sprun}")

    params["sp_experiment"] = spexp
    params["sp_run"] = sprun

    exprun = generate_vp_train(exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name=run_name)
    # *** generate the vp verify expruns FIXME: not implemented yet ***
    params_verify = {}

    expruns.append(exprun)
# *******************************************
# generate the comparisons: all, for latents 128 and 256
# *******************************************
exp_name = "visual_proprioception"
# all
run_name = "vp_comp_flow_all"
params = {}
params["name"] = run_name
params["tocompare"] = vpruns
exprun = generate_vp_compare(exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name=run_name)
expruns.append(exprun)
# by latent
for latent_size in [128, 256]:
    run_name = f"vp_comp_flow_{latent_size}"
    params = {}
    params["name"] = run_name
    params["tocompare"] = vpruns_latent[latent_size]
    exprun = generate_vp_compare(exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name=run_name)
    expruns.append(exprun)



# ### Run the flow
# 
# Run the flow, that is, run a series of notebooks with papermill. In order to follow the execution inside these notebooks, one needs to open the output notebook, which is in the output_filename. 

# In[11]:


print(f"***Starting automated running of the flow.\n The path for the output notebooks is\n{result_path}")

for exprun in tqdm.tqdm(expruns):
    print(f"***Automating {exprun['notebook']} :\n {exprun['experiment']}/{exprun['run']}")
    notebook_path = pathlib.Path("..", exprun["notebook"])
    output_filename = f"{notebook_path.stem}_{exprun['experiment']}_{exprun['run']}_output{notebook_path.suffix}"
    print(f"--> {output_filename}")
    # parameters that we are passing on to the notebook
    params = {}
    params["experiment"] = exprun["experiment"]
    params["run"] = exprun["run"]
    params["external_path"] = exprun["external_path"]
    params["data_path"] = exprun["data_path"]    
    output_path = pathlib.Path(result_path, output_filename)
    try:
        papermill.execute_notebook(
            notebook_path,
            output_path.absolute(),
            cwd=notebook_path.parent,
            parameters=params
        )
    except Exception as e:
        print(f"There was an exception {e}")  


# In[ ]:




