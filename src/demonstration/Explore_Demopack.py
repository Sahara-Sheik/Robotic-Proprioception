#!/usr/bin/env python
# coding: utf-8

# # Exploring a demopack
# 
# This notebook shows ways to interact with a demopack. 

# In[1]:


import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
import pathlib
import socket
import pprint

from demopack import import_demopack, group_chooser_sp_bc_trivial
from demonstration import list_demos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ## Importing a demopack from a folder

# In[2]:


# demopack_name = "touch-apple"
demopack_name = "random-both-cameras-video"

demopack_path = pathlib.Path(Config()["demopacks_path"], demopack_name)

flow_path = pathlib.Path(Config()["flows_path"], "Explore_Demopack")
exprun_path = pathlib.Path(flow_path, "exprun")
results_path = pathlib.Path(flow_path, "results")
exprun_path.mkdir(exist_ok=True, parents=True)
results_path.mkdir(exist_ok=True, parents=True)
Config().set_exprun_path(exprun_path)
Config().set_results_path(results_path)


# In[3]:


import_demopack(demopack_path, group_chooser_sp_bc_trivial)


# In[4]:


# Setting up the experiment for the demonstration
experiment = "demonstration"
run = demopack_name
exp = Config().get_experiment(experiment, run)


# In[5]:


# read out all the demonstrations
demos = list_demos(exp)
pprint.pprint(demos)


# In[ ]:




