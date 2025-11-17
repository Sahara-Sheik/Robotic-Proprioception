#!/usr/bin/env python
# coding: utf-8

# # Exploring a demonstration
# 
# This notebook shows several standard ways to interact with the demonstrations stored into the exp/run directories. 

# In[1]:


import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import matplotlib.pyplot as plt
import random
import pprint
import torch

from demonstration import Demonstration, list_demos, select_demo, get_simple_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[ ]:


experiment = "demonstration"
# run = "freeform"
# run = "random-both-cameras"
run = "random-both-cameras-video"
exp = Config().get_experiment(experiment, run)
# FIXME: this should be configurable 
exp_controller = Config().get_experiment("robot_al5d", "position_controller_00")


# ### Read out all the demonstrations from a run

# In[3]:


# read out all the demonstrations
demos = list_demos(exp)
pprint.pprint(demos)


# In[4]:


#demoname = select_demo(exp, force_name="testing")
# demoname = select_demo(exp)
demoname = select_demo(exp, force_choice=1)
print(f"You selected demo with name: {demoname}")


# ###  Read out all the pictures in the demonstration

# In[5]:


demo = Demonstration(exp, demoname)
# print(demo)


# In[10]:


print(f"Cameras found: {demo.metadata['cameras']}")
print(f"There are {demo.metadata['maxsteps']} steps in this demonstration")

# if demo.metadata["stored_as_images"]:
    # pick and show a random image from a random camera
print("Random image, read as anything")
cam = random.choice(demo.metadata["cameras"])
imgno = random.randint(0, demo.metadata["maxsteps"])
print(f"Chosen camera {cam} image {imgno}")

transform = get_simple_transform()

sensor_readings, image = demo.get_image(imgno, camera=cam, device=device, transform=transform)
fig, ax = plt.subplots()
ax.imshow(image)
actions = demo.get_action(imgno, "rc-position-target", exp_controller)
print(actions)


# In[9]:


demo.actions[20]["rc-position-target"]


# ### Compare the same image from the video and the image

# In[ ]:


if demo.metadata["stored_as_images"] and demo.metadata["stored_as_video"]:
    imgno = random.randint(0, demo.metadata["maxsteps"])
    print(f"Image number: {imgno}")
    vid_tensor, vid_image = demo.get_image_from_video(imgno, camera=cam, device=device, transform=transform)
    img_tensor, img_image = demo.get_image(imgno+1, camera=cam, device=device, transform=transform)
    fig, [ax1, ax2] = plt.subplots(1,2)
    ax1.imshow(img_image)
    ax1.set_title("Image from image file")
    ax2.imshow(vid_image)
    ax2.set_title("Image from video file")
    print("Difference between image and video tensors")
    print(vid_tensor - img_tensor)
else:
    print("This demo does not have both image and video files")


# ### Example of how to generate video files from the image files
# Note that this will delete the image files.

# In[ ]:


# demo.move_to_video(delete_img_files=True)
# demo.move_to_video(delete_img_files=False)


# In[ ]:


### How to access the actions
imgno = random.randint(0, demo.metadata["maxsteps"])
pprint.pprint(demo.actions[imgno])
a = demo.get_action(imgno)
print(f"Action: {a}")

