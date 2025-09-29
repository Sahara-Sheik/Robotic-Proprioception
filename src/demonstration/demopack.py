"""
demopack.py

A demo pack is a self contained file that contains a set of demonstrations that later can be used to be imported into an experiment, and used as training and validation data at various levels. 

Normally a demo pack is a zip file or a directory, containing a list of demonstrations
"""
import pathlib
import zipfile
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"


class DemoGroupChooser:
    pass

def import_demopack_from_folder(demo_path, group_chooser):
    assert(demo_path.is_dir())
    # get these from the config
    # exprun_path = 
    # result_path = 
    # get the list of demonstrations from the demo path
    # identify the target location result_path/demo_path
    # we have the groups: eg. "sp_training", "sp_valid", "sp_test", "vp_training" etc. We pass a function which chooses the corresponding demonstrations: eg: return a[0:half], a[half:] etc
    # copy the demonstrations under the name 

def import_demopack_from_zip(exprun_path, result_path):
