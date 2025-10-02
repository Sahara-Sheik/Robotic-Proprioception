"""
demopack.py

A demo pack is a self contained file that contains a set of demonstrations that later can be used to be imported into an experiment, and used as training and validation data at various levels. 

Normally a demo pack is a zip file or a directory, containing a list of demonstrations
"""
import pathlib
import zipfile
import shutil
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"


def group_chooser_sp_bc_trivial(demo_names):
    """Copy all the data to sp, bc both training and testing. Note that this overlaps the training and testing so it is not a good idea in general."""
    retval = {}
    # the sensorprocessing data
    for i, demo_name in enumerate(demo_names):
        retval[f"sp_training_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"sp_validation_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"sp_testing_{i:05d}"] = demo_name
    # the behavior cloninng data
    for i, demo_name in enumerate(demo_names):
        retval[f"bc_training_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"bc_validation_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"bc_testing_{i:05d}"] = demo_name
    return retval

def group_chooser_sp_vp_trivial(demo_names):
    """Copy all the data to sp, bc both training and testing. Note that this overlaps the training and testing so it is not a good idea in general."""
    retval = {}
    # the sensorprocessing data
    for i, demo_name in enumerate(demo_names):
        retval[f"sp_training_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"sp_validation_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"sp_testing_{i:05d}"] = demo_name
    # the behavior cloninng data
    for i, demo_name in enumerate(demo_names):
        retval[f"vp_training_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"vp_validation_{i:05d}"] = demo_name
    for i, demo_name in enumerate(demo_names):
        retval[f"vp_testing_{i:05d}"] = demo_name
    return retval

def import_demopack(demo_path, group_chooser):
    assert(demo_path.is_dir())
    # get these from the config
    exprun_path = Config().get_exprun_path()
    results_path = Config().get_results_path()
    demoname = demo_path.stem
    demonstration_yaml = pathlib.Path(demo_path, demoname + ".yaml")
    demo_names = [d.name for d in demo_path.iterdir() if d.is_dir()]
    target_yaml = pathlib.Path(exprun_path, "demonstration", demoname + ".yaml")
    target_dir = pathlib.Path(results_path, "demonstration", demoname)
    if target_dir.exists():
        print(f"*** import_demopack: {demo_path}, target directory {target_dir} already exists, returning")
        return
    target_dir.mkdir(exist_ok=True, parents=True)
    demo_exprun = pathlib.Path(exprun_path, "demonstration")
    demo_exprun.mkdir(exist_ok=True, parents=True)
    # create the defaults 
    defaults_yaml = pathlib.Path(demo_exprun, "_defaults_demonstration.yaml")
    defaults_yaml.touch()
    # copy the target yaml both to exprun and results
    target_yaml.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(demonstration_yaml, target_yaml)
    shutil.copy2(demonstration_yaml, pathlib.Path(demo_exprun, demoname + ".yaml"))
    # copy the different demonstrations with different names
    tocopy = group_chooser(demo_names)
    for target in tocopy:
        destdir = pathlib.Path(target_dir, target)
        sourcedir = pathlib.Path(demo_path, tocopy[target])
        shutil.copytree(sourcedir, destdir)

    