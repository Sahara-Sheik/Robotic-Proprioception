#!/usr/bin/env python
# coding: utf-8

"""
Create/Export a demopack from demonstration data
This script copies demonstration data from the data directory to the demopacks directory
"""

import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import pathlib
import shutil

# Configuration
demo_experiment = "demonstration"
demo_run = "random-both-cameras-video"  # or "freeform"
demopack_name = "random-both-cameras-video"  # Name for the demopack

# Get paths
source_path = pathlib.Path(Config()["experiment_data"], demo_experiment, demo_run).expanduser()
target_path = pathlib.Path(Config()["demopacks_path"], demopack_name).expanduser()

print("=" * 70)
print("CREATING DEMOPACK")
print("=" * 70)
print(f"\nSource (demonstration data): {source_path}")
print(f"Target (demopack location):  {target_path}")

# Check if source exists
if not source_path.exists():
    print(f"\n❌ ERROR: Source demonstration data doesn't exist!")
    print(f"   Looking for: {source_path}")
    print(f"\n   Available demonstrations in {source_path.parent}:")
    if source_path.parent.exists():
        for item in source_path.parent.iterdir():
            if item.is_dir():
                print(f"     - {item.name}")
    sys.exit(1)

# Check if target already exists
if target_path.exists():
    response = input(f"\n⚠️  Demopack already exists at {target_path}\n   Overwrite? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)
    print(f"Removing existing demopack...")
    shutil.rmtree(target_path)

# Create target directory if it doesn't exist
target_path.parent.mkdir(exist_ok=True, parents=True)

# Copy the demonstration data to create the demopack
print(f"\nCopying demonstration data to demopack location...")
shutil.copytree(source_path, target_path)

print(f"\n✅ Demopack created successfully!")
print(f"   Location: {target_path}")
print(f"\n   Contents:")
for item in target_path.iterdir():
    if item.is_dir():
        print(f"     📁 {item.name}/")
    else:
        print(f"     📄 {item.name}")

print("\n" + "=" * 70)
print("Now you can run: import_demopack(demopack_path, group_chooser)")
print("=" * 70)
