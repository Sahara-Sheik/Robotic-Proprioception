"""
demonstration_helper.py

Create training data from demonstrations with support for multiple camera views.
"""
import sys
sys.path.append("..")

import torch
import pprint
import pathlib
from sensorprocessing.sp_helper import load_picturefile_to_tensor


class Demonstration:
    """This class encapsulates all the convenience functions for a demonstration, including loading images etc. """
    
    def __init__(self, exp, demo):
        """Initializes the demonstration, based on an experiment"""
        self.exp = exp
        self.demo = demo
        self.demo_dir = pathlib.Path(exp.data_dir(), demo)
        self.maxsteps = -1
        cameraset = {}
        for a in self.demo_dir.iterdir():
            if a.name.endswith(".json") and a.name.startswith("0"):
                count = int(a.name.split(".")[0])
                self.maxsteps = max(self.maxsteps, count)
            if a.name.endswith(".jpg"):
                cameraname = a.name[6:-4]
                cameraset[cameraname] = cameraname
        self.cameras = sorted(cameraset.keys())

    def __str__(self):
        return pprint.pformat(self.__dict__)


    def get_image_path(self, i, camera=None):
        """Returns the Path to the image, if the demo is stored as independent image files."""
        if camera is None:
            camera = self.cameras[0]
        filepath = pathlib.Path(self.demo_dir, f"{i:05d}_{camera}.jpg")
        return filepath


    def get_image(self, i, camera=None, transform=None):
        """
        Gets the image as a torch batch

        Args:
            i: The timestep index
            camera: Which camera to use. If None, uses the first camera.
            transform: Optional transform to apply to the image
        """

        filepath = self.get_image_path(i, camera)
        sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
        return sensor_readings, image

    def get_all_images(self, i, transform=None):
        """
        Gets images from all cameras for a specific timestep

        Args:
            i: The timestep index
            transform: Optional transform to apply to the images

        Returns:
            Dictionary mapping camera names to (sensor_readings, image) tuples
        """
        images = {}
        for camera in self.cameras:
            filepath = pathlib.Path(self.demo_dir, f"{i:05d}_{camera}.jpg")
            if filepath.exists():
                sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
                images[camera] = (sensor_readings, image)

        return images