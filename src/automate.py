import pathlib
import papermill as pm

import os
print(os.getcwd())

params = {"run": "sp_vae_256"}
params["external_path"] = r"C:\Users\lboloni\Documents\Code\_TempData\BerryPicker-2\experiments"
#params["data_path"] = r"C:\Users\lboloni\Documents\Code\_TempData\BerryPicker-2\data"
params["epochs"] = 10

output = pathlib.Path(r"C:\Users\lboloni\Documents\Code\_TempData\BerryPicker-2\automation_output", "sensorprocessing_Train-Conv-VAE-output.ipynb")

# also set epochs from here.
# and use them from there

# FIXME: the output.ipynb should actually go into the data dir, or something...

try:
   pm.execute_notebook(
      'sensorprocessing/Train-Conv-VAE.ipynb',
      output.absolute(),
      cwd="sensorprocessing",
      parameters=params
   )
except Exception as e:
   print(f"There was an exception {e}")

# pm.execute_notebook(
#    'sensorprocessing/Train-Conv-VAE.ipynb',
#    'output.ipynb',
#    parameters=dict(run="sp_vae_256")
# )