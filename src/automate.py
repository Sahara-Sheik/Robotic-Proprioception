import papermill as pm

pm.execute_notebook(
   'sensorprocessing/Train-Conv-VAE.ipynb',
   'output.ipynb',
   parameters=dict(run="sp_vae_256")
)