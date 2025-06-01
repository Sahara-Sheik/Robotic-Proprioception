"""
sp_factory.py

Factory functions to create sensor processing objects based on an exp/run 
"""

from sensorprocessing import sp_conv_vae, sp_propriotuned_cnn, sp_aruco, sp_vit, sp_vit_multiview, sp_vit_concat_images

def create_sp(spexp, device):
    """Gets the sensor processing component specified by the
    visual_proprioception experiment."""
    # spexp = Config().get_experiment(exp['sp_experiment'], exp['sp_run'])
    if spexp["class"] == "ConvVaeSensorProcessing":
        return sp_conv_vae.ConvVaeSensorProcessing(spexp, device)
    if spexp['class']=="VGG19ProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.VGG19ProprioTunedSensorProcessing(spexp, device)
    if spexp['class']=="ResNetProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.ResNetProprioTunedSensorProcessing(spexp, device)
    if spexp['class']=="Aruco":
        return sp_aruco.ArucoSensorProcessing(spexp, device)
    if spexp['class']=="Vit":
        return sp_vit.VitSensorProcessing(spexp, device)
    if spexp['class'] == "Vit_multiview":
        return sp_vit_multiview.MultiViewVitSensorProcessing(spexp, device)
    # FIXME: I don't know which ones are these ones
    if spexp['class'] == "Vit_concat_images":
        return sp_vit_concat_images.ConcatImageVitSensorProcessing(spexp, device)
    raise Exception('Unknown sensor processing {exp["sensor_processing"]}')
