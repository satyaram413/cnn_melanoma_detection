import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



class MelanomaDetectionModel:
    ROOT_PATH="model/"
    MODEL_NAME="CNN_MELANOMIA.h5"
    
    def __init__(self) -> None:
        print("oppo "+os.getcwd())
        self.model = load_model(MelanomaDetectionModel.ROOT_PATH + MelanomaDetectionModel.MODEL_NAME, compile=False)

    def getClassOfCancer(self, uploadedImage):
        class_names=['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
        tf_image = image.img_to_array(uploadedImage)

        # Add an extra dimension to the array to make it a tensor with shape (1, *image_shape)
        image_tensor = tf_image[tf.newaxis,...]
        scaled_image=tf.keras.layers.Rescaling(scale=1./255)(image_tensor)
        prob=self.model.predict(scaled_image)
        prediction = np.argmax(prob)

        # Print the class label
        print(class_names[prediction])
