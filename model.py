import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



class MelanomaDetectionModel:
    ROOT_PATH="model/"
    MODEL_NAME="CNN_MELANOMIA2.h5"
    
    def __init__(self) -> None:
        print("oppo "+os.getcwd())
        self.model = load_model(MelanomaDetectionModel.ROOT_PATH + MelanomaDetectionModel.MODEL_NAME, compile=False)

    def getClassOfCancer(self, uploadedImage):
        class_names=['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
        input_img = image.load_img(uploadedImage, target_size=(180, 180), color_mode='rgb')
        tf_image = image.img_to_array(input_img)

        # Add an extra dimension to the array to make it a tensor with shape (1, *image_shape)
        image_tensor = tf_image[tf.newaxis,...]
        scaled_image=tf.keras.layers.Rescaling(scale=1./255)(image_tensor)
        prob=self.model.predict(scaled_image)
        prediction = np.argmax(prob)

        # Print the class label
        print(class_names[prediction])
