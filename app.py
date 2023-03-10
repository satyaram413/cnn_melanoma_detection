from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import MelanomaDetectionModel
from util import base64_to_pil
import numpy as np
from gevent.pywsgi import WSGIServer
import os

app = Flask(__name__)

cancer_model = MelanomaDetectionModel()

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predictClass', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        class_names=['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
        # image=request.files["upload-button"]
        img = base64_to_pil(request.json)
        img.save("uploads\image.jpg")
        img_path = os.path.join(os.path.dirname(__file__),'uploads\image.jpg')
        os.path.isfile(img_path)
        input_img = img.resize((180,180))
        cancer_class_probs=cancer_model.getClassOfCancer(input_img)

        result =class_names[np.argmax(cancer_class_probs)]             # Convert to string
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability="{:.3f}".format(np.amax(cancer_class_probs)))

if __name__ == '__main__':
    app.run()