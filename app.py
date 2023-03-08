from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import MelanomaDetectionModel
from util import base64_to_pil
from gevent.pywsgi import WSGIServer
import os

app = Flask(__name__)

cancer_model = MelanomaDetectionModel()

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predictClass', methods=['POST'])
def predict_sentiment():
    image=request.form["uploadImage"]
    img = base64_to_pil(request.json)
    img.save("uploads\image.jpg")
    img_path = os.path.join(os.path.dirname(__file__),'uploads\image.jpg')
    os.path.isfile(img_path)
    input_img = image.load_img(img, target_size=(180, 180), color_mode='rgb')
    cancer_class=cancer_model.getClassOfCancer(input_img)
    print(cancer_class)

if __name__ == '__main__':
    app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()