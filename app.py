from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import MelanomaDetectionModel


app = Flask(__name__)

cancer_model = MelanomaDetectionModel()

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predictClass', methods=['POST'])
def predict_sentiment():
    image=request.form["uploadImage"]
    cancer_class=cancer_model.getClassOfCancer(image)
    print(cancer_class)