from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
import cv2
import numpy as np
import json
import bcrypt
from flask_cors import CORS
from keras.models import load_model
from PIL import Image, ImageOps

np.set_printoptions(suppress=True)
model = load_model('model_v6.h5', compile=False)
class_names = open('labels.txt', 'r').readlines()


APP_ROOT = os.path.abspath(os.path.dirname(__file__))


# image pre-processing
def preprocess_image(_image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(_image_path).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    return data


# make the prediction based on the pre-processed image
def get_prediction(data):
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return {
            "prediction": class_name.rstrip("\n"),
            "probability": str(confidence_score)
        }


# Working as the toString method
def output_prediction(filename):
    _image_path = f"./images/{filename}"
    image_data = preprocess_image(_image_path)
    result = get_prediction(image_data)

    return result


# Init app
app = Flask(__name__)
CORS(app)


# Image prediction endpoint
@app.route('/api/predict', methods=['POST'])
def get_disease_prediction():
    target = os.path.join(APP_ROOT, 'images/')

    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files.get('file')
    filename = file.filename
    destination = '/'.join([target, filename])

    file.save(destination)

    response = output_prediction(filename)

    print("response", response)
    return jsonify(response)


# Run Server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=False)
    





