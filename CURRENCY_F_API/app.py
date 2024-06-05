import os
import pickle
import cv2
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained Keras model (replace 'currency_denomination_model.h5' with your model file)
model = load_model('keras_model.h5')

# Define a list of denomination labels
denomination_labels = ['10', '50', '200', '100', '20', '500']

@app.route('/classify_denomination', methods=['POST'])
def classify_denomination_endpoint():
    print(request.form)
    print(request.files)
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image file'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform image processing here to preprocess the image
    # You should adapt this part to your specific model and preprocessing steps.

    # Example: Resize the image to match the model's input size
    input_size = (224, 224)
    image = cv2.resize(image, input_size)

    # Convert the image to a NumPy array
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    print(prediction)
    score=tf.nn.softmax(prediction[0])
    # prediction_class = np.argmax(prediction)
    # print(prediction_class)
    # if not (0 <= prediction_class <= 6):
    #     return jsonify({'error': 'Invalid prediction value'}), 400
    
    denomination = denomination_labels[np.argmax(score)]
    print(denomination)
    return jsonify({'denomination': denomination})

if __name__ == '__main__':
    app.run(debug=True)