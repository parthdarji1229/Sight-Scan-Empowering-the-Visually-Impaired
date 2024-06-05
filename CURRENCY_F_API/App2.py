from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from sklearn.externals import joblib

app = Flask(__name)

# Load the trained currency detection model
model = joblib.load('currency_model.pkl')

# Define a function to preprocess the input image
def preprocess_image(image):
    # Convert to grayscale, resize, and flatten the image
    image = image.convert('L').resize((100, 100))
    image = np.array(image).flatten()
    return image

@app.route('/detect_currency', methods=['POST'])
def detect_currency():
    # Check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is valid
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Open and preprocess the image
            image = Image.open(file)
            preprocessed_image = preprocess_image(image)

            # Use the model to make a prediction
            prediction = model.predict([preprocessed_image])

            # Map prediction to currency denomination
            denominations = {0: '10 Rupees', 1: '100 Rupees', 2: '500 Rupees'}
            currency = denominations.get(prediction[0], 'Unknown')

            return jsonify({'currency': currency})
        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
