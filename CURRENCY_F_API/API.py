from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = ["10", "20", "50", "100", "200", "500"]

@app.route('/classify_denomination', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    
    # Open the image
    image = Image.open(file).convert("RGB")
    
    # Resize and preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Prepare response
    response = {
        "class_name": class_name,
        "confidence_score": float(confidence_score)
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
