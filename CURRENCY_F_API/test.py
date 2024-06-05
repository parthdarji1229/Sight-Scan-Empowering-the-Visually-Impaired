import requests

# Specify the path to your image file
image_path = "<PATH_TO_YOUR_IMAGE>"

# API endpoint
url = "http://localhost:5000/predict"

# Prepare the image file
files = {'image': open(r"C:\Users\Vimal\Downloads\archive (3)\Test\2Hundrednote\33.jpg", 'rb')}

# Send POST request to the API
response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Predicted Class:", result['class_name'])
    print("Confidence Score:", result['confidence_score'])
else:
    print("Error:", response.text)
