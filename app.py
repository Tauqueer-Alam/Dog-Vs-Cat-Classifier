from flask import Flask, render_template, request
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model

model = load_model("model.h5")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def form():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0  
    image = np.expand_dims(image, axis=0) 

    result = float(model.predict(image))
    prediction = "Dog" if result > 0.5 else "Cat"

    return render_template('predict.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
