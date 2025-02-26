from flask import Flask, render_template, request
import numpy as np
import pickle
import cv2
import os

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0  
    image = np.expand_dims(image, axis=0) 

    result = model.predict(image)[0][0]
    
    prediction = "Dog" if result > 0.5 else "Cat"

    return render_template('predict.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
