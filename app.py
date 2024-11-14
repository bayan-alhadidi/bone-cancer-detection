from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import os
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (replace the path with your saved model's path)
model = tf.keras.models.load_model('bone_cancer_detection_model.h5')

# Define class names for benign and malignant
class_names = ['benign', 'malignant']

# Folder where uploaded images will be saved temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Image preprocessing function
def preprocess_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # Apply Median filter
    filtered_image = cv2.medianBlur(image, 3)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_image = clahe.apply(l)
    lab = cv2.merge((clahe_image, a, b))
    contrast_enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(contrast_enhanced_image, -1, kernel)

    # Resize the image to match the input size expected by the model
    img_resized = cv2.resize(sharpened_image, (512, 512))

    # Convert the image to a format that the model can predict on
    img_resized = img_resized.astype('float32') / 255.0  # Normalize pixel values
    
    img_resized = np.expand_dims(img_resized, axis=0)     # Add batch dimension
    
    return img_resized

# Function to predict the image class (benign or malignant)
def predict_image_class(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)

    # Apply a threshold (e.g., 0.5) to decide between benign and malignant
    if prediction >= 0.5:
        diagnosis = 'malignant'
        confidence = prediction[0][0] * 100
    else:
        diagnosis = 'benign'
        confidence = (1 - prediction[0][0]) * 100
    
    return diagnosis, confidence

# Route to serve uploaded images for viewing
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Define the route for the web interface
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return "No file part"
        img_file = request.files['file']
        
        if img_file.filename == '':
            return "No selected file"
        
        if img_file:
            # Save the image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(img_path)

            # Predict the diagnosis
            diagnosis, confidence = predict_image_class(img_path)

            # Determine the color based on diagnosis
            if diagnosis == 'malignant':
                color = 'red'
            else:
                color = 'green'

            # Return the result as JSON for display on the same page
            return jsonify({
                'diagnosis': diagnosis,
                'confidence': f'{confidence:.2f}',
                'img_path': img_file.filename,
                'color': color
            })
    
    # Render the HTML template for uploading the image
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
