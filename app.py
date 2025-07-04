from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os

# Configuration
IMAGE_SIZE = 160
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
MODEL_PATH = 'facenet_vantoan_vanhau.pth'
CLASS_NAMES_PATH = 'class_names.txt'
CONFIDENCE_THRESHOLD = 0.5

# Initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Load class names
def load_class_names(class_names_path):
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        print(f"Class names file not found at {class_names_path}")
        return None

# Load the trained model
def load_model(model_path, num_classes, device):
    model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    image_size=IMAGE_SIZE, 
    margin=0, 
    min_face_size=MINSIZE,
    thresholds=THRESHOLD, 
    factor=FACTOR, 
    post_process=True,
    device=device
)

# Load class names and model
class_names = load_class_names(CLASS_NAMES_PATH)
if class_names is None:
    print("Error: Could not load class names!")
    class_names = []
else:
    print(f"Loaded class names: {class_names}")

if class_names and os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, len(class_names), device)
        print("FaceNet model successfully loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Model file not found: {MODEL_PATH}")
    model = None

# Prediction function
def predict_face(image):
    """
    Predict the class of a face in an image
    """
    try:
        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect and crop face
        img_cropped = mtcnn(image)
        
        if img_cropped is None:
            return "No face detected", 0.0
        
        # Add batch dimension and move to device
        img_cropped = img_cropped.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_cropped)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return "Error", 0.0

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
@cross_origin()
def index():
    return "FaceNet Recognition Server is Running!"

@app.route('/recog', methods=['POST'])
@cross_origin()
def upload_img_file():
    if request.method == 'POST':
        if model is None:
            return "Error: Model not loaded"
        
        if not class_names:
            return "Error: Class names not available"
        
        try:
            name = "Unknown"
            
            # Get base64 image from request
            f = request.form.get('image')
            if not f:
                return "Error: No image provided"
            
            # Optional: get image dimensions
            w = int(request.form.get('w', 100))
            h = int(request.form.get('h', 100))
            
            print(f"Received image data, dimensions: {w}x{h}")
            
            # Decode base64 image
            decoded_string = base64.b64decode(f)
            frame = np.frombuffer(decoded_string, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            if frame is None:
                return "Error: Failed to decode image"
            
            # Make prediction
            predicted_class, confidence = predict_face(frame)
            
            print(f"Prediction: {predicted_class}, Confidence: {confidence:.3f}")
            
            # Return result based on confidence threshold
            if confidence > CONFIDENCE_THRESHOLD and predicted_class not in ["No face detected", "Error"]:
                name = predicted_class
            else:
                name = "Unknown"
            
            return name
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return f"Error: {str(e)}"

@app.route('/recog_detailed', methods=['POST'])
@cross_origin()
def upload_img_file_detailed():
    """Alternative endpoint that returns detailed JSON response"""
    if request.method == 'POST':
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if not class_names:
            return jsonify({'error': 'Class names not available'}), 500
        
        try:
            name = "Unknown"
            
            # Get base64 image from request
            f = request.form.get('image')
            if not f:
                return jsonify({'error': 'No image provided'}), 400
            
            # Optional: get image dimensions
            w = int(request.form.get('w', 100))
            h = int(request.form.get('h', 100))
            
            # Decode base64 image
            decoded_string = base64.b64decode(f)
            frame = np.frombuffer(decoded_string, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Failed to decode image'}), 400
            
            # Make prediction
            predicted_class, confidence = predict_face(frame)
            
            print(f"Prediction: {predicted_class}, Confidence: {confidence:.3f}")
            
            # Return result based on confidence threshold
            if confidence > CONFIDENCE_THRESHOLD and predicted_class not in ["No face detected", "Error"]:
                name = predicted_class
            else:
                name = "Unknown"
            
            return jsonify({
                'name': name,
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'threshold': CONFIDENCE_THRESHOLD,
                'classes': class_names
            })
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': class_names,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
