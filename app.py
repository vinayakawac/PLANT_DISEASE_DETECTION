"""
Plant Disease Detection API
A production-ready REST API for detecting plant diseases from leaf images.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import json
import uuid
from functools import lru_cache
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MODEL_PATH'] = 'plant_disease_model.h5'
app.config['CLASSES_PATH'] = 'plant_disease_classes.npy'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load model and classes at startup
try:
    model = load_model(app.config['MODEL_PATH'])
    class_names = np.load(app.config['CLASSES_PATH'], allow_pickle=True)
    logger.info(f"Model loaded successfully. Found {len(class_names)} disease classes.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None
    class_names = None

# Load treatment recommendations
@lru_cache(maxsize=1)
def load_treatment_data():
    """Load treatment recommendations from JSON file."""
    try:
        with open('treatment_recommendations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Treatment recommendations file not found. Using default recommendations.")
        return {}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_path):
    """Validate that the file is a valid image."""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        return False

def predict_disease(image_path):
    """
    Predict plant disease from image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Prediction results including plant, disease, confidence, and recommendations
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        pred_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        predicted_class = class_names[pred_index]
        
        # Parse class name
        if "___" in predicted_class:
            plant, disease = predicted_class.split("___", 1)
        else:
            plant, disease = "Unknown", predicted_class
        
        plant = plant.replace("_", " ").strip()
        disease = disease.replace("_", " ").strip()
        
        # Get treatment recommendations
        treatment_data = load_treatment_data()
        recommendations = treatment_data.get(
            predicted_class,
            {
                "severity": "unknown",
                "treatment": "Consult with an agricultural expert for proper diagnosis and treatment.",
                "prevention": "Maintain good agricultural practices."
            }
        )
        
        # Determine confidence level
        if confidence >= 90:
            confidence_level = "High"
        elif confidence >= 70:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        result = {
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence, 2),
            "confidence_level": confidence_level,
            "is_healthy": "healthy" in disease.lower(),
            "recommendations": recommendations,
            "raw_class": predicted_class
        }
        
        logger.info(f"Prediction successful: {plant} - {disease} ({confidence:.2f}%)")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise

def save_result(prediction_result, image_path, result_id):
    """Save prediction result to JSON file."""
    try:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        result_data = {
            "id": result_id,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "prediction": prediction_result
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Result saved: {result_file}")
        return result_file
    except Exception as e:
        logger.error(f"Failed to save result: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy" if model is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "num_classes": len(class_names) if class_names is not None else 0
    }
    
    status_code = 200 if model is not None else 503
    return jsonify(health_status), status_code

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of all supported disease classes."""
    if class_names is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    classes_list = [
        {
            "raw_name": cls,
            "plant": cls.split("___")[0].replace("_", " ") if "___" in cls else "Unknown",
            "disease": cls.split("___")[1].replace("_", " ") if "___" in cls else cls.replace("_", " ")
        }
        for cls in class_names
    ]
    
    return jsonify({
        "total_classes": len(classes_list),
        "classes": classes_list
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict plant disease from uploaded image.
    
    Expected: multipart/form-data with 'file' field
    Returns: JSON with prediction results
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not available"}), 503
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate unique ID for this prediction
        result_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{result_id[:8]}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(file_path)
        
        # Validate image
        if not validate_image(file_path):
            os.remove(file_path)
            return jsonify({"error": "Invalid or corrupted image file"}), 400
        
        logger.info(f"Processing image: {saved_filename}")
        
        # Make prediction
        prediction_result = predict_disease(file_path)
        
        # Save result
        save_result(prediction_result, file_path, result_id)
        
        # Prepare response
        response = {
            "success": True,
            "result_id": result_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction_result
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error during prediction",
            "message": str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict plant diseases from multiple uploaded images.
    
    Expected: multipart/form-data with multiple 'files' fields
    Returns: JSON with batch prediction results
    """
    try:
        if model is None:
            return jsonify({"error": "Model not available"}), 503
        
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        
        if len(files) == 0:
            return jsonify({"error": "No files selected"}), 400
        
        if len(files) > 10:
            return jsonify({"error": "Maximum 10 files allowed per batch"}), 400
        
        results = []
        errors = []
        
        for idx, file in enumerate(files):
            try:
                if file.filename == '' or not allowed_file(file.filename):
                    errors.append({
                        "index": idx,
                        "filename": file.filename,
                        "error": "Invalid file"
                    })
                    continue
                
                # Generate unique ID
                result_id = str(uuid.uuid4())
                
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_filename = f"{timestamp}_{result_id[:8]}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
                file.save(file_path)
                
                # Validate and predict
                if not validate_image(file_path):
                    os.remove(file_path)
                    errors.append({
                        "index": idx,
                        "filename": filename,
                        "error": "Invalid image"
                    })
                    continue
                
                prediction_result = predict_disease(file_path)
                save_result(prediction_result, file_path, result_id)
                
                results.append({
                    "index": idx,
                    "filename": filename,
                    "result_id": result_id,
                    "prediction": prediction_result
                })
                
            except Exception as e:
                logger.error(f"Error processing file {idx}: {str(e)}")
                errors.append({
                    "index": idx,
                    "filename": file.filename if hasattr(file, 'filename') else "unknown",
                    "error": str(e)
                })
        
        response = {
            "success": len(results) > 0,
            "timestamp": datetime.now().isoformat(),
            "total_processed": len(results),
            "total_errors": len(errors),
            "results": results,
            "errors": errors
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error during batch prediction",
            "message": str(e)
        }), 500

@app.route('/api/result/<result_id>', methods=['GET'])
def get_result(result_id):
    """Retrieve a saved prediction result by ID."""
    try:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        
        if not os.path.exists(result_file):
            return jsonify({"error": "Result not found"}), 404
        
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        return jsonify(result_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving result: {str(e)}")
        return jsonify({"error": "Failed to retrieve result"}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics."""
    try:
        results_files = [f for f in os.listdir(app.config['RESULTS_FOLDER']) if f.endswith('.json')]
        
        disease_counts = {}
        plant_counts = {}
        total_predictions = len(results_files)
        
        for result_file in results_files:
            with open(os.path.join(app.config['RESULTS_FOLDER'], result_file), 'r') as f:
                data = json.load(f)
                disease = data['prediction']['disease']
                plant = data['prediction']['plant']
                
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
                plant_counts[plant] = plant_counts.get(plant, 0) + 1
        
        stats = {
            "total_predictions": total_predictions,
            "disease_distribution": disease_counts,
            "plant_distribution": plant_counts,
            "top_diseases": sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error generating stats: {str(e)}")
        return jsonify({"error": "Failed to generate statistics"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run in development mode
    # For production, use a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
