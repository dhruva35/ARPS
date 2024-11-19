from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys
import os
import logging
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.predict import AppRatingPredictor

# Initialize Flask app with correct template and static folders
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)

predictor = AppRatingPredictor()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    try:
        logger.debug("Received prediction request")
        
        # Get data from request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            })
            
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            })
        
        # Create app data dictionary
        app_data = {
            'name': data.get('name', 'Unknown App'),
            'app_size_mb': float(data.get('app_size_mb', 0)),
            'price_usd': float(data.get('price_usd', 0)),
            'downloads': int(data.get('downloads', 0)),
            'app_type': data.get('app_type', 'productivity'),
            'store': data.get('store', 'google_play')
        }
        
        logger.debug(f"Processed app_data: {app_data}")
        
        # Make prediction
        predicted_rating = predictor.predict_rating(app_data)
        logger.debug(f"Predicted rating: {predicted_rating}")
        
        # Get feature importance
        feature_importance = {
            'App Size': 46.91,
            'Downloads': 15.01,
            'Price': 11.91,
            'Store Platform': 3.0,
            'App Type': 2.61
        }
        
        # Convert numpy types to Python types
        response_data = {
            'success': True,
            'predicted_rating': float(predicted_rating),
            'app_details': app_data,
            'feature_importance': feature_importance
        }
        logger.debug(f"Sending response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(port=5002, debug=True)
