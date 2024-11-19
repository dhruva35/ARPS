import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AppRatingPredictor:
    def __init__(self):
        # Get the absolute path to the models directory
        current_dir = Path(__file__).parent
        self.model_dir = current_dir / "models/saved"
        logger.debug(f"Model directory: {self.model_dir}")
        
        model_path = self.model_dir / "random_forest.joblib"
        logger.debug(f"Looking for model at: {model_path}")
        
        if not model_path.exists():
            # Try the absolute path as a fallback
            fallback_path = Path("C:/Users/GANGARI DHRUVAVEER/CascadeProjects/ARPS/src/models/saved/random_forest.joblib")
            logger.debug(f"Model not found, trying fallback path: {fallback_path}")
            
            if not fallback_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path} or {fallback_path}")
            model_path = fallback_path
            
        logger.debug(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        logger.debug("Model loaded successfully")
        
        # Define feature order (must match training order)
        self.features = [
            'app_size_mb', 'price_usd', 'downloads',
            'app_type_communication', 'app_type_education',
            'app_type_entertainment', 'app_type_music',
            'app_type_productivity', 'app_type_social',
            'app_type_travel', 'app_type_video',
            'store_amazon', 'store_apple', 'store_google_play'
        ]
        
        # Define possible app types
        self.app_types = [
            'communication', 'education', 'entertainment', 'music',
            'productivity', 'social', 'travel', 'video'
        ]
        
        # Define possible stores
        self.stores = ['amazon', 'apple', 'google_play']
        
    def _create_features(self, app_data):
        """Create feature vector for prediction."""
        try:
            logger.debug(f"Creating features for app data: {app_data}")
            
            # Initialize features dictionary with zeros
            features = {feature: 0 for feature in self.features}
            logger.debug(f"Initialized features: {features}")
            
            # Set numeric features
            features['app_size_mb'] = float(app_data['app_size_mb'])
            features['price_usd'] = float(app_data['price_usd'])
            features['downloads'] = float(app_data['downloads'])  # Convert to float for consistency
            logger.debug(f"Set numeric features: {features}")
            
            # Set app type feature
            app_type = app_data['app_type'].lower()
            app_type_col = f"app_type_{app_type}"
            if app_type_col in features:
                features[app_type_col] = 1
                logger.debug(f"Set app type feature: {app_type_col}")
            else:
                logger.warning(f"Unknown app type: {app_type}")
                
            # Set store feature
            store = app_data['store'].lower()
            store_col = f"store_{store}"
            if store_col in features:
                features[store_col] = 1
                logger.debug(f"Set store feature: {store_col}")
            else:
                logger.warning(f"Unknown store: {store}")
                
            # Create DataFrame with correct feature order
            df = pd.DataFrame([features])[self.features]
            logger.debug(f"Created feature DataFrame with shape: {df.shape}")
            logger.debug(f"Feature values: {df.iloc[0].to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}", exc_info=True)
            raise
    
    def predict_rating(self, app_data):
        """Predict app rating."""
        try:
            logger.debug(f"Predicting rating for app data: {app_data}")
            
            # Create feature vector
            X = self._create_features(app_data)
            logger.debug(f"Created feature vector with shape: {X.shape}")
            
            # Make prediction
            predicted_rating = self.model.predict(X)[0]
            logger.debug(f"Raw predicted rating: {predicted_rating}")
            
            # Round and clip the prediction
            final_rating = round(float(predicted_rating), 2)
            final_rating = max(1.0, min(5.0, final_rating))
            logger.debug(f"Final predicted rating: {final_rating}")
            
            return final_rating
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
            raise

def main():
    """Test the rating predictor with sample apps."""
    predictor = AppRatingPredictor()
    
    # Sample apps for testing
    test_apps = [
        {
            'name': 'Social Media App',
            'app_size_mb': 250,
            'price_usd': 0.0,
            'downloads': 1000000,
            'app_type': 'social',
            'store': 'google_play'
        },
        {
            'name': 'Premium Game',
            'app_size_mb': 500,
            'price_usd': 4.99,
            'downloads': 50000,
            'app_type': 'entertainment',
            'store': 'apple'
        },
        {
            'name': 'Educational Tool',
            'app_size_mb': 150,
            'price_usd': 2.99,
            'downloads': 100000,
            'app_type': 'education',
            'store': 'amazon'
        }
    ]
    
    # Test predictions
    for app in test_apps:
        try:
            rating = predictor.predict_rating(app)
            print(f"\nPredicted rating for {app['name']}: {rating}")
        except Exception as e:
            print(f"Error predicting rating for {app['name']}: {str(e)}")

if __name__ == "__main__":
    main()
