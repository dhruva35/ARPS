import os
import logging
from data.collector import GooglePlayCollector, AppleAppStoreCollector, AmazonAppStoreCollector
from data.preprocessor import DataPreprocessor
from models.rating_predictor import RatingPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_data():
    """Collect data from all platforms."""
    collectors = [
        GooglePlayCollector(),
        AppleAppStoreCollector(),
        AmazonAppStoreCollector()
    ]
    
    all_app_data = []
    all_review_data = []
    
    for collector in collectors:
        try:
            # Collect app data
            app_data = collector.collect_app_data()
            all_app_data.append(app_data)
            
            # Collect reviews for each app
            for app_id in app_data['app_id']:
                reviews = collector.collect_reviews(app_id)
                all_review_data.append(reviews)
        except Exception as e:
            logger.error(f"Error collecting data from {collector.__class__.__name__}: {str(e)}")
    
    return all_app_data, all_review_data

def preprocess_data(app_data, review_data):
    """Preprocess collected data."""
    preprocessor = DataPreprocessor()
    
    # Preprocess app data
    processed_app_data = preprocessor.preprocess_app_data(app_data)
    
    # Preprocess review data
    processed_review_data = preprocessor.preprocess_reviews(review_data)
    
    return processed_app_data, processed_review_data

def train_model(processed_data):
    """Train the rating prediction model."""
    predictor = RatingPredictor(model_type='rf')
    
    # Prepare features and target
    X = predictor.prepare_features(processed_data)
    y = processed_data['rating']
    
    # Train model with hyperparameter optimization
    train_score, val_score = predictor.train(X, y, optimize=True)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    if feature_importance is not None:
        logger.info("\nFeature Importance:")
        logger.info(feature_importance)
    
    # Save the trained model
    model_path = os.path.join('models', 'rating_predictor.joblib')
    predictor.save_model(model_path)
    
    return predictor

def main():
    """Main execution function."""
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Step 1: Collect data
        logger.info("Starting data collection...")
        app_data, review_data = collect_data()
        
        # Step 2: Preprocess data
        logger.info("Preprocessing data...")
        processed_app_data, processed_review_data = preprocess_data(app_data, review_data)
        
        # Step 3: Train model
        logger.info("Training model...")
        predictor = train_model(processed_app_data)
        
        logger.info("ARPS pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in ARPS pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
