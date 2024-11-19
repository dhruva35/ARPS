from preprocessor import DataPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Process the app store datasets."""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        processed_data = preprocessor.preprocess_data()
        
        # Get and print feature statistics
        stats = preprocessor.get_feature_stats()
        
        logger.info("\nDataset Statistics:")
        logger.info(f"Total number of apps: {stats['total_apps']}")
        logger.info("\nApps by store:")
        for store, count in stats['apps_by_store'].items():
            logger.info(f"  {store}: {count}")
        
        logger.info("\nApps by type:")
        for app_type, count in stats['apps_by_type'].items():
            logger.info(f"  {app_type}: {count}")
        
        logger.info("\nRating statistics:")
        for stat, value in stats['rating_stats'].items():
            logger.info(f"  {stat}: {value:.2f}")
        
        # Get training data
        X, y = preprocessor.get_training_data()
        logger.info(f"\nTraining data shape: {X.shape}")
        logger.info(f"Number of features: {X.shape[1]}")
        
        return processed_data, X, y
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
