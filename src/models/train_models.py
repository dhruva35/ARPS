import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model_trainer import ModelTrainer
from data.preprocessor import DataPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train and evaluate models for app rating prediction."""
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        preprocessor = DataPreprocessor()
        preprocessor.preprocess_data()
        X, y = preprocessor.get_training_data()
        
        # Initialize and train models
        logger.info("Training models...")
        trainer = ModelTrainer()
        results = trainer.train_models(X, y)
        
        # Get and display feature importance
        logger.info("\nFeature Importance:")
        importance = trainer.get_feature_importance()
        for feature, score in importance.items():
            logger.info(f"{feature}: {score:.4f}")
        
        return results, importance
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

if __name__ == "__main__":
    main()
