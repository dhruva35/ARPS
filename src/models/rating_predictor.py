import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RatingPredictor:
    def __init__(self, model_type='rf'):
        """Initialize the rating predictor with specified model type."""
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'linear':
            self.model = LinearRegression()
            self.param_grid = {}
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_features(self, df):
        """Prepare feature matrix from preprocessed dataframe."""
        # Select relevant features
        feature_columns = [
            'price', 'size_mb', 'downloads', 'category', 'platform',
            'rating_count', 'review_count', 'sentiment_score'
        ]
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                df[col] = 0
        
        return df[feature_columns]
    
    def train(self, X, y, optimize=True):
        """Train the model with optional hyperparameter optimization."""
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if optimize and self.param_grid:
            # Perform grid search
            logger.info("Starting hyperparameter optimization...")
            grid_search = GridSearchCV(
                self.model,
                self.param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train)
        
        # Calculate feature importance for random forest
        if self.model_type == 'rf':
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Evaluate model
        train_score = self.evaluate(X_train, y_train)
        val_score = self.evaluate(X_val, y_val)
        
        logger.info(f"Training R² score: {train_score:.4f}")
        logger.info(f"Validation R² score: {val_score:.4f}")
        
        return train_score, val_score
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def get_feature_importance(self):
        """Get feature importance if available."""
        if self.feature_importance is None:
            logger.warning("Feature importance not available")
            return None
        return self.feature_importance
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self
