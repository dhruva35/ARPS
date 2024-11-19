import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import logging
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating app rating prediction models."""
    
    def __init__(self, model_dir: str = "src/models/saved"):
        """Initialize the model trainer."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.feature_importance = {}
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple models and evaluate their performance."""
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Store results
            results[name] = {
                'metrics': metrics,
                'cv_rmse': cv_rmse
            }
            
            # Store model
            self.models[name] = model
            
            # Calculate feature importance for random forest
            if name == 'random_forest':
                self.feature_importance = dict(zip(
                    X.columns,
                    model.feature_importances_
                ))
            
            # Save model
            self._save_model(name, model)
            
            # Log results
            logger.info(f"\nResults for {name}:")
            logger.info(f"MSE: {metrics['mse']:.4f}")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"R2 Score: {metrics['r2']:.4f}")
            logger.info(f"Cross-validation RMSE: {cv_rmse:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _save_model(self, name: str, model: Any) -> None:
        """Save model to disk."""
        model_path = self.model_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} model to {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from random forest model."""
        if not self.feature_importance:
            raise ValueError("No feature importance available. Train random forest model first.")
        
        # Sort feature importance
        sorted_importance = dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_importance
    
    def predict(self, X: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train model first.")
        
        return self.models[model_name].predict(X)
