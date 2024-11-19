import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing app store data."""
    
    def __init__(self, data_dir: str = "src/data/raw"):
        """Initialize the preprocessor with data directory path."""
        self.data_dir = Path(data_dir)
        self.store_data = {}
        self.combined_data = None
    
    def load_data(self) -> None:
        """Load data from all app stores."""
        store_files = {
            'google_play': 'google_play_store.csv',
            'apple': 'apple_app_store.csv',
            'amazon': 'amazon_app_store.csv'
        }
        
        for store, filename in store_files.items():
            file_path = self.data_dir / filename
            try:
                self.store_data[store] = pd.read_csv(file_path)
                logger.info(f"Loaded {store} data: {len(self.store_data[store])} records")
            except Exception as e:
                logger.error(f"Error loading {store} data: {str(e)}")
                self.store_data[store] = None
    
    def clean_app_size(self, size_str: str) -> float:
        """Convert app size string to float in MB."""
        try:
            return float(size_str.replace(' MB', ''))
        except:
            return np.nan
    
    def clean_app_price(self, price_str: str) -> float:
        """Convert price string to float in USD."""
        if price_str == 'Free':
            return 0.0
        elif price_str == 'Free with In-App Purchases':
            return 0.0
        try:
            return float(price_str.replace('$', ''))
        except:
            return np.nan
    
    def clean_downloads(self, downloads_str: str) -> int:
        """Convert downloads string to integer."""
        try:
            multiplier = {
                'K': 1000,
                'M': 1000000,
                'B': 1000000000
            }
            number = float(downloads_str.replace('+', '').split('K')[0].split('M')[0].split('B')[0])
            for suffix, mult in multiplier.items():
                if suffix in downloads_str:
                    return int(number * mult)
            return int(number)
        except:
            return 0
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess all datasets and combine them."""
        if not self.store_data:
            self.load_data()
        
        processed_data = []
        
        for store_name, df in self.store_data.items():
            if df is not None:
                # Clean numeric columns
                df['app_size_mb'] = df['App Size'].apply(self.clean_app_size)
                df['price_usd'] = df['App Price'].apply(self.clean_app_price)
                df['downloads'] = df['Downloads'].apply(self.clean_downloads)
                
                # Clean and standardize categorical columns
                df['app_type'] = df['App Type'].str.lower()
                df['store'] = store_name
                
                # Select and rename columns
                processed_df = df[[
                    'App Name', 'app_size_mb', 'price_usd', 'app_type',
                    'App Version', 'User Rating', 'downloads', 'store'
                ]].copy()
                
                processed_df.columns = [
                    'app_name', 'app_size_mb', 'price_usd', 'app_type',
                    'app_version', 'user_rating', 'downloads', 'store'
                ]
                
                processed_data.append(processed_df)
        
        # Combine all processed datasets
        self.combined_data = pd.concat(processed_data, ignore_index=True)
        
        # Handle missing values
        self.combined_data = self.handle_missing_values(self.combined_data)
        
        logger.info(f"Preprocessed {len(self.combined_data)} total records")
        return self.combined_data
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numeric missing values with median
        numeric_columns = ['app_size_mb', 'price_usd', 'downloads']
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        categorical_columns = ['app_type', 'app_version']
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def get_feature_stats(self) -> Dict:
        """Get basic statistics about the features."""
        if self.combined_data is None:
            self.preprocess_data()
        
        stats = {
            'total_apps': len(self.combined_data),
            'apps_by_store': self.combined_data['store'].value_counts().to_dict(),
            'apps_by_type': self.combined_data['app_type'].value_counts().to_dict(),
            'price_stats': self.combined_data['price_usd'].describe().to_dict(),
            'rating_stats': self.combined_data['user_rating'].describe().to_dict(),
            'size_stats': self.combined_data['app_size_mb'].describe().to_dict()
        }
        
        return stats
    
    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training."""
        if self.combined_data is None:
            self.preprocess_data()
        
        # Select features for training
        features = [
            'app_size_mb', 'price_usd', 'downloads',
            'app_type', 'store'
        ]
        
        # Create dummy variables for categorical features
        X = pd.get_dummies(self.combined_data[features], columns=['app_type', 'store'])
        y = self.combined_data['user_rating']
        
        return X, y
