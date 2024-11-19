import os
import pandas as pd
import logging
from abc import ABC, abstractmethod
from .platform_apis import GooglePlayAPI, AppleAppStoreAPI, AmazonAppStoreAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AppStoreCollector(ABC):
    """Abstract base class for app store data collection."""
    
    def __init__(self):
        self.max_apps = int(os.getenv('MAX_APPS_PER_PLATFORM', 1000))
        self.data_dir = os.path.join('data', self.__class__.__name__.lower())
        os.makedirs(self.data_dir, exist_ok=True)
    
    @abstractmethod
    def collect_app_data(self):
        """Collect app data from the store."""
        pass
    
    @abstractmethod
    def collect_reviews(self, app_id):
        """Collect reviews for a specific app."""
        pass
    
    def save_data(self, data, filename):
        """Save collected data to CSV file."""
        try:
            df = pd.DataFrame(data)
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f'Successfully saved {len(df)} records to {filepath}')
            return df
        except Exception as e:
            logger.error(f'Error saving data to {filename}: {str(e)}')
            return pd.DataFrame()

class GooglePlayCollector(AppStoreCollector):
    def __init__(self):
        super().__init__()
        self.api = GooglePlayAPI()
        
    def collect_app_data(self):
        """Collect app data from Google Play Store."""
        logger.info('Starting Google Play Store data collection')
        
        # Example app IDs (in practice, you would get these from a search or category listing)
        sample_app_ids = [
            'com.whatsapp',
            'com.facebook.katana',
            'com.instagram.android',
            'com.spotify.music',
            'com.netflix.mediaclient'
        ]
        
        app_data = []
        for app_id in sample_app_ids[:self.max_apps]:
            try:
                data = self.api.get_app_details(app_id)
                if data:
                    app_data.append(data)
            except Exception as e:
                logger.error(f"Error collecting data for {app_id}: {str(e)}")
        
        return self.save_data(app_data, 'apps.csv')
    
    def collect_reviews(self, app_id):
        """Collect reviews from Google Play Store."""
        logger.info(f'Collecting reviews for app {app_id}')
        
        try:
            reviews = self.api.get_app_reviews(app_id)
            return self.save_data(reviews, f'reviews_{app_id}.csv')
        except Exception as e:
            logger.error(f"Error collecting reviews for {app_id}: {str(e)}")
            return pd.DataFrame()

class AppleAppStoreCollector(AppStoreCollector):
    def __init__(self):
        super().__init__()
        self.api = AppleAppStoreAPI()
        
    def collect_app_data(self):
        """Collect app data from Apple App Store."""
        logger.info('Starting Apple App Store data collection')
        
        # Example app IDs
        sample_app_ids = [
            '310633997',  # WhatsApp
            '284882215',  # Facebook
            '389801252',  # Instagram
            '324684580',  # Spotify
            '363590051'   # Netflix
        ]
        
        app_data = []
        for app_id in sample_app_ids[:self.max_apps]:
            try:
                data = self.api.get_app_details(app_id)
                if data:
                    app_data.append(data)
            except Exception as e:
                logger.error(f"Error collecting data for {app_id}: {str(e)}")
        
        return self.save_data(app_data, 'apps.csv')
    
    def collect_reviews(self, app_id):
        """Collect reviews from Apple App Store."""
        logger.info(f'Collecting reviews for app {app_id}')
        
        try:
            reviews = self.api.get_app_reviews(app_id)
            return self.save_data(reviews, f'reviews_{app_id}.csv')
        except Exception as e:
            logger.error(f"Error collecting reviews for {app_id}: {str(e)}")
            return pd.DataFrame()

class AmazonAppStoreCollector(AppStoreCollector):
    def __init__(self):
        super().__init__()
        self.api = AmazonAppStoreAPI()
        
    def collect_app_data(self):
        """Collect app data from Amazon App Store."""
        logger.info('Starting Amazon App Store data collection')
        
        # Example app IDs (ASIN)
        sample_app_ids = [
            'B00YVBFAZG',  # WhatsApp
            'B0094BB4TW',  # Facebook
            'B00387DT2A',  # Instagram
            'B004DTBKRO',  # Spotify
            'B005ZXWMUS'   # Netflix
        ]
        
        app_data = []
        for app_id in sample_app_ids[:self.max_apps]:
            try:
                data = self.api.get_app_details(app_id)
                if data:
                    app_data.append(data)
            except Exception as e:
                logger.error(f"Error collecting data for {app_id}: {str(e)}")
        
        return self.save_data(app_data, 'apps.csv')
    
    def collect_reviews(self, app_id):
        """Collect reviews from Amazon App Store."""
        logger.info(f'Collecting reviews for app {app_id}')
        
        try:
            reviews = self.api.get_app_reviews(app_id)
            return self.save_data(reviews, f'reviews_{app_id}.csv')
        except Exception as e:
            logger.error(f"Error collecting reviews for {app_id}: {str(e)}")
            return pd.DataFrame()

def main():
    """Main function to run data collection from all platforms."""
    collectors = [
        GooglePlayCollector(),
        AppleAppStoreCollector(),
        AmazonAppStoreCollector()
    ]
    
    for collector in collectors:
        try:
            collector.collect_app_data()
        except Exception as e:
            logger.error(f'Error collecting data from {collector.__class__.__name__}: {str(e)}')

if __name__ == '__main__':
    main()
