import os
import json
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv

load_dotenv()

class GooglePlayAPI:
    """Google Play Store API wrapper"""
    
    def __init__(self):
        self.base_url = "https://play.google.com/store/apps"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def get_app_details(self, app_id):
        """Get app details using web scraping (since official API is restricted)"""
        url = f"{self.base_url}/details?id={app_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract app details
            app_data = {
                'app_id': app_id,
                'name': self._extract_text(soup, 'h1'),
                'category': self._extract_text(soup, '[itemprop="genre"]'),
                'rating': self._extract_rating(soup),
                'reviews': self._extract_reviews_count(soup),
                'size': self._extract_size(soup),
                'price': self._extract_price(soup),
                'downloads': self._extract_downloads(soup)
            }
            
            return app_data
            
        except Exception as e:
            print(f"Error fetching app details for {app_id}: {str(e)}")
            return None
    
    def get_app_reviews(self, app_id, limit=100):
        """Get app reviews using Selenium (since reviews are loaded dynamically)"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            url = f"{self.base_url}/details?id={app_id}&showAllReviews=true"
            driver.get(url)
            
            # Wait for reviews to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[jsname="fk8dgd"]'))
            )
            
            # Scroll to load more reviews
            reviews = []
            while len(reviews) < limit:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)  # Wait for new reviews to load
                
                review_elements = driver.find_elements(By.CSS_SELECTOR, '[jsname="fk8dgd"]')
                
                for element in review_elements:
                    review = {
                        'text': element.find_element(By.CSS_SELECTOR, '[jsname="bN97Pc"]').text,
                        'rating': len(element.find_elements(By.CSS_SELECTOR, 'span[aria-label="Rated"] > span')),
                        'date': element.find_element(By.CSS_SELECTOR, '[jsname="fk8dgd"] > div').text
                    }
                    reviews.append(review)
                
                if len(reviews) >= limit:
                    break
            
            return reviews[:limit]
            
        except Exception as e:
            print(f"Error fetching reviews for {app_id}: {str(e)}")
            return []
        
        finally:
            if 'driver' in locals():
                driver.quit()
    
    def _extract_text(self, soup, selector):
        """Helper method to extract text from HTML elements"""
        element = soup.select_one(selector)
        return element.text.strip() if element else ''
    
    def _extract_rating(self, soup):
        """Extract app rating"""
        element = soup.select_one('[itemprop="ratingValue"]')
        return float(element['content']) if element else 0.0
    
    def _extract_reviews_count(self, soup):
        """Extract number of reviews"""
        element = soup.select_one('[itemprop="reviewCount"]')
        return int(element['content'].replace(',', '')) if element else 0
    
    def _extract_size(self, soup):
        """Extract app size"""
        element = soup.select_one('div:contains("Size")')
        if element:
            size_text = element.find_next('div').text.strip()
            return self._parse_size(size_text)
        return 0
    
    def _extract_price(self, soup):
        """Extract app price"""
        element = soup.select_one('[itemprop="price"]')
        if element:
            price_text = element['content']
            return float(price_text.replace('$', '')) if price_text != 'Free' else 0.0
        return 0.0
    
    def _extract_downloads(self, soup):
        """Extract number of downloads"""
        element = soup.select_one('div:contains("Downloads")')
        if element:
            downloads_text = element.find_next('div').text.strip()
            return self._parse_downloads(downloads_text)
        return 0
    
    def _parse_size(self, size_text):
        """Convert size text to MB"""
        try:
            number = float(''.join(filter(str.isdigit, size_text)))
            if 'GB' in size_text:
                return number * 1024
            return number
        except:
            return 0
    
    def _parse_downloads(self, downloads_text):
        """Convert downloads text to number"""
        try:
            number = float(''.join(filter(str.isdigit, downloads_text)))
            if 'M' in downloads_text:
                return int(number * 1_000_000)
            if 'K' in downloads_text:
                return int(number * 1_000)
            return int(number)
        except:
            return 0

class AppleAppStoreAPI:
    """Apple App Store API wrapper"""
    
    def __init__(self):
        self.base_url = "https://itunes.apple.com/lookup"
        self.search_url = "https://itunes.apple.com/search"
    
    def get_app_details(self, app_id):
        """Get app details using iTunes API"""
        try:
            params = {
                'id': app_id,
                'entity': 'software'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['resultCount'] > 0:
                result = data['results'][0]
                
                app_data = {
                    'app_id': str(result['trackId']),
                    'name': result['trackName'],
                    'category': result['primaryGenreName'],
                    'rating': result.get('averageUserRating', 0),
                    'reviews': result.get('userRatingCount', 0),
                    'size': result['fileSizeBytes'] / (1024 * 1024),  # Convert to MB
                    'price': result['price'],
                    'downloads': 0  # Apple doesn't provide download counts
                }
                
                return app_data
                
            return None
            
        except Exception as e:
            print(f"Error fetching app details for {app_id}: {str(e)}")
            return None
    
    def get_app_reviews(self, app_id, limit=100):
        """Get app reviews using web scraping (since iTunes API doesn't provide reviews)"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            url = f"https://apps.apple.com/us/app/id{app_id}"
            driver.get(url)
            
            # Wait for reviews section
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.we-customer-review'))
            )
            
            reviews = []
            review_elements = driver.find_elements(By.CSS_SELECTOR, '.we-customer-review')
            
            for element in review_elements[:limit]:
                review = {
                    'text': element.find_element(By.CSS_SELECTOR, '.we-customer-review__body').text,
                    'rating': len(element.find_elements(By.CSS_SELECTOR, '.we-star-rating-stars-outlines')),
                    'date': element.find_element(By.CSS_SELECTOR, '.we-customer-review__date').text
                }
                reviews.append(review)
            
            return reviews
            
        except Exception as e:
            print(f"Error fetching reviews for {app_id}: {str(e)}")
            return []
        
        finally:
            if 'driver' in locals():
                driver.quit()

class AmazonAppStoreAPI:
    """Amazon App Store API wrapper"""
    
    def __init__(self):
        self.base_url = "https://www.amazon.com/gp/product"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def get_app_details(self, app_id):
        """Get app details using web scraping"""
        url = f"{self.base_url}/{app_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            app_data = {
                'app_id': app_id,
                'name': self._extract_text(soup, '#productTitle'),
                'category': self._extract_category(soup),
                'rating': self._extract_rating(soup),
                'reviews': self._extract_reviews_count(soup),
                'size': self._extract_size(soup),
                'price': self._extract_price(soup),
                'downloads': self._extract_downloads(soup)
            }
            
            return app_data
            
        except Exception as e:
            print(f"Error fetching app details for {app_id}: {str(e)}")
            return None
    
    def get_app_reviews(self, app_id, limit=100):
        """Get app reviews using web scraping"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            url = f"{self.base_url}/{app_id}/reviews"
            driver.get(url)
            
            # Wait for reviews to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-hook="review"]'))
            )
            
            reviews = []
            review_elements = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
            
            for element in review_elements[:limit]:
                review = {
                    'text': element.find_element(By.CSS_SELECTOR, '[data-hook="review-body"]').text,
                    'rating': float(element.find_element(By.CSS_SELECTOR, '[data-hook="review-star-rating"]')
                                 .get_attribute('textContent').split('.')[0]),
                    'date': element.find_element(By.CSS_SELECTOR, '[data-hook="review-date"]').text
                }
                reviews.append(review)
            
            return reviews
            
        except Exception as e:
            print(f"Error fetching reviews for {app_id}: {str(e)}")
            return []
        
        finally:
            if 'driver' in locals():
                driver.quit()
    
    def _extract_text(self, soup, selector):
        """Helper method to extract text from HTML elements"""
        element = soup.select_one(selector)
        return element.text.strip() if element else ''
    
    def _extract_category(self, soup):
        """Extract app category"""
        element = soup.select_one('#wayfinding-breadcrumbs_container')
        if element:
            categories = element.find_all('a')
            return categories[-1].text.strip() if categories else ''
        return ''
    
    def _extract_rating(self, soup):
        """Extract app rating"""
        element = soup.select_one('[data-hook="rating-out-of-text"]')
        if element:
            rating_text = element.text.split(' ')[0]
            return float(rating_text)
        return 0.0
    
    def _extract_reviews_count(self, soup):
        """Extract number of reviews"""
        element = soup.select_one('[data-hook="total-review-count"]')
        if element:
            return int(element.text.replace(',', ''))
        return 0
    
    def _extract_size(self, soup):
        """Extract app size"""
        element = soup.find('td', text='File Size')
        if element:
            size_text = element.find_next('td').text.strip()
            return self._parse_size(size_text)
        return 0
    
    def _extract_price(self, soup):
        """Extract app price"""
        element = soup.select_one('.a-price-whole')
        return float(element.text.replace('$', '')) if element else 0.0
    
    def _extract_downloads(self, soup):
        """Extract number of downloads"""
        # Amazon doesn't typically show download counts
        return 0
    
    def _parse_size(self, size_text):
        """Convert size text to MB"""
        try:
            number = float(''.join(filter(str.isdigit, size_text)))
            if 'GB' in size_text:
                return number * 1024
            return number
        except:
            return 0
