import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

def clean_text(text):
    """Clean and normalize text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def get_sentiment_score(text):
    """Calculate sentiment score for text using VADER."""
    sia = SentimentIntensityAnalyzer()
    
    try:
        sentiment_scores = sia.polarity_scores(text)
        return sentiment_scores['compound']  # Returns a score between -1 and 1
    except Exception as e:
        print(f"Error calculating sentiment: {str(e)}")
        return 0.0

def extract_keywords(text, top_n=5):
    """Extract most important keywords from text."""
    # Clean text
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Get frequency distribution
    freq_dist = nltk.FreqDist(tokens)
    
    # Return top N keywords
    return [word for word, _ in freq_dist.most_common(top_n)]
