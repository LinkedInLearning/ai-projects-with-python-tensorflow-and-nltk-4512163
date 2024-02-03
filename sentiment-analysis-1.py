# Importing necessary libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the vader_lexicon package
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    # Initialize the VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute and print the sentiment scores
    sentiment = sia.polarity_scores(text)
    print(sentiment)

# Test the function with a sample text
analyze_sentiment("NLTK is a great library for Natural Language Processing!")