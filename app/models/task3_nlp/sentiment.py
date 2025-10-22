from typing import Dict, Any
import re
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        # Sentiment lexicon for rule-based approach
        self.positive_words = {
            'great', 'good', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'love', 'lovely', 'perfect', 'nice', 'best', 'beautiful',
            'brilliant', 'outstanding', 'superb', 'terrific', 'fabulous',
            'stunning', 'enjoy', 'enjoyed', 'happy', 'pleased', 'satisfied',
            'impressed', 'recommend', 'worth', 'oozes', 'kill', 'gem', 'big'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointing', 'poor', 'waste', 'rubbish', 'useless',
            'broken', 'damaged', 'defective', 'fake', 'scam', 'avoid',
            'problem', 'issues', 'complaint', 'sorry', 'regret'
        }
    
    def rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Rule-based sentiment analysis"""
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Calculate sentiment score
        total_words = positive_count + negative_count
        if total_words > 0:
            score = (positive_count - negative_count) / total_words
        else:
            score = 0
        
        # Determine sentiment label
        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Check for intensifiers and negations
        sentiment = self._apply_negation_rules(text, sentiment)
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'total_sentiment_words': total_words
        }
    
    def _apply_negation_rules(self, text: str, current_sentiment: str) -> str:
        """Apply negation rules to sentiment analysis"""
        negations = {'not', "n't", 'no', 'never', 'nothing'}
        
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in negations and i + 1 < len(words):
                next_word = words[i + 1]
                if (next_word in self.positive_words and current_sentiment == "positive"):
                    return "negative"
                elif (next_word in self.negative_words and current_sentiment == "negative"):
                    return "positive"
        
        return current_sentiment
    
    def textblob_sentiment(self, text: str) -> Dict[str, Any]:
        """Sentiment analysis using TextBlob as backup"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }