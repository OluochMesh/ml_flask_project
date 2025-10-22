from .text_processor import TextProcessor
from .ner_extraction import NERExtractor
from .sentiment import SentimentAnalyzer
from typing import Dict, Any, List  # ADD THIS IMPORT

class NLPAnalyzer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.ner_extractor = NERExtractor(self.text_processor.nlp)
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Complete NLP analysis of a text"""
        cleaned_text = self.text_processor.clean_text(text)
        
        # Extract entities
        entities = self.ner_extractor.extract_entities(cleaned_text)
        entity_summary = self.ner_extractor.get_entity_summary(entities)
        
        # Analyze sentiment
        rule_sentiment = self.sentiment_analyzer.rule_based_sentiment(cleaned_text)
        textblob_sentiment = self.sentiment_analyzer.textblob_sentiment(cleaned_text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'entities': entities,
            'entity_summary': entity_summary,
            'sentiment': {
                'rule_based': rule_sentiment,
                'textblob': textblob_sentiment,
                'final_sentiment': self._combine_sentiments(rule_sentiment, textblob_sentiment)
            },
            'analysis_metadata': {
                'text_length': len(text),
                'cleaned_length': len(cleaned_text),
                'entity_count': entity_summary['total_entities']
            }
        }
    
    def _combine_sentiments(self, rule_sentiment: Dict, textblob_sentiment: Dict) -> str:
        """Combine results from both sentiment analysis methods"""
        # If both agree, use that
        if rule_sentiment['sentiment'] == textblob_sentiment['sentiment']:
            return rule_sentiment['sentiment']
        
        # If rule-based has more sentiment words, trust it more
        if rule_sentiment['total_sentiment_words'] >= 2:
            return rule_sentiment['sentiment']
        else:
            return textblob_sentiment['sentiment']
    
    def analyze_review_file(self, file_path: str, sample_size: int = 10) -> Dict[str, Any]:
        """Analyze multiple reviews from file"""
        df = self.text_processor.load_amazon_data(file_path, sample_size)
        
        if df.empty:
            return {'error': 'No data loaded'}
        
        analyses = []
        for _, review in df.iterrows():
            analysis = self.analyze_text(review['text'])
            analysis['label'] = review['label']
            analyses.append(analysis)
        
        return {
            'sample_analyses': analyses,
            'sample_size': len(analyses),
            'file_path': file_path
        }