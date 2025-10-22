"""
Test script for NLP Task 3
Tests Named Entity Recognition and Sentiment Analysis
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.task3_nlp.text_processor import TextProcessor
from app.models.task3_nlp.ner_extraction import NERExtractor
from app.models.task3_nlp.sentiment_analysis import SentimentAnalyzer

def print_separator():
    print("\n" + "="*70 + "\n")

def test_ner():
    """Test Named Entity Recognition"""
    print("="*70)
    print("TESTING NAMED ENTITY RECOGNITION (NER)")
    print("="*70)
    
    # Initialize NER extractor
    print("\nInitializing NER Extractor...")
    ner = NERExtractor()
    print("✓ NER Extractor initialized successfully!")
    
    # Test texts
    test_reviews = [
        "I love my new Samsung Galaxy S21! The camera quality is amazing and the battery lasts all day.",
        "The Apple MacBook Pro is overpriced. Dell laptops offer better value for money.",
        "Just bought Sony headphones from Amazon. Great sound quality but expensive."
    ]
    
    print("\nTest Reviews:")
    for i, review in enumerate(test_reviews, 1):
        print(f"\n{i}. {review}")
    
    print_separator()
    
    # Extract entities from each review
    for i, review in enumerate(test_reviews, 1):
        print(f"Review {i} Analysis:")
        print(f"Text: {review}\n")
        
        entities = ner.extract_entities(review)
        summary = ner.get_entity_summary(entities)
        
        print(f"Total Entities Found: {summary['total_entities']}")
        print(f"\nProducts: {summary['unique_products']}")
        print(f"Brands: {summary['unique_brands']}")
        
        if entities['ORG']:
            print(f"Organizations: {[e['text'] for e in entities['ORG']]}")
        
        print_separator()

def test_sentiment():
    """Test Sentiment Analysis"""
    print("="*70)
    print("TESTING SENTIMENT ANALYSIS")
    print("="*70)
    
    # Initialize sentiment analyzer
    print("\nInitializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    print("✓ Sentiment Analyzer initialized successfully!")
    
    # Test reviews with different sentiments
    test_reviews = {
        'positive': "This is an excellent product! I absolutely love it. Best purchase ever!",
        'negative': "Terrible quality. Complete waste of money. Very disappointed.",
        'neutral': "The product works as expected. Nothing special but does the job."
    }
    
    print_separator()
    
    for label, review in test_reviews.items():
        print(f"Expected Sentiment: {label.upper()}")
        print(f"Text: {review}\n")
        
        result = analyzer.analyze_sentiment(review)
        
        print(f"Detected Sentiment: {result['sentiment'].upper()} {result['emoji']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Polarity: {result['polarity']:.3f}")
        print(f"Subjectivity: {result['subjectivity']:.3f}")
        
        print(f"\nPositive Keywords: {result['details']['positive_words']}")
        print(f"Negative Keywords: {result['details']['negative_words']}")
        
        print_separator()

def test_combined():
    """Test combined NER and Sentiment Analysis"""
    print("="*70)
    print("TESTING COMBINED NER + SENTIMENT ANALYSIS")
    print("="*70)
    
    # Initialize both
    ner = NERExtractor()
    analyzer = SentimentAnalyzer()
    
    # Sample Amazon review
    review = """
    I recently purchased the Samsung Galaxy S22 from Amazon and I'm absolutely thrilled! 
    The camera quality is outstanding, battery life is excellent, and the display is gorgeous. 
    Compared to my old Apple iPhone, this is a huge upgrade. Samsung really knocked it out 
    of the park with this one. Highly recommended!
    """
    
    print("\nSample Review:")
    print(review)
    
    print_separator()
    
    # Extract entities
    print("NAMED ENTITIES:")
    entities = ner.extract_entities(review)
    summary = ner.get_entity_summary(entities)
    
    print(f"\n✓ Products Found: {summary['unique_products']}")
    print(f"✓ Brands Found: {summary['unique_brands']}")
    
    # Analyze sentiment
    print("\nSENTIMENT ANALYSIS:")
    sentiment = analyzer.analyze_sentiment(review)
    
    print(f"\n✓ Sentiment: {sentiment['sentiment'].upper()} {sentiment['emoji']}")
    print(f"✓ Score: {sentiment['score']:.3f}")
    print(f"✓ Confidence: {sentiment['confidence']:.3f}")
    print(f"✓ Positive Words: {sentiment['details']['positive_words']}")
    print(f"✓ Negative Words: {sentiment['details']['negative_words']}")
    
    print_separator()

def main():
    """Main test function"""
    print("\n")
    print("="*70)
    print(" "*15 + "TASK 3: NLP ANALYSIS TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: NER
        test_ner()
        
        # Test 2: Sentiment
        test_sentiment()
        
        # Test 3: Combined
        test_combined()
        
        print("="*70)
        print(" "*20 + "ALL TESTS COMPLETED!")
        print("="*70)
        print("\n✓ NER Extraction: WORKING")
        print("✓ Sentiment Analysis: WORKING")
        print("✓ Combined Analysis: WORKING")
        print("\nTask 3 is ready to use in the Flask application!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()