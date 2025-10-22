#!/usr/bin/env python3
"""
Test script for NLP components
"""

import os
import sys
import spacy

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.models.task3_nlp.text_processor import TextProcessor
    from app.models.task3_nlp.ner_extraction import NERExtractor
    from app.models.task3_nlp.sentiment import SentimentAnalyzer
    from app.models.task3_nlp import NLPAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all NLP components are properly set up")
    sys.exit(1)

def test_nlp_components():
    """Test all NLP components"""
    print("🧪 TESTING NLP COMPONENTS")
    print("=" * 50)
    
    # Test data from your screenshot
    test_text = """My lovely Pat has one of the GREAT voices of her generation. 
    I have listened to this CD for YEARS and I still LOVE it. 
    When I'm in a good mood it makes me feel better. 
    A bad mood just evaporates like sugar in the rain. 
    This CD just oozes LIFE. Vocals are jugs! STUNNING and lyrics just kill. 
    One of life's hidden gems. This is a desert is the CD in my book. 
    Why she never made it big is just beyond me. 
    EVERYONE I play this, no matter black, white, young, old, male, female 
    EVERYBODY says one thing "Who was that singing"?"""
    
    # Test 1: Text Processor
    print("\n1. Testing Text Processor...")
    try:
        text_processor = TextProcessor()
        cleaned_text = text_processor.clean_text(test_text)
        print(f"   ✓ TextProcessor initialized")
        print(f"   ✓ Original length: {len(test_text)}")
        print(f"   ✓ Cleaned length: {len(cleaned_text)}")
    except Exception as e:
        print(f"   ✗ TextProcessor failed: {e}")
        return False
    
    # Test 2: NER Extractor
    print("\n2. Testing NER Extractor...")
    try:
        ner_extractor = NERExtractor(text_processor.nlp)
        entities = ner_extractor.extract_entities(test_text)
        entity_summary = ner_extractor.get_entity_summary(entities)
        print(f"   ✓ NER Extractor initialized")
        print(f"   ✓ Entities found: {entity_summary['total_entities']}")
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_texts = [e['text'] for e in entity_list]
                print(f"     - {entity_type}: {entity_texts}")
    except Exception as e:
        print(f"   ✗ NER Extractor failed: {e}")
        return False
    
    # Test 3: Sentiment Analyzer
    print("\n3. Testing Sentiment Analyzer...")
    try:
        sentiment_analyzer = SentimentAnalyzer()
        sentiment = sentiment_analyzer.rule_based_sentiment(test_text)
        print(f"   ✓ Sentiment Analyzer initialized")
        print(f"   ✓ Sentiment: {sentiment['sentiment']}")
        print(f"   ✓ Score: {sentiment['score']:.3f}")
        print(f"   ✓ Positive words: {sentiment['positive_words']}")
        print(f"   ✓ Negative words: {sentiment['negative_words']}")
    except Exception as e:
        print(f"   ✗ Sentiment Analyzer failed: {e}")
        return False
    
    # Test 4: Complete NLP Analyzer
    print("\n4. Testing Complete NLP Analyzer...")
    try:
        nlp_analyzer = NLPAnalyzer()
        analysis = nlp_analyzer.analyze_text(test_text)
        print(f"   ✓ NLP Analyzer initialized")
        print(f"   ✓ Final sentiment: {analysis['sentiment']['final_sentiment']}")
        print(f"   ✓ Total entities: {analysis['entity_summary']['total_entities']}")
        print(f"   ✓ Text length: {analysis['analysis_metadata']['text_length']}")
    except Exception as e:
        print(f"   ✗ NLP Analyzer failed: {e}")
        return False
    
    # Test 5: Sample data analysis
    print("\n5. Testing Sample Data Analysis...")
    try:
        sample_file = 'data/processed/amazon_reviews/combined_sample.csv'
        if os.path.exists(sample_file):
            sample_df = pd.read_csv(sample_file)
            sample_text = sample_df.iloc[0]['text'] if len(sample_df) > 0 else test_text
            sample_analysis = nlp_analyzer.analyze_text(sample_text)
            print(f"   ✓ Sample data analysis completed")
            print(f"   ✓ Sample sentiment: {sample_analysis['sentiment']['final_sentiment']}")
        else:
            print(f"   ⚠ Sample file not found, using test text")
    except Exception as e:
        print(f"   ✗ Sample data analysis failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 NLP COMPONENTS TESTS PASSED!")
    return True

if __name__ == "__main__":
    test_nlp_components()