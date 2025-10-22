import re
import pandas as pd
from typing import List, Dict, Tuple
import spacy

class TextProcessor:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise Exception("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;]', '', text)
        return text.strip()
    
    def load_amazon_data(self, file_path: str, sample_size: int = 1000) -> pd.DataFrame:
        """Load Amazon reviews data"""
        try:
            # Amazon reviews format: __label__X review text
            reviews = []
            labels = []
            texts = []
            
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i >= sample_size:
                        break
                    
                    # Parse label and text
                    if line.startswith('__label__'):
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            label = parts[0].replace('__label__', '')
                            text = parts[1].strip()
                            reviews.append({
                                'label': label,
                                'text': text,
                                'cleaned_text': self.clean_text(text)
                            })
            
            return pd.DataFrame(reviews)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()