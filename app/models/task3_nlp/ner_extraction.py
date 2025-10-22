import re
import spacy
from typing import List, Dict, Any
from collections import defaultdict

class NERExtractor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        # Define entity types we're interested in for product reviews
        self.relevant_entities = ['PRODUCT', 'ORG', 'GPE', 'PERSON', 'MONEY']
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        
        entities = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ in self.relevant_entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Also look for potential product mentions using custom patterns
        product_mentions = self._extract_product_mentions(text)
        if product_mentions:
            entities['CUSTOM_PRODUCT'] = product_mentions
        
        return dict(entities)
    
    def _extract_product_mentions(self, text: str) -> List[Dict]:
        """Extract potential product mentions using patterns"""
        patterns = [
            r'\b(CD|DVD|album|song|book|novel|movie|film)\b',
            r'\b[iI]pod|[iI]phone|[iI]pad|MacBook|Android\b',
            r'\b\d+GB|\d+MB|\d+TB\b',
            r'\b\d+\s*inch|\d+"\b'
        ]
        
        mentions = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                mentions.append({
                    'text': match.group(),
                    'label': 'CUSTOM_PRODUCT',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return mentions
    
    def get_entity_summary(self, entities: Dict) -> Dict[str, Any]:
        """Generate summary of extracted entities"""
        summary = {
            'total_entities': 0,
            'by_type': {},
            'most_common_products': [],
            'most_common_brands': []
        }
        
        for entity_type, entity_list in entities.items():
            summary['by_type'][entity_type] = len(entity_list)
            summary['total_entities'] += len(entity_list)
            
            # Count frequencies for products and brands
            if entity_type in ['PRODUCT', 'ORG', 'CUSTOM_PRODUCT']:
                entity_texts = [entity['text'] for entity in entity_list]
                from collections import Counter
                common = Counter(entity_texts).most_common(5)
                
                if entity_type in ['PRODUCT', 'CUSTOM_PRODUCT']:
                    summary['most_common_products'].extend(common)
                else:  # ORG (often brands)
                    summary['most_common_brands'].extend(common)
        
        return summary