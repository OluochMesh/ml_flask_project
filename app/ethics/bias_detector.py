#!/usr/bin/env python3
"""
Bias detection for MNIST and Amazon Reviews models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

class BiasDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_mnist_biases(self, model, test_data, test_labels):
        """
        Analyze potential biases in MNIST model
        """
        biases = {}
        
        try:
            # Analyze performance across different digit classes
            predictions = model.predict(test_data)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(test_labels, axis=1)
            
            # Calculate accuracy per digit
            digit_accuracy = {}
            for digit in range(10):
                digit_mask = true_classes == digit
                if np.sum(digit_mask) > 0:
                    accuracy = np.mean(predicted_classes[digit_mask] == digit)
                    digit_accuracy[digit] = accuracy
            
            biases['digit_accuracy_disparity'] = {
                'max_accuracy': max(digit_accuracy.values()),
                'min_accuracy': min(digit_accuracy.values()),
                'accuracy_range': max(digit_accuracy.values()) - min(digit_accuracy.values()),
                'per_digit_accuracy': digit_accuracy
            }
            
            # Check for systematic misclassifications
            confusion = {}
            for true_digit in range(10):
                true_mask = true_classes == true_digit
                pred_for_true = predicted_classes[true_mask]
                confusion[true_digit] = {
                    str(pred_digit): np.sum(pred_for_true == pred_digit) 
                    for pred_digit in range(10)
                }
            
            biases['confusion_patterns'] = confusion
            
            self.logger.info("MNIST bias analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in MNIST bias analysis: {e}")
            biases['error'] = str(e)
        
        return biases
    
    def analyze_amazon_biases(self, reviews_df, analysis_results):
        """
        Analyze potential biases in Amazon reviews sentiment analysis
        """
        biases = {}
        
        try:
            # Analyze sentiment distribution by review length
            reviews_df['text_length'] = reviews_df['text'].str.len()
            reviews_df['word_count'] = reviews_df['text'].str.split().str.len()
            
            # Check if sentiment correlates with text length
            length_bias = reviews_df.groupby('sentiment').agg({
                'text_length': ['mean', 'std'],
                'word_count': ['mean', 'std']
            }).to_dict()
            
            biases['length_correlation'] = length_bias
            
            # Analyze entity extraction biases
            entity_biases = {}
            for result in analysis_results:
                sentiment = result['sentiment']['final_sentiment']
                entities = result['entity_summary']['total_entities']
                
                if sentiment not in entity_biases:
                    entity_biases[sentiment] = []
                entity_biases[sentiment].append(entities)
            
            # Calculate average entities per sentiment
            entity_stats = {}
            for sentiment, counts in entity_biases.items():
                entity_stats[sentiment] = {
                    'mean_entities': np.mean(counts),
                    'std_entities': np.std(counts),
                    'count': len(counts)
                }
            
            biases['entity_extraction_bias'] = entity_stats
            
            # Check for demographic term biases
            demographic_terms = {
                'gender_terms': ['he', 'she', 'him', 'her', 'his', 'hers', 'man', 'woman', 'boy', 'girl'],
                'age_terms': ['young', 'old', 'teen', 'senior', 'child', 'adult'],
                'location_terms': ['US', 'UK', 'Europe', 'Asia', 'America', 'China', 'India']
            }
            
            term_biases = {}
            for category, terms in demographic_terms.items():
                term_biases[category] = {}
                for term in terms:
                    term_count = reviews_df['text'].str.contains(term, case=False).sum()
                    term_biases[category][term] = term_count
            
            biases['demographic_term_frequency'] = term_biases
            
            self.logger.info("Amazon reviews bias analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in Amazon reviews bias analysis: {e}")
            biases['error'] = str(e)
        
        return biases
    
    def generate_fairness_report(self, mnist_biases=None, amazon_biases=None):
        """
        Generate comprehensive fairness report
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': {},
            'recommendations': []
        }
        
        if mnist_biases:
            # Analyze MNIST fairness
            accuracy_range = mnist_biases['digit_accuracy_disparity']['accuracy_range']
            if accuracy_range > 0.1:
                report['summary']['mnist_fairness'] = 'POOR'
                report['recommendations'].append(
                    "MNIST model shows significant accuracy disparity between digits. "
                    "Consider data augmentation for underrepresented digits."
                )
            elif accuracy_range > 0.05:
                report['summary']['mnist_fairness'] = 'MODERATE'
                report['recommendations'].append(
                    "MNIST model shows moderate accuracy disparity. "
                    "Monitor performance across all digits."
                )
            else:
                report['summary']['mnist_fairness'] = 'GOOD'
            
            report['mnist_details'] = mnist_biases
        
        if amazon_biases:
            # Analyze Amazon reviews fairness
            if 'entity_extraction_bias' in amazon_biases:
                entity_bias = amazon_biases['entity_extraction_bias']
                max_diff = max([v['mean_entities'] for v in entity_bias.values()]) - \
                          min([v['mean_entities'] for v in entity_bias.values()])
                
                if max_diff > 2:
                    report['summary']['nlp_fairness'] = 'POOR'
                    report['recommendations'].append(
                        "Significant entity extraction bias detected across sentiments. "
                        "Review NER model training data for balance."
                    )
                else:
                    report['summary']['nlp_fairness'] = 'GOOD'
            
            report['amazon_details'] = amazon_biases
        
        # General recommendations
        report['recommendations'].extend([
            "Use TensorFlow Fairness Indicators for continuous monitoring",
            "Implement spaCy's rule-based systems for demographic term handling",
            "Regularly audit training data for representation balance",
            "Consider adversarial debiasing techniques"
        ])
        
        return report