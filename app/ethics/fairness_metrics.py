#!/usr/bin/env python3
"""
Fairness metrics calculation inspired by TensorFlow Fairness Indicators
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Any

class FairnessMetrics:
    def __init__(self):
        pass
    
    def demographic_parity(self, predictions, sensitive_attributes):
        """
        Calculate demographic parity difference
        """
        groups = np.unique(sensitive_attributes)
        positive_rates = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            if np.sum(group_mask) > 0:
                positive_rates[group] = np.mean(predictions[group_mask])
        
        if len(positive_rates) < 2:
            return 0
        
        return max(positive_rates.values()) - min(positive_rates.values())
    
    def equal_opportunity(self, predictions, true_labels, sensitive_attributes):
        """
        Calculate equal opportunity difference
        """
        groups = np.unique(sensitive_attributes)
        true_positive_rates = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            true_positive_mask = (true_labels == 1) & group_mask
            
            if np.sum(true_positive_mask) > 0:
                true_positive_rates[group] = np.mean(predictions[true_positive_mask])
        
        if len(true_positive_rates) < 2:
            return 0
        
        return max(true_positive_rates.values()) - min(true_positive_rates.values())
    
    def accuracy_equality(self, predictions, true_labels, sensitive_attributes):
        """
        Calculate accuracy equality across groups
        """
        groups = np.unique(sensitive_attributes)
        accuracies = {}
        
        for group in groups:
            group_mask = sensitive_attributes == group
            if np.sum(group_mask) > 0:
                accuracies[group] = np.mean(predictions[group_mask] == true_labels[group_mask])
        
        if len(accuracies) < 2:
            return 0
        
        return max(accuracies.values()) - min(accuracies.values())
    
    def calculate_all_metrics(self, predictions, true_labels, sensitive_attributes):
        """
        Calculate all fairness metrics
        """
        return {
            'demographic_parity': self.demographic_parity(predictions, sensitive_attributes),
            'equal_opportunity': self.equal_opportunity(predictions, true_labels, sensitive_attributes),
            'accuracy_equality': self.accuracy_equality(predictions, true_labels, sensitive_attributes)
        }