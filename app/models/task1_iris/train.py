import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os
import logging
from datetime import datetime

class IrisModelTrainer:
    """
    Trainer for Iris classification models
    """
    
    def __init__(self, model_type='decision_tree', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.feature_importance = None
        
    def initialize_model(self, **kwargs):
        """Initialize the selected model type"""
        if self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                random_state=self.random_state,
                **kwargs
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=self.random_state,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        logging.info(f"Initialized {self.model_type} model")
    
    def get_hyperparameter_grid(self):
        """Get hyperparameter grid for GridSearchCV"""
        if self.model_type == 'decision_tree':
            return {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
    
    def train_with_hyperparameter_tuning(self, X_train, y_train, cv=5):
        """
        Train model with hyperparameter tuning using GridSearchCV
        (Compatibility method for your routes)
        """
        return self.train_with_grid_search(X_train, y_train, cv)
    
    def train_with_grid_search(self, X_train, y_train, cv=5):
        """Train model with hyperparameter tuning"""
        if self.model is None:
            self.initialize_model()
        
        param_grid = self.get_hyperparameter_grid()
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        logging.info(f"Starting GridSearchCV for {self.model_type}")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_scores = grid_search.cv_results_
        
        logging.info(f"Best parameters: {self.best_params}")
        logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def train_basic_model(self, X_train, y_train, **kwargs):
        """
        Train model without hyperparameter tuning
        (Compatibility method for your routes)
        """
        return self.train_simple(X_train, y_train, **kwargs)
    
    def train_simple(self, X_train, y_train, **kwargs):
        """Train model without hyperparameter tuning"""
        self.initialize_model(**kwargs)
        self.model.fit(X_train, y_train)
        
        logging.info(f"Trained {self.model_type} model with {len(X_train)} samples")
        return self.model
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            self.initialize_model()
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'cv_folds': cv
        }
        
        logging.info(f"Cross-validation scores: {cv_scores}")
        logging.info(f"Mean CV accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, feature_names):
        """Extract feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importance))
            
            logging.info("Feature importance calculated")
            return self.feature_importance
        else:
            logging.warning("Model does not support feature importance")
            return None
    
    def save_model(self, file_path):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().isoformat(),
            'model_class': self.model.__class__.__name__
        }
        
        metadata_path = file_path.replace('.pkl', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Model saved to: {file_path}")
        logging.info(f"Metadata saved to: {metadata_path}")
    
    def load_model(self, file_path):
        """Load trained model from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        self.model = joblib.load(file_path)
        
        metadata_path = file_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_type = metadata.get('model_type', 'decision_tree')
            self.best_params = metadata.get('best_params')
            self.feature_importance = metadata.get('feature_importance')
        
        logging.info(f"Model loaded from: {file_path}")
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.model is None:
            return {"status": "No model trained"}
        
        info = {
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'n_features': getattr(self.model, 'n_features_in_', 'Unknown')
        }
        
        return info