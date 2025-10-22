import numpy as np
import joblib

class IrisPredictor:
    """
    Predictor for Iris Classification
    """
    
    def __init__(self, model_path=None, label_mapping=None):
        self.model = None
        self.label_mapping = label_mapping
        self.reverse_mapping = None
        
        if label_mapping:
            self.reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load trained model
        
        Args:
            model_path: Path to saved model
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
    
    def predict_single(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Predict species for a single iris flower
        
        Args:
            sepal_length: Sepal length in cm
            sepal_width: Sepal width in cm
            petal_length: Petal length in cm
            petal_width: Petal width in cm
            
        Returns:
            Dictionary with prediction and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")
        
        # Prepare input
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get species name
        species_name = self.reverse_mapping.get(prediction, f"Class {prediction}")
        
        # Create probability dictionary
        prob_dict = {}
        if self.reverse_mapping:
            for class_idx, prob in enumerate(probabilities):
                class_name = self.reverse_mapping.get(class_idx, f"Class {class_idx}")
                prob_dict[class_name] = float(prob)
        
        result = {
            'predicted_class': int(prediction),
            'predicted_species': species_name,
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'input_features': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        }
        
        return result
    
    def predict_batch(self, features_list):
        """
        Predict species for multiple flowers
        
        Args:
            features_list: List of feature arrays or list of dicts
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")
        
        results = []
        
        for features in features_list:
            if isinstance(features, dict):
                # If features is a dictionary
                result = self.predict_single(
                    features['sepal_length'],
                    features['sepal_width'],
                    features['petal_length'],
                    features['petal_width']
                )
            else:
                # If features is an array
                result = self.predict_single(*features)
            
            results.append(result)
        
        return results
    
    def validate_input(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Validate input features
        
        Args:
            sepal_length, sepal_width, petal_length, petal_width: Input features
            
        Returns:
            Tuple (is_valid, error_message)
        """
        errors = []
        
        # Check if values are numeric
        try:
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)
        except (ValueError, TypeError):
            return False, "All features must be numeric values"
        
        # Check reasonable ranges (based on Iris dataset)
        if not (4.0 <= sepal_length <= 8.0):
            errors.append("Sepal length should be between 4.0 and 8.0 cm")
        
        if not (2.0 <= sepal_width <= 5.0):
            errors.append("Sepal width should be between 2.0 and 5.0 cm")
        
        if not (1.0 <= petal_length <= 7.0):
            errors.append("Petal length should be between 1.0 and 7.0 cm")
        
        if not (0.1 <= petal_width <= 3.0):
            errors.append("Petal width should be between 0.1 and 3.0 cm")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, None