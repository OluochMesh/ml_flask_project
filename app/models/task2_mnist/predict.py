import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import json

class MNISTPredictor:
    """
    Predictor for MNIST handwritten digit classification
    """
    
    def __init__(self, model_path=None, metadata_path=None):
        self.model = None
        self.metadata = None
        self.img_height = 28
        self.img_width = 28
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, metadata_path)
    
    def load_model(self, model_path, metadata_path=None):
        """
        Load trained model and metadata
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Load metadata if available
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print("Metadata loaded successfully")
            else:
                self.metadata = {
                    'model_type': 'CNN',
                    'input_shape': (28, 28, 1),
                    'num_classes': 10
                }
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for prediction
        
        Args:
            image: Can be file path, numpy array, or PIL Image
            
        Returns:
            Preprocessed image ready for prediction
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # Load from file path
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                img = Image.open(image).convert('L')  # Convert to grayscale
            elif isinstance(image, np.ndarray):
                # Handle numpy array
                if len(image.shape) == 3 and image.shape[2] == 3:
                    img = Image.fromarray(image).convert('L')
                else:
                    img = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                # Already PIL Image
                img = image.convert('L')
            else:
                raise ValueError("Unsupported image format")
            
            # Resize to 28x28
            img = img.resize((self.img_width, self.img_height))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize pixel values
            img_array = img_array.astype('float32') / 255.0
            
            # Invert colors if background is white (MNIST has black background)
            if np.mean(img_array) > 0.5:  # If background is bright
                img_array = 1.0 - img_array
            
            # Reshape for model input
            img_array = img_array.reshape(1, self.img_height, self.img_width, 1)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def predict_single(self, image):
        """
        Predict digit for a single image
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Get probabilities for all classes
            probabilities = {
                str(i): float(prediction[0][i]) for i in range(10)
            }
            
            result = {
                'predicted_digit': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': probabilities,
                'all_predictions': prediction[0].tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, images):
        """
        Predict digits for multiple images
        
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")
        
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.predict_single(image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'predicted_digit': -1,
                    'confidence': 0.0,
                    'error': str(e),
                    'image_index': i
                })
        
        return results
    
    def validate_image(self, image):
        """
        Validate if image is suitable for prediction
        
        Returns:
            (is_valid, error_message)
        """
        try:
            processed = self.preprocess_image(image)
            
            # Check if image has reasonable content (not all black/white)
            if np.std(processed) < 0.01:
                return False, "Image appears to be blank or has low contrast"
            
            return True, "Image is valid"
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            'model_type': self.metadata.get('model_type', 'CNN'),
            'input_shape': self.metadata.get('input_shape', (28, 28, 1)),
            'num_classes': self.metadata.get('num_classes', 10),
            'training_date': self.metadata.get('training_date', 'Unknown'),
            'parameters': self.metadata.get('total_parameters', 'Unknown')
        }
        
        return info