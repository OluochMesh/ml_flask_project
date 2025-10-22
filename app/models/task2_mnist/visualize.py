import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class MNISTVisualizer:
    """
    Visualization utilities for MNIST predictions and results
    """
    
    def __init__(self):
        self.class_names = [str(i) for i in range(10)]
    
    def plot_sample_predictions(self, x_test, y_test, predictions, num_samples=5, save_path=None):
        """
        Plot sample predictions with true and predicted labels
        
        Args:
            x_test: Test images
            y_test: True labels
            predictions: Model predictions
            num_samples: Number of samples to display
            save_path: Path to save the plot
        """
        # Convert predictions to class labels
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        # Convert y_test if it's one-hot encoded
        if len(y_test.shape) > 1:
            true_classes = np.argmax(y_test, axis=1)
        else:
            true_classes = y_test
        
        plt.figure(figsize=(15, 8))
        
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            
            # Plot image
            plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            
            # Determine title color (green for correct, red for wrong)
            title_color = 'green' if true_classes[i] == pred_classes[i] else 'red'
            
            plt.title(f'True: {true_classes[i]}, Pred: {pred_classes[i]}', 
                     color=title_color, fontsize=12, fontweight='bold')
            plt.axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample predictions plot saved to: {save_path}")
        
        return save_path
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        # Convert to class labels if one-hot encoded
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        return save_path
    
    def plot_probability_distribution(self, probabilities, true_label=None, save_path=None):
        """
        Plot probability distribution for a single prediction
        
        Args:
            probabilities: Probability distribution over classes
            true_label: True label (optional)
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(range(10), probabilities, color='skyblue', alpha=0.7)
        
        # Highlight predicted class
        predicted_class = np.argmax(probabilities)
        bars[predicted_class].set_color('red')
        
        # Highlight true class if provided
        if true_label is not None:
            bars[true_label].set_color('green')
        
        plt.xlabel('Digit Class', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(10))
        plt.grid(axis='y', alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, color='red', alpha=0.7, label='Predicted'),
            plt.Rectangle((0,0), 1, 1, color='green', alpha=0.7, label='True') if true_label is not None else None,
            plt.Rectangle((0,0), 1, 1, color='skyblue', alpha=0.7, label='Other')
        ]
        legend_elements = [elem for elem in legend_elements if elem is not None]
        plt.legend(handles=legend_elements)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Probability distribution plot saved to: {save_path}")
        
        return save_path
    
    def plot_feature_maps(self, model, image, layer_names=None, save_path=None):
        """
        Plot feature maps from convolutional layers
        
        Args:
            model: Trained CNN model
            image: Input image
            layer_names: Names of layers to visualize
            save_path: Path to save the plot
        """
        if layer_names is None:
            # Get first few convolutional layers
            layer_names = []
            for layer in model.layers:
                if 'conv' in layer.name:
                    layer_names.append(layer.name)
                if len(layer_names) >= 4:  # Limit to 4 layers
                    break
        
        # Create a model that outputs the feature maps
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations
        activations = activation_model.predict(image.reshape(1, 28, 28, 1), verbose=0)
        
        # Plot feature maps
        fig, axes = plt.subplots(len(layer_names), 8, figsize=(20, 3 * len(layer_names)))
        
        for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
            # Get number of feature maps in this layer
            n_features = layer_activation.shape[-1]
            
            # Plot first 8 feature maps
            for j in range(min(8, n_features)):
                if len(layer_names) > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                
                ax.imshow(layer_activation[0, :, :, j], cmap='viridis')
                ax.set_title(f'{layer_name}\nMap {j+1}')
                ax.axis('off')
        
        plt.suptitle('CNN Feature Maps Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature maps plot saved to: {save_path}")
        
        return save_path
    
    def generate_comprehensive_report(self, model, x_test, y_test, predictions, save_dir):
        """
        Generate comprehensive visualization report
        
        Returns:
            Dictionary with paths to all generated plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        plot_paths = {}
        
        # Sample predictions
        plot_paths['sample_predictions'] = self.plot_sample_predictions(
            x_test, y_test, predictions, 
            save_path=os.path.join(save_dir, 'sample_predictions.png')
        )
        
        # Confusion matrix
        plot_paths['confusion_matrix'] = self.plot_confusion_matrix(
            y_test, predictions,
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        return plot_paths