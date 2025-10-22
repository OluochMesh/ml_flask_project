import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

class MNISTTrainer:
    """
    Trainer for MNIST CNN model
    """
    
    def __init__(self):
        self.history = None
        self.model = None
        self.training_time = None
    
    def setup_callbacks(self, checkpoint_path, logs_dir):
        """
        Setup training callbacks
        
        Returns:
            List of callbacks
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        callbacks = [
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_dir,
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model, train_generator, validation_generator, epochs=50, callbacks=None):
        """
        Train the CNN model
        
        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs...")
        start_time = datetime.now()
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = datetime.now() - start_time
        self.history = history
        self.model = model
        
        print(f"Training completed in {self.training_time}")
        return history
    
    def evaluate_model(self, model, test_generator):
        """
        Evaluate model on test set
        
        Args:
            model: Trained Keras model
            test_generator: Test data generator
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating model on test set...")
        
        # Evaluate using the generator
        results = model.evaluate(test_generator, verbose=1)
        
        # Get metric names
        metric_names = model.metrics_names
        
        # Create metrics dictionary
        metrics = {}
        for name, value in zip(metric_names, results):
            metrics[f'test_{name}'] = float(value)
        
        print("\nTest Results:")
        print(f"  Loss: {metrics['test_loss']:.4f}")
        print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
        if 'test_precision' in metrics:
            print(f"  Precision: {metrics['test_precision']:.4f}")
        if 'test_recall' in metrics:
            print(f"  Recall: {metrics['test_recall']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot precision
        ax3.plot(self.history.history['precision'], label='Training Precision')
        ax3.plot(self.history.history['val_precision'], label='Validation Precision')
        ax3.set_title('Model Precision')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True)
        
        # Plot recall
        ax4.plot(self.history.history['recall'], label='Training Recall')
        ax4.plot(self.history.history['val_recall'], label='Validation Recall')
        ax4.set_title('Model Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        return fig
    
    def save_model(self, model, model_path, metadata_path):
        """
        Save trained model and metadata
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'CNN',
            'input_shape': model.input_shape[1:],
            'num_classes': model.output_shape[1],
            'training_date': datetime.now().isoformat(),
            'training_time': str(self.training_time),
            'total_parameters': self.count_parameters(model),
            'test_accuracy': self.history.history['val_accuracy'][-1] if self.history else None
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
    
    def count_parameters(self, model):
        """
        Count trainable parameters
        """
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        return int(trainable_count)