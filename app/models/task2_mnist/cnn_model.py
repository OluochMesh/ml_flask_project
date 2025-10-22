import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class MNISTCNNModel:
    """
    CNN model for MNIST handwritten digit classification
    """
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_basic_cnn(self):
        """
        Build a basic CNN architecture
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_advanced_cnn(self):
        """
        Build a more advanced CNN architecture
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, model=None, learning_rate=0.001):
        """
        Compile the model with appropriate optimizer and metrics
        
        Args:
            model: Keras model to compile (uses self.model if None)
            learning_rate: Learning rate for optimizer
        """
        if model is None:
            model = self.model
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # FIXED: Use metric objects instead of strings for precision and recall
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',  # String works fine for accuracy
                tf.keras.metrics.Precision(name='precision'),  # Use metric object
                tf.keras.metrics.Recall(name='recall')  # Use metric object
            ]
        )
        
        print("Model compiled successfully!")
        print(f"Optimizer: Adam (lr={learning_rate})")
        print("Loss: categorical_crossentropy")
        print("Metrics: accuracy, precision, recall")
        
        return model
    
    def get_model_summary(self, model=None):
        """
        Get model summary as string
        """
        if model is None:
            model = self.model
        
        import io
        import contextlib
        
        # Capture model summary
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                model.summary()
            summary_str = buf.getvalue()
        
        return summary_str
    
    def count_parameters(self, model=None):
        """
        Count total trainable parameters in the model
        """
        if model is None:
            model = self.model
        
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        return {
            'trainable': int(trainable_count),
            'non_trainable': int(non_trainable_count),
            'total': int(trainable_count + non_trainable_count)
        }