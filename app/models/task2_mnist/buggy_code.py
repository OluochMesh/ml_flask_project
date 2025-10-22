#!/usr/bin/env python3
"""
BUGGY CODE - TensorFlow script with intentional errors for debugging challenge
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def create_buggy_cnn():
    """
    Buggy CNN implementation with multiple errors
    """
    model = models.Sequential()
    
    # ERROR 1: Incorrect input shape for MNIST (28,28,1) but specified (32,32,3)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    # ERROR 2: Missing padding causing dimension issues
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    # ERROR 3: Too aggressive dropout causing underfitting
    model.add(layers.Dropout(0.8))
    
    model.add(layers.Flatten())
    
    # ERROR 4: Incorrect dimension for dense layer
    model.add(layers.Dense(128, activation='relu'))
    
    # ERROR 5: Wrong activation function for multi-class classification
    model.add(layers.Dense(10, activation='sigmoid'))
    
    # ERROR 6: Incorrect loss function for multi-class classification
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Should be categorical_crossentropy
                  metrics=['accuracy'])
    
    return model

def buggy_data_preprocessing(x_train, y_train, x_test, y_test):
    """
    Buggy data preprocessing function
    """
    # ERROR 7: Incorrect normalization range
    x_train = x_train / 255.0 * 2 - 1  # Should be simple division by 255.0
    
    # ERROR 8: Wrong test data preprocessing (inconsistent with training)
    x_test = x_test / 127.5 - 1
    
    # ERROR 9: Incorrect reshaping causing dimension mismatch
    x_train = x_train.reshape(-1, 32, 32, 3)  # MNIST is 28x28x1
    
    # ERROR 10: Wrong test data reshaping
    x_test = x_test.reshape(-1, 32, 32, 3)
    
    # ERROR 11: Incorrect one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)  # Should be 10 classes
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)
    
    return x_train, y_train, x_test, y_test

def buggy_training_loop(model, x_train, y_train, x_val, y_val):
    """
    Buggy training loop with issues
    """
    # ERROR 12: Wrong batch size causing memory issues
    batch_size = 2048  # Too large for typical systems
    
    # ERROR 13: Too many epochs causing overfitting
    epochs = 100
    
    # ERROR 14: Missing validation data in fit call
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    
    return history

# ERROR 15: Main execution block with dimension mismatches
if __name__ == "__main__":
    # This will fail due to multiple bugs
    from tensorflow.keras.datasets import mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Process data with bugs
    x_train, y_train, x_test, y_test = buggy_data_preprocessing(x_train, y_train, x_test, y_test)
    
    # Create buggy model
    model = create_buggy_cnn()
    
    # Try training with bugs
    history = buggy_training_loop(model, x_train, y_train, x_test, y_test)