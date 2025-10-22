#!/usr/bin/env python3
"""
DEBUGGED CODE - Fixed version of the buggy TensorFlow script
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def create_fixed_cnn():
    """
    Fixed CNN implementation
    """
    model = models.Sequential()
    
    # FIX 1: Correct input shape for MNIST (28,28,1)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    
    # FIX 2: Added padding to maintain dimensions
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    # FIX 3: Reasonable dropout rate
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    
    # FIX 4: Appropriate dense layer dimension
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # FIX 5: Correct activation for multi-class classification
    model.add(layers.Dense(10, activation='softmax'))
    
    # FIX 6: Correct loss function for multi-class classification
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def fixed_data_preprocessing(x_train, y_train, x_test, y_test):
    """
    Fixed data preprocessing function
    """
    # FIX 7: Correct normalization
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # FIX 8: Consistent preprocessing for test data
    # x_test already normalized same as x_train
    
    # FIX 9: Correct reshaping for MNIST
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # FIX 10: Correct one-hot encoding with 10 classes
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    return x_train, y_train, x_test, y_test

def fixed_training_loop(model, x_train, y_train, x_val, y_val):
    """
    Fixed training loop
    """
    # FIX 11: Reasonable batch size
    batch_size = 128
    
    # FIX 12: Appropriate number of epochs
    epochs = 10
    
    # FIX 13: Include validation data and early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, restore_best_weights=True
    )
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1)
    
    return history

def debug_explanation():
    """
    Explanation of all bugs fixed
    """
    bugs_fixed = [
        {
            'bug': 'Incorrect input shape (32,32,3) for MNIST',
            'fix': 'Changed to (28,28,1)',
            'impact': 'Dimension mismatch would prevent model from working'
        },
        {
            'bug': 'Missing padding in convolutional layers',
            'fix': 'Added padding="same"',
            'impact': 'Prevents dimension reduction and information loss'
        },
        {
            'bug': 'Too aggressive dropout (0.8)',
            'fix': 'Reduced to 0.25 and 0.5',
            'impact': 'Prevents underfitting and allows model to learn'
        },
        {
            'bug': 'Wrong activation in output layer (sigmoid)',
            'fix': 'Changed to softmax',
            'impact': 'Sigmoid is for binary classification, softmax for multi-class'
        },
        {
            'bug': 'Incorrect loss function (binary_crossentropy)',
            'fix': 'Changed to categorical_crossentropy',
            'impact': 'Binary crossentropy is for binary classification'
        },
        {
            'bug': 'Inconsistent data preprocessing',
            'fix': 'Standardized normalization to /255.0',
            'impact': 'Ensures consistent feature scaling'
        },
        {
            'bug': 'Incorrect reshaping dimensions',
            'fix': 'Changed to (28,28,1) for MNIST',
            'impact': 'Fixes dimension mismatches'
        },
        {
            'bug': 'Wrong number of classes in one-hot encoding',
            'fix': 'Changed from 5 to 10 classes',
            'impact': 'MNIST has 10 digit classes (0-9)'
        },
        {
            'bug': 'Too large batch size (2048)',
            'fix': 'Reduced to 128',
            'impact': 'Prevents memory issues and allows better convergence'
        },
        {
            'bug': 'Missing validation data in training',
            'fix': 'Added validation_data parameter',
            'impact': 'Enables proper model evaluation and prevents overfitting'
        }
    ]
    
    return bugs_fixed

# Fixed main execution
if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Process data correctly
    x_train, y_train, x_test, y_test = fixed_data_preprocessing(x_train, y_train, x_test, y_test)
    
    # Create fixed model
    model = create_fixed_cnn()
    
    # Train model correctly
    history = fixed_training_loop(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Print debugging explanation
    print("\n" + "="*50)
    print("DEBUGGING EXPLANATION")
    print("="*50)
    fixes = debug_explanation()
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['bug']}")
        print(f"   Fix: {fix['fix']}")
        print(f"   Impact: {fix['impact']}")