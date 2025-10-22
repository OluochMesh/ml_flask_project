import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class MNISTDataLoader:
    """
    Data loader for MNIST handwritten digits dataset
    """
    
    def __init__(self):
        self.img_height = 28
        self.img_width = 28
        self.num_classes = 10
        self.class_names = [str(i) for i in range(10)]
    
    def load_data(self):
        """
        Load MNIST dataset from TensorFlow
        
        Returns:
            (x_train, y_train), (x_test, y_test)
        """
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to include channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_data(self, x_train, y_train, x_test, y_test):
        """
        Preprocess the data for CNN training
        
        Returns:
            Preprocessed datasets
        """
        # One-hot encode labels
        y_train_encoded = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test_encoded = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        print("Data preprocessing completed:")
        print(f"Training labels encoded: {y_train_encoded.shape}")
        print(f"Test labels encoded: {y_test_encoded.shape}")
        
        return x_train, y_train_encoded, x_test, y_test_encoded
    
    def get_data_statistics(self, x_train, y_train, x_test, y_test):
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'training_samples': len(x_train),
            'test_samples': len(x_test),
            'image_shape': x_train[0].shape,
            'num_classes': self.num_classes,
            'class_distribution_train': np.bincount(y_train),
            'class_distribution_test': np.bincount(y_test),
            'pixel_value_range': {
                'min': float(x_train.min()),
                'max': float(x_train.max()),
                'mean': float(x_train.mean()),
                'std': float(x_train.std())
            }
        }
        return stats
    
    def visualize_samples(self, x_data, y_data, num_samples=10, save_path=None):
        """
        Visualize sample images from the dataset
        
        Args:
            x_data: Image data
            y_data: Labels
            num_samples: Number of samples to display
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 6))
        
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_data[i].reshape(28, 28), cmap='gray')
            plt.title(f'Label: {y_data[i]}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample visualization saved to: {save_path}")
        
        return save_path
    
    def create_data_generators(self, x_train, y_train, x_test, y_test, batch_size=32, validation_split=0.1):
        """
        Create data generators with augmentation for training
        
        Returns:
            train_generator, validation_generator, test_generator
        """
        from keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            validation_split=validation_split
        )
        
        # No augmentation for validation and test
        test_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=batch_size,
            subset='training'
        )
        
        validation_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=batch_size,
            subset='validation'
        )
        
        test_generator = test_datagen.flow(
            x_test, y_test,
            batch_size=batch_size,
            shuffle=False
        )
        
        print("Data generators created:")
        print(f"Training batches: {len(train_generator)}")
        print(f"Validation batches: {len(validation_generator)}")
        print(f"Test batches: {len(test_generator)}")
        
        return train_generator, validation_generator, test_generator