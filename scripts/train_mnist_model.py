"""
Script to train the MNIST CNN model
Run this script to train the model before starting the Flask app
This avoids timeout issues during web training
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.task2_mnist.data_loader import MNISTDataLoader
from app.models.task2_mnist.cnn_model import MNISTCNNModel
from app.models.task2_mnist.train import MNISTTrainer
from app.models.task2_mnist.visualize import MNISTVisualizer
from config import config
from tensorflow.keras.callbacks import EarlyStopping

def main():
    """Main training function"""
    print("="*60)
    print("MNIST CNN MODEL TRAINING")
    print("="*60)
    
    # Load configuration
    cfg = config['development']
    
    # Paths
    model_path = cfg.MNIST_MODEL_PATH
    metadata_path = cfg.MNIST_METADATA_PATH
    figures_dir = os.path.join(cfg.RESULTS_DIR, 'figures', 'mnist')
    
    # Create directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n[1/5] Loading MNIST dataset...")
    data_loader = MNISTDataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    
    print("\n[2/5] Preprocessing data...")
    x_train, y_train_encoded, x_test, y_test_encoded = data_loader.preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # Create data generators with larger batch size for faster training
    train_gen, val_gen, test_gen = data_loader.create_data_generators(
        x_train, y_train_encoded, x_test, y_test_encoded,
        batch_size=128,
        validation_split=0.1
    )
    
    # Step 2: Build and compile model
    print("\n[3/5] Building CNN model...")
    model_builder = MNISTCNNModel()
    
    # Use basic model for faster training
    model = model_builder.build_basic_cnn()
    model = model_builder.compile_model(model, learning_rate=0.001)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Step 3: Train model
    print("\n[4/5] Training model...")
    trainer = MNISTTrainer()
    
    # Simple callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train with reasonable number of epochs
    epochs = 10
    print(f"\nTraining for {epochs} epochs...")
    
    history = trainer.train_model(
        model, train_gen, val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Step 4: Evaluate model
    print("\n[5/5] Evaluating model...")
    metrics = trainer.evaluate_model(model, test_gen)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    if 'test_precision' in metrics:
        print(f"Test Precision: {metrics['test_precision']:.4f}")
    if 'test_recall' in metrics:
        print(f"Test Recall: {metrics['test_recall']:.4f}")
    
    # Check if target accuracy is met
    if metrics['test_accuracy'] >= 0.95:
        print("\n✓ Target accuracy (>95%) ACHIEVED!")
    else:
        print(f"\n⚠ Target accuracy not met. Current: {metrics['test_accuracy']*100:.2f}%")
        print("  Consider training for more epochs or using advanced model.")
    
    # Step 5: Generate visualizations
    print("\n[6/6] Generating visualizations...")
    visualizer = MNISTVisualizer()
    
    # Get predictions
    test_predictions = model.predict(test_gen, verbose=0)
    
    # Generate comprehensive report
    plot_paths = visualizer.generate_comprehensive_report(
        model, x_test, y_test, test_predictions, figures_dir
    )
    
    print(f"\nVisualizations saved to: {figures_dir}")
    
    # Step 6: Save model
    print("\nSaving model...")
    trainer.save_model(model, model_path, metadata_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Visualizations: {figures_dir}")
    print(f"\nFinal Test Accuracy: {metrics['test_accuracy']:.4f}")
    print("\nYou can now start the Flask application and use the model!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
