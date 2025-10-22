"""
Script to train the Iris classification model
Run this script to train the initial model before starting the Flask app
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.task1_iris.preprocessing import IrisPreprocessor
from app.models.task1_iris.train import IrisModelTrainer
from app.models.task1_iris.evaluate import IrisModelEvaluator
from config import config

def main():
    """Main training function"""
    print("="*60)
    print("IRIS CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Load configuration
    cfg = config['development']
    
    # Paths
    data_path = cfg.IRIS_DATA_PATH
    model_path = cfg.IRIS_MODEL_PATH
    results_dir = cfg.FIGURES_DIR
    metrics_dir = cfg.METRICS_DIR
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"\nError: Data file not found at {data_path}")
        print("Please ensure the iris.csv file is in the data/raw/ directory")
        return
    
    # Step 1: Preprocess data
    print("\n[1/4] Preprocessing data...")
    preprocessor = IrisPreprocessor()
    X_train, X_test, y_train, y_test, label_mapping, statistics = preprocessor.preprocess(
        data_path, test_size=0.2, random_state=42
    )
    
    # Save label mapping
    mapping_path = os.path.join(os.path.dirname(model_path), 'iris_label_mapping.json')
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Label mapping saved to: {mapping_path}")
    
    # Step 2: Train model
    print("\n[2/4] Training model...")
    trainer = IrisModelTrainer()
    
    # Train with hyperparameter tuning for best performance
    model = trainer.train_with_hyperparameter_tuning(X_train, y_train, cv=5)
    
    # Get feature importance
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    feature_importance = trainer.get_feature_importance(feature_names)
    
    # Perform cross-validation
    cv_scores = trainer.cross_validate(X_train, y_train, cv=5)
    
    # Step 3: Evaluate model
    print("\n[3/4] Evaluating model...")
    evaluator = IrisModelEvaluator(label_mapping)
    metrics = evaluator.evaluate(model, X_test, y_test)
    
    # Generate plots
    print("\n[4/4] Generating visualizations...")
    plot_paths = evaluator.generate_all_plots(
        metrics, 
        feature_importance, 
        results_dir
    )
    
    # Save model
    trainer.save_model(model_path)
    
    # Save metrics to file
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'iris_metrics.json')
    
    # Convert numpy types to native Python types for JSON serialization
    metrics_to_save = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'precision_weighted': float(metrics['precision_weighted']),
        'recall_macro': float(metrics['recall_macro']),
        'recall_weighted': float(metrics['recall_weighted']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_importance': feature_importance,
        'best_params': trainer.best_params if trainer.best_params else {}
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"Visualizations saved to: {results_dir}")
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\nYou can now start the Flask application!")

if __name__ == '__main__':
    main()