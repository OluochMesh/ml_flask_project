from flask import Blueprint, render_template, request, jsonify, current_app
import os
import json
import numpy as np
from app.models.task2_mnist.data_loader import MNISTDataLoader
from app.models.task2_mnist.cnn_model import MNISTCNNModel
from app.models.task2_mnist.train import MNISTTrainer
from app.models.task2_mnist.predict import MNISTPredictor
from app.models.task2_mnist.visualize import MNISTVisualizer

task2_bp = Blueprint('task2', __name__)

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create MNIST predictor instance"""
    global _predictor
    if _predictor is None:
        model_path = current_app.config['MNIST_MODEL_PATH']
        metadata_path = current_app.config['MNIST_METADATA_PATH']
        
        if os.path.exists(model_path):
            _predictor = MNISTPredictor(model_path, metadata_path)
        else:
            _predictor = None
    
    return _predictor

@task2_bp.route('/')
def index():
    """Task 2 home page"""
    return render_template('task2/mnist_input.html')

@task2_bp.route('/dataset-info')
def dataset_info():
    """Get MNIST dataset information"""
    try:
        data_loader = MNISTDataLoader()
        (x_train, y_train), (x_test, y_test) = data_loader.load_data()
        stats = data_loader.get_data_statistics(x_train, y_train, x_test, y_test)
        
        # Convert numpy types to JSON serializable
        serializable_stats = {
            'training_samples': int(stats['training_samples']),
            'test_samples': int(stats['test_samples']),
            'image_shape': [int(dim) for dim in stats['image_shape']],
            'num_classes': int(stats['num_classes']),
            'class_distribution_train': [int(count) for count in stats['class_distribution_train']],
            'class_distribution_test': [int(count) for count in stats['class_distribution_test']],
            'pixel_value_range': stats['pixel_value_range']
        }
        
        return jsonify({
            'success': True,
            'statistics': serializable_stats
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Dataset info error: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Failed to load dataset info: {str(e)}'
        }), 500

@task2_bp.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train MNIST CNN model"""
    if request.method == 'GET':
        return render_template('task2/train.html')
    
    try:
        # Get training parameters from JSON (for AJAX requests)
        if request.is_json:
            data = request.get_json()
            model_type = data.get('model_type', 'basic')
            epochs = int(data.get('epochs', 5))  # Reduced default for testing
            batch_size = int(data.get('batch_size', 128))  # Increased batch size
            learning_rate = float(data.get('learning_rate', 0.001))
        else:
            # Get from form data
            model_type = request.form.get('model_type', 'basic')
            epochs = int(request.form.get('epochs', 5))
            batch_size = int(request.form.get('batch_size', 128))
            learning_rate = float(request.form.get('learning_rate', 0.001))
        
        # Limit epochs to prevent timeout
        if epochs > 10:
            epochs = 10
            print(f"Warning: Limiting epochs to 10 to prevent timeout")
        
        # Paths
        model_path = current_app.config['MNIST_MODEL_PATH']
        metadata_path = current_app.config['MNIST_METADATA_PATH']
        logs_dir = current_app.config['MNIST_LOGS_DIR']
        figures_dir = os.path.join(current_app.static_folder, 'figures', 'mnist')
        
        # Step 1: Load and preprocess data
        data_loader = MNISTDataLoader()
        (x_train, y_train), (x_test, y_test) = data_loader.load_data()
        x_train, y_train_encoded, x_test, y_test_encoded = data_loader.preprocess_data(
            x_train, y_train, x_test, y_test
        )
        
        # Create data generators
        train_gen, val_gen, test_gen = data_loader.create_data_generators(
            x_train, y_train_encoded, x_test, y_test_encoded,
            batch_size=batch_size
        )
        
        # Step 2: Build and compile model
        model_builder = MNISTCNNModel()
        
        if model_type == 'advanced':
            model = model_builder.build_advanced_cnn()
        else:
            model = model_builder.build_basic_cnn()
        
        model = model_builder.compile_model(model, learning_rate=learning_rate)
        
        # Get model info
        model_summary = model_builder.get_model_summary(model)
        model_params = model_builder.count_parameters(model)
        
        # Step 3: Train model
        trainer = MNISTTrainer()
        checkpoint_path = os.path.join(logs_dir, 'best_model.h5')
        callbacks = trainer.setup_callbacks(checkpoint_path, logs_dir)
        
        history = trainer.train_model(
            model, train_gen, val_gen, 
            epochs=epochs, callbacks=callbacks
        )
        
        # Step 4: Evaluate model
        metrics = trainer.evaluate_model(model, test_gen)
        
        # Step 5: Generate visualizations
        visualizer = MNISTVisualizer()
        
        # Get predictions for visualization
        test_predictions = model.predict(test_gen, verbose=0)
        
        # Generate plots in static directory
        os.makedirs(figures_dir, exist_ok=True)
        plot_paths = visualizer.generate_comprehensive_report(
            model, x_test, y_test, test_predictions, figures_dir
        )
        
        # Step 6: Save model
        trainer.save_model(model, model_path, metadata_path)
        
        # Reset predictor
        global _predictor
        _predictor = None
        
        # Prepare response
        response = {
            'success': True,
            'message': 'MNIST model trained successfully!',
            'metrics': metrics,
            'model_info': {
                'type': model_type,
                'parameters': model_params,
                'test_accuracy': metrics['test_accuracy']
            },
            'plots': {
                'sample_predictions': '/static/figures/mnist/sample_predictions.png',
                'confusion_matrix': '/static/figures/mnist/confusion_matrix.png'
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"MNIST Training error: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Model training failed: {str(e)}'
        }), 500

@task2_bp.route('/predict', methods=['POST'])
def predict():
    """Predict digit from uploaded image"""
    try:
        predictor = get_predictor()
        
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained yet. Please train the model first.'
            }), 400
        
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        # Save uploaded file temporarily
        upload_dir = current_app.config['UPLOAD_DIR']
        os.makedirs(upload_dir, exist_ok=True)
        
        temp_path = os.path.join(upload_dir, f'mnist_temp_{os.urandom(8).hex()}.png')
        image_file.save(temp_path)
        
        try:
            # Validate image
            is_valid, error_msg = predictor.validate_image(temp_path)
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 400
            
            # Make prediction
            result = predictor.predict_single(temp_path)
            
            # Add model info
            result['model_info'] = predictor.get_model_info()
            
            return jsonify({
                'success': True,
                'prediction': result
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"MNIST Prediction error: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@task2_bp.route('/model-info')
def model_info():
    """Get information about the current MNIST model"""
    try:
        predictor = get_predictor()
        
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'No trained model found'
            }), 404
        
        info = predictor.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@task2_bp.route('/draw')
def draw_digit():
    """Page for drawing digits"""
    return render_template('task2/draw.html')

@task2_bp.route('/samples')
def sample_images():
    """Page with sample MNIST images"""
    return render_template('task2/samples.html')