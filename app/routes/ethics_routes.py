from flask import Blueprint, render_template, request, jsonify
import os
import json
from app.ethics.bias_detector import BiasDetector
from app.ethics.fairness_metrics import FairnessMetrics

ethics_bp = Blueprint('ethics', __name__, url_prefix='/ethics')

@ethics_bp.route('/')
def ethics_dashboard():
    return render_template('ethics/dashboard.html')

@ethics_bp.route('/analyze_biases', methods=['POST'])
def analyze_biases():
    try:
        bias_detector = BiasDetector()
        
        # This would typically load your actual models and data
        # For demonstration, we'll use sample data
        
        # Sample MNIST analysis
        mnist_biases = {
            'digit_accuracy_disparity': {
                'max_accuracy': 0.98,
                'min_accuracy': 0.92, 
                'accuracy_range': 0.06,
                'per_digit_accuracy': {i: 0.95 + (i-4.5)*0.005 for i in range(10)}
            }
        }
        
        # Sample Amazon reviews analysis
        amazon_biases = {
            'entity_extraction_bias': {
                'positive': {'mean_entities': 3.2, 'std_entities': 1.1, 'count': 150},
                'negative': {'mean_entities': 2.8, 'std_entities': 1.3, 'count': 120},
                'neutral': {'mean_entities': 2.9, 'std_entities': 1.0, 'count': 80}
            }
        }
        
        # Generate fairness report
        report = bias_detector.generate_fairness_report(mnist_biases, amazon_biases)
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ethics_bp.route('/debug_challenge')
def debug_challenge():
    return render_template('ethics/debug_challenge.html')

@ethics_bp.route('/run_buggy_code')
def run_buggy_code():
    try:
        # Import and run buggy code
        from app.models.task2_mnist.buggy_code import create_buggy_cnn, buggy_data_preprocessing
        
        # This will intentionally fail
        model = create_buggy_cnn()
        
        return jsonify({
            'success': False,
            'error': 'Buggy code should have failed but somehow ran'
        })
        
    except Exception as e:
        return jsonify({
            'success': True,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'message': 'Buggy code failed as expected - ready for debugging!'
        })

@ethics_bp.route('/run_fixed_code')
def run_fixed_code():
    try:
        # Import and run fixed code
        from app.models.task2_mnist.debugged_code import create_fixed_cnn, fixed_data_preprocessing, debug_explanation
        
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Process data correctly
        x_train, y_train, x_test, y_test = fixed_data_preprocessing(x_train, y_train, x_test, y_test)
        
        # Create fixed model
        model = create_fixed_cnn()
        
        # Just compile, don't actually train for demo
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        fixes = debug_explanation()
        
        return jsonify({
            'success': True,
            'message': 'Fixed code runs successfully!',
            'model_summary': 'CNN model compiled successfully',
            'fixes_applied': fixes
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500