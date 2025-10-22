from flask import Blueprint, render_template, request, jsonify, current_app
import os
import json
from app.models.task1_iris.preprocessing import IrisPreprocessor
from app.models.task1_iris.train import IrisModelTrainer
from app.models.task1_iris.evaluate import IrisModelEvaluator
from app.models.task1_iris.predict import IrisPredictor

task1_bp = Blueprint('task1', __name__)

# Global variables to store model and mappings
_predictor = None
_label_mapping = None

def get_predictor():
    """Get or initialize the predictor"""
    global _predictor, _label_mapping
    
    if _predictor is None:
        model_path = current_app.config['IRIS_MODEL_PATH']
        
        # Load label mapping if exists
        mapping_path = os.path.join(
            current_app.config['MODELS_DIR'], 
            'iris_label_mapping.json'
        )
        
        _label_mapping = None
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, 'r') as f:
                    _label_mapping = json.load(f)
                print(f"✅ Label mapping loaded: {_label_mapping}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️  Error loading label mapping: {e}")
                _label_mapping = None
        
        # Check if model exists
        if os.path.exists(model_path) and _label_mapping is not None:
            try:
                _predictor = IrisPredictor(model_path, _label_mapping)
                print("✅ Predictor initialized successfully")
            except Exception as e:
                print(f"⚠️  Error initializing predictor: {e}")
                _predictor = None
        else:
            _predictor = None
            if not os.path.exists(model_path):
                print("⚠️  Model file not found")
            if _label_mapping is None:
                print("⚠️  Label mapping not available")
    
    return _predictor

@task1_bp.route('/')
def index():
    """Task 1 home page"""
    return render_template('task1/iris_input.html')

@task1_bp.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train the Iris classification model"""
    if request.method == 'GET':
        return render_template('task1/train.html')
    
    try:
        # Get training parameters
        use_tuning = request.form.get('use_tuning', 'false') == 'true'
        test_size = float(request.form.get('test_size', 20)) / 100.0
        
        # Paths
        data_path = current_app.config['IRIS_DATA_PATH']
        model_path = current_app.config['IRIS_MODEL_PATH']
        results_dir = current_app.config['FIGURES_DIR']
        
        # Step 1: Preprocess data
        preprocessor = IrisPreprocessor()
        X_train, X_test, y_train, y_test, label_mapping, statistics = preprocessor.preprocess(
            data_path, test_size=test_size
        )
        
        # FIX: Convert numpy int64 to regular int for JSON serialization
        serializable_label_mapping = {}
        for key, value in label_mapping.items():
            # Convert numpy types to native Python types
            if hasattr(value, 'item'):  # Check if it's a numpy type
                serializable_label_mapping[str(key)] = value.item()
            else:
                serializable_label_mapping[str(key)] = int(value)
        
        # Save label mapping
        mapping_path = os.path.join(
            current_app.config['MODELS_DIR'], 
            'iris_label_mapping.json'
        )
        with open(mapping_path, 'w') as f:
            json.dump(serializable_label_mapping, f, indent=2)  # Added indent for readability
        
        # Step 2: Train model
        trainer = IrisModelTrainer()
        
        if use_tuning:
            model = trainer.train_with_hyperparameter_tuning(X_train, y_train)
        else:
            model = trainer.train_basic_model(X_train, y_train)
        
        # Get feature importance
        feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        feature_importance = trainer.get_feature_importance(feature_names)
        
        # Step 3: Evaluate model
        evaluator = IrisModelEvaluator(label_mapping)
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        # FIX: Use static directory for images instead of results directory
        static_figures_dir = os.path.join(current_app.static_folder, 'figures')
        os.makedirs(static_figures_dir, exist_ok=True)  # Ensure directory exists
        
        # Generate plots in static directory
        plot_paths = evaluator.generate_all_plots(
            metrics, 
            feature_importance, 
            static_figures_dir  # Use static directory instead of results_dir
        )
        
        # Step 4: Save model
        trainer.save_model(model_path)
        
        # Reset predictor to reload new model
        global _predictor
        _predictor = None
        
        # Prepare response WITHOUT statistics to avoid serialization issues
        response = {
            'success': True,
            'message': 'Model trained successfully',
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision_macro': float(metrics['precision_macro']),
                'recall_macro': float(metrics['recall_macro']),
                'f1_macro': float(metrics['f1_macro'])
            },
            'feature_importance': feature_importance,
            'plots': {
                'confusion_matrix': '/static/figures/confusion_matrix.png',
                'feature_importance': '/static/figures/feature_importance.png',
                'metrics_comparison': '/static/figures/metrics_comparison.png'
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"🔴 FULL TRAINING ERROR: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Model training failed: {str(e)}'
        }), 500

@task1_bp.route('/predict', methods=['POST'])
def predict():
    """Make prediction for iris species"""
    try:
        # Get predictor
        predictor = get_predictor()
        
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained yet. Please train the model first.'
            }), 400
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            }), 400
            
        sepal_length = float(data.get('sepal_length', 0))
        sepal_width = float(data.get('sepal_width', 0))
        petal_length = float(data.get('petal_length', 0))
        petal_width = float(data.get('petal_width', 0))
        
        # Validate input
        is_valid, error_msg = predictor.validate_input(
            sepal_length, sepal_width, petal_length, petal_width
        )
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Make prediction
        result = predictor.predict_single(
            sepal_length, sepal_width, petal_length, petal_width
        )
        
        result['success'] = True
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"🔴 FULL PREDICTION ERROR: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500
    
@task1_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get model metrics"""
    try:
        metrics_path = os.path.join(
            current_app.config['METRICS_DIR'], 
            'iris_metrics.json'
        )
        
        if not os.path.exists(metrics_path):
            return jsonify({
                'success': False,
                'error': 'Metrics not available. Train the model first.'
            }), 404
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@task1_bp.route('/results')
def iris_results():
    """Display prediction results page"""
    return render_template('task1/iris_results.html')

@task1_bp.route('/save-prediction', methods=['POST'])
def save_prediction():
    """Save prediction results to session"""
    try:
        data = request.get_json()
        # Store in session for the results page
        from flask import session
        session['iris_prediction'] = data
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@task1_bp.route('/dataset-info', methods=['GET'])
def dataset_info():
    """Get information about the dataset"""
    try:
        data_path = current_app.config['IRIS_DATA_PATH']
        
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'error': 'Dataset not found'
            }), 404
        
        preprocessor = IrisPreprocessor()
        df = preprocessor.load_data(data_path)
        stats = preprocessor.get_statistics(df)
        
        # Convert numpy types to Python native types for JSON serialization
        return jsonify({
            'success': True,
            'statistics': {
                'shape': [int(stats['shape'][0]), int(stats['shape'][1])],
                'columns': [str(col) for col in stats['columns']],
                'species_distribution': {str(k): int(v) for k, v in stats['species_distribution'].items()},
                'missing_values': {str(k): int(v) for k, v in stats['missing_values'].items()}
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500