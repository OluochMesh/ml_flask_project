from flask import Flask
import os
import logging
from config import config

def create_app(config_name=None):
    """
    Application factory pattern for creating Flask app
    """
    if config_name is None:
        config_name = os.getenv('FLASK_CONFIG', 'default')
    
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize directories
    with app.app_context():
        initialize_directories(app)
        setup_logging(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register context processors
    register_context_processors(app)
    
    # Register before/after request handlers
    register_request_handlers(app)
    
    return app

def initialize_directories(app):
    """Create necessary directories if they don't exist"""
    directories = [
        app.config['DATA_DIR'],
        app.config['RAW_DATA_DIR'],
        app.config['PROCESSED_DATA_DIR'],
        app.config['UPLOAD_DIR'],
        app.config['MODELS_DIR'],
        app.config['RESULTS_DIR'],
        app.config['FIGURES_DIR'],
        app.config['METRICS_DIR'],
        app.config['LOGS_DIR']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        app.logger.info(f"Ensured directory exists: {directory}")

def setup_logging(app):
    """Configure application logging"""
    log_file = os.path.join(app.config['LOGS_DIR'], 'app.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def register_blueprints(app):
    """Register all application blueprints"""
    from app.routes.main import main_bp
    from app.routes.task1_routes import task1_bp
    from app.routes.task2_routes import task2_bp
    from app.routes.task3_routes import task3_bp
    from app.routes.ethics_routes import ethics_bp
    
    blueprints = [
        (main_bp, ''),
        (task1_bp, '/task1'),
        (task2_bp, '/task2'),
        (task3_bp, '/task3'),
        (ethics_bp, '/ethics')
    ]
    
    for blueprint, url_prefix in blueprints:
        app.register_blueprint(blueprint, url_prefix=url_prefix)
        app.logger.info(f"Registered blueprint: {blueprint.name} with prefix: {url_prefix}")

def register_error_handlers(app):
    """Register custom error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        app.logger.warning(f'404 error: {error}')
        return {
            'error': 'Not Found',
            'message': 'The requested resource was not found.',
            'status_code': 404
        }, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'500 error: {error}')
        return {
            'error': 'Internal Server Error',
            'message': 'An internal server error occurred.',
            'status_code': 500
        }, 500
    
    @app.errorhandler(413)
    def too_large(error):
        app.logger.warning('File upload too large')
        return {
            'error': 'File Too Large',
            'message': f'File size exceeds {app.config["MAX_CONTENT_LENGTH"]} bytes.',
            'status_code': 413
        }, 413
    
    @app.errorhandler(400)
    def bad_request(error):
        app.logger.warning(f'Bad request: {error}')
        return {
            'error': 'Bad Request',
            'message': 'The request could not be understood.',
            'status_code': 400
        }, 400

def register_context_processors(app):
    """Register template context processors"""
    
    @app.context_processor
    def inject_config():
        """Inject configuration into templates"""
        return {
            'app_name': 'ML Flask Portfolio',
            'app_version': '1.0.0',
            'debug_mode': app.config['DEBUG']
        }
    
    @app.context_processor
    def utility_processor():
        """Inject utility functions into templates"""
        def format_float(value, precision=4):
            """Format float for display"""
            try:
                return f"{float(value):.{precision}f}"
            except (ValueError, TypeError):
                return value
        
        return {
            'format_float': format_float
        }

def register_request_handlers(app):
    """Register before/after request handlers"""
    
    @app.before_request
    def before_request():
        """Execute before each request"""
        app.logger.info(f'Processing request: {request.endpoint}')
    
    @app.after_request
    def after_request(response):
        """Execute after each request"""
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response

# Import at the end to avoid circular imports
from flask import request