import os
from datetime import timedelta

class Config:
    """Base configuration"""
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Base directory
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Data directories
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
    
    # Model directories
    MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
    
    # Results directories
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    
    # Logs
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv', 'txt'}
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Model settings
    IRIS_MODEL_PATH = os.path.join(MODELS_DIR, 'iris_decision_tree.pkl')
    MNIST_MODEL_PATH = os.path.join(MODELS_DIR, 'mnist_cnn.h5')
    MNIST_METADATA_PATH = os.path.join(MODELS_DIR, 'mnist_metadata.json')
    MNIST_LOGS_DIR = os.path.join(LOGS_DIR, 'mnist')
    
    # Dataset paths
    IRIS_DATA_PATH = os.path.join(RAW_DATA_DIR, 'iris.csv')
    AMAZON_REVIEWS_PATH = os.path.join(RAW_DATA_DIR, 'amazon_reviews.csv')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


