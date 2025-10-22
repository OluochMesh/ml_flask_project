import os
import logging
from flask import current_app

def ensure_directories_exist():
    """Ensure all required directories exist"""
    directories = [
        current_app.config['DATA_DIR'],
        current_app.config['RAW_DATA_DIR'],
        current_app.config['PROCESSED_DATA_DIR'],
        current_app.config['UPLOAD_DIR'],
        current_app.config['MODELS_DIR'],
        current_app.config['RESULTS_DIR'],
        current_app.config['FIGURES_DIR'],
        current_app.config['METRICS_DIR'],
        current_app.config['LOGS_DIR']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")
    
    return True