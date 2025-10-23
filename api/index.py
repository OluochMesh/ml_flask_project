import os
from flask import Flask, jsonify

app = Flask(__name__)

# Basic configuration
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', False)
app.config['UPLOAD_DIR'] = '/tmp/uploads'  # Vercel has ephemeral storage

@app.route('/')
def home():
    return jsonify({
        'message': 'ML Flask API is running!',
        'status': 'success'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# Import your routes if they exist
try:
    from app.routes import *
except ImportError:
    @app.route('/predict', methods=['POST'])
    def predict():
        return jsonify({'error': 'ML features not available on Vercel'})

if __name__ == '__main__':
    app.run()