import os
from app import create_app
from config import config

# Get configuration from environment
config_name = os.getenv('FLASK_CONFIG', 'default')

# Create application instance
app = create_app(config_name)

@app.shell_context_processor
def make_shell_context():
    """Make variables available in Flask shell"""
    return {
        'app': app,
        'config': app.config
    }

if __name__ == '__main__':
    print(f"Starting ML Flask project with {config_name} configuration...")
    print(f"Debug mode: {app.config['DEBUG']}")
    print(f"Upload folder: {app.config['UPLOAD_DIR']}")
    
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=app.config['DEBUG']
    )