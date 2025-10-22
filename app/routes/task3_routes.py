from flask import Blueprint, render_template, request, jsonify
import os
from app.models.task3_nlp import NLPAnalyzer

task3_bp = Blueprint('task3', __name__, url_prefix='/task3')
nlp_analyzer = NLPAnalyzer()


@task3_bp.route("/index")
def index():
    return nlp_input()
@task3_bp.route('/')
def nlp_input():
    return render_template('task3/nlp_input.html')

@task3_bp.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        if 'text' not in request.form:
            return jsonify({'error': 'No text provided'}), 400
        
        text = request.form['text']
        if not text.strip():
            return jsonify({'error': 'Empty text'}), 400
        
        # Perform NLP analysis
        analysis = nlp_analyzer.analyze_text(text)
        
        return render_template('task3/nlp_results.html', analysis=analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@task3_bp.route('/analyze_file', methods=['POST'])
def analyze_file():
    try:
        # Analyze sample from the uploaded file
        file_path = os.path.join('data', 'raw', 'test.ft.txt')
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Data file not found'}), 404
        
        sample_size = int(request.form.get('sample_size', 5))
        analysis = nlp_analyzer.analyze_review_file(file_path, sample_size)
        
        return render_template('task3/nlp_results.html', 
                             batch_analysis=analysis,
                             sample_size=sample_size)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@task3_bp.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    analysis = nlp_analyzer.analyze_text(text)
    
    return jsonify(analysis)