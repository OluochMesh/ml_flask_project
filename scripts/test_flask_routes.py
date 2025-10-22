#!/usr/bin/env python3
"""
Test script for Flask routes
"""

import os
import sys
import requests
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_flask_app():
    """Test if Flask app starts and responds"""
    print("🧪 TESTING FLASK APP")
    print("=" * 50)
    
    # Check if run.py exists
    if not os.path.exists('run.py'):
        print("❌ run.py not found")
        return False
    
    # Try to import and test Flask app
    try:
        # Import the app
        from run import app
        
        # Test basic routes
        print("\n1. Testing Flask app initialization...")
        with app.test_client() as client:
            # Test home route
            print("\n2. Testing routes...")
            
            # Test Task 3 routes
            response = client.get('/task3/')
            if response.status_code == 200:
                print("   ✓ Task 3 input page loaded")
            else:
                print(f"   ✗ Task 3 input page failed: {response.status_code}")
                return False
            
            # Test NLP analysis with POST
            test_data = {
                'text': 'This is a great product! I love it.'
            }
            response = client.post('/task3/analyze', data=test_data, follow_redirects=True)
            if response.status_code == 200:
                print("   ✓ NLP analysis route working")
            else:
                print(f"   ✗ NLP analysis route failed: {response.status_code}")
                return False
            
            # Test API endpoint
            response = client.post('/task3/api/analyze', 
                                 json={'text': 'Test review text'},
                                 content_type='application/json')
            if response.status_code == 200:
                print("   ✓ API endpoint working")
            else:
                print(f"   ✗ API endpoint failed: {response.status_code}")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 FLASK APP TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

if __name__ == "__main__":
    test_flask_app()