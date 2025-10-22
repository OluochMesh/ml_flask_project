#!/usr/bin/env python3
"""
Comprehensive test runner for the ML Flask project
"""

import os
import sys
import importlib.util
from datetime import datetime

def run_test(test_script):
    """Run a single test script"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {test_script}")
    print(f"{'='*60}")
    
    try:
        # Dynamically import and run the test module
        spec = importlib.util.spec_from_file_location("test_module", test_script)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Run the test function
        if hasattr(test_module, 'test_data_preparation'):
            return test_module.test_data_preparation()
        elif hasattr(test_module, 'test_nlp_components'):
            return test_module.test_nlp_components()
        elif hasattr(test_module, 'test_flask_app'):
            return test_module.test_flask_app()
        else:
            print(f"❌ No test function found in {test_script}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {test_script}: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE ML FLASK PROJECT TEST SUITE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_scripts = [
        'scripts/test_data_preparation.py',
        'scripts/test_nlp_components.py', 
        'scripts/test_flask_routes.py'
    ]
    
    results = {}
    
    for test_script in test_scripts:
        if os.path.exists(test_script):
            results[test_script] = run_test(test_script)
        else:
            print(f"⚠️  Test script not found: {test_script}")
            results[test_script] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_script, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {os.path.basename(test_script)}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! You're ready to run the app!")
        print("\nNext steps:")
        print("1. Install spaCy model: python -m spacy download en_core_web_sm")
        print("2. Run the app: python run.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("❌ Some tests failed. Please fix the issues before running the app.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)