#!/usr/bin/env python3
"""
Test script for data preparation components
"""

import os
import sys
import pandas as pd
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_amazon_data import AmazonDataPreprocessor

def test_data_preparation():
    """Test the data preparation pipeline"""
    print("🧪 TESTING DATA PREPARATION")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = AmazonDataPreprocessor('data/raw', 'data/processed')
    
    # Test 1: Parse a small sample
    print("\n1. Testing file parsing...")
    test_file = 'data/raw/test.ft.txt'
    if os.path.exists(test_file):
        df = preprocessor.parse_amazon_file(test_file, sample_size=10)
        print(f"   ✓ Parsed {len(df)} reviews from {test_file}")
        print(f"   ✓ Columns: {list(df.columns)}")
        print(f"   ✓ Labels found: {df['label'].unique().tolist()}")
    else:
        print(f"   ✗ File not found: {test_file}")
        return False
    
    # Test 2: Text cleaning
    print("\n2. Testing text cleaning...")
    test_text = "This   is  a   test!!! text with   extra spaces...   "
    cleaned = preprocessor.clean_review_text(test_text)
    print(f"   ✓ Original: '{test_text}'")
    print(f"   ✓ Cleaned:  '{cleaned}'")
    
    # Test 3: Check output files
    print("\n3. Checking output files...")
    output_dir = 'data/processed/amazon_reviews'
    expected_files = [
        'test_processed.csv',
        'train_processed.csv', 
        'combined_processed.csv',
        'test_sample.csv',
        'train_sample.csv',
        'combined_sample.csv',
        'data_analysis.json'
    ]
    
    all_files_exist = True
    for file in expected_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            print(f"   ✓ {file} exists")
            
            # Check file content for CSV files
            if file.endswith('.csv'):
                try:
                    df_check = pd.read_csv(file_path)
                    print(f"     - Rows: {len(df_check)}, Columns: {len(df_check.columns)}")
                except Exception as e:
                    print(f"     - Error reading {file}: {e}")
                    all_files_exist = False
            # Check JSON file
            elif file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"     - JSON loaded successfully")
                except Exception as e:
                    print(f"     - Error reading {file}: {e}")
                    all_files_exist = False
        else:
            print(f"   ✗ {file} missing")
            all_files_exist = False
    
    print("\n" + "=" * 50)
    if all_files_exist:
        print("🎉 DATA PREPARATION TESTS PASSED!")
        return True
    else:
        print("❌ DATA PREPARATION TESTS FAILED!")
        return False

if __name__ == "__main__":
    test_data_preparation()