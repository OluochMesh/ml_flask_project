#!/usr/bin/env python3
"""
Amazon Product Reviews Data Preparation Script

This script processes the raw Amazon review data from:
- data/raw/test.ft.txt
- data/raw/train.ft.txt

And creates cleaned, structured datasets for NLP analysis.
"""

import os
import pandas as pd
import re
from typing import Dict, List, Tuple, Any
import json
import argparse
from collections import Counter
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmazonDataPreprocessor:
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_dir, 'amazon_reviews'), exist_ok=True)
    
    def parse_amazon_file(self, file_path: str, sample_size: int = None) -> pd.DataFrame:
        """
        Parse Amazon review files in the format: __label__X review text
        
        Args:
            file_path: Path to the .txt file
            sample_size: Number of reviews to sample (None for all)
        
        Returns:
            DataFrame with columns: ['label', 'text', 'source_file']
        """
        logger.info(f"Parsing Amazon file: {file_path}")
        
        reviews = []
        labels = []
        line_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if sample_size and line_count >= sample_size:
                        break
                    
                    # Parse label and text
                    if line.startswith('__label__'):
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            label = parts[0].replace('__label__', '').strip()
                            text = parts[1].strip()
                            
                            # Clean text
                            cleaned_text = self.clean_review_text(text)
                            
                            if cleaned_text:  # Only add non-empty reviews
                                reviews.append(cleaned_text)
                                labels.append(label)
                                line_count += 1
                    
                    # Progress logging
                    if line_count % 10000 == 0 and line_count > 0:
                        logger.info(f"Processed {line_count} reviews...")
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame({
            'label': labels,
            'text': reviews,
            'source_file': os.path.basename(file_path)
        })
        
        logger.info(f"Successfully parsed {len(df)} reviews from {file_path}")
        return df
    
    def clean_review_text(self, text: str) -> str:
        """
        Clean and preprocess review text
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation and letters
        text = re.sub(r'[^\w\s.,!?;:()\-&\'"]', '', text)
        
        # Fix common encoding issues
        text = re.sub(r'&#\d+;', '', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([.,!?;:)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        
        return text.strip()
    
    def analyze_labels(self, df: pd.DataFrame) -> Dict:
        """
        Analyze label distribution
        
        Args:
            df: DataFrame with labels
            
        Returns:
            Dictionary with label analysis
        """
        label_counts = Counter(df['label'])
        total_reviews = len(df)
        
        analysis = {
            'total_reviews': int(total_reviews),  # Convert to native int
            'label_distribution': {label: int(count) for label, count in label_counts.items()},  # Convert to native types
            'label_percentages': {label: float(count/total_reviews * 100) 
                                for label, count in label_counts.items()}  # Convert to float
        }
        
        return analysis
    
    def extract_sample_reviews(self, df: pd.DataFrame, n_per_label: int = 5) -> pd.DataFrame:
        """
        Extract sample reviews for each label
        
        Args:
            df: Full DataFrame
            n_per_label: Number of samples per label
            
        Returns:
            DataFrame with samples
        """
        samples = []
        for label in df['label'].unique():
            label_samples = df[df['label'] == label].head(n_per_label)
            samples.append(label_samples)
        
        return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    
    def analyze_text_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Analyze text statistics
        
        Args:
            df: DataFrame with review texts
            
        Returns:
            Dictionary with text statistics
        """
        text_lengths = df['text'].str.len()
        word_counts = df['text'].str.split().str.len()
        
        stats = {
            'avg_text_length': float(text_lengths.mean()),
            'max_text_length': int(text_lengths.max()),
            'min_text_length': int(text_lengths.min()),
            'avg_word_count': float(word_counts.mean()),
            'max_word_count': int(word_counts.max()),
            'min_word_count': int(word_counts.min()),
            'total_words': int(word_counts.sum())
        }
        
        return stats
    
    def convert_to_serializable(self, obj: Any) -> Any:
        """
        Convert numpy/pandas types to native Python types for JSON serialization
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def prepare_datasets(self, sample_size: int = None, create_samples: bool = True):
        """
        Main method to prepare all datasets
        
        Args:
            sample_size: Number of reviews to process from each file (None for all)
            create_samples: Whether to create sample datasets
        """
        logger.info("Starting Amazon data preparation...")
        
        # File paths
        test_file = os.path.join(self.raw_data_dir, 'test.ft.txt')
        train_file = os.path.join(self.raw_data_dir, 'train.ft.txt')
        
        # Check if files exist
        if not os.path.exists(test_file):
            logger.error(f"Test file not found: {test_file}")
            return
        if not os.path.exists(train_file):
            logger.error(f"Train file not found: {train_file}")
            return
        
        # Parse files
        test_df = self.parse_amazon_file(test_file, sample_size)
        train_df = self.parse_amazon_file(train_file, sample_size)
        
        if test_df.empty or train_df.empty:
            logger.error("Failed to parse one or more files")
            return
        
        # Combine datasets for analysis
        combined_df = pd.concat([test_df, train_df], ignore_index=True)
        
        # Analyze datasets
        label_analysis = self.analyze_labels(combined_df)
        text_stats = self.analyze_text_statistics(combined_df)
        
        # Save processed datasets
        output_dir = os.path.join(self.processed_data_dir, 'amazon_reviews')
        
        # Save full datasets
        test_df.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
        train_df.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
        combined_df.to_csv(os.path.join(output_dir, 'combined_processed.csv'), index=False)
        
        # Save sample datasets for quick testing
        if create_samples:
            test_sample = self.extract_sample_reviews(test_df, n_per_label=10)
            train_sample = self.extract_sample_reviews(train_df, n_per_label=10)
            combined_sample = self.extract_sample_reviews(combined_df, n_per_label=10)
            
            test_sample.to_csv(os.path.join(output_dir, 'test_sample.csv'), index=False)
            train_sample.to_csv(os.path.join(output_dir, 'train_sample.csv'), index=False)
            combined_sample.to_csv(os.path.join(output_dir, 'combined_sample.csv'), index=False)
        
        # Save analysis results
        analysis_results = {
            'label_analysis': label_analysis,
            'text_statistics': text_stats,
            'dataset_sizes': {
                'test': int(len(test_df)),
                'train': int(len(train_df)),
                'combined': int(len(combined_df))
            },
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Convert all numpy types to native Python types
        analysis_results = self.convert_to_serializable(analysis_results)
        
        with open(os.path.join(output_dir, 'data_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Print summary
        self.print_summary(analysis_results, output_dir)
        
        logger.info("Data preparation completed successfully!")
    
    def print_summary(self, analysis_results: Dict, output_dir: str):
        """Print summary of the data preparation"""
        print("\n" + "="*50)
        print("AMAZON DATA PREPARATION SUMMARY")
        print("="*50)
        
        label_analysis = analysis_results['label_analysis']
        text_stats = analysis_results['text_statistics']
        dataset_sizes = analysis_results['dataset_sizes']
        
        print(f"\nTotal Reviews: {label_analysis['total_reviews']:,}")
        print(f"Test set: {dataset_sizes['test']:,}")
        print(f"Train set: {dataset_sizes['train']:,}")
        
        print(f"\nLabel Distribution:")
        for label, count in label_analysis['label_distribution'].items():
            percentage = label_analysis['label_percentages'][label]
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nText Statistics:")
        print(f"  Average text length: {text_stats['avg_text_length']:.1f} characters")
        print(f"  Average word count: {text_stats['avg_word_count']:.1f} words")
        print(f"  Total words: {text_stats['total_words']:,}")
        
        print(f"\nOutput files saved to: {output_dir}")
        print("="*50 + "\n")

def main():
    """Main function to run data preparation"""
    parser = argparse.ArgumentParser(description='Prepare Amazon review data for NLP analysis')
    parser.add_argument('--raw-dir', default='data/raw', 
                       help='Directory containing raw data files')
    parser.add_argument('--processed-dir', default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of reviews to sample from each file (for testing)')
    parser.add_argument('--no-samples', action='store_true',
                       help='Skip creating sample datasets')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = AmazonDataPreprocessor(args.raw_dir, args.processed_dir)
    
    # Run data preparation
    preprocessor.prepare_datasets(
        sample_size=args.sample_size,
        create_samples=not args.no_samples
    )

if __name__ == "__main__":
    main()