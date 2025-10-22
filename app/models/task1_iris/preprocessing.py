import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class IrisPreprocessor:
    """
    Preprocessor for Iris dataset
    Handles missing values, label encoding, and data splitting
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.target_column = 'Species'
    
    def load_data(self, file_path):
        """
        Load Iris dataset from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df: pandas DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # Check for missing values
        missing_count = df.isnull().sum()
        print("\nMissing values per column:")
        print(missing_count)
        
        # For numerical columns, fill with median
        for col in self.feature_columns:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled {col} missing values with median: {median_value}")
        
        # For target column, drop rows with missing values
        if df[self.target_column].isnull().any():
            before_rows = len(df)
            df = df.dropna(subset=[self.target_column])
            print(f"Dropped {before_rows - len(df)} rows with missing target values")
        
        return df
    
    def encode_labels(self, df):
        """
        Encode species labels to numerical values
        
        Args:
            df: pandas DataFrame
            
        Returns:
            DataFrame with encoded labels and mapping dictionary
        """
        # Encode the target variable
        df['Species_Encoded'] = self.label_encoder.fit_transform(df[self.target_column])
        
        # Create mapping dictionary
        label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        print("\nLabel Encoding Mapping:")
        for species, code in label_mapping.items():
            print(f"  {species}: {code}")
        
        return df, label_mapping
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            df: pandas DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df[self.feature_columns].values
        y = df['Species_Encoded'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\nData split:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_statistics(self, df):
        """
        Get basic statistics of the dataset
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'describe': df[self.feature_columns].describe().to_dict(),
            'species_distribution': df[self.target_column].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return stats
    
    def preprocess(self, file_path, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path: Path to CSV file
            test_size: Proportion for testing
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test, label_mapping, statistics
        """
        # Load data
        df = self.load_data(file_path)
        
        # Get initial statistics
        stats_before = self.get_statistics(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode labels
        df, label_mapping = self.encode_labels(df)
        
        # Get final statistics
        stats_after = self.get_statistics(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df, test_size, random_state)
        
        statistics = {
            'before_preprocessing': stats_before,
            'after_preprocessing': stats_after,
            'label_mapping': label_mapping
        }
        
        return X_train, X_test, y_train, y_test, label_mapping, statistics