"""
Data Preprocessing Pipeline for CICIDS2017 Dataset
Loads, cleans, encodes, and normalizes network traffic data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import glob

class DataPreprocessor:
    def __init__(self, data_dir='data/MachineLearningCSV', output_dir='data/processed'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load and merge all CSV files from CICIDS2017 dataset"""
        print("Loading CSV files...")
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return None
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Read and concatenate all CSVs
        df_list = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                print(f"Loaded: {os.path.basename(file)} - {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not df_list:
            return None
        
        df = pd.concat(df_list, ignore_index=True)
        print(f"\nTotal rows after merging: {len(df)}")
        return df
    
    def clean_data(self, df):
        """Remove duplicates, handle missing values, and clean data"""
        print("\nCleaning data...")
        
        # Check initial shape
        print(f"Initial shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        print(f"Removed {missing_before} missing values")
        print(f"Final shape after cleaning: {df.shape}")
        
        return df
    
    def encode_labels(self, df, label_column='Label'):
        """Encode attack type labels"""
        print("\nEncoding labels...")
        
        if label_column not in df.columns:
            # Try to find label column (sometimes it's ' Label' with space)
            possible_cols = [col for col in df.columns if 'label' in col.lower()]
            if possible_cols:
                label_column = possible_cols[0]
            else:
                print(f"Warning: Could not find label column")
                return df, None
        
        # Show label distribution
        print(f"Label distribution:")
        print(df[label_column].value_counts())
        
        # Encode labels
        df['Label_Encoded'] = self.label_encoder.fit_transform(df[label_column])
        
        # Save label mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"\nLabel mapping: {label_mapping}")
        
        return df, label_mapping
    
    def normalize_features(self, df, label_column='Label'):
        """Normalize numeric features using MinMaxScaler"""
        print("\nNormalizing features...")
        
        # Identify feature columns (exclude label columns)
        feature_cols = [col for col in df.columns 
                       if col not in [label_column, 'Label_Encoded']]
        
        # Select only numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        print(f"Normalizing {len(numeric_cols)} numeric features")
        
        # Normalize
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df, numeric_cols
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\nSplitting data...")
        
        # Prepare features and labels
        X = df.drop(['Label_Encoded'], axis=1, errors='ignore')
        
        # Remove original label column if exists
        label_cols = [col for col in X.columns if 'label' in col.lower()]
        X = X.drop(label_cols, axis=1, errors='ignore')
        
        y = df['Label_Encoded']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data to CSV files"""
        print("\nSaving processed data...")
        
        # Save train data
        train_df = X_train.copy()
        train_df['Label'] = y_train.values
        train_df.to_csv(os.path.join(self.output_dir, 'train_data.csv'), index=False)
        
        # Save test data
        test_df = X_test.copy()
        test_df['Label'] = y_test.values
        test_df.to_csv(os.path.join(self.output_dir, 'test_data.csv'), index=False)
        
        print(f"Saved to {self.output_dir}/")
        print("- train_data.csv")
        print("- test_data.csv")
    
    def run_pipeline(self):
        """Execute complete preprocessing pipeline"""
        print("="*60)
        print("CICIDS2017 Data Preprocessing Pipeline")
        print("="*60)
        
        # Load data
        df = self.load_data()
        if df is None:
            print("Failed to load data. Exiting.")
            return
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode labels
        df, label_mapping = self.encode_labels(df)
        
        # Normalize features
        df, feature_cols = self.normalize_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test)
        
        # Save metadata
        metadata = {
            'num_features': len(feature_cols),
            'num_classes': len(label_mapping),
            'label_mapping': label_mapping,
            'feature_names': feature_cols
        }
        
        import json
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)
        print(f"Features: {metadata['num_features']}")
        print(f"Classes: {metadata['num_classes']}")
        
        return metadata

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline()