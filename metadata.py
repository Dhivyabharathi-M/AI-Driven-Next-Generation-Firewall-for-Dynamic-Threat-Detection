import json
import pandas as pd

# Load the processed data to get info
train_df = pd.read_csv('data/processed/train_data.csv')
test_df = pd.read_csv('data/processed/test_data.csv')

# Create metadata
feature_cols = [col for col in train_df.columns if col != 'Label']
num_classes = train_df['Label'].nunique()

metadata = {
    'num_features': len(feature_cols),
    'num_classes': int(num_classes),
    'label_mapping': {
        'BENIGN': 0, 'Bot': 1, 'DDoS': 2, 'DoS GoldenEye': 3, 
        'DoS Hulk': 4, 'DoS Slowhttptest': 5, 'DoS slowloris': 6,
        'FTP-Patator': 7, 'Heartbleed': 8, 'Infiltration': 9,
        'PortScan': 10, 'SSH-Patator': 11, 'Web Attack - Brute Force': 12,
        'Web Attack - Sql Injection': 13, 'Web Attack - XSS': 14
    },
    'feature_names': feature_cols
}

with open('data/processed/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Metadata file created successfully!")