import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def check_distribution():
    print("Checking Test Set Distribution...")
    processed_ddi_file = os.path.join(Config.PROCESSED_DATA_DIR, 'master_ddi_dataset.csv')
    
    if not os.path.exists(processed_ddi_file):
        print("Error: Processed data file not found.")
        return

    df = pd.read_csv(processed_ddi_file)
    test_df = df[df['Split'] == 'test']
    
    print(f"Total Test Samples: {len(test_df)}")
    
    # Binary Distribution
    print("\n[Binary Label Distribution]")
    print(test_df['Binary_Label'].value_counts())
    print(test_df['Binary_Label'].value_counts(normalize=True))
    
    # Risk Distribution (excluding -1)
    print("\n[Risk Label Distribution (excluding -1)]")
    risk_df = test_df[test_df['Risk_Label'] != -1]
    print(risk_df['Risk_Label'].value_counts())
    print(risk_df['Risk_Label'].value_counts(normalize=True))
    
    # Check for overlap with Train
    train_df = df[df['Split'] == 'train']
    
    # Create set of pairs (sorted tuple to handle undirected nature if applicable, 
    # but usually DDI datasets treat (A,B) same as (B,A). Let's check both directions)
    # Assuming dataset might have (A,B) and (B,A) or just one.
    
    print("\nChecking for Data Leakage (Train/Test Overlap)...")
    train_pairs = set(zip(train_df['ID_A'], train_df['ID_B']))
    test_pairs = set(zip(test_df['ID_A'], test_df['ID_B']))
    
    overlap = train_pairs.intersection(test_pairs)
    print(f"Direct Overlap Count: {len(overlap)}")
    
    # Check reverse pairs overlap
    train_pairs_rev = set(zip(train_df['ID_B'], train_df['ID_A']))
    overlap_rev = train_pairs_rev.intersection(test_pairs)
    print(f"Reverse Overlap Count: {len(overlap_rev)}")

if __name__ == "__main__":
    check_distribution()