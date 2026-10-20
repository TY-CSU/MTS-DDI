import pandas as pd
import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.utils import smiles_to_graph, parse_biofeat

def preprocess_data():
    print("Loading Bio-features for ...")
    if not os.path.exists(Config.BIO_FEATURES_FILE):
        print(f"Error: Bio-features file not found at {Config.BIO_FEATURES_FILE}")
        return

    bio_df = pd.read_csv(Config.BIO_FEATURES_FILE)
    print(f"Found {len(bio_df)} drugs with bio-features.")
    
    drug_data = {}
    
    print("Processing drugs (Generating Extended Graphs and Parsing Bio-feats)...")
    for idx, row in tqdm(bio_df.iterrows(), total=len(bio_df)):
        drug_id = row['DrugBank_ID']
        smiles = row['Smiles']
        bio_feat_str = row['BioFeat']
        
        # 1. Generate Graph using new rich features function
        graph = smiles_to_graph(smiles)
        
        # 2. Parse Bio-feat
        bio_feat = parse_biofeat(bio_feat_str)
        
        if graph is not None:
            drug_data[drug_id] = {
                'graph': graph,
                'bio_feat': bio_feat
            }
            
    print(f"Successfully processed {len(drug_data)} drugs with extended features.")
    
    # Save processed data avoiding overwrite of the old dataset
    save_path = os.path.join(Config.PROCESSED_DATA_DIR, 'drug_data.pt')
    torch.save(drug_data, save_path)
    print(f"Saved processed extended drug data to {save_path}")

if __name__ == "__main__":
    preprocess_data()
