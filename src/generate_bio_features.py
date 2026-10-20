import pandas as pd
import os
import sys
import logging
from signaturizer import Signaturizer
from rdkit import Chem
import json
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MAPPING_FILE = os.path.join(PROCESSED_DATA_DIR, 'drugbank_smiles_mapping.csv')
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'drugbank_bio_features.csv')

# Raw files
DDINTER_FILE = os.path.join(RAW_DATA_DIR, 'ddinter_merged_with_drugbank.csv')
DRUGBANK_DDI_FILE = os.path.join(RAW_DATA_DIR, 'drugbank_ddi.csv')
MASTER_DATASET_FILE = os.path.join(PROCESSED_DATA_DIR, 'master_ddi_dataset.csv')

# Bio signature base dir
BIO_MODEL_DIR = os.environ.get("MTS_DDI_BIO_MODEL_DIR", os.path.join(PROJECT_ROOT, "external", "biosignature"))

def get_unique_ids():
    ids = set()
    
    # Priority: Use Master Dataset if available
    if os.path.exists(MASTER_DATASET_FILE):
        logging.info(f"Reading IDs from {MASTER_DATASET_FILE}...")
        df = pd.read_csv(MASTER_DATASET_FILE)
        if 'ID_A' in df.columns and 'ID_B' in df.columns:
            ids.update(df['ID_A'].dropna().unique())
            ids.update(df['ID_B'].dropna().unique())
        logging.info(f"Found {len(ids)} unique IDs in master dataset.")
        return ids

    # Fallback: 1. DDInter
    if os.path.exists(DDINTER_FILE):
        logging.info(f"Reading {DDINTER_FILE}...")
        df = pd.read_csv(DDINTER_FILE)
        # Columns: DrugBank_ID_A, DrugBank_ID_B
        if 'DrugBank_ID_A' in df.columns and 'DrugBank_ID_B' in df.columns:
            ids.update(df['DrugBank_ID_A'].dropna().unique())
            ids.update(df['DrugBank_ID_B'].dropna().unique())
    
    # 2. DrugBank DDI
    if os.path.exists(DRUGBANK_DDI_FILE):
        logging.info(f"Reading {DRUGBANK_DDI_FILE}...")
        df = pd.read_csv(DRUGBANK_DDI_FILE)
        # Columns: drugbank_id, partner_drugbank_id
        if 'drugbank_id' in df.columns and 'partner_drugbank_id' in df.columns:
            ids.update(df['drugbank_id'].dropna().unique())
            ids.update(df['partner_drugbank_id'].dropna().unique())
            
    logging.info(f"Total unique DrugBank IDs found in raw data: {len(ids)}")
    return ids

def initialize_signaturizer():
    model_paths = []
    for letter in 'ABCDE':
        for number in '12345':
            model_name = f"{letter}{number}"
            model_path = os.path.join(BIO_MODEL_DIR, model_name)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                model_paths.append(model_path)
    
    model_paths = sorted(model_paths)
    if not model_paths:
        logging.error(f"No models found in {BIO_MODEL_DIR}. Please check the path.")
        return None

    logging.info(f"Initializing Signaturizer with {len(model_paths)} models...")
    # local=True means load from local path
    return Signaturizer(model_name=model_paths, local=True, verbose=True)

def main():
    # 1. Get relevant IDs
    target_ids = get_unique_ids()
    logging.info(f"Total target IDs: {len(target_ids)}")
    
    # 2. Check existing output
    existing_ids = set()
    existing_df = None
    if os.path.exists(OUTPUT_FILE):
        logging.info(f"Reading existing output from {OUTPUT_FILE}...")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if 'DrugBank_ID' in existing_df.columns:
                existing_ids = set(existing_df['DrugBank_ID'].unique())
            logging.info(f"Found {len(existing_ids)} existing IDs.")
        except Exception as e:
            logging.error(f"Error reading existing file: {e}")
    
    # 3. Identify missing IDs
    missing_ids = target_ids - existing_ids
    logging.info(f"Missing IDs to process: {len(missing_ids)}")
    
    if len(missing_ids) == 0:
        logging.info("All IDs already processed. Exiting.")
        return

    # 4. Load Mapping
    if not os.path.exists(MAPPING_FILE):
        logging.error(f"Mapping file not found: {MAPPING_FILE}")
        return
        
    logging.info(f"Loading mapping from {MAPPING_FILE}...")
    mapping_df = pd.read_csv(MAPPING_FILE)
    
    # Filter for missing IDs
    filtered_df = mapping_df[mapping_df['DrugBank_ID'].isin(missing_ids)].copy()
    logging.info(f"Found SMILES for {len(filtered_df)} missing drugs.")
    
    if len(filtered_df) == 0:
        logging.warning("No SMILES found for missing drugs.")
        return

    # Remove invalid SMILES
    filtered_df = filtered_df.dropna(subset=['Smiles'])
    filtered_df = filtered_df[filtered_df['Smiles'].str.len() > 0]
    
    # 5. Initialize Signaturizer
    sign = initialize_signaturizer()
    if sign is None:
        return

    # 6. Generate Features (Safe Mode)
    logging.info("Generating bio-signatures (Safe Mode: One-by-One)...")
    
    new_rows = []
    
    # Iterate one by one to avoid TF batching crashes
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        did = row['DrugBank_ID']
        smi = row['Smiles']
        
        try:
            # Predict for single item
            res = sign.predict([smi])
            if res.signature is not None:
                # signature is (1, 128)
                feat_str = json.dumps(res.signature[0].tolist())
                new_rows.append({
                    'DrugBank_ID': did,
                    'Smiles': smi,
                    'BioFeat': feat_str
                })
        except Exception as e:
            logging.error(f"Error processing {did}: {e}")
            continue
            
    if not new_rows:
        logging.warning("No new features generated.")
        return
        
    new_df = pd.DataFrame(new_rows)
    logging.info(f"Generated {len(new_df)} new features.")
    
    # 7. Append and Save
    if existing_df is not None:
        # Append to existing file
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df
        
    # Remove duplicates just in case
    final_df = final_df.drop_duplicates(subset=['DrugBank_ID'])
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved updated bio-features to {OUTPUT_FILE}. Total records: {len(final_df)}")

if __name__ == "__main__":
    main()
