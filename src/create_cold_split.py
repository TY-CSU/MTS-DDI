import pandas as pd
import os
import sys
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def create_cold_split():
    print("Loading Data...")
    # Load SMILES mapping
    smiles_df = pd.read_csv(Config.ID_TO_SMILES_FILE)
    # Load Master Dataset to ensure we only include relevant drugs
    ddi_df = pd.read_csv(Config.DDI_FILE) # Or processed one? Let's use processed to be safe
    processed_ddi_file = os.path.join(Config.PROCESSED_DATA_DIR, 'master_ddi_dataset.csv')
    if os.path.exists(processed_ddi_file):
        ddi_df = pd.read_csv(processed_ddi_file)
    
    # Get set of all drugs in the dataset
    relevant_drugs = set(ddi_df['ID_A']).union(set(ddi_df['ID_B']))
    print(f"Total unique drugs in DDI dataset: {len(relevant_drugs)}")
    
    # Filter SMILES df
    smiles_df = smiles_df[smiles_df['DrugBank_ID'].isin(relevant_drugs)].reset_index(drop=True)
    print(f"Drugs with SMILES: {len(smiles_df)}")
    
    # Compute Scaffolds
    print("Computing Scaffolds...")
    scaffold_to_drugs = defaultdict(list)
    drug_to_scaffold = {}
    
    for idx, row in tqdm(smiles_df.iterrows(), total=len(smiles_df)):
        drug_id = row['DrugBank_ID']
        smiles = row['Smiles']
        
        scaffold = generate_scaffold(smiles)
        if scaffold is None:
            print(f"Warning: Invalid SMILES for {drug_id}, excluding from split.")
            continue
            
        scaffold_to_drugs[scaffold].append(drug_id)
        drug_to_scaffold[drug_id] = scaffold
        
    print(f"Total unique scaffolds: {len(scaffold_to_drugs)}")
    
    # Convert to list of clusters (scaffolds)
    # Each cluster is a list of drug_ids
    clusters = list(scaffold_to_drugs.values())
    
    # Shuffle clusters
    random.seed(42)
    random.shuffle(clusters)
    
    # Split: 20% Test, 80% Dev
    num_clusters = len(clusters)
    num_test = int(num_clusters * 0.2)
    
    test_clusters = clusters[:num_test]
    dev_clusters = clusters[num_test:]
    
    # Flatten to get drug lists
    test_drugs = set([d for cluster in test_clusters for d in cluster])
    dev_drugs = set([d for cluster in dev_clusters for d in cluster])
    
    print(f"Test Drugs: {len(test_drugs)}")
    print(f"Dev Drugs: {len(dev_drugs)}")
    
    # Split Dev into 5 Folds
    # We split the CLUSTERS, not the drugs directly, to maintain scaffold separation
    num_dev_clusters = len(dev_clusters)
    fold_size = num_dev_clusters // 5
    
    drug_split_map = {} # drug_id -> split_label (test, 0, 1, 2, 3, 4)
    
    # Assign Test
    for drug in test_drugs:
        drug_split_map[drug] = 'test'
        
    # Assign Folds
    for i in range(5):
        start = i * fold_size
        # For the last fold, take all remaining
        if i == 4:
            fold_clusters = dev_clusters[start:]
        else:
            end = (i + 1) * fold_size
            fold_clusters = dev_clusters[start:end]
            
        fold_drugs = set([d for cluster in fold_clusters for d in cluster])
        for drug in fold_drugs:
            drug_split_map[drug] = i
            
    # Save Mapping
    print("Saving Split Mapping...")
    split_df = pd.DataFrame(list(drug_split_map.items()), columns=['Drug_ID', 'Split_Group'])
    output_file = os.path.join(Config.PROCESSED_DATA_DIR, 'drug_cold_split.csv')
    split_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    # Verify
    print("\nDistribution:")
    print(split_df['Split_Group'].value_counts())

if __name__ == "__main__":
    create_cold_split()