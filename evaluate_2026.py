#!/usr/bin/env python3
"""
MTS-DDI 2026 New Drug Evaluation with a Fixed Binary Threshold of 0.5
Uses the latest model weights (with V1-aligned hyperparams: class weights, OHNM, etc.)
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from src.model import MTS_DDI
from src.utils import smiles_to_graph
from signaturizer import Signaturizer
from torch_geometric.data import Batch, Data

BIO_MODEL_DIR = os.environ.get("MTS_DDI_BIO_MODEL_DIR", os.path.join(Config.PROJECT_ROOT, "external", "biosignature"))
OUTPUT_DIR = os.path.join(Config.PROJECT_ROOT, 'results', '2026')


def initialize_signaturizer():
    model_paths = []
    for letter in 'ABCDE':
        for number in '12345':
            mp = os.path.join(BIO_MODEL_DIR, f"{letter}{number}")
            if os.path.isdir(mp): model_paths.append(mp)
    return Signaturizer(model_paths, local=True)


def evaluate_binary(y_true, y_prob):
    """Evaluate binary classification at the fixed training threshold of 0.5."""
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'cm': cm}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Load 2026 Data ---
    new_csv = os.environ.get("MTS_DDI_2026_FILE", os.path.join(Config.PROJECT_ROOT, 'data', '2026', 'ddi_enriched.csv'))
    df_new = pd.read_csv(new_csv)
    print(f"Loaded new dataset with {len(df_new)} pairs.")
    
    severity_map = {'minor': 0, 'moderate': 1, 'major': 2}
    df_new['Risk_Label'] = df_new['severity'].str.lower().map(severity_map)
    df_new = df_new.dropna(subset=['Risk_Label', 'target_smiles', 'interacting_smiles'])
    df_new['Risk_Label'] = df_new['Risk_Label'].astype(int)
    df_new['Binary_Label'] = 1
    df_new['ID_A'] = "NEW_" + df_new['target_norm'].astype(str)
    df_new['ID_B'] = "NEW_" + df_new['interacting_norm'].astype(str)
    df_new['Source'] = '2026_External'
    df_new = df_new.rename(columns={'target_smiles': 'SMILES_A', 'interacting_smiles': 'SMILES_B'})
    df_new = df_new[['ID_A', 'ID_B', 'SMILES_A', 'SMILES_B', 'Risk_Label', 'Binary_Label', 'Source']]
    
    df = df_new.copy()
    print(f"Dataset Size: {len(df)}")
    
    # --- 2. Build drug data dict (model format: graph with edge_attr + bio_feat) ---
    unique_drugs = {}
    for _, row in df.iterrows():
        t_id, i_id = str(row['ID_A']), str(row['ID_B'])
        if t_id not in unique_drugs: unique_drugs[t_id] = row['SMILES_A']
        if i_id not in unique_drugs: unique_drugs[i_id] = row['SMILES_B']
    
    print(f"Extracted {len(unique_drugs)} unique drugs.")
    
    # Load precomputed DrugBank bio features (if new drugs happen to be in DrugBank)
    bio_csv_path = os.environ.get("MTS_DDI_BIO_CACHE", os.path.join(Config.PROJECT_ROOT, 'data', 'optional', 'drugbank_bio_features.csv'))
    precomputed_bio = {}
    if os.path.exists(bio_csv_path):
        print(f"Loading precomputed bio features from {bio_csv_path}")
        bio_df = pd.read_csv(bio_csv_path)
        for _, row in bio_df.iterrows():
            db_id = str(row['DrugBank_ID']).strip()
            precomputed_bio[db_id] = row['BioFeat']
    print(f"Loaded {len(precomputed_bio)} precomputed bio feature entries.")
    
    # Also load  drug_data.pt for any DrugBank drugs that might overlap
    drug_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'drug_data.pt')
    drugbank_cache = {}
    if os.path.exists(drug_data_path):
        print(f"Loading drug data from {drug_data_path}")
        drugbank_cache = torch.load(drug_data_path, map_location='cpu', weights_only=False)
        print(f"  Loaded {len(drugbank_cache)} DrugBank drugs with features")
    
    # Process drugs: build  graphs + bio features
    sign = None
    drug_data = {}
    keys_list = list(unique_drugs.keys())
    
    for k in tqdm(keys_list, desc="Processing Drugs (graphs + bio)"):
        raw_id = k.replace("NEW_", "").strip()
        smi = str(unique_drugs[k]).strip()
        
        try:
            # Check if this drug already exists in drugbank_cache (by DrugBank ID)
            if raw_id in drugbank_cache:
                d = drugbank_cache[raw_id]
                graph = d['graph'].clone()
                bio_feat = d['bio_feat'].clone()
            else:
                # Build  graph with edge_attr
                graph = smiles_to_graph(smi)
                if graph is None:
                    continue
                
                # Compute bio features
                if raw_id in precomputed_bio:
                    feat_str = str(precomputed_bio[raw_id]).strip()
                    feat_list = json.loads(feat_str)
                    if isinstance(feat_list[0], list):
                        feat_list = feat_list[0]
                    bio_feat = torch.tensor(feat_list, dtype=torch.float32)
                else:
                    if sign is None:
                        print(f"\nDrug '{k}' not in cache. Initializing Signaturizer...")
                        sign = initialize_signaturizer()
                    res = sign.predict([smi])
                    bio_feat = torch.tensor(res.signature[0], dtype=torch.float32)
                
                if bio_feat.dim() == 1:
                    bio_feat = bio_feat.unsqueeze(0)  # [1, 3200]
            
            # Store in model-compatible format
            # graph already has x [N, 79], edge_index [2, E], edge_attr [E, 6]
            graph.bio_feat = bio_feat  # [1, 3200] or [3200]
            drug_data[k] = graph
            
        except Exception as e:
            if len(drug_data) == 0:
                print(f"Error processing first drug {k}: {e}")
            continue
    
    print(f"Successfully processed {len(drug_data)} drugs (model format).")
    
    eval_df = df[df['ID_A'].isin(drug_data.keys()) & df['ID_B'].isin(drug_data.keys())].copy()
    print(f"Evaluable pairs: {len(eval_df)}")
    
    # --- 3. Load  Cold-Start 5-Fold Ensemble Models ---
    models = []
    print("Loading cold-start 5-Fold Ensemble models (latest version)...")
    
    for fold in range(5):
        m_path = os.path.join(Config.PROJECT_ROOT, 'trained_models', f'cold_fold_{fold}', 'best_model.pth')
        if os.path.exists(m_path):
            try:
                m = MTS_DDI(
                    atom_feature_dim=79, edge_feature_dim=6,
                    bio_input_dim=3200, hidden_dim=256,
                    use_bio=True, use_gate=True
                ).to(device)
                state_dict = torch.load(m_path, map_location=device, weights_only=False)
                m.load_state_dict(state_dict, strict=False)
                m.eval()
                models.append(m)
                print(f"  Loaded fold {fold}")
            except Exception as e:
                print(f"  Error loading fold {fold}: {e}")
        else:
            print(f"  Fold {fold} not found at {m_path}")
    
    if not models:
        print("Error: No models loaded!")
        return
    
    print(f"Loaded {len(models)} model(s)")
    
    # --- 4. Run Predictions ( model expects edge_attr + bio_feat) ---
    all_preds, all_probs, all_targets = [], [], []
    all_bin_probs, all_bin_targets = [], []
    
    batch_size = 128
    for i in tqdm(range(0, len(eval_df), batch_size), desc="Evaluating"):
        batch_df = eval_df.iloc[i:i+batch_size]
        graphs_A, graphs_B, labels, bin_labels = [], [], [], []
        bio_A_list, bio_B_list = [], []
        
        for _, row in batch_df.iterrows():
            dA = drug_data[row['ID_A']].clone()
            dB = drug_data[row['ID_B']].clone()
            labels.append(row['Risk_Label'])
            bin_labels.append(row['Binary_Label'])
            
            # Extract bio_feat before batching (it's not a standard PyG attribute)
            bio_A_list.append(dA.bio_feat.flatten())
            bio_B_list.append(dB.bio_feat.flatten())
            
            # Create clean PyG Data for batching (with edge_attr)
            gA = Data(x=dA.x, edge_index=dA.edge_index, edge_attr=dA.edge_attr)
            gB = Data(x=dB.x, edge_index=dB.edge_index, edge_attr=dB.edge_attr)
            graphs_A.append(gA)
            graphs_B.append(gB)
        
        batch_A = Batch.from_data_list(graphs_A).to(device)
        batch_B = Batch.from_data_list(graphs_B).to(device)
        
        # Attach bio_feat ( model expects it)
        batch_A.bio_feat = torch.stack(bio_A_list).to(device)  # [batch, 3200]
        batch_B.bio_feat = torch.stack(bio_B_list).to(device)  # [batch, 3200]
        
        with torch.no_grad():
            batch_probs_list, batch_bin_probs_list = [], []
            for m in models:
                out_bin, out_risk = m(batch_A, batch_B)
                batch_probs_list.append(torch.softmax(out_risk, dim=1).cpu().numpy())
                # Binary: out_bin shape [batch, 1], apply sigmoid
                b_probs_val = torch.sigmoid(out_bin).cpu().numpy().flatten()
                batch_bin_probs_list.append(b_probs_val)
            
            avg_probs = np.mean(batch_probs_list, axis=0)
            avg_bin_probs = np.mean(batch_bin_probs_list, axis=0)
            
            all_preds.extend(np.argmax(avg_probs, axis=1).tolist())
            all_probs.extend(avg_probs)
            all_targets.extend(labels)
            all_bin_probs.extend(avg_bin_probs.tolist())
            all_bin_targets.extend(bin_labels)
    
    # --- 5. Binary Classification at the Fixed Threshold of 0.5 ---
    y_bin_true = np.array(all_bin_targets)
    y_bin_prob = np.array(all_bin_probs)
    
    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION (fixed threshold: 0.5)")
    print("=" * 60)
    print(f"Total samples: {len(y_bin_true)}")
    print(f"Positive samples: {y_bin_true.sum()} ({y_bin_true.mean()*100:.1f}%)")
    print(f"Mean predicted probability: {y_bin_prob.mean():.4f}")
    print(f"Median predicted probability: {np.median(y_bin_prob):.4f}")
    
    binary_metrics = evaluate_binary(y_bin_true, y_bin_prob)
    print(f"Accuracy:  {binary_metrics['accuracy']:.4f}")
    print(f"F1:        {binary_metrics['f1']:.4f}")
    print(f"Precision: {binary_metrics['precision']:.4f}")
    print(f"Recall:    {binary_metrics['recall']:.4f}")
    
    # --- 6. Multi-class Risk Evaluation ---
    print("\n" + "=" * 60)
    print("MULTI-CLASS RISK CLASSIFICATION (model)")
    print("=" * 60)
    
    y_true_all = np.array(all_targets)
    y_pred_all = np.array(all_preds)
    y_prob_all = np.array(all_probs)
    
    valid_mask = y_true_all >= 0
    y_true = y_true_all[valid_mask]
    y_pred = y_pred_all[valid_mask]
    y_prob = y_prob_all[valid_mask]
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        roc_auc = float('nan')
    
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        pr_auc = average_precision_score(y_true_bin, y_prob, average='macro')
    except:
        pr_auc = float('nan')
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"ROC-AUC (Macro): {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC: N/A")
    print(f"PR-AUC (Macro): {pr_auc:.4f}" if not np.isnan(pr_auc) else "PR-AUC: N/A")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Minor', 'Moderate', 'Major']))
    
    # Per-class F1
    class_f1 = f1_score(y_true, y_pred, average=None)
    print(f"Per-class F1: Minor={class_f1[0]:.4f}, Moderate={class_f1[1]:.4f}, Major={class_f1[2]:.4f}")
    
    # --- 7. Save Results ---
    # Save detailed report
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write("MTS-DDI 2026 New Drug Evaluation (Latest Model with V1-aligned Hyperparams)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Models: cold-start 5-Fold Ensemble (lr=1e-4, OHNM ratio=2.0, risk_weights=[10,1,3])\n")
        f.write(f"Total evaluable pairs: {len(eval_df)}\n")
        f.write(f"Unique drugs processed: {len(drug_data)}\n\n")
        
        f.write("Binary Classification Probability Statistics:\n")
        f.write(f"  Mean probability: {y_bin_prob.mean():.4f}\n")
        f.write(f"  Median probability: {np.median(y_bin_prob):.4f}\n")
        f.write(f"  Std probability: {y_bin_prob.std():.4f}\n\n")
        
        f.write("Binary Classification (fixed threshold: 0.5):\n")
        f.write(f"  Accuracy: {binary_metrics['accuracy']:.4f}\n")
        f.write(f"  F1: {binary_metrics['f1']:.4f}\n")
        f.write(f"  Precision: {binary_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {binary_metrics['recall']:.4f}\n\n")
        
        f.write("Multi-class Risk Classification:\n")
        f.write(f"  Accuracy: {acc:.4f}\n")
        f.write(f"  Macro F1: {f1_macro:.4f}\n")
        f.write(f"  Weighted F1: {f1_weighted:.4f}\n")
        f.write(f"  ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"  PR-AUC: {pr_auc:.4f}\n")
        f.write(f"  Per-class F1: Minor={class_f1[0]:.4f}, Moderate={class_f1[1]:.4f}, Major={class_f1[2]:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=['Minor', 'Moderate', 'Major']))
    
    # Save per-pair predictions for further analysis
    pred_df = eval_df.copy()
    pred_df['bin_prob'] = all_bin_probs
    pred_df['risk_pred'] = all_preds
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions_2026.csv'), index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
