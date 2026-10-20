import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.dataset import DDIDataset, collate_fn
from src.model import MTS_DDI

def compute_binary_loss_ohnm(pred, target, neg_ratio=2.0):
    """Online Hard Negative Mining for Binary Classification (same as V1)."""
    loss_all = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    
    pos_mask = (target == 1).squeeze()
    neg_mask = (target == 0).squeeze()
    
    loss_pos = loss_all[pos_mask]
    loss_neg = loss_all[neg_mask]
    
    num_pos = loss_pos.numel()
    num_neg = loss_neg.numel()
    
    if num_pos == 0:
        return loss_neg.mean()
    if num_neg == 0:
        return loss_pos.mean()
    
    num_neg_keep = int(num_pos * neg_ratio)
    num_neg_keep = min(num_neg_keep, num_neg)
    
    if num_neg_keep > 0:
        loss_neg_top, _ = torch.topk(loss_neg.squeeze(), num_neg_keep)
        loss_neg = loss_neg_top
        
    total_loss = (loss_pos.sum() + loss_neg.sum()) / (num_pos + loss_neg.numel())
    return total_loss

def calculate_metrics_local(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.5
    f1 = f1_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    return {'Accuracy': acc, 'AUROC': auroc, 'F1_score': f1, 'AUPRC': auprc}

def train_cold():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    drug_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'drug_data.pt')
    split_file = os.path.join(Config.PROCESSED_DATA_DIR, 'drug_cold_split.csv')
    # Use master_ddi_dataset.csv as it has ID_A and ID_B (DrugBank IDs)
    processed_ddi_file = os.path.join(Config.PROCESSED_DATA_DIR, 'master_ddi_dataset.csv')
    
    if not os.path.exists(drug_data_path):
        print(f"Error: {drug_data_path} not found.")
        return

    drug_data = torch.load(drug_data_path)
    split_df = pd.read_csv(split_file)
    ddi_df = pd.read_csv(processed_ddi_file)

    # Convert Split_Group to string for uniform comparison
    split_df['Split_Group'] = split_df['Split_Group'].astype(str)
    drug_split = dict(zip(split_df['Drug_ID'], split_df['Split_Group']))

    all_fold_metrics = []

    for fold in range(5):
        print(f"\n===== Starting Cold-Start Training Fold {fold} () =====")
        fold_str = str(fold)
        
        train_pairs = []
        val_pairs = []
        
        # Filtering logic based on original project
        # Only keep drugs available in drug_data (the rich features dict)
        relevant_df = ddi_df[ddi_df['ID_A'].isin(drug_data) & ddi_df['ID_B'].isin(drug_data)]
        
        for _, row in relevant_df.iterrows():
            id_a, id_b = row['ID_A'], row['ID_B']
            g_a, g_b = drug_split.get(id_a), drug_split.get(id_b)
            
            # Skip if any drug is in the holdout 'test'
            if g_a == 'test' or g_b == 'test':
                continue
                
            # Train: Neither drug is in this fold's val set
            if g_a != fold_str and g_b != fold_str:
                train_pairs.append(row)
            # Val: At least one drug is in this fold
            elif g_a == fold_str or g_b == fold_str:
                val_pairs.append(row)
                
        train_df = pd.DataFrame(train_pairs).reset_index(drop=True)
        val_df = pd.DataFrame(val_pairs).reset_index(drop=True)
        
        print(f"Fold {fold}: Train Pairs={len(train_df)}, Val Pairs={len(val_df)}")
        
        train_loader = DataLoader(DDIDataset(train_df, drug_data), batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(DDIDataset(val_df, drug_data), batch_size=256, shuffle=False, collate_fn=collate_fn)

        save_root = os.path.join(Config.PROJECT_ROOT, "trained_models", f"cold_fold_{fold}")
        os.makedirs(save_root, exist_ok=True)

        model = MTS_DDI(atom_feature_dim=79, edge_feature_dim=6).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        criterion_binary = nn.BCEWithLogitsLoss()
        risk_weights = torch.tensor([10.0, 1.0, 3.0]).to(device)
        criterion_risk = nn.CrossEntropyLoss(weight=risk_weights, ignore_index=-1)
        
        best_val_loss = float('inf')
        early_stopping_counter = 0
        patience = 10
        
        for epoch in range(50):
            # === Training ===
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Fold {fold} Ep {epoch+1} [Train]", leave=False)
            for bA, bB, yb, yr in pbar:
                bA, bB, yb, yr = bA.to(device), bB.to(device), yb.to(device), yr.to(device)
                optimizer.zero_grad()
                ob, or_logits = model(bA, bB)
                
                # Binary loss with OHNM (same as V1)
                loss_bin = compute_binary_loss_ohnm(ob, yb.unsqueeze(1) if yb.dim() == 1 else yb, neg_ratio=2.0)
                
                # Risk loss with class weights
                mask = yr != -1
                if mask.any():
                    loss_risk = criterion_risk(or_logits[mask], yr[mask])
                    if torch.isnan(loss_risk):
                        loss = loss_bin
                    else:
                        loss = loss_bin + 0.5 * loss_risk
                else:
                    loss = loss_bin
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / len(train_loader)
            
            # === Validation ===
            model.eval()
            val_loss = 0
            v_preds, v_labels = [], []
            with torch.no_grad():
                for bA, bB, yb, yr in val_loader:
                    bA, bB, yb, yr = bA.to(device), bB.to(device), yb.to(device), yr.to(device)
                    ob, or_logits = model(bA, bB)
                    
                    v_loss_bin = criterion_binary(ob.squeeze(), yb)
                    v_mask = yr != -1
                    if v_mask.any():
                        v_loss_risk = criterion_risk(or_logits[v_mask], yr[v_mask])
                        if not torch.isnan(v_loss_risk):
                            val_loss += (v_loss_bin + 0.5 * v_loss_risk).item()
                        else:
                            val_loss += v_loss_bin.item()
                    else:
                        val_loss += v_loss_bin.item()
                    
                    v_preds.extend(torch.sigmoid(ob).cpu().numpy())
                    v_labels.extend(yb.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            if not v_labels:
                continue
                
            m = calculate_metrics_local(np.array(v_labels), np.array(v_preds))
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{save_root}/best_model.pth")
                early_stopping_counter = 0
                print(f"  Fold {fold} Ep {epoch+1}: val_loss={avg_val_loss:.4f}, Bin AUC={m['AUROC']:.4f} (saved)")
            else:
                early_stopping_counter += 1
                print(f"  Fold {fold} Ep {epoch+1}: val_loss={avg_val_loss:.4f}, Bin AUC={m['AUROC']:.4f} (no improve {early_stopping_counter}/{patience})")
            
            if early_stopping_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Final Evaluation for this Fold
        if os.path.exists(f"{save_root}/best_model.pth"):
            model.load_state_dict(torch.load(f"{save_root}/best_model.pth"))
            model.eval()
            p_bin, l_bin, p_risk, l_risk = [], [], [], []
            with torch.no_grad():
                for bA, bB, yb, yr in val_loader:
                    ob, or_l = model(bA.to(device), bB.to(device))
                    p_bin.extend(torch.sigmoid(ob).cpu().numpy())
                    l_bin.extend(yb.numpy())
                    mask = yr != -1
                    if mask.any():
                        p_risk.extend(torch.softmax(or_l[mask], dim=1).cpu().numpy())
                        l_risk.extend(yr[mask].numpy())
            
            bin_m = calculate_metrics_local(np.array(l_bin), np.array(p_bin))
            risk_acc = accuracy_score(l_risk, np.argmax(p_risk, axis=1)) if l_risk else 0
            try: risk_auc = roc_auc_score(pd.get_dummies(l_risk), p_risk, multi_class='ovr', average='macro')
            except: risk_auc = 0
            risk_f1 = f1_score(l_risk, np.argmax(p_risk, axis=1), average='macro') if l_risk else 0
            risk_f1_micro = f1_score(l_risk, np.argmax(p_risk, axis=1), average='micro') if l_risk else 0
            try: risk_pr = roc_auc_score(pd.get_dummies(l_risk), p_risk, multi_class='ovr', average='macro') 
            except: risk_pr = 0
            
            # Per-class risk metrics
            risk_class_f1 = f1_score(l_risk, np.argmax(p_risk, axis=1), average=None) if l_risk else [0,0,0]
            
            fold_res = {
                'Fold': fold,
                'Bin_Acc': bin_m['Accuracy'], 'Bin_AUC': bin_m['AUROC'], 'Bin_F1': bin_m['F1_score'], 'Bin_PR': bin_m['AUPRC'],
                'Risk_Acc': risk_acc, 'Risk_AUC': risk_auc, 'Risk_F1_Macro': risk_f1, 'Risk_F1_Micro': risk_f1_micro,
                'Risk_Minor_F1': risk_class_f1[0] if len(risk_class_f1)>0 else 0,
                'Risk_Moderate_F1': risk_class_f1[1] if len(risk_class_f1)>1 else 0,
                'Risk_Major_F1': risk_class_f1[2] if len(risk_class_f1)>2 else 0,
            }
            all_fold_metrics.append(fold_res)
            print(f"Fold {fold}: Bin AUC={bin_m['AUROC']:.4f}, Risk ACC={risk_acc:.4f}, Risk F1={risk_f1:.4f}")
            print(f"  Risk per-class F1: Minor={risk_class_f1[0]:.4f}, Moderate={risk_class_f1[1]:.4f}, Major={risk_class_f1[2]:.4f}")

    if all_fold_metrics:
        summary_df = pd.DataFrame(all_fold_metrics)
        summary_df.to_csv("cold_performance_summary.csv", index=False)
        print("\nFinal Performance Summary (Mean over Folds):")
        print(summary_df.mean())
    else:
        print("Error: No folds were successfully trained.")

if __name__ == "__main__":
    train_cold()
