"""
 Cold-Start Ablation Training Script.
Supports: --no_bio, --no_e5, --no_gate, --no_ohnm
Usage:
    python src/train_cold_ablation.py --fold 0 --no_bio
    python src/train_cold_ablation.py --fold 0 --no_e5
    python src/train_cold_ablation.py --fold 0 --no_gate
    python src/train_cold_ablation.py --fold 0 --no_ohnm
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
import random
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.dataset import DDIDataset, collate_fn
from src.model import MTS_DDI
from src.e5_experiment import data_counts, file_manifest, run_metadata, save_json


BIO_INPUT_DIM = 3200
E5_BIO_SLICE = (3072, 3200)


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


def train_cold_ablation(fold_idx, use_bio=True, use_ohnm=True, use_gate=True, use_e5=True, seed=None):
    if fold_idx not in range(5):
        raise ValueError('fold_idx must be 0-4')
    if not use_e5 and not (use_bio and use_ohnm and use_gate):
        raise ValueError('The E5 experiment must keep Bio, Gate and OHNM enabled')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Bio Features: {'Enabled' if use_bio else 'Disabled'}")
    print(f"E5 DDI Signature Block: {'Enabled' if use_e5 else 'Masked'}")
    print(f"OHNM: {'Enabled' if use_ohnm else 'Disabled'}")
    print(f"Gated Fusion: {'Enabled' if use_gate else 'Disabled (Concat)'}")
    
    # Checkpoint Directory
    suffix = ""
    if not use_bio:
        suffix += "_no_bio"
    if not use_e5:
        suffix += "_no_e5"
    if not use_ohnm:
        suffix += "_no_ohnm"
    if not use_gate:
        suffix += "_no_gate"
    
    ckpt_dir = os.path.join(Config.PROJECT_ROOT, 'trained_models', f'cold_fold_{fold_idx}{suffix}')
    
    # Skip fold if already completed (best_model.pth + fold_metrics.csv both exist)
    best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
    fold_metrics_path = os.path.join(ckpt_dir, 'fold_metrics.csv')
    if os.path.exists(best_model_path) and os.path.exists(fold_metrics_path):
        if not use_e5:
            for name in ('experiment_metadata.json', 'dataset_metadata.json', 'val_predictions.npz'):
                if not os.path.isfile(os.path.join(ckpt_dir, name)):
                    raise RuntimeError(f'Incomplete E5 provenance: {ckpt_dir}/{name}')
        print(f"Fold {fold_idx}{suffix} already completed (found {best_model_path} and {fold_metrics_path}), skipping.")
        return

    if not use_e5 and os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
        raise FileExistsError(f'Refusing to overwrite an incomplete E5 run: {ckpt_dir}')

    fold_seed = None if seed is None else seed + fold_idx
    if fold_seed is not None:
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")
    
    # Load Data
    drug_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'drug_data.pt')
    split_file = os.path.join(Config.PROCESSED_DATA_DIR, 'drug_cold_split.csv')
    processed_ddi_file = os.path.join(Config.PROCESSED_DATA_DIR, 'master_ddi_dataset.csv')
    
    if not os.path.exists(drug_data_path):
        print(f"Error: {drug_data_path} not found.")
        return

    drug_data = torch.load(drug_data_path, map_location='cpu', weights_only=False)
    if not use_e5:
        bad_shapes = []
        feature_shapes = {}
        for drug_id, record in drug_data.items():
            feat = record['bio_feat']
            feature_shapes[str(tuple(feat.shape))] = feature_shapes.get(str(tuple(feat.shape)), 0) + 1
            if tuple(feat.shape) not in ((3200,), (1, 3200)) or not torch.isfinite(feat).all():
                bad_shapes.append({'Drug_ID': drug_id, 'shape': list(feat.shape), 'numel': int(feat.numel())})
        if bad_shapes:
            raise ValueError(
                f'E5 ablation requires {BIO_INPUT_DIM}-dimensional bio features; '
                f'found {len(bad_shapes)} invalid records, examples={bad_shapes[:3]}'
            )
        print(f"Verified {len(drug_data)} cached bio features; shapes={feature_shapes}")
    split_df = pd.read_csv(split_file)
    ddi_df = pd.read_csv(processed_ddi_file)

    split_df['Split_Group'] = split_df['Split_Group'].astype(str)
    drug_split = dict(zip(split_df['Drug_ID'], split_df['Split_Group']))

    # Filter data for this fold
    fold_str = str(fold_idx)
    relevant_df = ddi_df[ddi_df['ID_A'].isin(drug_data) & ddi_df['ID_B'].isin(drug_data)]
    
    train_pairs = []
    val_pairs = []
    
    for _, row in relevant_df.iterrows():
        id_a, id_b = row['ID_A'], row['ID_B']
        g_a, g_b = drug_split.get(id_a), drug_split.get(id_b)
        
        if g_a == 'test' or g_b == 'test':
            continue
            
        if g_a != fold_str and g_b != fold_str:
            train_pairs.append(row)
        elif g_a == fold_str or g_b == fold_str:
            val_pairs.append(row)
            
    train_df = pd.DataFrame(train_pairs).reset_index(drop=True)
    val_df = pd.DataFrame(val_pairs).reset_index(drop=True)
    
    print(f"Fold {fold_idx}: Train Pairs={len(train_df)}, Val Pairs={len(val_df)}")
    
    bio_zero_slice = E5_BIO_SLICE if (use_bio and not use_e5) else None
    train_loader = DataLoader(
        DDIDataset(train_df, drug_data, bio_zero_slice=bio_zero_slice),
        batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        DDIDataset(val_df, drug_data, bio_zero_slice=bio_zero_slice),
        batch_size=256, shuffle=False, collate_fn=collate_fn
    )

    if not use_e5:
        risk_values = pd.to_numeric(val_df['Risk_Label'], errors='coerce')
        fold_metadata = {
            'ablation': 'no_e5',
            'fold': fold_idx,
            'bio_input_dim': BIO_INPUT_DIM,
            'masked_slice_0_based': list(E5_BIO_SLICE),
            'masked_dimensions': E5_BIO_SLICE[1] - E5_BIO_SLICE[0],
            'train_pairs': int(len(train_df)),
            'val_pairs': int(len(val_df)),
            'val_binary_positive': int((val_df['Binary_Label'] == 1).sum()),
            'val_binary_negative': int((val_df['Binary_Label'] == 0).sum()),
            'val_risk_labeled': int((risk_values.notna() & (risk_values != -1)).sum()),
            'val_risk_missing': int((risk_values.isna() | (risk_values == -1)).sum()),
            'train': data_counts(train_df),
            'val': data_counts(val_df),
            'data_files': {
                'drug_data': drug_data_path,
                'cold_split': split_file,
                'master_dataset': processed_ddi_file,
            },
            'created_at': datetime.now().astimezone().isoformat(),
        }
        validation_path = os.path.join(Config.PROJECT_ROOT, 'analysis_results', 'e5_ablation_validation.json')
        if not os.path.isfile(validation_path):
            raise RuntimeError('Run scripts/validate_e5_ablation.py before starting E5 training')
        with open(validation_path, encoding='utf-8') as f:
            expected = json.load(f)
        if expected['status'] != 'passed':
            raise RuntimeError('E5 preflight validation did not pass')
        for name in ('train', 'val'):
            if fold_metadata[name] != expected['folds'][str(fold_idx)][name]:
                raise RuntimeError(f'Fold {fold_idx} {name} differs from the validated data')
        save_json(os.path.join(ckpt_dir, 'dataset_metadata.json'), fold_metadata)
        metadata = run_metadata(Config.PROJECT_ROOT, fold_idx, fold_seed, device)
        metadata['data_files'] = file_manifest([drug_data_path, split_file, processed_ddi_file])
        metadata['status'] = 'training'
        save_json(os.path.join(ckpt_dir, 'experiment_metadata.json'), metadata)

    # Model -  uses atom_feature_dim=79, edge_feature_dim=6
    model = MTS_DDI(
        atom_feature_dim=79, edge_feature_dim=6,
        bio_input_dim=BIO_INPUT_DIM, hidden_dim=256,
        use_bio=use_bio, use_gate=use_gate if use_bio else False
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    criterion_binary = nn.BCEWithLogitsLoss()
    risk_weights = torch.tensor([10.0, 1.0, 3.0]).to(device)
    criterion_risk = nn.CrossEntropyLoss(weight=risk_weights, ignore_index=-1)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 10
    best_epoch = None
    
    print(f"\n===== Starting  Cold-Start Ablation Fold {fold_idx}{suffix} =====")
    
    for epoch in range(50):
        # === Training ===
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Ep {epoch+1} [Train]", leave=False)
        
        for bA, bB, yb, yr in pbar:
            bA, bB, yb, yr = bA.to(device), bB.to(device), yb.to(device), yr.to(device)
            optimizer.zero_grad()
            ob, or_logits = model(bA, bB)
            
            # Binary loss
            if use_ohnm:
                loss_bin = compute_binary_loss_ohnm(ob, yb.unsqueeze(1) if yb.dim() == 1 else yb, neg_ratio=2.0)
            else:
                loss_bin = criterion_binary(ob.squeeze(), yb)
            
            # Risk loss
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
        with torch.no_grad():
            for bA, bB, yb, yr in val_loader:
                bA, bB, yb, yr = bA.to(device), bB.to(device), yb.to(device), yr.to(device)
                ob, or_logits = model(bA, bB)
                
                loss_bin = criterion_binary(ob.squeeze(), yb)
                mask = yr != -1
                if mask.any():
                    loss_risk = criterion_risk(or_logits[mask], yr[mask])
                    if torch.isnan(loss_risk):
                        loss = loss_bin
                    else:
                        loss = loss_bin + 0.5 * loss_risk
                else:
                    loss = loss_bin
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{patience}")
        if not use_e5:
            pd.DataFrame([{'epoch': epoch + 1, 'train_loss': avg_train_loss,
                           'val_loss': avg_val_loss, 'best_epoch': best_epoch,
                           'learning_rate': optimizer.param_groups[0]['lr']}]).to_csv(
                os.path.join(ckpt_dir, 'training_history.csv'), mode='a', header=(epoch == 0), index=False)

        if early_stopping_counter >= patience:
            print("  Early stopping triggered!")
            break
    
    # === Final Evaluation ===
    if os.path.exists(os.path.join(ckpt_dir, 'best_model.pth')):
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pth'), map_location=device, weights_only=True))
        model.eval()
        
        p_bin, l_bin, p_risk, l_risk = [], [], [], []
        with torch.no_grad():
            for bA, bB, yb, yr in val_loader:
                ob, or_l = model(bA.to(device), bB.to(device))
                p_bin.extend(torch.sigmoid(ob).cpu().numpy().flatten())
                l_bin.extend(yb.numpy())
                mask = yr != -1
                if mask.any():
                    p_risk.extend(torch.softmax(or_l[mask], dim=1).cpu().numpy())
                    l_risk.extend(yr[mask].numpy())
        
        # Binary metrics
        bin_m = calculate_metrics_local(np.array(l_bin), np.array(p_bin))
        
        # Risk metrics
        risk_acc = accuracy_score(l_risk, np.argmax(p_risk, axis=1)) if l_risk else 0
        try:
            risk_auc = roc_auc_score(pd.get_dummies(l_risk), p_risk, multi_class='ovr', average='macro')
        except:
            risk_auc = 0
        risk_f1 = f1_score(l_risk, np.argmax(p_risk, axis=1), average='macro') if l_risk else 0
        risk_f1_micro = f1_score(l_risk, np.argmax(p_risk, axis=1), average='micro') if l_risk else 0
        risk_class_f1 = f1_score(l_risk, np.argmax(p_risk, axis=1), average=None) if l_risk else [0,0,0]
        
        result = {
            'Fold': fold_idx,
            'Ablation': suffix.strip('_') if suffix else 'full',
            'Bin_Acc': bin_m['Accuracy'],
            'Bin_AUC': bin_m['AUROC'],
            'Bin_F1': bin_m['F1_score'],
            'Bin_PR': bin_m['AUPRC'],
            'Risk_Acc': risk_acc,
            'Risk_AUC': risk_auc,
            'Risk_F1_Macro': risk_f1,
            'Risk_F1_Micro': risk_f1_micro,
            'Risk_Minor_F1': risk_class_f1[0] if len(risk_class_f1)>0 else 0,
            'Risk_Moderate_F1': risk_class_f1[1] if len(risk_class_f1)>1 else 0,
            'Risk_Major_F1': risk_class_f1[2] if len(risk_class_f1)>2 else 0,
        }
        
        # Save per-fold result
        result_df = pd.DataFrame([result])
        result_file = os.path.join(ckpt_dir, 'fold_metrics.csv')
        result_df.to_csv(result_file, index=False)

        if not use_e5:
            np.savez_compressed(
                os.path.join(ckpt_dir, 'val_predictions.npz'),
                id_a=val_df['ID_A'].to_numpy(dtype=str),
                id_b=val_df['ID_B'].to_numpy(dtype=str),
                binary_label=np.asarray(l_bin, dtype=np.int64),
                binary_probability=np.asarray(p_bin),
                risk_label_all=val_df['Risk_Label'].fillna(-1).to_numpy(dtype=np.int64),
                risk_label=np.asarray(l_risk, dtype=np.int64),
                risk_probability=np.asarray(p_risk),
            )
            metadata.update(status='complete', best_epoch=best_epoch, epochs_completed=epoch + 1,
                            best_val_loss=best_val_loss, metrics=result,
                            checkpoint_sha256=file_manifest([best_model_path])[best_model_path]['sha256'],
                            completed_at=datetime.now().astimezone().isoformat())
            save_json(os.path.join(ckpt_dir, 'experiment_metadata.json'), metadata)
        
        print(f"\n===== Fold {fold_idx}{suffix} Final Results =====")
        print(f"  Binary  - Acc: {bin_m['Accuracy']:.4f}, AUC: {bin_m['AUROC']:.4f}, F1: {bin_m['F1_score']:.4f}")
        print(f"  Risk    - Acc: {risk_acc:.4f}, AUC: {risk_auc:.4f}, F1: {risk_f1:.4f}")
        print(f"  Risk per-class F1: Minor={risk_class_f1[0]:.4f}, Moderate={risk_class_f1[1]:.4f}, Major={risk_class_f1[2]:.4f}")
    else:
        print("No best model saved, skipping final evaluation.")


def collect_ablation_results(ablation_suffix, project_root):
    """Collect per-fold results into a summary CSV after all folds complete."""
    if ablation_suffix == '_no_e5':
        from src.e5_experiment import collect_e5_results
        return collect_e5_results(project_root)
    all_results = []
    for fold in range(5):
        ckpt_dir = os.path.join(project_root, 'trained_models', f'cold_fold_{fold}{ablation_suffix}')
        csv_file = os.path.join(ckpt_dir, 'fold_metrics.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_results.append(df)
        else:
            print(f"  WARNING: {csv_file} not found, skipping fold {fold}")
    
    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        # Add mean row
        mean_row = summary_df.mean(numeric_only=True).to_dict()
        mean_row['Fold'] = 'Mean'
        mean_row['Ablation'] = summary_df['Ablation'].iloc[0] if len(summary_df) > 0 else ''
        summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)
        
        # Save summary
        ablation_name = ablation_suffix.strip('_') if ablation_suffix else 'full'
        summary_file = os.path.join(project_root, f'cold_ablation_{ablation_name}_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n===== Ablation Summary ({ablation_name}) =====")
        print(summary_df.to_string(index=False))
        print(f"Saved to: {summary_file}")

        if ablation_name == 'no_e5':
            metadata = {
                'status': 'complete' if len(all_results) == 5 else 'partial',
                'ablation': 'no_e5',
                'description': 'E5 DDI-derived signature block masked at model input',
                'bio_input_dim': BIO_INPUT_DIM,
                'masked_slice_0_based': list(E5_BIO_SLICE),
                'masked_dimensions': E5_BIO_SLICE[1] - E5_BIO_SLICE[0],
                'folds_found': [int(x) for x in summary_df.loc[summary_df['Fold'] != 'Mean', 'Fold'].tolist()],
                'summary_file': summary_file,
                'fold_metadata': [],
                'training': {
                    'protocol': 'five-fold scaffold cold-start',
                    'optimizer': 'Adam',
                    'learning_rate': 1e-4,
                    'batch_size': 256,
                    'max_epochs': 50,
                    'early_stopping_patience': 10,
                    'ohnm_neg_ratio': 2.0,
                    'risk_loss_weight': 0.5,
                    'risk_class_weights': [10.0, 1.0, 3.0],
                },
                'created_at': datetime.now().astimezone().isoformat(),
            }
            for fold in range(5):
                metadata_path = os.path.join(
                    project_root, 'trained_models', f'cold_fold_{fold}_no_e5', 'dataset_metadata.json'
                )
                if os.path.exists(metadata_path):
                    with open(metadata_path, encoding='utf-8') as f:
                        metadata['fold_metadata'].append(json.load(f))
            metadata_path = os.path.join(project_root, 'analysis_results', 'e5_ablation_metadata.json')
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Saved E5 metadata to: {metadata_path}")
    else:
        print("No fold results found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Cold-Start Ablation Training')
    parser.add_argument('--fold', type=int, default=-1, help='Fold index (0-4), or -1 to run all folds sequentially')
    parser.add_argument('--no_bio', action='store_true', help='Disable biological features')
    parser.add_argument('--no_e5', action='store_true', help='Mask the E5 DDI signature block (dimensions 3072:3200)')
    parser.add_argument('--no_ohnm', action='store_true', help='Disable Online Hard Negative Mining')
    parser.add_argument('--no_gate', action='store_true', help='Disable Gated Fusion (use Concat instead)')
    parser.add_argument('--seed', type=int, default=None, help='Optional base seed; each fold uses seed + fold')
    parser.add_argument('--require_cuda', action='store_true', help='Fail if no GPU is available')
    parser.add_argument('--collect_only', action='store_true', help='Collect completed folds without training')
    args = parser.parse_args()
    if args.fold not in range(-1, 5):
        parser.error('--fold must be -1 or 0-4')
    if args.no_e5 and (args.no_bio or args.no_gate or args.no_ohnm):
        parser.error('--no_e5 is a standalone ablation; keep Bio, Gate and OHNM enabled')
    if args.require_cuda and not torch.cuda.is_available():
        parser.error('CUDA is required; submit this experiment to a GPU node')
    
    use_bio = not args.no_bio
    use_e5 = not args.no_e5
    use_ohnm = not args.no_ohnm
    use_gate = not args.no_gate
    
    suffix = ""
    if not use_bio: suffix += "_no_bio"
    if not use_e5: suffix += "_no_e5"
    if not use_ohnm: suffix += "_no_ohnm"
    if not use_gate: suffix += "_no_gate"
    
    if args.collect_only:
        collect_ablation_results(suffix, Config.PROJECT_ROOT)
    elif args.fold >= 0:
        # Single fold
        train_cold_ablation(args.fold, use_bio=use_bio, use_e5=use_e5, use_ohnm=use_ohnm, use_gate=use_gate, seed=args.seed)
    else:
        # Run all 5 folds sequentially
        for fold in range(5):
            train_cold_ablation(fold, use_bio=use_bio, use_e5=use_e5, use_ohnm=use_ohnm, use_gate=use_gate, seed=args.seed)
        # Collect results
        collect_ablation_results(suffix, Config.PROJECT_ROOT)
