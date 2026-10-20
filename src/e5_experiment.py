"""Provenance and independently recomputable results for the E5 experiment."""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def file_manifest(paths):
    result = {}
    for item in paths:
        path = Path(item).resolve()
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        result[str(path)] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    return result


def data_counts(frame):
    """Canonical ordered pair/label digest, independent of DataFrame dtypes/index."""
    canonical = frame[['ID_A', 'ID_B', 'Binary_Label', 'Risk_Label']].copy()
    canonical['ID_A'] = canonical['ID_A'].astype(str)
    canonical['ID_B'] = canonical['ID_B'].astype(str)
    canonical['Binary_Label'] = canonical['Binary_Label'].astype('int64')
    canonical['Risk_Label'] = canonical['Risk_Label'].fillna(-1).astype('int64')
    binary = canonical['Binary_Label']
    risk = canonical['Risk_Label']
    if not binary.isin([0, 1]).all() or not risk.isin([-1, 0, 1, 2]).all():
        raise ValueError('Unexpected binary or severity label')
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    return {
        'pairs': len(frame),
        'binary_counts': {str(k): int((binary == k).sum()) for k in (0, 1)},
        'risk_counts': {str(k): int((risk == k).sum()) for k in (-1, 0, 1, 2)},
        'ordered_pair_label_sha256': hashlib.sha256(row_hashes.tobytes()).hexdigest(),
    }


def run_metadata(project_root, fold, seed, device):
    root = Path(project_root)
    sources = ['src/dataset.py', 'src/model.py', 'src/train_cold_ablation.py',
               'src/e5_experiment.py', 'scripts/validate_e5_ablation.py',
               'src/generate_bio_features.py', 'sbatch_ablation/run_no_e5.sh']
    return {
        'ablation': 'no_e5', 'fold': fold,
        'display_name': 'w/o E5 (DDI space)',
        'bio_input_dim': 3200, 'masked_slice_0_based': [3072, 3200],
        'masked_dimensions': 128, 'mask_applies_to': ['drug_A', 'drug_B'],
        'mask_stages': ['train', 'validation', 'final_evaluation'],
        'training': {
            'optimizer': 'Adam', 'learning_rate': 1e-4, 'weight_decay': 0,
            'batch_size': 256, 'max_epochs': 50, 'early_stopping_patience': 10,
            'lr_scheduler': 'ReduceLROnPlateau', 'lr_factor': 0.1, 'lr_patience': 5,
            'ohnm_neg_ratio': 2.0, 'risk_loss_weight': 0.5,
            'risk_class_weights': [10.0, 1.0, 3.0], 'risk_ignore_index': -1,
            'checkpoint_selection': 'minimum validation loss',
            'validation_used_for_checkpoint_selection_and_metrics': True,
            'binary_threshold_rule': 'probability > 0.5',
            'risk_decision_rule': 'argmax over minor/moderate/major',
            'fold_seed': seed, 'torch_initial_seed': torch.initial_seed(),
            'historical_full_and_no_bio_seeds': 'not recorded',
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        },
        'environment': {
            'python': sys.version, 'torch': torch.__version__,
            'torch_geometric': torch_geometric.__version__, 'numpy': np.__version__,
            'pandas': pd.__version__, 'sklearn': sklearn.__version__,
            'cuda_runtime': torch.version.cuda, 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'hostname': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        },
        'source_files': file_manifest([root / path for path in sources]),
        'command': sys.argv,
        'started_at': datetime.now().astimezone().isoformat(),
        'interpretation': (
            'Retrained ablation of the E5 DDI-space input block. Removing this block '
            'does not establish that all remaining pretrained spaces are free of DDI-derived information.'
        ),
    }


def metrics_from_predictions(predictions):
    binary = predictions['binary_label']
    probability = predictions['binary_probability']
    risk_all = predictions['risk_label_all']
    risk = predictions['risk_label']
    risk_probability = predictions['risk_probability']
    if probability.shape != binary.shape or len(risk_all) != len(binary):
        raise ValueError('Binary prediction/label shapes differ')
    if risk_probability.shape != (len(risk), 3):
        raise ValueError('Risk prediction/label shapes differ')
    np.testing.assert_array_equal(risk_all[risk_all != -1], risk)
    if not np.isfinite(probability).all() or not np.isfinite(risk_probability).all():
        raise ValueError('Non-finite prediction')
    if not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Binary probabilities outside [0,1]')
    if not ((risk_probability >= 0) & (risk_probability <= 1)).all():
        raise ValueError('Risk probabilities outside [0,1]')
    np.testing.assert_allclose(risk_probability.sum(axis=1), 1, atol=1e-6)
    prediction = (probability > 0.5).astype(int)
    risk_prediction = risk_probability.argmax(axis=1)
    precision, recall, _ = precision_recall_curve(binary, probability)
    class_f1 = f1_score(risk, risk_prediction, labels=[0, 1, 2], average=None)
    metrics = {
        'Bin_Acc': accuracy_score(binary, prediction),
        'Bin_AUC': roc_auc_score(binary, probability),
        'Bin_F1': f1_score(binary, prediction), 'Bin_PR': auc(recall, precision),
        'Risk_Acc': accuracy_score(risk, risk_prediction),
        'Risk_AUC': roc_auc_score(np.eye(3)[risk], risk_probability,
                                  multi_class='ovr', average='macro'),
        'Risk_F1_Macro': float(class_f1.mean()),
        'Risk_F1_Micro': f1_score(risk, risk_prediction, average='micro'),
        'Risk_Minor_F1': class_f1[0], 'Risk_Moderate_F1': class_f1[1],
        'Risk_Major_F1': class_f1[2],
    }
    confusion = {'binary': confusion_matrix(binary, prediction, labels=[0, 1]).tolist(),
                 'risk': confusion_matrix(risk, risk_prediction, labels=[0, 1, 2]).tolist()}
    return metrics, confusion


def _write_csv_once(frame, path):
    """Allow repeat verification, but never silently replace different results."""
    if Path(path).exists():
        prior = pd.read_csv(path)
        normalized = frame.copy()
        if 'Fold' in normalized:
            normalized['Fold'] = normalized['Fold'].astype(str)
            prior['Fold'] = prior['Fold'].astype(str)
        pd.testing.assert_frame_equal(prior, normalized, check_dtype=False, atol=1e-10, rtol=1e-8)
    else:
        frame.to_csv(path, index=False, mode='x')


def collect_e5_results(project_root):
    root = Path(project_root)
    report_path = root / 'analysis_results/e5_ablation_validation.json'
    with report_path.open(encoding='utf-8') as handle:
        preflight = json.load(handle)
    if preflight['status'] != 'passed':
        raise ValueError('Preflight validation must pass before result collection')
    current_inputs = file_manifest(preflight['data_files'])
    if current_inputs != preflight['data_files']:
        raise ValueError('Input files changed after preflight validation')
    rows, checked_folds = [], []
    source_manifest = None
    for fold in range(5):
        directory = root / 'trained_models' / f'cold_fold_{fold}_no_e5'
        with (directory / 'experiment_metadata.json').open(encoding='utf-8') as handle:
            metadata = json.load(handle)
        with (directory / 'dataset_metadata.json').open(encoding='utf-8') as handle:
            dataset = json.load(handle)
        if metadata['status'] != 'complete' or metadata['fold'] != fold:
            raise ValueError(f'Fold {fold} is incomplete or misidentified')
        if metadata['masked_slice_0_based'] != [3072, 3200] or metadata['bio_input_dim'] != 3200:
            raise ValueError(f'Fold {fold} has the wrong feature mask')
        if metadata['data_files'] != current_inputs:
            raise ValueError(f'Fold {fold} used different input files')
        if source_manifest is None:
            source_manifest = metadata['source_files']
        elif metadata['source_files'] != source_manifest:
            raise ValueError('Code changed between E5 folds')
        for subset in ('train', 'val'):
            if dataset[subset] != preflight['folds'][str(fold)][subset]:
                raise ValueError(f'Fold {fold} {subset} membership/labels differ')
        checkpoint = str(directory / 'best_model.pth')
        if file_manifest([checkpoint])[checkpoint]['sha256'] != metadata['checkpoint_sha256']:
            raise ValueError(f'Fold {fold} checkpoint changed after evaluation')
        with np.load(directory / 'val_predictions.npz', allow_pickle=False) as predictions:
            prediction_rows = pd.DataFrame({
                'ID_A': predictions['id_a'], 'ID_B': predictions['id_b'],
                'Binary_Label': predictions['binary_label'], 'Risk_Label': predictions['risk_label_all'],
            })
            if data_counts(prediction_rows) != dataset['val']:
                raise ValueError(f'Fold {fold} predictions do not match its validation rows')
            metrics, confusion = metrics_from_predictions(predictions)
        csv = pd.read_csv(directory / 'fold_metrics.csv')
        if len(csv) != 1 or int(csv.iloc[0]['Fold']) != fold or csv.iloc[0]['Ablation'] != 'no_e5':
            raise ValueError(f'Invalid metrics row for fold {fold}')
        for key, value in metrics.items():
            np.testing.assert_allclose(value, csv.iloc[0][key], atol=1e-10, rtol=1e-8,
                                       err_msg=f'Fold {fold}: {key}')
        rows.append({'Fold': fold, 'Ablation': 'no_e5', **metrics})
        checked_folds.append({'fold': fold, 'metrics_recomputed': True, 'confusion_matrices': confusion,
                              'best_epoch': metadata['best_epoch'], 'epochs_completed': metadata['epochs_completed'],
                              'checkpoint': checkpoint, 'seed': metadata['training']['fold_seed']})
    folds = pd.DataFrame(rows)
    metric_columns = [key for key in folds if key not in ('Fold', 'Ablation')]
    summary = pd.concat([folds, pd.DataFrame([
        {'Fold': 'Mean', 'Ablation': 'no_e5', **folds[metric_columns].mean().to_dict()},
        {'Fold': 'Std', 'Ablation': 'no_e5', **folds[metric_columns].std(ddof=1).to_dict()},
    ])], ignore_index=True)
    summary_path = root / 'cold_ablation_no_e5_summary.csv'
    _write_csv_once(summary, summary_path)
    comparisons = []
    sources = [('Full', root / 'cold_performance_summary.csv'),
               ('w/o E5 (DDI space)', summary_path),
               ('w/o Bio', root / 'cold_ablation_no_bio_summary.csv')]
    for variant, path in sources:
        frame = pd.read_csv(path)
        frame = frame[pd.to_numeric(frame['Fold'], errors='coerce').notna()]
        if sorted(frame['Fold'].astype(int)) != list(range(5)):
            raise ValueError(f'{variant} must contain exactly folds 0-4')
        for metric in metric_columns:
            comparisons.append({'Variant': variant, 'Metric': metric, 'N_Folds': 5,
                                'Mean': frame[metric].mean(), 'Sample_SD': frame[metric].std(ddof=1)})
    comparison_path = root / 'analysis_results/e5_ablation_comparison.csv'
    _write_csv_once(pd.DataFrame(comparisons), comparison_path)
    result = {
        'status': 'complete_and_verified', 'ablation': 'no_e5',
        'bio_input_dim': 3200, 'masked_slice_0_based': [3072, 3200],
        'folds': checked_folds, 'summary_file': str(summary_path),
        'comparison_file': str(comparison_path), 'standard_deviation_ddof': 1,
        'data_files': current_inputs, 'source_files': source_manifest,
        'comparison_sources': file_manifest([path for _, path in sources]),
        'limits': ['Historical full/no-bio training seeds are not recorded.',
                   'Development-fold validation is also used for checkpoint selection.',
                   'This experiment removes the E5 input block; it does not audit all pretraining overlap.'],
    }
    save_json(root / 'analysis_results/e5_ablation_metadata.json', result)
    print(summary.to_string(index=False))
    print(f'Verified five folds. Saved {summary_path} and {comparison_path}')
    return summary
