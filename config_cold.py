import os

class ConfigCold:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    # External Data Paths
    DDI_FILE = os.path.join(RAW_DATA_DIR, 'ddinter_merged_with_drugbank.csv')
    DRUGBANK_DDI_FILE = os.path.join(RAW_DATA_DIR, 'drugbank_ddi.csv')
    
    # Mappings
    ID_TO_SMILES_FILE = os.path.join(PROCESSED_DATA_DIR, 'drugbank_smiles_mapping.csv')
    BIO_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'drugbank_bio_features.csv')
    
    # Model Hyperparameters
    GNN_HIDDEN_DIM = 256
    BIO_INPUT_DIM = 3200 
    BIO_HIDDEN_DIM = 256
    JOINT_DIM = 1024
    
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128 
    EPOCHS = 6
    
    # Training Strategy
    PATIENCE = 10          
    LR_PATIENCE = 5        
    LR_FACTOR = 0.1        
    
    # Loss Weights & Hyperparameters
    LAMBDA_RISK = 0.5
    
    # OHNM Settings
    # Lowering ratio from 3.0 to 1.0 or 2.0 might help Recall (less aggressive on negatives)
    OHNM_NEG_RATIO = 2.0 
    
    # Class Weights for Risk (Will be calculated dynamically or set manually)
    # Approximate counts: Minor: ~5%, Moderate: ~75%, Major: ~20%
    # Inverse weights: 1/0.05=20, 1/0.75=1.33, 1/0.2=5
    # Normalized roughly: [10.0, 1.0, 3.0]
    RISK_CLASS_WEIGHTS = [10.0, 1.0, 3.0] 
    
    # Device
    DEVICE = 'cuda' 