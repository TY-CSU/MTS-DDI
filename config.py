import os

class Config:
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
    BIO_INPUT_DIM = 3200 # From signaturizer
    BIO_HIDDEN_DIM = 256
    JOINT_DIM = 1024
    
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128 # Adjust based on GPU memory
    EPOCHS = 50
    
    # Training Strategy
    PATIENCE = 10          # Early stopping patience
    LR_PATIENCE = 5        # Learning rate scheduler patience
    LR_FACTOR = 0.1        # Learning rate reduction factor
    
    # Loss Weights
    LAMBDA_RISK = 0.5
    
    # Device
    DEVICE = 'cuda' # or 'cpu'
