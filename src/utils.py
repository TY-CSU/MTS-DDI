import torch
import numpy as np
import json
from rdkit import Chem
from torch_geometric.data import Data

def parse_biofeat(biofeat_str):
    """Parses BioFeat string (JSON) to Tensor."""
    try:
        feat_list = json.loads(biofeat_str)
        return torch.tensor(feat_list, dtype=torch.float32)
    except (ValueError, json.JSONDecodeError):
        return torch.zeros(3200, dtype=torch.float32)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom, explicit_H=True, use_chirality=False):
    atom_symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
              ]) + [atom.GetIsAromatic()]
              
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results).astype(np.float32)

def edge_features(bond):
    bond_type = bond.GetBondType()
    return [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]

def smiles_to_graph(smiles):
    """
    Converts SMILES to PyG Data object with extended rich features, including edge features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    x = []
    for atom in mol.GetAtoms():
        feat = atom_features(atom, explicit_H=True, use_chirality=True)
        x.append(feat)
        
    x = torch.tensor(np.array(x), dtype=torch.float32)
    
    # Edge Index and Edge Attr
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feat = edge_features(bond)
        
        edge_index.append([u, v])
        edge_attr.append(bond_feat)
        
        # Undirected graph requires adding the reverse direction
        edge_index.append([v, u])
        edge_attr.append(bond_feat)
        
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
