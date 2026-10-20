import torch
import pandas as pd
from torch_geometric.data import Dataset, Batch, Data

class DDIDataset(Dataset):
    def __init__(self, df, drug_data, bio_zero_slice=None):
        super(DDIDataset, self).__init__()
        self.df = df
        self.drug_data = drug_data
        self.bio_zero_slice = bio_zero_slice

        if bio_zero_slice is not None:
            if len(bio_zero_slice) != 2:
                raise ValueError('bio_zero_slice must be a (start, end) pair')
            start, end = bio_zero_slice
            if not (0 <= start < end <= 3200):
                raise ValueError(f'Invalid bio feature slice: {bio_zero_slice}')

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        id_A = row['ID_A']
        id_B = row['ID_B']

        binary_label = torch.tensor([row['Binary_Label']], dtype=torch.float32)
        risk_value = -1 if pd.isna(row['Risk_Label']) else row['Risk_Label']
        risk_label = torch.tensor([risk_value], dtype=torch.long)

        data_A = self._get_drug_data(id_A)
        data_B = self._get_drug_data(id_B)

        return data_A, data_B, binary_label, risk_label

    def _get_drug_data(self, drug_id):
        drug = self.drug_data[drug_id]
        
        x = drug['graph'].x.clone()
        edge_index = drug['graph'].edge_index.clone()
        edge_attr = drug['graph'].edge_attr.clone()
        bio_feat = drug['bio_feat'].clone().squeeze(0) # [3200]
        # E5 ablation operates on a cloned per-sample tensor, so the cached
        # drug representation is never modified. The default path is unchanged.
        if self.bio_zero_slice is not None:
            if bio_feat.dim() != 1 or bio_feat.numel() != 3200:
                raise ValueError(
                    f'Expected a 3200-dimensional bio feature for {drug_id}, '
                    f'got shape {tuple(bio_feat.shape)}'
                )
            start, end = self.bio_zero_slice
            bio_feat[start:end] = 0
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.bio_feat = bio_feat
        return data

def collate_fn(data_list):
    batch_A_list = []
    batch_B_list = []
    bio_A_list = []
    bio_B_list = []
    binary_labels = []
    risk_labels = []
    
    for a, b, y_bin, y_risk in data_list:
        bio_A_list.append(a.bio_feat)
        bio_B_list.append(b.bio_feat)
        
        graph_a = Data(x=a.x, edge_index=a.edge_index, edge_attr=a.edge_attr)
        graph_b = Data(x=b.x, edge_index=b.edge_index, edge_attr=b.edge_attr)
        
        batch_A_list.append(graph_a)
        batch_B_list.append(graph_b)
        binary_labels.append(y_bin)
        risk_labels.append(y_risk)
        
    batch_A = Batch.from_data_list(batch_A_list)
    batch_B = Batch.from_data_list(batch_B_list)
    
    batch_A.bio_feat = torch.stack(bio_A_list)
    batch_B.bio_feat = torch.stack(bio_B_list)
    
    binary_labels = torch.cat(binary_labels)
    risk_labels = torch.cat(risk_labels)
    
    return batch_A, batch_B, binary_labels, risk_labels
