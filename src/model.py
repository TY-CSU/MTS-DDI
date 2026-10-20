import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class GNN_Encoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(GNN_Encoder, self).__init__()
        # Using GATv2Conv to support edge_attr
        self.conv1 = GATv2Conv(input_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim, edge_dim=edge_dim, heads=2, concat=True)
        self.conv3 = GATv2Conv(hidden_dim * 2, output_dim, edge_dim=edge_dim, heads=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        
        # Global Pooling (Mean + Max) - Original model used this
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return x

class Bio_Projector(nn.Module):
    def __init__(self, input_dim=3200, hidden_dim=1024, output_dim=256):
        super(Bio_Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Gated_Fusion(nn.Module):
    def __init__(self, dim=256):
        super(Gated_Fusion, self).__init__()
        self.gate_weight = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_struct, h_bio):
        combined = torch.cat([h_struct, h_bio], dim=1)
        z = self.sigmoid(self.gate_weight(combined))
        h_fused = z * h_struct + (1 - z) * h_bio
        return h_fused

class MTS_DDI(nn.Module):
    def __init__(self, atom_feature_dim=79, edge_feature_dim=6, bio_input_dim=3200, hidden_dim=256, use_bio=True, use_gate=True):
        super(MTS_DDI, self).__init__()
        self.use_bio = use_bio
        self.use_gate = use_gate
        
        # GNN dimension logic: 
        # Original used output_dim = hidden_dim // 2, then cat(mean, max) -> hidden_dim
        self.gnn = GNN_Encoder(atom_feature_dim, edge_feature_dim, hidden_dim, hidden_dim // 2) 
        
        if self.use_bio:
            self.bio_proj = Bio_Projector(bio_input_dim, 1024, hidden_dim)
            if self.use_gate:
                self.gate = Gated_Fusion(hidden_dim)
            else:
                self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Interaction Layer Dimension: 4 * hidden_dim (e.g. 4 * 256 = 1024)
        self.interaction_dim = 4 * hidden_dim
        
        # Heads - EXACTLY the same as V1
        self.head_binary = nn.Sequential(
            nn.Linear(self.interaction_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
        self.head_risk = nn.Sequential(
            nn.Linear(self.interaction_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3) 
        )

    def encode(self, x, edge_index, edge_attr, batch, bio_feat):
        h_struct = self.gnn(x, edge_index, edge_attr, batch)
        
        if not self.use_bio:
            return h_struct
        
        # Ensure bio_feat is [Batch, Dim]
        if bio_feat.dim() == 3:
             bio_feat = bio_feat.squeeze(1)
             
        h_bio = self.bio_proj(bio_feat)
        
        if self.use_gate:
            h_fused = self.gate(h_struct, h_bio)
        else:
            combined = torch.cat([h_struct, h_bio], dim=1)
            h_fused = self.concat_proj(combined)
            
        return h_fused

    def forward(self, data_A, data_B):
        # data_A and data_B are PyG Batch objects
        H_A = self.encode(data_A.x, data_A.edge_index, data_A.edge_attr, data_A.batch, data_A.bio_feat)
        H_B = self.encode(data_B.x, data_B.edge_index, data_B.edge_attr, data_B.batch, data_B.bio_feat)
        
        # Interaction - EXACTLY the same as V1
        diff = torch.abs(H_A - H_B)
        mult = H_A * H_B
        V_joint = torch.cat([H_A, H_B, mult, diff], dim=1)
        
        return self.head_binary(V_joint), self.head_risk(V_joint)
