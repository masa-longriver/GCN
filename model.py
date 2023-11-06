import torch
import torch_geometric.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

class GCN(torch.nn.Module):
    def __init__(self, config, dataset):
        super(GCN, self).__init__()
        self.conv1 = nn.GCNConv(
            dataset.num_node_features,
            config['model']['hidden_dim']
        )
        self.conv2 = nn.GCNConv(
            config['model']['hidden_dim'],
            dataset.num_classes
        )
    
    def forward(self, data):
        x = data.x
        edge_index = add_self_loops(data.edge_index)[0]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)