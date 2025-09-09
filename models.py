"""
Model definitions for MIA attacks (WeMeM-main style)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIAFC(nn.Module):
    """Fully Connected MIA Attack Model"""
    def __init__(self, input_dim, output_dim=2):
        super(MIAFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ColumnFC(nn.Module):
    """Column Fully Connected Model"""
    def __init__(self, input_dim, output_dim):
        super(ColumnFC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)