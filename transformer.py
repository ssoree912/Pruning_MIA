"""
Transformer-based MIA attack model (simplified version for WeMeM-main compatibility)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    """Simplified Transformer for MIA attacks (SAMIA-style)"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_heads=4, num_layers=2):
        super(Transformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # Project input to hidden dimension
        x = self.input_proj(x)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, 1, hidden_dim]
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # Classification
        x = self.classifier(x)
        
        return x