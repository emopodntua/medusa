import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import *



class DeepLeg(nn.Module):
    def __init__(self, embed_dim, mlp_out):
        """
        Args:
            embed_dim (int): Dimensionality of input and output embeddings.
            mlp_out (int): Dimensionality of the feed-forward layer.
        """
        super(DeepLeg, self).__init__()

        self.linear1 = nn.Linear(embed_dim, mlp_out)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out, mlp_out)

        encoder_layer = nn.TransformerEncoderLayer(d_model=mlp_out, nhead=1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

        self.pooling = MeanPooling()


    def forward(self, x):
        """
        Forward pass of the Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        h1 = self.transformer.layers[0](x)
        h2 = self.transformer.layers[-1](h1)
        h3 = self.pooling(h2)
        
        return h1, h2, h3

