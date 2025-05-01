import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
logger = logging.getLogger(__name__)

class SpatialAttention(nn.Module):
    """
    Spatial attention module for capturing relationships between locations
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        """
        Initialize the spatial attention module
        
        Args:
            hidden_dim (int): Dimension of hidden features
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, spatial_matrix=None, mask=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_dim]
            spatial_matrix (torch.Tensor, optional): Spatial relation matrix [batch_size, seq_len, seq_len]
            mask (torch.Tensor, optional): Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Output tensor after spatial attention
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Project input to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply spatial bias if provided
        if spatial_matrix is not None:
            try:
                # Process spatial matrix to match required dimensions
                if spatial_matrix.dim() == 2:  # [seq_len, seq_len]
                    # Reshape to [1, 1, seq_len, seq_len] and expand to batch_size
                    spatial_bias = spatial_matrix.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
                elif spatial_matrix.dim() == 3:  # [batch_size, seq_len, seq_len]
                    # Reshape to [batch_size, 1, seq_len, seq_len] and expand heads dimension
                    spatial_bias = spatial_matrix.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
                else:
                    raise ValueError(f"Unsupported spatial matrix dimension: {spatial_matrix.dim()}")
                
                # Normalize spatial bias and add to attention scores
                spatial_bias = F.softmax(spatial_bias, dim=-1)
                scores = scores + spatial_bias
            except Exception as e:
                logger.warning(f"Error applying spatial bias: {str(e)}. Skipping spatial bias.")
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        output = self.out_proj(attn_output)
        
        return output


class MultiMatrixSpatialAttention(nn.Module):
    """
    Spatial attention module using multiple spatial matrices
    """
    def __init__(self, hidden_dim, num_matrices=3, num_heads=4, dropout=0.1):
        """
        Initialize the multi-matrix spatial attention
        
        Args:
            hidden_dim (int): Dimension of hidden features
            num_matrices (int): Number of spatial matrices
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(MultiMatrixSpatialAttention, self).__init__()
        self.num_matrices = num_matrices
        
        # Create attention modules for each matrix
        self.attentions = nn.ModuleList([
            SpatialAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_matrices)
        ])
        
        # Matrix weights (learnable)
        self.matrix_weights = nn.Parameter(torch.ones(num_matrices) / num_matrices)
        
        # Final projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, spatial_matrices=None, mask=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_dim]
            spatial_matrices (list, optional): List of spatial matrices
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor after combined spatial attention
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Apply softmax to matrix weights
        matrix_weights = F.softmax(self.matrix_weights, dim=0)
        
        # If no spatial matrices provided, use default attention
        if spatial_matrices is None or len(spatial_matrices) == 0:
            # Just use the first attention module without spatial bias
            output = self.attentions[0](x, None, mask)
            output = self.out_proj(output)
            output = self.dropout(output)
            return output
        
        # Apply each attention module with its corresponding matrix
        outputs = []
        
        for i, attention in enumerate(self.attentions):
            if i < len(spatial_matrices) and spatial_matrices[i] is not None:
                try:
                    matrix_output = attention(x, spatial_matrices[i], mask)
                    outputs.append(matrix_output * matrix_weights[i])
                except Exception as e:
                    logger.warning(f"Error in attention {i}: {str(e)}. Falling back to standard attention.")
                    matrix_output = attention(x, None, mask)
                    outputs.append(matrix_output * matrix_weights[i])
            else:
                # If matrix is not provided, use regular attention
                matrix_output = attention(x, None, mask)
                outputs.append(matrix_output * matrix_weights[i])
        
        # Combine outputs
        combined_output = sum(outputs)
        
        # Final projection
        output = self.out_proj(combined_output)
        output = self.dropout(output)
        
        return output


class CoordinateEmbedding(nn.Module):
    """
    Module for embedding spatial coordinates
    """
    def __init__(self, coord_dim=2, hidden_dim=32):
        """
        Initialize coordinate embedding
        
        Args:
            coord_dim (int): Number of coordinate dimensions (usually 2 for X, Y)
            hidden_dim (int): Dimension of hidden features
        """
        super(CoordinateEmbedding, self).__init__()
        self.coord_embed = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, coords):
        """
        Forward pass
        
        Args:
            coords (torch.Tensor): Coordinates tensor [batch_size, coord_dim]
            
        Returns:
            torch.Tensor: Embedded coordinates
        """
        return self.coord_embed(coords)