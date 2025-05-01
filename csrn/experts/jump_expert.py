import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


class MultiEthnicJumpProcessExpert(nn.Module):
    """
    Expert module for handling abrupt changes in population for multiple ethnicities simultaneously.
    
    This expert focuses on modeling discontinuous jumps in population values
    that cannot be well-captured by continuous diffusion processes. It implements
    a stochastic jump process model that can handle large, sudden population changes
    across multiple ethnic groups.
    """
    
    def __init__(self, 
                 num_features: int,
                 num_ethnicities: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 log_space: bool = False,
                 epsilon: float = 1e-6,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Multi-Ethnic Jump Process Expert.
        
        Args:
            num_features: Number of input features
            num_ethnicities: Number of ethnicities to model
            hidden_dim: Hidden dimension for internal representations
            num_heads: Number of attention heads
            dropout: Dropout rate
            log_space: Whether to operate in log space
            epsilon: Small value for numerical stability
            device: Device to use for computations
        """
        super(MultiEthnicJumpProcessExpert, self).__init__()
        self.num_features = num_features
        self.num_ethnicities = num_ethnicities
        self.hidden_dim = hidden_dim
        self.device = device
        self.log_space = log_space
        self.epsilon = epsilon
        self.dauid_list = None
        # Shared feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Ethnicity-specific feature enhancement
        self.ethnicity_feature_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout/2)
            ) for _ in range(num_ethnicities)
        ])
        
        # Population encoding - combined for all ethnicities
        self.pop_encoder = nn.Sequential(
            nn.Linear(num_ethnicities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for contextual understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Jump probability network for all ethnicities at once
        self.jump_probability = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()
        )
        
        # Jump magnitude network for all ethnicities at once
        self.jump_magnitude = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities)
        )
        
        # Extreme jump detector for all ethnicities
        self.extreme_jump_detector = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()
        )
        
        # Extreme jump magnitude for all ethnicities
        self.extreme_jump_magnitude = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Softplus()  # Ensures positive values
        )
        
        # Ethnicity interaction module - models how ethnicities influence each other's jumps
        self.ethnicity_interaction = nn.Sequential(
            nn.Linear(num_ethnicities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities * num_ethnicities),
            nn.Sigmoid()
        )
        
        # Scale parameters based on log_space (one per ethnicity)
        base_scaler_value = 2.0 if log_space else 100.0
        self.extreme_jump_base_scaler = nn.Parameter(torch.ones(num_ethnicities) * base_scaler_value)
        
        # Feature-based jump classification for all ethnicities
        self.feature_jump_classifier = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3 * num_ethnicities),  # 3 classes per ethnicity: no jump, moderate jump, extreme jump
            nn.Softmax(dim=1)
        )
        
        # Feature-based magnitude scaling factors for different jump types for all ethnicities
        self.feature_jump_scaling = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3 * num_ethnicities),  # 3 scaling factors per ethnicity
            nn.Softplus()  # Ensure positive scaling
        )
        
        # Ethnicity-specific growth/decline tendency (learned)
        self.ethnicity_tendency = nn.Parameter(torch.zeros(num_ethnicities))
        
        # Interpretability trackers
        self.jump_probabilities = None
        self.jump_magnitudes = None
        self.extreme_jump_probs = None
        self.extreme_jump_magnitudes = None
        self.ethnicity_interactions = None
        self.feature_jump_classes = None
        self.feature_jump_scales = None
        self.final_predictions = None
        self.feature_importance = None
        
        # Move model to device
        self.to(device)
    
    def compute_feature_importance(self):
        """
        Compute feature importance scores based on the feature encoder weights.
        
        Returns:
            np.ndarray: Feature importance scores
        """
        # Get feature weights from first layer of feature encoder
        feature_weights = self.feature_encoder[0].weight.data
        
        # Compute importance as L1 norm of weights
        importance = torch.abs(feature_weights).sum(dim=0)
        
        # Normalize to sum to 1
        importance = importance / importance.sum()
        
        self.feature_importance = importance.detach().cpu().numpy()
        return self.feature_importance
    
    def forward(self, 
                population: torch.Tensor,
                features: torch.Tensor,
                spatial_context: Optional[torch.Tensor] = None,
                dauid_list=None) -> torch.Tensor:
        """
        Forward pass of the Multi-Ethnic Jump Process Expert.
        
        Args:
            population: Base population values [batch_size, num_ethnicities]
            features: Input features [batch_size, num_features]
            spatial_context: Optional spatial context features
            
        Returns:
            Updated population tensor [batch_size, num_ethnicities]
        """
        batch_size = features.shape[0]
        self.dauid_list = dauid_list
        # Ensure proper dimensions
        if len(population.shape) != 2:
            raise ValueError(f"Expected population shape [batch_size, num_ethnicities], got {population.shape}")
            
        if population.shape[1] != self.num_ethnicities:
            raise ValueError(f"Population has {population.shape[1]} ethnicities, but model expects {self.num_ethnicities}")
        
        # Encode shared features
        shared_features = self.feature_encoder(features)  # [batch_size, hidden_dim]
        
        # Create ethnicity-enhanced features
        eth_enhanced_features = []
        for eth_idx in range(self.num_ethnicities):
            enhanced = self.ethnicity_feature_enhancers[eth_idx](shared_features)
            eth_enhanced_features.append(enhanced)
        
        # Stack ethnicity-enhanced features
        stacked_eth_features = torch.stack(eth_enhanced_features, dim=1)  # [batch_size, num_ethnicities, hidden_dim]
        
        # Encode population
        pop_encoding = self.pop_encoder(population)  # [batch_size, hidden_dim]
        
        # Initialize context encoding
        if spatial_context is None:
            # Create context via attention
            # Reshape for attention (expects sequence dimension)
            feat_seq = shared_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            pop_seq = pop_encoding.unsqueeze(1)      # [batch_size, 1, hidden_dim]
            
            # Compute attention
            context, _ = self.attention(
                query=pop_seq,
                key=feat_seq,
                value=feat_seq
            )
            
            # Remove sequence dimension
            context_encoding = context.squeeze(1)  # [batch_size, hidden_dim]
        else:
            # Use provided spatial context
            # Ensure spatial_context has correct dimensions
            if spatial_context.shape[-1] != self.hidden_dim:
                # Project spatial context to hidden_dim if dimensions don't match
                context_encoding = nn.Linear(spatial_context.shape[-1], self.hidden_dim).to(self.device)(spatial_context)
            else:
                context_encoding = spatial_context
        
        # Combine the three encodings
        combined_encoding = torch.cat([
            shared_features,      # [batch_size, hidden_dim]
            pop_encoding,         # [batch_size, hidden_dim]
            context_encoding      # [batch_size, hidden_dim]
        ], dim=1)                 # Result: [batch_size, hidden_dim*3]
        
        # Compute jump probability for all ethnicities
        jump_prob = self.jump_probability(combined_encoding)  # [batch_size, num_ethnicities]
        
        # Compute jump magnitude for all ethnicities
        jump_mag = self.jump_magnitude(combined_encoding)  # [batch_size, num_ethnicities]
        
        # Apply ethnicity tendency (some ethnicities are more likely to increase/decrease)
        jump_mag = jump_mag + self.ethnicity_tendency.unsqueeze(0).expand(batch_size, -1)
        
        # Compute extreme jump probability and magnitude for all ethnicities
        extreme_jump_prob = self.extreme_jump_detector(combined_encoding)  # [batch_size, num_ethnicities]
        extreme_jump_base = self.extreme_jump_magnitude(combined_encoding)  # [batch_size, num_ethnicities]
        
        # Model ethnicity interactions - how ethnicities influence each other's jumps
        eth_interactions = self.ethnicity_interaction(population)  # [batch_size, num_ethnicities*num_ethnicities]
        eth_interactions = eth_interactions.view(batch_size, self.num_ethnicities, self.num_ethnicities)
        
        # Apply ethnicity interactions to jump magnitudes
        # Create new tensors for the updated values
        updated_jump_mag = jump_mag.clone()
        updated_extreme_jump_base = extreme_jump_base.clone()

        # Apply ethnicity interactions to jump magnitudes
        for eth_idx in range(self.num_ethnicities):
            # Get influence from all ethnicities on this ethnicity
            influences = eth_interactions[:, eth_idx, :]  # [batch_size, num_ethnicities]
            
            # Apply influence of other ethnicities on this ethnicity's jump
            # Higher influence increases the jump magnitude
            updated_jump_mag[:, eth_idx] = jump_mag[:, eth_idx] * (1.0 + influences.mean(dim=1))
            updated_extreme_jump_base[:, eth_idx] = extreme_jump_base[:, eth_idx] * (1.0 + influences.mean(dim=1))

        # Use the updated tensors for further computation
        jump_mag = updated_jump_mag
        extreme_jump_base = updated_extreme_jump_base
        
        # Population statistics for adaptive scaling
        current_pop_mean = population.mean(dim=1, keepdim=True) + self.epsilon
        current_pop_std = population.std(dim=1, keepdim=True) + self.epsilon
        
        # Adaptive scaling with awareness of log space
        if self.log_space:
            # In log space, population metrics have different meaning
            # We need much smaller scaling to avoid exploding values
            adaptive_scale = self.extreme_jump_base_scaler.unsqueeze(0) * torch.sigmoid(current_pop_std)
        else:
            # Original behavior for normal space
            adaptive_scale = self.extreme_jump_base_scaler.unsqueeze(0) * (1.0 + current_pop_std / current_pop_mean)
        
        # Apply adaptive scaling
        extreme_jump_mag = extreme_jump_base * adaptive_scale
        
        # Feature-based scaling
        growth_potential_logits = self.feature_encoder[0](features).mean(dim=1, keepdim=True)
        
        if self.log_space:
            # Much smaller range for log space (e.g., 1.1x to 2.0x)
            growth_potential_base = 1.1 + 0.9 * torch.sigmoid(growth_potential_logits)
        else:
            # Original range for normal space (1.5x to 5x)
            growth_potential_base = 1.5 + 3.5 * torch.sigmoid(growth_potential_logits)
        
        # Apply scaling to positive jumps only
        jump_mag = torch.where(
            jump_mag > 0,
            jump_mag * growth_potential_base.expand(-1, self.num_ethnicities),
            jump_mag
        )
        
        # Store for interpretability
        self.jump_probabilities = jump_prob.detach().cpu().numpy()
        self.jump_magnitudes = jump_mag.detach().cpu().numpy()
        self.extreme_jump_probs = extreme_jump_prob.detach().cpu().numpy()
        self.extreme_jump_magnitudes = extreme_jump_mag.detach().cpu().numpy()
        self.ethnicity_interactions = eth_interactions.detach().cpu().numpy()
        
        # Apply jumps with awareness of log space
        if self.log_space:
            # In log space, jumps should be limited to reasonable ranges
            jump_mag = torch.clamp(jump_mag, -3.0, 3.0)
            extreme_jump_mag = torch.clamp(extreme_jump_mag, 0.0, 3.0)
        
        # Get feature-based jump classification
        jump_class_probs_flat = self.feature_jump_classifier(features)  # [batch, 3*num_ethnicities]
        jump_class_probs = jump_class_probs_flat.view(batch_size, self.num_ethnicities, 3)
        
        # Get feature-based scaling factors for different jump types
        jump_type_scales_flat = self.feature_jump_scaling(features)  # [batch, 3*num_ethnicities]
        jump_type_scales = jump_type_scales_flat.view(batch_size, self.num_ethnicities, 3)
        
        # Store for interpretability
        self.feature_jump_classes = jump_class_probs.detach().cpu().numpy()
        self.feature_jump_scales = jump_type_scales.detach().cpu().numpy()
        
        # Apply area-specific scaling based on jump class for each ethnicity
        # Create new tensors for updated values
        updated_jump_prob = jump_prob.clone()
        updated_jump_mag = jump_mag.clone()
        updated_extreme_jump_mag = extreme_jump_mag.clone()

        # Apply area-specific scaling based on jump class for each ethnicity
        for eth_idx in range(self.num_ethnicities):
            # Combine jump probability from attention pathway with direct feature classification
            jump_prob_boost = 0.5 + 0.5 * (jump_class_probs[:, eth_idx, 1] + jump_class_probs[:, eth_idx, 2])
            updated_jump_prob[:, eth_idx] = jump_prob[:, eth_idx] * jump_prob_boost
            
            # Scale jump magnitude based on feature-predicted jump type
            jump_magnitude_scale = (
                jump_class_probs[:, eth_idx, 0] * jump_type_scales[:, eth_idx, 0] +  # No jump scaling
                jump_class_probs[:, eth_idx, 1] * jump_type_scales[:, eth_idx, 1] +  # Moderate jump scaling
                jump_class_probs[:, eth_idx, 2] * jump_type_scales[:, eth_idx, 2]    # Extreme jump scaling
            )
            
            updated_jump_mag[:, eth_idx] = jump_mag[:, eth_idx] * jump_magnitude_scale
            updated_extreme_jump_mag[:, eth_idx] = extreme_jump_mag[:, eth_idx] * jump_magnitude_scale

        # Use the updated tensors for further computation
        jump_prob = updated_jump_prob
        jump_mag = updated_jump_mag
        extreme_jump_mag = updated_extreme_jump_mag
        
        # Ensure the range stays reasonable based on log_space setting
        if self.log_space:
            jump_mag = torch.clamp(jump_mag, -5.0, 5.0)
            extreme_jump_mag = torch.clamp(extreme_jump_mag, 0.0, 5.0)
        
        # Apply jumps to population (all ethnicities at once)
        predicted_population = (
            population +  # Starting population
            jump_prob * jump_mag +  # Regular jumps
            extreme_jump_prob * extreme_jump_mag  # Extreme jumps
        )
        
        # Identify potential high-value areas using feature information
        high_value_indicator = torch.sigmoid(self.feature_encoder[0](features).mean(dim=1, keepdim=True) * 2 - 4)
        
        # Apply a boost to high-value areas identified by features
        high_value_boost = torch.where(
            high_value_indicator.expand(-1, self.num_ethnicities) > 0.7,
            population * 0.2,  # Add 20% boost for high-value areas
            torch.zeros_like(population)
        )
        
        # Add the high-value boost
        predicted_population = predicted_population + high_value_boost
        if self.log_space:
            predicted_population = torch.clamp(predicted_population, -10.0, 10.0)
        else:
            # Clamp to a reasonable maximum relative to input population
            max_multiplier = 3.0  # Maximum 3x growth in a single step
            max_values = population * max_multiplier
            min_values = torch.zeros_like(population)  # Ensure non-negative
            predicted_population = torch.clamp(predicted_population, min=min_values, max=max_values)
        
        # Store final predictions for interpretability
        self.final_predictions = predicted_population.detach().cpu().numpy()
            
        return predicted_population
    
    def get_interpretability_data(self):
        """
        Get interpretability data from the jump process expert with DAUIDs as keys.
        
        Returns:
            dict: Interpretability data
        """
        if self.feature_importance is None:
            self.compute_feature_importance()
            
        # Get model parameters
        parameters = {
            'extreme_jump_base_scaler': self.extreme_jump_base_scaler.detach().cpu().numpy(),
            'ethnicity_tendency': self.ethnicity_tendency.detach().cpu().numpy(),
            'log_space': self.log_space
        }
        
        interp_data = {
            'feature_importance': self.feature_importance,
            'parameters': parameters
        }
        
        # Convert numpy arrays to dictionaries with DAUIDs as keys if available
        if self.dauid_list is not None:
            jump_prob_dict = {}
            jump_mag_dict = {}
            extreme_prob_dict = {}
            extreme_mag_dict = {}
            eth_interactions_dict = {}
            feat_jump_classes_dict = {}
            feat_jump_scales_dict = {}
            final_pred_dict = {}
            
            for i, dauid in enumerate(self.dauid_list):
                # Skip if index is out of bounds
                if i >= self.jump_probabilities.shape[0]:
                    continue
                    
                dauid_key = str(dauid)
                
                # Store each parameter for this DAUID
                jump_prob_dict[dauid_key] = self.jump_probabilities[i].tolist()
                jump_mag_dict[dauid_key] = self.jump_magnitudes[i].tolist()
                extreme_prob_dict[dauid_key] = self.extreme_jump_probs[i].tolist()
                extreme_mag_dict[dauid_key] = self.extreme_jump_magnitudes[i].tolist()
                eth_interactions_dict[dauid_key] = self.ethnicity_interactions[i].tolist()
                feat_jump_classes_dict[dauid_key] = self.feature_jump_classes[i].tolist()
                feat_jump_scales_dict[dauid_key] = self.feature_jump_scales[i].tolist()
                final_pred_dict[dauid_key] = self.final_predictions[i].tolist()
            
            interp_data['predictions'] = {
                'jump_probabilities': jump_prob_dict,
                'jump_magnitudes': jump_mag_dict,
                'extreme_jump_probabilities': extreme_prob_dict,
                'extreme_jump_magnitudes': extreme_mag_dict,
                'ethnicity_interactions': eth_interactions_dict,
                'feature_jump_classes': feat_jump_classes_dict,
                'feature_jump_scales': feat_jump_scales_dict,
                'final_predictions': final_pred_dict
            }
        else:
            # If DAUIDs are not available, use original arrays
            interp_data['predictions'] = {
                'jump_probabilities': self.jump_probabilities,
                'jump_magnitudes': self.jump_magnitudes,
                'extreme_jump_probabilities': self.extreme_jump_probs,
                'extreme_jump_magnitudes': self.extreme_jump_magnitudes,
                'ethnicity_interactions': self.ethnicity_interactions,
                'feature_jump_classes': self.feature_jump_classes,
                'feature_jump_scales': self.feature_jump_scales,
                'final_predictions': self.final_predictions
            }
        
        return interp_data