import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


class MultiEthnicColonizationExpert(nn.Module):
    """
    Enhanced specialized expert for zero-to-nonzero population transitions
    that handles multiple ethnicities simultaneously.
    
    This expert is designed to predict new population values for previously
    uninhabited areas, using a specific modeling approach for colonization dynamics.
    """
    
    def __init__(self, 
                 num_features: int,
                 num_ethnicities: int,
                 hidden_dim: int = 128,
                 min_population: float = 10.0,
                 max_population: float = 2000.0,
                 device: str = 'cuda'):
        super(MultiEthnicColonizationExpert, self).__init__()
        self.num_features = num_features
        self.num_ethnicities = num_ethnicities
        self.hidden_dim = hidden_dim
        self.min_population = min_population
        self.max_population = max_population
        self.device = device
        self.dauid_list = None
        # Deeper shared network with increased capacity
        self.shared_network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Additional layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
        
        # Ethnicity-specific encoders - learns unique patterns for each ethnicity
        self.ethnicity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU()
            ) for _ in range(num_ethnicities)
        ]).to(device)
        
        # First stage: predict colonization size in four categories for all ethnicities at once
        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4 * num_ethnicities),  # Four outputs per ethnicity: small, medium, large, very large
            nn.Softmax(dim=1)  # Probabilities of different sizes
        ).to(device)
        
        # Four separate predictors for different colonization scales - shared across ethnicities
        self.small_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_ethnicities),
            nn.ReLU()  # Ensures non-negative
        ).to(device)
        
        self.medium_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_ethnicities),
            nn.ReLU()  # Ensures non-negative
        ).to(device)
        
        self.large_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),  # Even wider network for large predictions
            nn.ReLU(),
            nn.Dropout(0.15),  # Slightly higher dropout
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.ReLU()  # Ensures non-negative
        ).to(device)

        self.very_large_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),  # Widest network
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.ReLU()
        ).to(device)

        # Ethnicity-specific scaling factors (learned)
        # Each ethnicity will have its own set of scaling factors
        self.small_scale = nn.Parameter(torch.ones(num_ethnicities) * 75.0)
        self.medium_scale = nn.Parameter(torch.ones(num_ethnicities) * 350.0)
        self.large_scale = nn.Parameter(torch.ones(num_ethnicities) * 900.0)
        self.very_large_scale = nn.Parameter(torch.ones(num_ethnicities) * 2000.0)
        
        # Ethnicity-specific bias factors (learned)
        self.small_bias = nn.Parameter(torch.ones(num_ethnicities) * 25.0)
        self.medium_bias = nn.Parameter(torch.ones(num_ethnicities) * 100.0)
        self.large_bias = nn.Parameter(torch.ones(num_ethnicities) * 200.0)
        self.very_large_bias = nn.Parameter(torch.ones(num_ethnicities) * 500.0)
        
        # Ethnicity importance model - learns which ethnicities are more likely to colonize an area
        self.ethnicity_importance = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Softmax(dim=1)  # Normalize importance across ethnicities
        ).to(device)
        
        # Initialize with higher prediction capacity
        with torch.no_grad():
            # Increase the output range of the large predictor
            if isinstance(self.large_predictor[-2], nn.Linear):
                self.large_predictor[-2].bias.data.fill_(3.0)
            if isinstance(self.very_large_predictor[-2], nn.Linear):
                self.very_large_predictor[-2].bias.data.fill_(4.0)
                
        # For interpretability
        self.size_probabilities = None
        self.eth_importance_values = None
        self.small_predictions = None
        self.medium_predictions = None
        self.large_predictions = None
        self.very_large_predictions = None
        self.feature_importance = None
    
    def compute_feature_importance(self):
        """
        Compute feature importance scores based on the shared network weights.
        
        Returns:
            np.ndarray: Feature importance scores
        """
        # Get feature weights from first layer of shared network
        feature_weights = self.shared_network[0].weight.data
        
        # Compute importance as L1 norm of weights
        importance = torch.abs(feature_weights).sum(dim=0)
        
        # Normalize to sum to 1
        importance = importance / importance.sum()
        
        self.feature_importance = importance.detach().cpu().numpy()
        return self.feature_importance
    
    def forward(self, population, features, spatial_context, dauid_list=None):
        """
        Forward pass with mixture of small, medium, large, and very large colonization models
        for all ethnicities at once.
        
        Args:
            population: Population tensor [batch_size, num_ethnicities]
            features: Feature tensor [batch_size, num_features]
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
        
        # Shared feature processing
        shared_features = self.shared_network(features)
        
        # Get ethnicity importance - which ethnicities are more likely to colonize here
        eth_importance = self.ethnicity_importance(features)  # [batch_size, num_ethnicities]
        
        # Process ethnicities in parallel
        # Predict colonization size probabilities (small, medium, large, very large) for all ethnicities
        size_probs_flat = self.size_predictor(shared_features)  # [batch_size, 4*num_ethnicities]
        
        # Reshape to [batch_size, num_ethnicities, 4] for easier processing
        size_probs = size_probs_flat.view(batch_size, self.num_ethnicities, 4)
        
        # Get predictions from the four size models with learned biases for all ethnicities
        small_pop = self.small_predictor(shared_features)  # [batch_size, num_ethnicities]
        medium_pop = self.medium_predictor(shared_features)  # [batch_size, num_ethnicities]
        large_pop = self.large_predictor(shared_features)  # [batch_size, num_ethnicities]
        very_large_pop = self.very_large_predictor(shared_features)  # [batch_size, num_ethnicities]
        
        # Apply ethnicity-specific scales and biases
        small_pop = small_pop * self.small_scale + self.small_bias
        medium_pop = medium_pop * self.medium_scale + self.medium_bias
        large_pop = large_pop * self.large_scale + self.large_bias
        very_large_pop = very_large_pop * self.very_large_scale + self.very_large_bias
        
        # Store for interpretability
        self.size_probabilities = size_probs.detach().cpu().numpy()
        self.eth_importance_values = eth_importance.detach().cpu().numpy()
        self.small_predictions = small_pop.detach().cpu().numpy()
        self.medium_predictions = medium_pop.detach().cpu().numpy()
        self.large_predictions = large_pop.detach().cpu().numpy()
        self.very_large_predictions = very_large_pop.detach().cpu().numpy()
        
        # Initialize predicted population tensor
        predicted_population = torch.zeros_like(population)
        
        # Blend predictions based on size probabilities for each ethnicity
        for eth_idx in range(self.num_ethnicities):
            small_prob = size_probs[:, eth_idx, 0].unsqueeze(1)
            medium_prob = size_probs[:, eth_idx, 1].unsqueeze(1)
            large_prob = size_probs[:, eth_idx, 2].unsqueeze(1)
            very_large_prob = size_probs[:, eth_idx, 3].unsqueeze(1)
            
            # Get this ethnicity's predictions
            eth_small_pop = small_pop[:, eth_idx].unsqueeze(1)
            eth_medium_pop = medium_pop[:, eth_idx].unsqueeze(1)
            eth_large_pop = large_pop[:, eth_idx].unsqueeze(1)
            eth_very_large_pop = very_large_pop[:, eth_idx].unsqueeze(1)
            
            # Blend different size predictions
            eth_predicted = (
                small_prob * eth_small_pop + 
                medium_prob * eth_medium_pop + 
                large_prob * eth_large_pop +
                very_large_prob * eth_very_large_pop
            )
            
            # Store in predicted population tensor
            predicted_population[:, eth_idx] = eth_predicted.squeeze(1)
        
        # Apply colonization logic - only colonize zero/near-zero populations
        is_zero = population <= 0.1
        
        # Apply minimum and maximum population thresholds
        colonized_population = torch.clamp(predicted_population, min=self.min_population, max=self.max_population)
        
        # Modulate by ethnicity importance - ethnicities with higher importance have stronger colonization
        colonized_population = colonized_population * eth_importance
        
        # Only replace zero/near-zero populations
        output_population = torch.where(
            is_zero,
            colonized_population,
            population
        )
        
        return output_population
    
    def get_interpretability_data(self):
        """
        Get interpretability data from the colonization expert.
        
        Returns:
            dict: Interpretability data
        """
        if self.feature_importance is None:
            self.compute_feature_importance()
            
        # Get learned scaling parameters
        scales_data = {
            'small_scale': self.small_scale.detach().cpu().numpy(),
            'medium_scale': self.medium_scale.detach().cpu().numpy(),
            'large_scale': self.large_scale.detach().cpu().numpy(),
            'very_large_scale': self.very_large_scale.detach().cpu().numpy(),
            'small_bias': self.small_bias.detach().cpu().numpy(),
            'medium_bias': self.medium_bias.detach().cpu().numpy(),
            'large_bias': self.large_bias.detach().cpu().numpy(),
            'very_large_bias': self.very_large_bias.detach().cpu().numpy()
        }
        
        interp_data = {
            'feature_importance': self.feature_importance,
            'min_population': self.min_population,
            'max_population': self.max_population,
            'scales': scales_data
        }
        
        # Convert numpy arrays to dictionaries with DAUIDs as keys if available
        if self.dauid_list is not None:
            size_probs_dict = {}
            eth_importance_dict = {}
            small_pred_dict = {}
            medium_pred_dict = {}
            large_pred_dict = {}
            very_large_pred_dict = {}
            
            for i, dauid in enumerate(self.dauid_list):
                # Skip if index is out of bounds
                if i >= self.size_probabilities.shape[0]:
                    continue
                    
                dauid_key = str(dauid)
                
                # Store each parameter for this DAUID
                size_probs_dict[dauid_key] = self.size_probabilities[i].tolist()
                eth_importance_dict[dauid_key] = self.eth_importance_values[i].tolist()
                small_pred_dict[dauid_key] = self.small_predictions[i].tolist()
                medium_pred_dict[dauid_key] = self.medium_predictions[i].tolist()
                large_pred_dict[dauid_key] = self.large_predictions[i].tolist()
                very_large_pred_dict[dauid_key] = self.very_large_predictions[i].tolist()
            
            interp_data.update({
                'size_probabilities': size_probs_dict,
                'ethnicity_importance': eth_importance_dict,
                'predictor_outputs': {
                    'small': small_pred_dict,
                    'medium': medium_pred_dict,
                    'large': large_pred_dict,
                    'very_large': very_large_pred_dict
                }
            })
        else:
            # If DAUIDs are not available, use original arrays
            interp_data.update({
                'size_probabilities': self.size_probabilities,
                'ethnicity_importance': self.eth_importance_values,
                'predictor_outputs': {
                    'small': self.small_predictions,
                    'medium': self.medium_predictions,
                    'large': self.large_predictions,
                    'very_large': self.very_large_predictions
                }
            })
        
        return interp_data