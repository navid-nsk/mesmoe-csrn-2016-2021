import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


class MultiEthnicDeclineExpert(nn.Module):
    """
    Expert module for modeling rapid population decline (exodus) across multiple ethnicities.
    
    This expert specializes in predicting when areas experience significant
    population decline or complete exodus for different ethnic groups simultaneously.
    It models the probability of decline events and the magnitude of population loss
    for each ethnicity in parallel.
    """
    
    def __init__(self, 
                 num_features: int,
                 num_ethnicities: int,
                 hidden_dim: int = 64,
                 dropout: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Multi-Ethnic Decline Expert.
        
        Args:
            num_features: Number of census features
            num_ethnicities: Number of ethnicities to model
            hidden_dim: Hidden dimension for internal representations
            dropout: Dropout rate
            device: Device to use for computations
        """
        super(MultiEthnicDeclineExpert, self).__init__()
        self.num_features = num_features
        self.num_ethnicities = num_ethnicities
        self.hidden_dim = hidden_dim
        self.device = device
        self.dauid_list = None
        # Shared feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
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
        
        # Population encoding - handles all ethnicities together
        self.pop_encoder = nn.Sequential(
            nn.Linear(num_ethnicities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decline probability network - predicts decline probability for all ethnicities
        self.decline_probability = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()
        )
        
        # Decline factor network - how much population is retained, between 0 and 1
        self.decline_factor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()
        )
        
        # Ethnicity interaction matrix - how decline in one ethnicity affects others
        self.ethnicity_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities * num_ethnicities),
            nn.Sigmoid()
        )
        
        # Feature-based decline risk assessment for all ethnicities
        self.feature_decline_risk = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()
        )
        
        # Feature-based decline severity estimator for all ethnicities
        self.feature_decline_severity = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()  # 0 = total decline, 1 = no decline
        )
        
        # Feature-based extreme decline detector for all ethnicities
        self.feature_extreme_decline = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()
        )
        
        # Population sensitivity to features for all ethnicities
        self.feature_population_sensitivity = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ethnicities),
            nn.Sigmoid()
        )
        
        # Ethnicity-specific decline tendencies (some ethnicities are more prone to decline)
        self.ethnicity_decline_tendency = nn.Parameter(torch.zeros(num_ethnicities))
        
        # Interpretability trackers
        self.decline_probs = None
        self.decline_factors = None
        self.feature_importance = None
        self.feature_risk = None
        self.feature_severity = None
        self.extreme_decline_prob = None
        self.pop_sensitivity = None
        self.eth_interaction_weights = None
        self.final_predictions = None
        
        # Move model to device
        self.to(device)
    
    def compute_feature_importance(self) -> np.ndarray:
        """
        Compute feature importance scores.
        
        Returns:
            Array of feature importance scores
        """
        # Get feature weights from feature encoder
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
                spatial_context: Optional[torch.Tensor] = None, dauid_list=None) -> torch.Tensor:
        """
        Forward pass of the Multi-Ethnic Decline Expert.
        
        Args:
            population: Population tensor of shape [batch_size, num_ethnicities]
            features: Feature tensor of shape [batch_size, num_features]
            spatial_context: Optional spatial context tensor
            
        Returns:
            Predicted population tensor after potential decline [batch_size, num_ethnicities]
        """
        batch_size = population.shape[0]
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
        
        # Initialize context tensor
        if spatial_context is None:
            # Create fake spatial context with same dimension as hidden_dim
            context = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        else:
            # If spatial context dimension doesn't match hidden_dim, project it
            if spatial_context.shape[-1] != self.hidden_dim:
                projection = nn.Linear(spatial_context.shape[-1], self.hidden_dim).to(self.device)
                context = projection(spatial_context)
            else:
                context = spatial_context
        
        # Combine encodings
        combined_encoding = torch.cat([
            shared_features,  # [batch_size, hidden_dim]
            pop_encoding,     # [batch_size, hidden_dim]
            context           # [batch_size, hidden_dim]
        ], dim=1)  # Result: [batch_size, hidden_dim * 3]
        
        # Compute decline probability for all ethnicities
        decline_prob = self.decline_probability(combined_encoding)  # [batch_size, num_ethnicities]
        
        # Apply ethnicity-specific decline tendencies
        decline_prob = torch.sigmoid(
            torch.logit(decline_prob) + self.ethnicity_decline_tendency.unsqueeze(0)
        )
        
        # Compute decline factor for all ethnicities
        decline_factor = self.decline_factor(combined_encoding)  # [batch_size, num_ethnicities]
        
        # Store for interpretability
        self.decline_probs = decline_prob.detach().cpu().numpy()
        self.decline_factors = decline_factor.detach().cpu().numpy()
        
        # Calculate feature-based decline risk for all ethnicities
        feature_risk = self.feature_decline_risk(features)  # [batch_size, num_ethnicities]
        
        # Calculate feature-based decline severity for all ethnicities
        feature_severity = self.feature_decline_severity(features)  # [batch_size, num_ethnicities]
        
        # Identify potential extreme decline areas for all ethnicities
        extreme_decline_prob = self.feature_extreme_decline(features)  # [batch_size, num_ethnicities]
        
        # Get population sensitivity to features for all ethnicities
        pop_sensitivity = self.feature_population_sensitivity(features)  # [batch_size, num_ethnicities]
        
        # Store for interpretability
        self.feature_risk = feature_risk.detach().cpu().numpy()
        self.feature_severity = feature_severity.detach().cpu().numpy()
        self.extreme_decline_prob = extreme_decline_prob.detach().cpu().numpy()
        self.pop_sensitivity = pop_sensitivity.detach().cpu().numpy()
        
        # Model ethnicity interactions - how decline in one affects others
        eth_interaction_weights = self.ethnicity_interaction(shared_features)  # [batch_size, num_ethnicities*num_ethnicities]
        eth_interaction_weights = eth_interaction_weights.view(batch_size, self.num_ethnicities, self.num_ethnicities)
        
        # Store for interpretability
        self.eth_interaction_weights = eth_interaction_weights.detach().cpu().numpy()
        
        # Apply ethnicity interactions
        # For each ethnicity, compute how its decline affects other ethnicities
        updated_decline_prob = decline_prob.clone()

        # Apply ethnicity interactions
        # For each ethnicity, compute how its decline affects other ethnicities
        for source_eth in range(self.num_ethnicities):
            for target_eth in range(self.num_ethnicities):
                if source_eth == target_eth:
                    continue  # Skip self-interaction
                
                # Get interaction weight from source to target
                interaction_weight = eth_interaction_weights[:, target_eth, source_eth]
                
                # Apply influence - high decline probability in source increases decline in target
                # Use updated_decline_prob for the result, but keep using original decline_prob on the right side
                updated_decline_prob[:, target_eth] = torch.clamp(
                    decline_prob[:, target_eth] + interaction_weight * decline_prob[:, source_eth] * 0.2,
                    0.0, 1.0
                )

        # Replace the original tensor with the updated one
        decline_prob = updated_decline_prob
        
        # Combine model-based and feature-based decline probabilities
        combined_decline_prob = (
            (1 - pop_sensitivity) * decline_prob +  # Model-based
            pop_sensitivity * feature_risk          # Feature-based
        )
        
        # Combine model-based and feature-based decline factors
        combined_decline_factor = (
            (1 - pop_sensitivity) * decline_factor +  # Model-based
            pop_sensitivity * feature_severity        # Feature-based
        )
        
        # Apply special handling for extreme decline areas
        extreme_factor = torch.where(
            extreme_decline_prob > 0.8,
            torch.zeros_like(combined_decline_factor),  # Total decline to zero
            combined_decline_factor                     # Regular decline factor
        )
        
        # Apply enhanced decline model to all ethnicities at once
        predicted_population = population * (
            1 - combined_decline_prob +  # Keep original population if no decline
            combined_decline_prob * (
                (1 - extreme_decline_prob) * combined_decline_factor +  # Regular decline
                extreme_decline_prob * extreme_factor                   # Potential extreme decline
            )
        )
        
        # Explicitly set very small values to zero (important for true zeros)
        predicted_population = torch.where(
            predicted_population < 0.5,  # Small values should be true zeros
            torch.zeros_like(predicted_population),
            predicted_population
        )
        
        # Store final predictions for interpretability
        self.final_predictions = predicted_population.detach().cpu().numpy()
        
        return predicted_population
    
    def get_interpretability_data(self):
        """
        Get interpretability data with DAUIDs as keys.
        
        Returns:
            Dict of interpretability data
        """
        if self.feature_importance is None:
            self.compute_feature_importance()
            
        # Get model parameters
        parameters = {
            'ethnicity_decline_tendency': self.ethnicity_decline_tendency.detach().cpu().numpy()
        }
        
        interp_data = {
            'feature_importance': self.feature_importance,
            'parameters': parameters
        }
        
        # Convert numpy arrays to dictionaries with DAUIDs as keys if available
        if self.dauid_list is not None:
            decline_probs_dict = {}
            decline_factors_dict = {}
            feature_risk_dict = {}
            feature_severity_dict = {}
            extreme_prob_dict = {}
            pop_sensitivity_dict = {}
            eth_interactions_dict = {}
            final_pred_dict = {}
            
            for i, dauid in enumerate(self.dauid_list):
                # Skip if index is out of bounds
                if i >= self.decline_probs.shape[0]:
                    continue
                    
                dauid_key = str(dauid)
                
                # Store each parameter for this DAUID
                decline_probs_dict[dauid_key] = self.decline_probs[i].tolist()
                decline_factors_dict[dauid_key] = self.decline_factors[i].tolist()
                feature_risk_dict[dauid_key] = self.feature_risk[i].tolist()
                feature_severity_dict[dauid_key] = self.feature_severity[i].tolist()
                extreme_prob_dict[dauid_key] = self.extreme_decline_prob[i].tolist()
                pop_sensitivity_dict[dauid_key] = self.pop_sensitivity[i].tolist()
                eth_interactions_dict[dauid_key] = self.eth_interaction_weights[i].tolist()
                final_pred_dict[dauid_key] = self.final_predictions[i].tolist()
            
            interp_data['predictions'] = {
                'decline_probabilities': decline_probs_dict,
                'decline_factors': decline_factors_dict,
                'feature_risk': feature_risk_dict,
                'feature_severity': feature_severity_dict,
                'extreme_decline_probability': extreme_prob_dict,
                'population_sensitivity': pop_sensitivity_dict,
                'ethnicity_interaction_weights': eth_interactions_dict,
                'final_predictions': final_pred_dict
            }
        else:
            # If DAUIDs are not available, use original arrays
            interp_data['predictions'] = {
                'decline_probabilities': self.decline_probs,
                'decline_factors': self.decline_factors,
                'feature_risk': self.feature_risk,
                'feature_severity': self.feature_severity,
                'extreme_decline_probability': self.extreme_decline_prob,
                'population_sensitivity': self.pop_sensitivity,
                'ethnicity_interaction_weights': self.eth_interaction_weights,
                'final_predictions': self.final_predictions
            }
        
        return interp_data