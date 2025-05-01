import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging

from csrn.experts.pde_solver_cuda import MultiEthnicCSRNDiffusionSolverCUDA
from csrn.spatial_attention.spatial_attention import CoordinateEmbedding, MultiMatrixSpatialAttention
from csrn.experts.colonization_expert import MultiEthnicColonizationExpert
from csrn.experts.jump_expert import MultiEthnicJumpProcessExpert
from csrn.experts.decline_expert import MultiEthnicDeclineExpert

logger = logging.getLogger(__name__)

class MultiEthnicLearnableExpertRouter(nn.Module):
    """Router network that learns to assign experts for multiple ethnicities simultaneously"""
    def __init__(self, input_dim, num_ethnicities, hidden_dim=64, dropout_rate=0.2, num_experts=4,
                 colonization_threshold=0.1, decline_threshold=50.0, jump_threshold=100.0):
        super(MultiEthnicLearnableExpertRouter, self).__init__()
        
        self.num_experts = num_experts
        self.num_ethnicities = num_ethnicities
        self.colonization_threshold = colonization_threshold
        self.decline_threshold = decline_threshold
        self.jump_threshold = jump_threshold
        
        # Feature encoder - processes input features
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Population encoder - processes population vectors
        self.population_encoder = nn.Sequential(
            nn.Linear(num_ethnicities, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Combined encoder
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Expert selector - outputs probabilities for each expert per ethnicity
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_experts * num_ethnicities),  # Predicts for all ethnicities at once
            nn.Softmax(dim=1)  # Convert to probabilities
        )
        
        # Classification head - helps with training signal
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_experts * num_ethnicities),  # Predicts for all ethnicities at once
            nn.Sigmoid()
        )
        
        # For interpretability
        self.feature_importance = None
    
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
    
    def forward(self, features, population_vectors):
        """
        Forward pass for multi-ethnic expert router
        
        Args:
            features (torch.Tensor): Input features [batch_size, input_dim]
            population_vectors (torch.Tensor): Population vectors [batch_size, num_ethnicities]
            
        Returns:
            tuple: (expert_weights, expert_classes)
                - expert_weights: [batch_size, num_ethnicities, num_experts]
                - expert_classes: [batch_size, num_ethnicities, num_experts]
        """
        batch_size = features.size(0)
        
        # Encode features
        feature_encoding = self.feature_encoder(features)  # [batch_size, hidden_dim]
        
        # Encode population vectors
        population_encoding = self.population_encoder(population_vectors)  # [batch_size, hidden_dim]
        
        # Combine encodings
        combined = torch.cat([feature_encoding, population_encoding], dim=1)  # [batch_size, hidden_dim*2]
        combined_encoding = self.combined_encoder(combined)  # [batch_size, hidden_dim]
        
        # Get expert probabilities for all ethnicities
        expert_probs_flat = self.selector(combined_encoding)  # [batch_size, num_experts*num_ethnicities]
        
        # Reshape to [batch_size, num_ethnicities, num_experts]
        expert_probs = expert_probs_flat.view(batch_size, self.num_ethnicities, self.num_experts)
        
        # Get classification probabilities for all ethnicities
        class_probs_flat = self.classifier(combined_encoding)  # [batch_size, num_experts*num_ethnicities]
        class_probs = class_probs_flat.view(batch_size, self.num_ethnicities, self.num_experts)
        
        # Create masks based on population thresholds for each ethnicity
        # Shape: [batch_size, num_ethnicities, 1]
        is_colonization = (population_vectors <= self.colonization_threshold).float().unsqueeze(-1)
        
        is_jump = ((population_vectors > self.decline_threshold) & 
                   (population_vectors > self.jump_threshold)).float().unsqueeze(-1)
        
        is_decline = ((population_vectors > self.colonization_threshold) & 
                      (population_vectors <= self.decline_threshold)).float().unsqueeze(-1) * 0.5
        
        is_pde = ((population_vectors > self.colonization_threshold) & 
                  (population_vectors <= self.jump_threshold * 2)).float().unsqueeze(-1) * 1.5
        
        # Combine hints for all ethnicities
        # Shape: [batch_size, num_ethnicities, num_experts]
        hint_tensor = torch.cat([
            is_colonization,  # [batch_size, num_ethnicities, 1]
            is_jump,          # [batch_size, num_ethnicities, 1]
            is_decline,       # [batch_size, num_ethnicities, 1]
            is_pde            # [batch_size, num_ethnicities, 1]
        ], dim=-1)  # Result: [batch_size, num_ethnicities, 4]
        
        # Blend learned probabilities with rule-based hints
        hint_weight = 0.2  # Reduced to give more weight to learning
        blended_probs = (1 - hint_weight) * expert_probs + hint_weight * hint_tensor
        
        return blended_probs, class_probs
    
    def get_interpretability_data(self):
        """
        Get interpretability data from the router.
        
        Returns:
            dict: Interpretability data
        """
        if self.feature_importance is None:
            self.compute_feature_importance()
            
        return {
            'feature_importance': self.feature_importance
        }

class MultiEthnicSpatialMoEPredictor(nn.Module):
    """
    Enhanced Mixture of Experts neural network model for predicting multi-ethnic population changes,
    with specialized handling for colonization cases, significant population jumps,
    population decline cases, and PDE-based growth/decline.
    
    This version processes all ethnicities in parallel for much better performance.
    """
    def __init__(self, input_dim, num_ethnicities, ethnicity_names=None, hidden_dims=[128, 64, 32], coord_embed_dim=32, 
                num_attention_heads=4, num_matrices=3, dropout_rate=0.2,
                colonization_threshold=0.1, colonization_expert_weight=0.8,
                jump_threshold=100.0, jump_expert_weight=0.7,
                decline_threshold=50.0, decline_expert_weight=0.8,
                pde_expert_weight=0.6, global_params=None,
                log_space=False, device='cuda' if torch.cuda.is_available() else 'cpu',
                skip_dim_adjustments=False):
        """
        Initialize the neural network for multi-ethnic population prediction
        
        Args:
            input_dim (int): Number of input features
            num_ethnicities (int): Number of ethnicities to predict
            hidden_dims (list): List of hidden dimensions
            coord_embed_dim (int): Dimension for coordinate embeddings
            num_attention_heads (int): Number of attention heads
            num_matrices (int): Number of spatial matrices to use
            dropout_rate (float): Dropout rate
            colonization_threshold (float): Threshold below which population is considered zero
            colonization_expert_weight (float): Base weight for colonization expert (0-1)
            jump_threshold (float): Threshold for identifying potential jump cases
            jump_expert_weight (float): Base weight for jump expert influence
            decline_threshold (float): Threshold for identifying potential decline cases
            decline_expert_weight (float): Base weight for decline expert influence
            pde_expert_weight (float): Base weight for PDE expert influence
            global_params (dict, optional): Global parameters for PDE expert
            log_space (bool): Whether the model operates in log space
            device (str): Device to use for computation
        """
        super(MultiEthnicSpatialMoEPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.num_ethnicities = num_ethnicities
        self.colonization_threshold = colonization_threshold
        self.colonization_expert_weight = colonization_expert_weight
        self.jump_threshold = jump_threshold
        self.jump_expert_weight = jump_expert_weight
        self.decline_threshold = decline_threshold
        self.decline_expert_weight = decline_expert_weight
        self.pde_expert_weight = pde_expert_weight
        self.log_space = log_space
        self.device = device
        self.dauid_list = None
        # Main model components
        # Coordinate embedding
        self.coord_embedding = CoordinateEmbedding(coord_dim=2, hidden_dim=coord_embed_dim)
        self.num_ethnicities = num_ethnicities
    
        # Store ethnicity names with a fallback to generic names
        if ethnicity_names is None or len(ethnicity_names) != num_ethnicities:
            self.ethnicity_names = [f"ethnicity_{i}" for i in range(num_ethnicities)]
        else:
            self.ethnicity_names = ethnicity_names
        
        # Initial feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout_rate)
        )
        
        # Learnable expert router for multi-ethnic data
        self.expert_router = MultiEthnicLearnableExpertRouter(
            input_dim=input_dim,
            num_ethnicities=num_ethnicities,
            hidden_dim=hidden_dims[0],
            dropout_rate=dropout_rate,
            num_experts=4,  # 4 experts: colonization, jump, decline, PDE
            colonization_threshold=colonization_threshold,
            decline_threshold=decline_threshold,
            jump_threshold=jump_threshold
        )
        
        # Combine feature and coordinate embeddings
        self.combined_dim = hidden_dims[0] + coord_embed_dim
        self.combine_embeddings = nn.Linear(self.combined_dim, hidden_dims[0])
        
        # Ensure coord_embed_dim is divisible by num_attention_heads
        if not skip_dim_adjustments and coord_embed_dim % num_attention_heads != 0:
            # Find the nearest multiple of num_attention_heads
            old_dim = coord_embed_dim
            coord_embed_dim = (coord_embed_dim // num_attention_heads) * num_attention_heads
            logger.warning(f"Adjusted coord_embed_dim from {old_dim} to {coord_embed_dim} to ensure divisibility by {num_attention_heads} heads")

        # Also make sure hidden_dims[0] is divisible by num_attention_heads
        if not skip_dim_adjustments and hidden_dims[0] % num_attention_heads != 0:
            old_dim = hidden_dims[0]
            # Round to nearest multiple that's divisible
            hidden_dims[0] = (hidden_dims[0] // num_attention_heads) * num_attention_heads
            logger.warning(f"Adjusted hidden_dims[0] from {old_dim} to {hidden_dims[0]} to ensure divisibility by {num_attention_heads} heads")

        # Update combined_dim based on the potentially adjusted dimensions
        self.combined_dim = hidden_dims[0] + coord_embed_dim
        
        # Multi-matrix spatial attention
        self.spatial_attention = MultiMatrixSpatialAttention(
            hidden_dim=hidden_dims[0],
            num_matrices=num_matrices,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Build the main network layers
        self.layers = nn.ModuleList()
        current_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Output layer for main model - now predicts multiple ethnicities at once
        self.output_layer = nn.Sequential(
            nn.Linear(current_dim, num_ethnicities),
            nn.ReLU()  # Ensure non-negative population predictions
        )
        
        # Create specialized experts that handle all ethnicities at once
        # Colonization Expert - specialized for zero-to-nonzero transitions
        self.colonization_expert = MultiEthnicColonizationExpert(
            num_features=input_dim,
            num_ethnicities=num_ethnicities,
            hidden_dim=128,
            min_population=10.0,
            max_population=2000.0,
            device=device
        )
        
        # Jump Process Expert - specialized for significant population increases
        self.jump_expert = MultiEthnicJumpProcessExpert(
            num_features=input_dim,
            num_ethnicities=num_ethnicities,
            hidden_dim=64,
            num_heads=4,
            dropout=dropout_rate,
            log_space=log_space,
            device=device
        )
        
        # Decline Expert - specialized for population decline cases
        self.decline_expert = MultiEthnicDeclineExpert(
            num_features=input_dim,
            num_ethnicities=num_ethnicities,
            hidden_dim=64,
            dropout=dropout_rate,
            device=device
        )
        
        # PDE Expert - physics-informed prediction for regular growth/decline
        self.pde_expert = MultiEthnicCSRNDiffusionSolverCUDA(
            num_features=input_dim,
            num_ethnicities=num_ethnicities,
            hidden_dim=64,
            num_steps=2,  # Number of integration steps
            global_params=global_params,
            enable_debug=False,
            log_space=log_space,
            device=device
        )
        
        # Ethnicity interaction module - learns how ethnicities influence each other
        self.ethnicity_interaction = nn.Sequential(
            nn.Linear(num_ethnicities, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], num_ethnicities * num_ethnicities),
            nn.Sigmoid()
        )
        
        # Interpretation storage
        self.expert_outputs = None
        self.expert_weights = None
        self.expert_classes = None
        self.main_model_output = None
        self.interaction_weights = None
        self.interaction_effect = None
        self.spatial_features = None
        
        # Move to specified device
        self.to(device)
    
    def forward(self, features, population_vectors, coordinates, 
                walking_matrix=None, transit_matrix=None, proximity_matrix=None, 
                store_interpretations=False, dauid_list=None):
        """
        Forward pass with learnable mixture of experts for multi-ethnic population prediction
        
        Args:
            features (torch.Tensor): Input features [batch_size, input_dim]
            population_vectors (torch.Tensor): Population vectors [batch_size, num_ethnicities]
            coordinates (torch.Tensor): Spatial coordinates [batch_size, 2]
            walking_matrix (torch.Tensor, optional): Walking distance matrix [batch_size, batch_size]
            transit_matrix (torch.Tensor, optional): Transit distance matrix [batch_size, batch_size]
            proximity_matrix (torch.Tensor, optional): Proximity matrix [batch_size, batch_size]
            store_interpretations (bool): Whether to store intermediate outputs for interpretation
            
        Returns:
            tuple: (predicted_populations, expert_weights, expert_classes)
                - predicted_populations: [batch_size, num_ethnicities]
                - expert_weights: [batch_size, num_ethnicities, num_experts]
                - expert_classes: [batch_size, num_ethnicities, num_experts]
        """
        # pred_start_time = time.time()
        batch_size = features.size(0)
        self.dauid_list = dauid_list
        # Calculate output from main model
        main_output = self._run_main_model(features, coordinates, 
                                           walking_matrix, transit_matrix, proximity_matrix)
        
        # Create spatial features from available matrices if possible
        spatial_features = self._create_spatial_features(
            walking_matrix, transit_matrix, proximity_matrix
        )
        
        # Get expert assignments using the learnable router
        expert_weights, expert_classes = self.expert_router(features, population_vectors)
        
        # Get predictions from all expert models for all ethnicities at once
        
        # Colonization expert - processes all ethnicities at once
        colonization_output = self.colonization_expert(
            population_vectors, features, spatial_features, dauid_list=dauid_list
        )
        
        # Jump process expert - processes all ethnicities at once
        jump_output = self.jump_expert(
            population_vectors, features, spatial_features, dauid_list=dauid_list
        )
        max_jump = population_vectors * 3.0 + 100.0  # Maximum 3x growth plus 100
        jump_output = torch.where(jump_output > max_jump, max_jump, jump_output)
        jump_output = torch.where(jump_output < 0.0, torch.zeros_like(jump_output), jump_output)
        
        # Decline expert - processes all ethnicities at once
        decline_output = self.decline_expert(
            population_vectors, features, spatial_features, dauid_list=dauid_list
        )
        max_decline = population_vectors + 50.0  # Maximum growth plus 50

        decline_output = torch.where(decline_output > max_decline, max_decline, decline_output)
        decline_output = torch.where(decline_output < 0.0, torch.zeros_like(decline_output), decline_output)
        
        # PDE expert - processes all ethnicities at once
        try:
            # Create a dictionary for spatial matrices
            pde_spatial_matrices = {}
            if walking_matrix is not None:
                pde_spatial_matrices['walking'] = walking_matrix
            if transit_matrix is not None:
                pde_spatial_matrices['transit'] = transit_matrix
            if proximity_matrix is not None:
                pde_spatial_matrices['proximity'] = proximity_matrix
            
            # If no matrices are available, create dummy ones
            if not pde_spatial_matrices:
                identity = torch.eye(batch_size, device=self.device)
                pde_spatial_matrices = {
                    'walking': identity,
                    'transit': identity,
                    'proximity': identity
                }
            
            # FIXED: Create area-specific features
            # Instead of expanding and duplicating the same features everywhere,
            # we create a proper batch×batch tensor where each row contains the unique features for that DA
            pde_features = torch.zeros(batch_size, batch_size, features.shape[1], device=self.device)
            for i in range(batch_size):
                # Each row contains the features of the corresponding DA
                pde_features[i] = features
            
            # FIXED: Similarly for population
            pde_population = torch.zeros(batch_size, batch_size, population_vectors.shape[1], device=self.device)
            for i in range(batch_size):
                # Each row contains the population data of the corresponding DA
                pde_population[i] = population_vectors
            
            pde_population = pde_population.contiguous()
            pde_features = pde_features.contiguous()
            
            # Run PDE solver
            pde_output = self.pde_expert(
                pde_population,
                pde_features,
                pde_spatial_matrices,
                coordinates,
                time_period=1.0,
                dauid_list=dauid_list
            )
            
            # The PDE solver now has access to a proper tensor where each DA has its unique features
            # We still extract the diagonal to get our predictions
            pde_output = torch.diagonal(pde_output, dim1=0, dim2=1).t()
            
            # Clamp PDE output to reasonable range as before
            max_pde = population_vectors * 5.0 + 100.0
            pde_output = torch.where(pde_output > max_pde, max_pde, pde_output)
            pde_output = torch.where(pde_output < 0.0, torch.zeros_like(pde_output), pde_output)

        except Exception as e:
            logger.warning(f"Error in PDE expert processing: {str(e)}")
            pde_output = torch.zeros_like(population_vectors)
            raise e
        
        # Ensure all expert outputs have the same shape as population_vectors
        all_outputs = [colonization_output, jump_output, decline_output, pde_output]
        expert_names = ['colonization', 'jump', 'decline', 'pde']
        
        for i, (output, name) in enumerate(zip(all_outputs, expert_names)):
            if output.shape != population_vectors.shape:
                logger.warning(f"{name} output shape {output.shape} doesn't match population shape {population_vectors.shape}")
                if len(output.shape) > len(population_vectors.shape):
                    all_outputs[i] = output.squeeze()
                elif len(output.shape) < len(population_vectors.shape):
                    all_outputs[i] = output.unsqueeze(dim=list(range(len(population_vectors.shape) - len(output.shape))))
                
                # If shapes still don't match, reshape to match
                if all_outputs[i].shape != population_vectors.shape:
                    corrected_output = torch.zeros_like(population_vectors, device=self.device)
                    # Copy as much data as possible
                    min_batch = min(all_outputs[i].shape[0], population_vectors.shape[0])
                    min_eth = min(all_outputs[i].shape[1] if len(all_outputs[i].shape) > 1 else 1, 
                                  population_vectors.shape[1])
                    
                    if len(all_outputs[i].shape) > 1:
                        corrected_output[:min_batch, :min_eth] = all_outputs[i][:min_batch, :min_eth]
                    else:
                        corrected_output[:min_batch, 0] = all_outputs[i][:min_batch]
                    
                    all_outputs[i] = corrected_output
        
        colonization_output, jump_output, decline_output, pde_output = all_outputs

        # Ensure all expert outputs have the same shape before stacking
        # Stack all expert outputs along a new dimension - shape: [batch_size, num_ethnicities, num_experts]
        stacked_expert_outputs = torch.stack([
            colonization_output, jump_output, decline_output, pde_output
        ], dim=2)

        # Calculate main model weight
        expert_sum = expert_weights.sum(dim=2, keepdim=True)
        main_weight = torch.clamp(1.0 - expert_sum, min=0.1, max=0.3)

        prev_weights = getattr(self, '_prev_expert_weights', None)
        if prev_weights is not None and prev_weights.shape == expert_weights.shape:
            # Smooth rapid transitions between experts
            smoothing_factor = 0.2  
            expert_weights = smoothing_factor * prev_weights + (1 - smoothing_factor) * expert_weights
        # Store for next iteration
        self._prev_expert_weights = expert_weights.clone().detach()
        
        # Calculate weighted sum of expert outputs for each ethnicity
        # Shape: [batch_size, num_ethnicities]
        weighted_expert_sum = torch.einsum('bne,bne->bn', stacked_expert_outputs, expert_weights)
        
        # Add main model contribution - shape: [batch_size, num_ethnicities]
        final_output = weighted_expert_sum + main_output * main_weight.squeeze(2)
        
        # Model ethnicity interactions - how population changes in one ethnicity affect others
        interaction_weights = self.ethnicity_interaction(population_vectors)  # [batch_size, num_ethnicities*num_ethnicities]
        interaction_weights = interaction_weights.view(batch_size, self.num_ethnicities, self.num_ethnicities)
        
        # Apply ethnicity interactions
        interaction_effect = torch.zeros_like(final_output)
        
        for i in range(self.num_ethnicities):
            for j in range(self.num_ethnicities):
                if i == j:
                    continue  # Skip self-interaction
                
                # Get population change for source ethnicity j
                pop_change_j = final_output[:, j] - population_vectors[:, j]
                
                # Apply influence from j to i based on the interaction weight
                interaction_effect[:, i] += interaction_weights[:, i, j] * pop_change_j * 0.2
        
        # Add interaction effect to final output
        final_output = final_output + interaction_effect
        
        # Ensure non-negative output
        final_output = torch.clamp(final_output, min=0.0)
        
        # Store interpretations if requested
        if store_interpretations:
            # Store expert outputs and weights for interpretation
            self.expert_outputs = {
                'colonization': colonization_output.detach().cpu().numpy(),
                'jump': jump_output.detach().cpu().numpy(),
                'decline': decline_output.detach().cpu().numpy(),
                'pde': pde_output.detach().cpu().numpy()
            }
            self.expert_weights = expert_weights.detach().cpu().numpy()
            self.expert_classes = expert_classes.detach().cpu().numpy()
            self.main_model_output = main_output.detach().cpu().numpy()
            self.interaction_weights = interaction_weights.detach().cpu().numpy()
            self.interaction_effect = interaction_effect.detach().cpu().numpy()
            self.spatial_features = spatial_features.detach().cpu().numpy()
            
        # pred_end_time = time.time()
        # print(f"Total Prediction computation time: {pred_end_time - pred_start_time:.4f} seconds")
        return final_output, expert_weights, expert_classes
    
    def _create_spatial_features(self, walking_matrix=None, transit_matrix=None, proximity_matrix=None):
        """
        Create spatial features from available matrices
        
        Args:
            walking_matrix (torch.Tensor, optional): Walking distance matrix
            transit_matrix (torch.Tensor, optional): Transit distance matrix
            proximity_matrix (torch.Tensor, optional): Proximity matrix
            
        Returns:
            torch.Tensor: Spatial features tensor
        """
        batch_size = walking_matrix.shape[0] if walking_matrix is not None else (
            transit_matrix.shape[0] if transit_matrix is not None else 
            proximity_matrix.shape[0] if proximity_matrix is not None else 0
        )
        
        if batch_size == 0:
            # No spatial matrices available
            return torch.zeros((batch_size, 1), device=self.device)
        
        # Compute mean distance to other nodes for each node
        spatial_features = []
        
        if walking_matrix is not None:
            # Get mean walking distance (excluding self-connections)
            mask = torch.ones_like(walking_matrix) - torch.eye(batch_size, device=self.device)
            walking_means = (walking_matrix * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
            spatial_features.append(walking_means.unsqueeze(1))
        
        if transit_matrix is not None:
            # Get mean transit distance
            mask = torch.ones_like(transit_matrix) - torch.eye(batch_size, device=self.device)
            transit_means = (transit_matrix * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
            spatial_features.append(transit_means.unsqueeze(1))
        
        if proximity_matrix is not None:
            # Get mean proximity
            mask = torch.ones_like(proximity_matrix) - torch.eye(batch_size, device=self.device)
            proximity_means = (proximity_matrix * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
            spatial_features.append(proximity_means.unsqueeze(1))
        
        # Combine features
        if spatial_features:
            return torch.cat(spatial_features, dim=1)
        else:
            return torch.zeros((batch_size, 1), device=self.device)
    
    def _run_main_model(self, features, coordinates, 
                      walking_matrix=None, transit_matrix=None, proximity_matrix=None):
        """
        Run the main population prediction model
        
        Args:
            features (torch.Tensor): Input features [batch_size, input_dim]
            coordinates (torch.Tensor): Spatial coordinates [batch_size, 2]
            walking_matrix (torch.Tensor, optional): Walking distance matrix
            transit_matrix (torch.Tensor, optional): Transit distance matrix
            proximity_matrix (torch.Tensor, optional): Proximity matrix
            
        Returns:
            torch.Tensor: Predicted population values from the main model for all ethnicities
                         [batch_size, num_ethnicities]
        """
        batch_size = features.size(0)
        
        # Embed features
        feature_embed = self.feature_embedding(features)
        
        # Embed coordinates
        coord_embed = self.coord_embedding(coordinates)
        
        # Combine embeddings
        combined = torch.cat([feature_embed, coord_embed], dim=1)
        combined = self.combine_embeddings(combined)
        
        # Add sequence dimension for spatial attention
        seq_input = combined.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Prepare available matrices
        available_matrices = []
        if walking_matrix is not None:
            available_matrices.append(walking_matrix)
        if transit_matrix is not None:
            available_matrices.append(transit_matrix)
        if proximity_matrix is not None:
            available_matrices.append(proximity_matrix)
        
        try:
            if available_matrices:
                # Apply spatial attention with available matrices
                reshaped_matrices = []
                
                for mat in available_matrices:
                    # Use the batch diagonal as the context for each element
                    diag = torch.diagonal(mat, dim1=0, dim2=1).unsqueeze(1).unsqueeze(1)
                    reshaped_matrices.append(diag)
                
                attended = self.spatial_attention(seq_input, reshaped_matrices)
            else:
                # No spatial information, use default attention
                attended = self.spatial_attention(seq_input)
            
            # Remove sequence dimension
            x = attended.squeeze(1)
        except Exception as e:
            logger.warning(f"Error in spatial attention: {str(e)}. Using combined embeddings directly.")
            x = combined
        
        # Apply remaining layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer - predicts values for all ethnicities
        output = self.output_layer(x)  # [batch_size, num_ethnicities]
        
        return output
    
    def get_interpretability_data(self):
        """
        Get comprehensive interpretability data from the model and all experts.
        
        Returns:
            dict: Comprehensive interpretability data from model and all experts
        """
        interpretability_data = {
            'model_architecture': {
                'input_dim': self.input_dim,
                'num_ethnicities': self.num_ethnicities,
                'hidden_dims': [module.out_features for module in self.layers if isinstance(module, nn.Linear)],
                'thresholds': {
                    'colonization': self.colonization_threshold,
                    'jump': self.jump_threshold,
                    'decline': self.decline_threshold,
                },
                'expert_weights': {
                    'colonization': self.colonization_expert_weight,
                    'jump': self.jump_expert_weight,
                    'decline': self.decline_expert_weight,
                    'pde': self.pde_expert_weight,
                }
            },
            'expert_predictions': self.expert_outputs,
            'expert_weights': self.expert_weights,
            'expert_classes': self.expert_classes,
            'main_model_output': self.main_model_output,
            'interaction_weights': self.interaction_weights,
            'interaction_effect': self.interaction_effect,
            'spatial_features': self.spatial_features
        }
        
        # Add expert-specific interpretation data if available
        try:
            # Router interpretability data
            router_data = self.expert_router.get_interpretability_data()
            interpretability_data['router'] = router_data
        except Exception as e:
            logger.warning(f"Could not get router interpretability data: {str(e)}")
        
        try:
            # Colonization expert interpretability data
            if hasattr(self.colonization_expert, 'get_interpretability_data'):
                colonization_data = self.colonization_expert.get_interpretability_data()
                interpretability_data['colonization_expert'] = colonization_data
        except Exception as e:
            logger.warning(f"Could not get colonization expert interpretability data: {str(e)}")
        
        try:
            # Jump expert interpretability data
            if hasattr(self.jump_expert, 'get_interpretability_data'):
                jump_data = self.jump_expert.get_interpretability_data()
                interpretability_data['jump_expert'] = jump_data
        except Exception as e:
            logger.warning(f"Could not get jump expert interpretability data: {str(e)}")
        
        try:
            # Decline expert interpretability data
            if hasattr(self.decline_expert, 'get_interpretability_data'):
                decline_data = self.decline_expert.get_interpretability_data()
                interpretability_data['decline_expert'] = decline_data
        except Exception as e:
            logger.warning(f"Could not get decline expert interpretability data: {str(e)}")
        
        try:
            # PDE expert interpretability data
            pde_data = self.pde_expert.get_interpretability_data()
            interpretability_data['pde_expert'] = pde_data
        except Exception as e:
            logger.warning(f"Could not get PDE expert interpretability data: {str(e)}")
        
        # Add feature importance with DAUID mapping
        if self.dauid_list is not None:
            feature_importance_by_dauid = {}
            feature_names = [f"feature_{i}" for i in range(self.input_dim)]  # Replace with actual feature names if available
            
            for i, dauid in enumerate(self.dauid_list):
                dauid_key = str(dauid)
                # For each DAUID, create a dictionary mapping feature names to importance scores
                if hasattr(self, 'router') and self.router.feature_importance is not None:
                    feature_importance_by_dauid[dauid_key] = {
                        feature_names[j]: float(self.router.feature_importance[j]) 
                        for j in range(min(len(feature_names), len(self.router.feature_importance)))
                    }
            
            interpretability_data['feature_importance_by_dauid'] = feature_importance_by_dauid
        
        if self.interaction_weights is not None and self.dauid_list is not None:
            interaction_weights_by_dauid = {}
            
            for i, dauid in enumerate(self.dauid_list):
                if i >= self.interaction_weights.shape[0]:
                    continue
                    
                dauid_key = str(dauid)
                
                # Create a more structured representation of interactions
                eth_interactions = {}
                for source_idx in range(self.num_ethnicities):
                    for target_idx in range(self.num_ethnicities):
                        source_eth = self.ethnicity_names[source_idx] if self.ethnicity_names else f"ethnicity_{source_idx}"
                        target_eth = self.ethnicity_names[target_idx] if self.ethnicity_names else f"ethnicity_{target_idx}"
                        
                        interaction_key = f"{source_eth}_to_{target_eth}"
                        eth_interactions[interaction_key] = float(self.interaction_weights[i, source_idx, target_idx])
                
                interaction_weights_by_dauid[dauid_key] = eth_interactions
            
            interpretability_data['ethnicity_interactions_by_dauid'] = interaction_weights_by_dauid
    
        if self.interaction_effect is not None and self.dauid_list is not None:
            interaction_effect_by_dauid = {}
            
            for i, dauid in enumerate(self.dauid_list):
                if i >= self.interaction_effect.shape[0]:
                    continue
                    
                dauid_key = str(dauid)
                
                # Store the effect of interactions on each ethnicity
                effect_dict = {}
                for eth_idx in range(self.num_ethnicities):
                    eth_name = self.ethnicity_names[eth_idx] if self.ethnicity_names else f"ethnicity_{eth_idx}"
                    effect_dict[eth_name] = float(self.interaction_effect[i, eth_idx])
                
                interaction_effect_by_dauid[dauid_key] = effect_dict
            
            interpretability_data['interaction_effect_by_dauid'] = interaction_effect_by_dauid
    
        return interpretability_data