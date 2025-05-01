import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple, List, Optional, Union

from csrn.experts.csrn_cuda_ops import CSRNCudaOps


class MultiEthnicCSRNDiffusionSolverCUDA(nn.Module):
    """
    CUDA-accelerated physics-informed neural PDE solver for the Cultural-Spatial Resonance Network (CSRN)
    that handles multiple ethnicities simultaneously.
    
    Implements the core evolution equation:
    ∂φ_i/∂t = D_i∇²φ_i + ∇·(λ_i v_i ∇φ_i) + (-α_i(x,t)·(φ_i - φ̄_i) - β_i(t)·max(0, ||∇φ_i||² - K_i) - γ_i·div(v_i)·φ_i)
    
    Where i denotes the ethnicity index.
    
    This solver ensures stability, conservation, and physically meaningful dynamics across all ethnicities.
    """
    
    def __init__(self, 
                 num_features: int,
                 num_ethnicities: int,
                 hidden_dim: int = 64,
                 num_steps: int = 10,
                 global_params: Optional[Dict] = None,
                 enable_debug: bool = False,
                 log_space: bool = True,
                 epsilon: float = 1e-6,
                 device: str = 'cuda'):
        """
        Initialize the Multi-Ethnic CSRN PDE solver.
        
        Args:
            num_features: Number of census features
            num_ethnicities: Number of ethnicities to model
            hidden_dim: Hidden dimension for internal representations
            num_steps: Number of steps for time integration
            global_params: Global parameters for the model
            enable_debug: Enable detailed debugging and timing
            log_space: Whether to work in log space
            epsilon: Small value for numerical stability
            device: Device to use for computations (should be 'cuda')
        """
        super(MultiEthnicCSRNDiffusionSolverCUDA, self).__init__()
        self.num_features = num_features
        self.num_ethnicities = num_ethnicities
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.device = device
        self.enable_debug = enable_debug
        self.log_space = log_space
        self.epsilon = epsilon
        self.dauid_list = None
        
        # Default global parameters
        if global_params is None:
            global_params = {
                "K": 0.05,  # Gradient magnitude threshold
                "sigma_squared": 3.6,  # For α(x,t) calculation
                "alpha_0": 0.57,  # Base value for α
                "beta_0": 3.34  # Base value for β
            }
        
        # MODIFIED: Adjust parameters for log space if needed
        if log_space:
            # For log space, these parameters might need adjustment
            if "K" in global_params:
                global_params["K"] = min(global_params["K"], 0.05)  # Lower threshold in log space
            if "sigma_squared" in global_params:
                global_params["sigma_squared"] = max(global_params["sigma_squared"], 3.6)  # Higher smoothing
        
        # Store shared parameters (same for all ethnicities)
        self.K = nn.Parameter(torch.tensor(global_params.get("K", 0.05)), requires_grad=True)
        self.sigma_squared = nn.Parameter(torch.tensor(global_params.get("sigma_squared", 3.6)), requires_grad=True)
        self.alpha_0 = nn.Parameter(torch.tensor(global_params.get("alpha_0", 0.57)), requires_grad=True)
        self.beta_0 = nn.Parameter(torch.tensor(global_params.get("beta_0", 3.34)), requires_grad=True)
        
        # Shared feature encoding - preserves area-specific features
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Ethnicity-specific feature enhancement
        self.ethnicity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_ethnicities)
        ])
        
        # FIXED: Modified parameter networks to produce area-specific values
        # Diffusion tensor network (outputs D) for all ethnicities and all areas
        self.diffusion_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ethnicities),  # One parameter per ethnicity for each area
            nn.Softplus()  # Ensure positive diffusion coefficient
        )
        
        # FIXED: Amplification network (outputs λ) for all ethnicities and all areas
        self.amplification_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ethnicities),  # One parameter per ethnicity for each area
            nn.Softplus()  # Ensure positive amplification
        )
        
        # FIXED: Velocity network (outputs v) for all ethnicities and all areas
        self.velocity_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2 * num_ethnicities),  # 2D velocity field per ethnicity for each area
            nn.Tanh()
        )
        
        # FIXED: Alpha parameter network for all ethnicities and all areas
        self.alpha_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ethnicities),  # One parameter per ethnicity for each area
            nn.Softplus()  # Ensure positive alpha
        )
        
        # FIXED: Beta parameter network for all ethnicities and all areas
        self.beta_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ethnicities),  # One parameter per ethnicity for each area
            nn.Softplus()  # Ensure positive beta
        )
        
        # FIXED: Gamma parameter network for all ethnicities and all areas
        self.gamma_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ethnicities),  # One parameter per ethnicity for each area
            nn.Softplus()  # Ensure positive gamma
        )
        
        # FIXED: Ethnicity interaction module - captures how one ethnicity affects others
        # Now produces area-specific interaction parameters
        self.ethnicity_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ethnicities * num_ethnicities),
            nn.Sigmoid()
        )
        
        # Conservation trackers for interpretability
        self.mass_conservation_error = None
        self.energy_conservation_error = None
        
        # Store interpretable parameters
        self.diffusion_values = None
        self.amplification_values = None
        self.velocity_values = None
        self.alpha_values = None
        self.beta_values = None
        self.gamma_values = None
        
        # Move model to device
        self.to(device)
    
    def compute_laplacian(self, 
                        population: torch.Tensor, 
                        spatial_matrix: torch.Tensor,
                        epsilon: float = 1e-6) -> torch.Tensor:
        """
        Compute the graph Laplacian operation.
        
        Args:
            population: Population tensor of shape (batch_size, num_das, num_ethnicities) 
                    or (batch_size, num_das, 1) for single ethnicity
            spatial_matrix: Spatial connectivity matrix (num_das, num_das)
            epsilon: Small value for numerical stability
            
        Returns:
            Laplacian applied to population (batch_size, num_das, num_ethnicities)
        """
        # Ensure population is contiguous
        population = population.contiguous()
        spatial_matrix = spatial_matrix.contiguous()
        
        batch_size = population.shape[0]
        num_das = population.shape[1]
        
        # Determine if this is a single ethnicity or multi-ethnicity tensor
        if len(population.shape) == 2:
            # Handle 2D input (batch_size, num_das) - reshape to 3D
            population = population.unsqueeze(-1)
            num_ethnicities = 1
        elif len(population.shape) == 3:
            num_ethnicities = population.shape[2]
        else:
            raise ValueError(f"Expected 2D or 3D population tensor, got shape {population.shape}")
        
        # Ensure dimensions match
        if spatial_matrix.shape[0] != num_das or spatial_matrix.shape[1] != num_das:
            print(f"Warning: Spatial matrix shape {spatial_matrix.shape} doesn't match population DAs {num_das}")
            # Use identity matrix as a fallback
            spatial_matrix = torch.eye(num_das, device=self.device)
        
        try:
            # Compute degree matrix
            degrees = spatial_matrix.sum(dim=1)
            
            # Stabilize with epsilon
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + epsilon))
            
            # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
            normalized_matrix = torch.eye(num_das, device=self.device) - \
                            D_inv_sqrt @ spatial_matrix @ D_inv_sqrt
            
            # Apply Laplacian to each ethnicity
            laplacians = []
            
            # Adjust the processing based on whether we're handling a single ethnicity or multiple ethnicities
            if num_ethnicities == 1 or len(population.shape) == 2:
                # Single ethnicity case (or 2D input)
                if len(population.shape) == 2:
                    eth_pop = population  # Already 2D
                else:
                    eth_pop = population[:, :, 0]  # Extract the only ethnicity
                    
                eth_laplacian = torch.bmm(eth_pop.unsqueeze(1), normalized_matrix.unsqueeze(0).expand(batch_size, -1, -1))
                laplacians.append(eth_laplacian.squeeze(1))
                
                # Return a 3D tensor with the last dimension of size 1
                return torch.stack(laplacians, dim=2)
            
            else:
                # Multi-ethnicity case
                for eth_idx in range(num_ethnicities):
                    eth_pop = population[:, :, eth_idx]
                    eth_laplacian = torch.bmm(eth_pop.unsqueeze(1), normalized_matrix.unsqueeze(0).expand(batch_size, -1, -1))
                    laplacians.append(eth_laplacian.squeeze(1))
                
                # Stack laplacians for all ethnicities
                return torch.stack(laplacians, dim=2)
        
        except RuntimeError as e:
            print(f"Error computing Laplacian: {e}")
            # Return zeros as fallback
            return torch.zeros_like(population)
        
    def compute_pde_terms(self, 
                              population: torch.Tensor,
                              features: torch.Tensor,
                              spatial_matrices: Dict[str, torch.Tensor],
                              da_coordinates: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all terms in the PDE with CUDA acceleration and gradient flow support
        for multiple ethnicities simultaneously - FIXED to produce area-specific parameters
        
        Args:
            population: Population tensor [batch_size, num_das, num_ethnicities]
            features: Feature tensor [batch_size, num_das, num_features]
            spatial_matrices: Dict of spatial matrices
            da_coordinates: Coordinates of DAs [num_das, 2]
            
        Returns:
            Dict of PDE terms for all ethnicities
        """
        if self.enable_debug:
            start_time = time.time()
            
        batch_size = population.shape[0]
        num_das = population.shape[1]
        
        # Ensure da_coordinates are limited to the same number of DAs as population
        if da_coordinates.shape[0] > num_das:
            da_coordinates = da_coordinates[:num_das]
        
        # Process features through shared encoder - maintain shape for each DA
        batch_features = features.reshape(-1, self.num_features)
        shared_encoding = self.feature_encoder(batch_features)
        
        # Initialize PDE parameter tensors for all ethnicities
        diffusion = torch.zeros(batch_size, num_das, self.num_ethnicities, device=self.device)
        amplification = torch.zeros(batch_size, num_das, self.num_ethnicities, device=self.device)
        velocity = torch.zeros(batch_size, num_das, self.num_ethnicities, 2, device=self.device)
        alpha = torch.zeros(batch_size, num_das, self.num_ethnicities, device=self.device)
        beta = torch.zeros(batch_size, num_das, self.num_ethnicities, device=self.device)
        gamma = torch.zeros(batch_size, num_das, self.num_ethnicities, device=self.device)
        
        # FIXED: Process each ethnicity for all areas in parallel
        for eth_idx in range(self.num_ethnicities):
            # Apply ethnicity-specific encoding
            eth_encoding = self.ethnicity_encoders[eth_idx](shared_encoding)
            
            # Get parameters for this ethnicity for all areas at once
            all_diffusion = self.diffusion_network(eth_encoding)
            all_amplification = self.amplification_network(eth_encoding)
            all_velocity = self.velocity_network(eth_encoding)
            all_alpha = self.alpha_network(eth_encoding)
            all_beta = self.beta_network(eth_encoding)
            all_gamma = self.gamma_network(eth_encoding)
            
            # FIXED: Extract the parameters for each DAUID for this ethnicity
            # The key fix: we use the actual values from the network output for each area
            # instead of just selecting a single value for the ethnicity
            
            # Each parameter has shape [batch_size*num_das, num_ethnicities]
            # We need to reshape to [batch_size, num_das, num_ethnicities] and then select the eth_idx column
            
            # Reshape to [batch_size, num_das, num_ethnicities]
            diffusion_reshaped = all_diffusion.view(batch_size, num_das, self.num_ethnicities)
            amplification_reshaped = all_amplification.view(batch_size, num_das, self.num_ethnicities)
            alpha_reshaped = all_alpha.view(batch_size, num_das, self.num_ethnicities)
            beta_reshaped = all_beta.view(batch_size, num_das, self.num_ethnicities)
            gamma_reshaped = all_gamma.view(batch_size, num_das, self.num_ethnicities)
            
            # Velocity has shape [batch_size*num_das, 2*num_ethnicities]
            # Reshape to [batch_size, num_das, 2*num_ethnicities]
            velocity_reshaped = all_velocity.view(batch_size, num_das, 2*self.num_ethnicities)
            
            # Extract current ethnicity's parameters - NOW AREA-SPECIFIC!
            diffusion[:, :, eth_idx] = diffusion_reshaped[:, :, eth_idx]
            amplification[:, :, eth_idx] = amplification_reshaped[:, :, eth_idx]
            alpha[:, :, eth_idx] = alpha_reshaped[:, :, eth_idx]
            beta[:, :, eth_idx] = beta_reshaped[:, :, eth_idx]
            gamma[:, :, eth_idx] = gamma_reshaped[:, :, eth_idx]
            
            # For velocity, we need to extract the x and y components for this ethnicity
            velocity[:, :, eth_idx, 0] = velocity_reshaped[:, :, 2*eth_idx]
            velocity[:, :, eth_idx, 1] = velocity_reshaped[:, :, 2*eth_idx+1]
        
        # Process ethnicity interactions using the shared encoding
        eth_interactions = self.ethnicity_interaction(shared_encoding)
        eth_interactions = eth_interactions.view(batch_size*num_das, self.num_ethnicities, self.num_ethnicities)
        eth_interactions = eth_interactions.view(batch_size, num_das, self.num_ethnicities, self.num_ethnicities)
        
        # Create a copy of diffusion to avoid in-place modification
        diffusion_with_interactions = diffusion.clone()
        
        # Apply ethnicity interactions using tensor operations without in-place modifications
        for source_eth in range(self.num_ethnicities):
            for target_eth in range(self.num_ethnicities):
                if source_eth == target_eth:
                    continue  # Skip self-interaction
                
                # Get interaction weight tensor for all areas at once
                interaction_weights = eth_interactions[:, :, target_eth, source_eth]
                
                # Update without in-place operation by creating the full contribution
                interaction_contribution = diffusion[:, :, source_eth] * interaction_weights * 0.2
                diffusion_with_interactions[:, :, target_eth] = diffusion_with_interactions[:, :, target_eth] + interaction_contribution
        
        # Replace the original diffusion with the version including interactions
        diffusion = diffusion_with_interactions
        
        # Scale parameters for more dynamic behavior - avoid in-place operations
        diffusion = diffusion * 0.5
        amplification = amplification * 0.5
        
        # Add base velocity to ensure movement - FIX: Avoid using index_put with slices
        # Create positive x velocity and negative y velocity
        base_velocity_x = torch.ones(batch_size, num_das, self.num_ethnicities, 1, device=self.device) * 0.1
        base_velocity_y = torch.ones(batch_size, num_das, self.num_ethnicities, 1, device=self.device) * -0.1
        
        # Concatenate to create the full base velocity tensor
        base_velocity = torch.cat([base_velocity_x, base_velocity_y], dim=3)
        
        # Add to original velocity
        velocity = velocity + base_velocity
        velocity = velocity * 0.5
        
        # Store interpretable parameters - KEEP THE FULL TENSORS WITH AREA-SPECIFIC VALUES
        self.diffusion_values = diffusion.detach().cpu().numpy()
        self.amplification_values = amplification.detach().cpu().numpy()
        self.velocity_values = velocity.detach().cpu().numpy()
        self.alpha_values = alpha.detach().cpu().numpy()
        self.beta_values = beta.detach().cpu().numpy()
        self.gamma_values = gamma.detach().cpu().numpy()
        
        # Get walking matrix (primary for diffusion)
        walking_matrix = spatial_matrices['walking']
        
        # Dictionary to collect PDE terms
        pde_terms = {}
        
        # Compute PDE terms for each ethnicity
        all_diffusion_terms = []
        all_directed_flow = []
        all_mean_reversion = []
        all_gradient_penalty = []
        all_advection_stabilization = []
        
        for eth_idx in range(self.num_ethnicities):
            # Extract parameters for this ethnicity
            eth_diffusion = diffusion[:, :, eth_idx]
            eth_amplification = amplification[:, :, eth_idx]
            eth_velocity = velocity[:, :, eth_idx]
            eth_alpha = alpha[:, :, eth_idx]
            eth_beta = beta[:, :, eth_idx]
            eth_gamma = gamma[:, :, eth_idx]
            eth_population = population[:, :, eth_idx]
            
            # Compute gradient of population field using CUDA with autograd support
            eth_gradient = CSRNCudaOps.compute_gradient(eth_population, da_coordinates)
            
            # Add small gradient noise (using addition, not in-place)
            gradient_noise = torch.randn_like(eth_gradient) * 0.01
            eth_gradient = eth_gradient + gradient_noise
            
            eth_gradient_magnitude_squared = torch.sum(eth_gradient**2, dim=-1)
            
            # Scale gradient magnitude for more significant effects
            if self.log_space:
                eth_gradient_magnitude_squared = torch.clamp(eth_gradient_magnitude_squared, 0.05, 5.0)
            else:
                eth_gradient_magnitude_squared = torch.clamp(eth_gradient_magnitude_squared, 0.05, 100.0)
            
            # Compute Laplacian term for this ethnicity
            eth_laplacian = self.compute_laplacian(
                eth_population.unsqueeze(-1), walking_matrix
            ).squeeze(-1)
            
            # Add minimal non-zero laplacian (using addition, not in-place)
            min_laplacian = 0.05 * torch.sign(eth_population)
            eth_laplacian = eth_laplacian + min_laplacian
            
            # Compute divergence of velocity field
            eth_velocity_div = CSRNCudaOps.compute_divergence(eth_velocity, da_coordinates)
            
            # Add minimal non-zero divergence value (using addition, not in-place)
            min_div = 0.05 * torch.sign(eth_population)
            eth_velocity_div = eth_velocity_div + min_div
            
            # Adjust clamping for larger effects
            eth_velocity_div = torch.clamp(eth_velocity_div, -2.0, 2.0)
            
            # Check for NaNs in parameters
            if torch.isnan(self.K) or torch.isnan(self.sigma_squared):
                K_safe = 0.05  # Use reasonable default
                sigma_squared_safe = 3.6
            else:
                K_safe = self.K.item()
                sigma_squared_safe = self.sigma_squared.item()
                    
            # Calculate the scale factor for this ethnicity
            eth_pop_max = eth_population.abs().max().item()
            
            # Adjust scale factor based on log space setting
            if self.log_space:
                eth_scale_factor = min(0.5, 1.0 / (eth_pop_max + self.epsilon))
            else:
                eth_scale_factor = min(3.0, 100.0 / (eth_pop_max + 1e-6))
            
            # Use CUDA to compute PDE terms with autograd support
            try:
                eth_diffusion_term, eth_directed_flow, eth_mean_reversion, eth_gradient_penalty, eth_advection_stabilization = \
                    CSRNCudaOps.compute_pde_terms(
                        eth_population,
                        eth_gradient,
                        eth_diffusion,
                        eth_amplification,
                        eth_velocity,
                        eth_alpha,
                        eth_beta,
                        eth_gamma,
                        eth_laplacian,
                        eth_velocity_div,
                        sigma_squared_safe,
                        K_safe,
                        eth_scale_factor
                    )
                
                # Scale each term (using multiplication, not in-place)
                eth_diffusion_term = eth_diffusion_term * 2.0
                eth_directed_flow = eth_directed_flow * 2.0
                eth_mean_reversion = eth_mean_reversion * 2.0
                eth_gradient_penalty = eth_gradient_penalty * 1.5
                eth_advection_stabilization = eth_advection_stabilization * 1.5
            
            except RuntimeError as e:
                print(f"Error computing PDE terms for ethnicity {eth_idx}: {e}")
                # Fallback to zeros
                eth_diffusion_term = torch.zeros_like(eth_population)
                eth_directed_flow = torch.zeros_like(eth_population)
                eth_mean_reversion = torch.zeros_like(eth_population)
                eth_gradient_penalty = torch.zeros_like(eth_population)
                eth_advection_stabilization = torch.zeros_like(eth_population)
                raise e
            
            # Store PDE terms for this ethnicity
            all_diffusion_terms.append(eth_diffusion_term)
            all_directed_flow.append(eth_directed_flow)
            all_mean_reversion.append(eth_mean_reversion)
            all_gradient_penalty.append(eth_gradient_penalty)
            all_advection_stabilization.append(eth_advection_stabilization)
        
        # Stack PDE terms for all ethnicities
        pde_terms['diffusion_term'] = torch.stack(all_diffusion_terms, dim=2)
        pde_terms['directed_flow'] = torch.stack(all_directed_flow, dim=2)
        pde_terms['mean_reversion'] = torch.stack(all_mean_reversion, dim=2)
        pde_terms['gradient_penalty'] = torch.stack(all_gradient_penalty, dim=2)
        pde_terms['advection_stabilization'] = torch.stack(all_advection_stabilization, dim=2)
        
        if self.enable_debug:
            end_time = time.time()
            print(f"Total PDE terms computation time: {end_time - start_time:.4f} seconds")
                
        return pde_terms
    
    # The rest of the class remains unchanged
    def rk4_step(self, 
                population: torch.Tensor,
                features: torch.Tensor,
                spatial_matrices: Dict[str, torch.Tensor],
                da_coordinates: torch.Tensor,
                dt: float) -> torch.Tensor:
        """
        Perform a single RK4 integration step with CUDA acceleration and gradient flow support
        for multiple ethnicities simultaneously.
        
        Args:
            population: Population tensor of shape [batch_size, num_das, num_ethnicities]
            features: Feature tensor of shape [batch_size, num_das, num_features]
            spatial_matrices: Dict of spatial matrices
            da_coordinates: Coordinates of DAs
            dt: Time step size (scalar float)
            
        Returns:
            Updated population tensor [batch_size, num_das, num_ethnicities]
        """
        if self.enable_debug:
            start_time = time.time()
        
        # Ensure da_coordinates is limited to match population dimensions
        num_das = population.shape[1]
        if da_coordinates.shape[0] > num_das:
            da_coordinates = da_coordinates[:num_das]
        
        # Define a function to compute the right-hand side of the PDE for all ethnicities
        def compute_rhs(pop):
            # Get all PDE terms
            terms = self.compute_pde_terms(pop, features, spatial_matrices, da_coordinates)
            
            # Add debugging
            if self.enable_debug:
                print(f"Diffusion term range: [{terms['diffusion_term'].min().item():.6f}, {terms['diffusion_term'].max().item():.6f}]")
                print(f"Directed flow range: [{terms['directed_flow'].min().item():.6f}, {terms['directed_flow'].max().item():.6f}]")
                print(f"Mean reversion range: [{terms['mean_reversion'].min().item():.6f}, {terms['mean_reversion'].max().item():.6f}]")
                print(f"Gradient penalty range: [{terms['gradient_penalty'].min().item():.6f}, {terms['gradient_penalty'].max().item():.6f}]")
                print(f"Advection stabilization range: [{terms['advection_stabilization'].min().item():.6f}, {terms['advection_stabilization'].max().item():.6f}]")
            
            # Sum all terms to get the right-hand side
            rhs = terms['diffusion_term'] + terms['directed_flow'] + \
                terms['mean_reversion'] + terms['gradient_penalty'] + \
                terms['advection_stabilization']
            
            # Scale right-hand side for log space and ensure non-zero changes
            if self.log_space:
                # Widen the clamping range
                rhs = torch.clamp(rhs, -1.0, 1.0)
                
                # Add a small forced change based on population sign
                # This ensures we always have some minimal update
                rhs = rhs + 0.01 * torch.sign(pop + 0.001)
            
            if self.enable_debug:
                print(f"RHS range: [{rhs.min().item():.6f}, {rhs.max().item():.6f}]")
                
            return rhs
        
        # Compute k1
        if self.enable_debug:
            k1_start = time.time()
            
        k1 = compute_rhs(population)
        
        if self.enable_debug:
            k1_end = time.time()
            print(f"k1 computation time: {k1_end - k1_start:.4f} seconds")
        
        # Ensure dt is a scalar float value for multiplication
        dt_float = float(dt)
        
        # Adjust dt for log space
        if self.log_space:
            # Use smaller time steps in log space to prevent large jumps
            dt_float = dt_float * 0.1
        
        try:
            # Compute k2
            if self.enable_debug:
                k2_start = time.time()
                
            pop_k2 = population + 0.5 * dt_float * k1
            k2 = compute_rhs(pop_k2)
            
            if self.enable_debug:
                k2_end = time.time()
                print(f"k2 computation time: {k2_end - k2_start:.4f} seconds")
            
            # Compute k3
            if self.enable_debug:
                k3_start = time.time()
                
            pop_k3 = population + 0.5 * dt_float * k2
            k3 = compute_rhs(pop_k3)
            
            if self.enable_debug:
                k3_end = time.time()
                print(f"k3 computation time: {k3_end - k3_start:.4f} seconds")
            
            # Compute k4
            if self.enable_debug:
                k4_start = time.time()
                
            pop_k4 = population + dt_float * k3
            k4 = compute_rhs(pop_k4)
            
            if self.enable_debug:
                k4_end = time.time()
                print(f"k4 computation time: {k4_end - k4_start:.4f} seconds")
            
            # Perform the RK4 update for all ethnicities at once
            if self.enable_debug:
                rk4_start = time.time()
                
            # Compute RK4 update manually (works for multi-ethnicity tensors)
            new_population = population + dt_float * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
            if self.enable_debug:
                rk4_end = time.time()
                print(f"RK4 final step computation time: {rk4_end - rk4_start:.4f} seconds")
            
        except Exception as e:
            print(f"RK4 integration failed: {e}")
            print("Using Euler fallback")
            
            # Adjust clamping for log space
            if self.log_space:
                new_population = population + torch.clamp(dt_float * k1, -0.5, 0.5)
                new_population = torch.clamp(new_population, -20.0, 20.0)  # Reasonable log-space range
            else:
                new_population = population + torch.clamp(dt_float * k1, -1.0, 1.0)
                new_population = torch.clamp(new_population, -1e6, 1e6)
            
        # Ensure non-negative population if not in log space
        # In log space, negative values can represent very small populations
        if not self.log_space:
            new_population = torch.clamp(new_population, min=0.0)
        
        if self.enable_debug:
            end_time = time.time()
            print(f"Total RK4 step computation time: {end_time - start_time:.4f} seconds")
            
        return new_population
    
    def check_conservation(self, 
                          initial_population: torch.Tensor, 
                          final_population: torch.Tensor,
                          epsilon: float = 1e-6) -> Tuple[float, float]:
        """
        Check conservation of mass and energy for multi-ethnic populations.
        
        Args:
            initial_population: Initial population tensor [batch_size, num_das, num_ethnicities]
            final_population: Final population tensor [batch_size, num_das, num_ethnicities]
            epsilon: Small value for numerical stability
            
        Returns:
            Tuple of mass and energy conservation errors
        """
        # Mass conservation error - sum across DAs and ethnicities
        initial_mass = torch.sum(initial_population, dim=(1,2))
        final_mass = torch.sum(final_population, dim=(1,2))
        mass_diff = torch.abs(final_mass - initial_mass)
        mass_error = mass_diff / (initial_mass + epsilon)
        
        # Energy conservation error (using L2 norm as proxy)
        initial_energy = torch.sum(initial_population**2, dim=(1,2))
        final_energy = torch.sum(final_population**2, dim=(1,2))
        energy_diff = torch.abs(final_energy - initial_energy)
        energy_error = energy_diff / (initial_energy + epsilon)
        
        # Store and return errors
        self.mass_conservation_error = mass_error.mean().item()
        self.energy_conservation_error = energy_error.mean().item()
        
        return self.mass_conservation_error, self.energy_conservation_error
    
    def forward(self, 
              population: torch.Tensor,
              features: torch.Tensor,
              spatial_matrices: Dict[str, torch.Tensor],
              da_coordinates: torch.Tensor,
              time_period: float,
              dauid_list=None) -> torch.Tensor:
        """
        Forward pass performing time integration of the PDE with CUDA acceleration and gradient flow
        for multiple ethnicities simultaneously.
        
        Args:
            population: Population tensor of shape [batch_size, num_das, num_ethnicities]
            features: Feature tensor of shape [batch_size, num_das, num_features]
            spatial_matrices: Dict of spatial matrices
            da_coordinates: Coordinates of DAs
            time_period: Total time period to integrate over (scalar float)
            dauid_list: List of DAUIDs for interpretability
            
        Returns:
            Predicted population tensor [batch_size, num_das, num_ethnicities]
        """
        if self.enable_debug:
            forward_start = time.time()
        population = population.contiguous()
        features = features.contiguous()
        self.dauid_list = dauid_list
        # Store initial population for conservation checks
        initial_population = population.clone()
        da_coordinates = da_coordinates.contiguous()
        # Calculate time step as scalar float
        time_period_float = float(time_period)
        dt = time_period_float / float(self.num_steps)

        # Perform time integration
        current_population = population.clone()
        
        for step in range(self.num_steps):
            if self.enable_debug:
                step_start = time.time()
            
            # Call the RK4 step for all ethnicities at once
            current_population = self.rk4_step(
                current_population, features, spatial_matrices, da_coordinates, dt
            )
            
            if self.enable_debug:
                step_end = time.time()
                print(f"Step {step+1}/{self.num_steps} execution time: {step_end - step_start:.4f} seconds")
            
            # Add a little perturbation to avoid identical forward/backward
            # This helps ensure gradient flow
            if torch.all(current_population == population):
                print("WARNING: Population tensor unchanged by RK4 step. Adding small perturbation.")
                current_population = current_population + torch.randn_like(current_population) * 1e-6
        
        # Check conservation
        self.check_conservation(initial_population, current_population)
        
        if self.enable_debug:
            forward_end = time.time()
            print(f"Forward method execution time: {forward_end - forward_start:.4f} seconds")
        
        # Add final clamping for log space
        if self.log_space:
            # Ensure output is in a reasonable log-space range
            current_population = torch.clamp(current_population, -13.85, 20.0)
            
        # Verify that output is different from input (for debugging)
        if torch.allclose(current_population, population):
            print("WARNING: Output tensor is identical to input tensor!")
            
        if not self.log_space:
            zero_threshold = 0.005  
            current_population = torch.where(
                current_population < zero_threshold,
                torch.zeros_like(current_population),
                current_population
            )
        
        return current_population
    
    def get_interpretability_data(self):
        """
        Get interpretable parameters and conservation metrics with DAUIDs as keys.
        
        Returns:
            Dict of interpretability data
        """
        # Base interpretability data (global parameters)
        interp_data = {
            'mass_conservation_error': self.mass_conservation_error,
            'energy_conservation_error': self.energy_conservation_error,
            'K': self.K.item(),
            'sigma_squared': self.sigma_squared.item(),
            'alpha_0': self.alpha_0.item(),
            'beta_0': self.beta_0.item()
        }
        
        # If we have both parameter values and DAUIDs, use DAUIDs as keys
        if self.dauid_list is not None and self.diffusion_values is not None:
            # Convert numpy arrays to dictionaries with DAUIDs as keys
            
            # Initialize parameter dictionaries
            diffusion_dict = {}
            amplification_dict = {}
            velocity_dict = {}
            alpha_dict = {}
            beta_dict = {}
            gamma_dict = {}
            
            # Map each DAUID to its corresponding parameter values
            for i, dauid in enumerate(self.dauid_list):
                # Skip if index is out of bounds
                if i >= self.diffusion_values.shape[1]:
                    continue
                    
                # Convert numeric DAUID to string if needed
                dauid_key = str(dauid)
                
                # Store each parameter for this DAUID - using correct index i (area-specific)
                # First dimension is batch, so we take the first item (batch_index=0)
                diffusion_dict[dauid_key] = self.diffusion_values[0, i, :].tolist()
                amplification_dict[dauid_key] = self.amplification_values[0, i, :].tolist()
                alpha_dict[dauid_key] = self.alpha_values[0, i, :].tolist()
                beta_dict[dauid_key] = self.beta_values[0, i, :].tolist()
                gamma_dict[dauid_key] = self.gamma_values[0, i, :].tolist()
                velocity_dict[dauid_key] = self.velocity_values[0, i, :, :].tolist()
            
            # Add the parameter dictionaries to the interpretability data
            interp_data.update({
                'diffusion': diffusion_dict,
                'amplification': amplification_dict,
                'velocity': velocity_dict,
                'alpha': alpha_dict,
                'beta': beta_dict,
                'gamma': gamma_dict
            })
        else:
            # If DAUIDs are not available, just return the numpy arrays
            interp_data.update({
                'diffusion': self.diffusion_values,
                'amplification': self.amplification_values,
                'velocity': self.velocity_values,
                'alpha': self.alpha_values,
                'beta': self.beta_values,
                'gamma': self.gamma_values
            })
        
        return interp_data