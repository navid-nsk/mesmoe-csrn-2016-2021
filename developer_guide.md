# Developer Guide: Multi-Ethnic CSRN

This guide provides details for researchers and developers who want to extend or modify the Multi-Ethnic Cultural-Spatial Resonance Network (CSRN) model.

## Architecture Overview

The Multi-Ethnic Spatial Mixture of Experts (MESMoE) model combines neural networks with physics-informed modeling using four specialized experts:

```
                     ┌───────────────────┐
                     │  Feature Encoder  │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │ Population Encoder│
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
┌───────────────┐    │  Expert Router    │    ┌───────────────┐
│ Spatial Data   ├───►                   ◄────┤ Ethnicity Data│
└───────────────┘    └┬────────┬────────┬┘    └───────────────┘
                      │        │        │
         ┌────────────┴─┐  ┌───┴────┐ ┌─┴──────────┐ ┌────────────┐
         │ Colonization │  │  Jump  │ │  Decline   │ │ PDE Solver │
         │    Expert    │  │ Expert │ │   Expert   │ │   Expert   │
         └──────┬───────┘  └───┬────┘ └─────┬──────┘ └─────┬──────┘
                │              │            │              │
                └──────────────┼────────────┼──────────────┘
                               │            │
                      ┌────────▼────────────▼───────┐
                      │      Weighted Sum          │
                      └────────┬────────────┬──────┘
                               │            │
                      ┌────────▼───┐  ┌─────▼───────┐
                      │ Ethnicity  │  │  Ethnicity  │
                      │Interactions│  │  Outputs    │
                      └────────────┘  └─────────────┘
```

## Core Components

### 1. Expert Models

Each expert model is designed to handle a specific population dynamic regime:

#### Colonization Expert (`experts/colonization_expert.py`)

Handles transitions from zero to non-zero population. To modify this expert:
- Adjust the colonization threshold in `MultiEthnicSpatialMoEPredictor`
- Modify the colonization size distributions in `ColonizationExpert`
- Update the importance function for ethnicity-specific colonization behavior

#### Jump Process Expert (`experts/jump_expert.py`)

Models sudden population increases. To modify this expert:
- Adjust jump detection thresholds
- Modify the jump magnitude estimators
- Update inter-ethnicity influence matrices for jump behavior

#### Decline Expert (`experts/decline_expert.py`)

Models population decreases and exodus events. To modify this expert:
- Adjust decline detection thresholds
- Update the decline magnitude estimators
- Modify risk assessment functions

#### CSRN Diffusion Solver (`experts/pde_expert.py`)

Implements the physics-informed neural PDE solver. To modify this expert:
- Update the PDE equation in `CSRNDiffusionSolver`
- Modify the numerical integration method
- Adjust stability constraints or parameter bounds

### 2. Expert Router (`routing/expert_router.py`)

Determines which experts handle each spatial region. To modify the router:
- Update the routing architecture in `ExpertRouter`
- Modify the inductive biases in `compute_hints`
- Change the blending parameter for rule-based routing

### 3. Spatial Attention (`spatial/multi_matrix_attention.py`)

Processes spatial relationships between dissemination areas. To modify:
- Change the attention mechanism in `MultiMatrixSpatialAttention`
- Update matrix weighting schemes
- Add new spatial matrices (e.g., economic, social, communication)

### 4. Ethnicity Interactions (`spatial_moe_model.py`)

Models how different ethnic populations influence each other. To modify:
- Update the interaction matrix in `compute_ethnicity_interactions`
- Change how interactions are applied to final predictions
- Modify the scaling factors for interactions

## Adding a New Expert

To add a new expert model (e.g., for modeling segregation patterns):

1. Create a new expert class in `experts/` that implements:
   - `__init__` method to initialize the expert
   - `forward` method that takes features and population as input

```python
class SegregationExpert(nn.Module):
    def __init__(self, input_dim, num_ethnicities, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_ethnicities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ethnicities)
        )
        
    def forward(self, features, population, coordinates, spatial_matrices=None):
        # Implement segregation patterns prediction logic
        inputs = torch.cat([features, population], dim=1)
        output = self.encoder(inputs)
        return output
```

2. Add the new expert to `MultiEthnicSpatialMoEPredictor` in `spatial_moe_model.py`:
   - Initialize the expert in `__init__`
   - Add the expert weight parameter
   - Update the routing logic to include the new expert
   - Include the new expert in the weighted sum

## Modifying the PDE Equation

To modify the Cultural-Spatial Resonance Network PDE equation:

1. Update the equation in the `CSRNDiffusionSolver` class:

```python
def compute_pde_rhs(self, population, coordinates, spatial_matrices):
    # Calculate spatial derivatives
    gradient = self.compute_gradient(population, coordinates)
    laplacian = self.compute_laplacian(population, spatial_matrices)
    
    # Compute area-specific parameters
    diffusion = self.diffusion_network(features)
    
    # YOUR MODIFIED PDE TERMS HERE
    # Example: Add a new term for cultural exchange
    cultural_exchange = self.cultural_exchange_net(features) * self.compute_cultural_similarity(population)
    
    # Combine all terms
    rhs = diffusion * laplacian + directed_flow + mean_reversion + gradient_penalty + cultural_exchange
    
    return rhs
```

2. If you add new terms that require CUDA acceleration, update the CUDA extensions:
   - Add new kernels to `cuda_ext/csrn_cuda_kernels.cu`
   - Update the C++ interface in `cuda_ext/csrn_cuda.cpp`
   - Add Python wrappers in the appropriate files

## Adding New Spatial Matrices

To incorporate new spatial relationships (e.g., economic connectivity):

1. Add the new matrix to `SpatialMatrixHandler` in `spatial_data_loader.py`:

```python
def load_spatial_data(self):
    # Load existing matrices
    walking_path = os.path.join(self.data_dir, 'walking_matrix.npy')
    # ...
    
    # Load new economic connectivity matrix
    economic_path = os.path.join(self.data_dir, 'economic_matrix.npy')
    if os.path.exists(economic_path):
        self.economic_matrix = np.load(economic_path)
        logger.info(f"Loaded economic matrix with shape {self.economic_matrix.shape}")
    else:
        logger.warning(f"Economic matrix not found at {economic_path}")
```

2. Update the `get_matrices_for_batch` method to include the new matrix

3. Modify the spatial attention mechanism to incorporate the new matrix:

```python
class MultiMatrixSpatialAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_matrices=4):  # Updated for 4 matrices
        # ...
        
    def forward(self, x, walking_matrix, transit_matrix, proximity_matrix, economic_matrix):
        # Process with existing matrices
        
        # Process with new economic matrix
        if economic_matrix is not None:
            economic_attention = self.compute_attention(economic_matrix)
            attention_outputs.append(economic_attention)
        
        # Combine all attention outputs
        # ...
```

## Hyperparameter Optimization

To modify the hyperparameter optimization process:

1. Update parameter ranges in `hyperopt.py`:

```python
# Define hyperparameters to optimize
attention_heads = trial.suggest_int('attention_heads', 2, 8)  # Expanded range
hidden_dim_1_base = trial.suggest_int('hidden_dim_1_base', 8, 64)  # Expanded range

# Add new hyperparameters
cultural_exchange_weight = trial.suggest_float('cultural_exchange_weight', 0.1, 2.0)
ethnicity_embedding_dim = trial.suggest_int('ethnicity_embedding_dim', 8, 32)
```

2. Pass the new hyperparameters to the model:

```python
model = MultiEthnicSpatialMoEPredictor(
    # Existing parameters
    cultural_exchange_weight=cultural_exchange_weight,
    ethnicity_embedding_dim=ethnicity_embedding_dim
)
```

## Training on New Data

To adapt the model to a new city or region:

1. Prepare data in the expected format:
   - Features CSV: Demographic features for each area
   - Population CSV: Population by ethnicity for each area
   - Spatial matrices: Walking, transit, and proximity relationships

2. Update data paths in your training script:

```python
args.data_dir = "./data/new_city"
args.spatial_dir = "./spatial_data/new_city"
```

3. You may need to adjust expert thresholds based on the new region's characteristics:

```python
model = MultiEthnicSpatialMoEPredictor(
    # ...
    colonization_threshold=0.2,  # Adjusted for new city
    jump_threshold=150.0,  # Adjusted for new city
    # ...
)
```

## Debugging

For debugging complex numerical issues:

1. Use the built-in visualization tools:

```python
python model_interpretation.py --model_path ./model/your_model.pt --debug_mode
```

2. Add gradient checks during training:

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Gradient issue in {name}: {grad_norm}")
```

3. Use the CUDA profiler for performance issues:

```python
with torch.cuda.profiler.profile():
    with torch.cuda.nvtx.range("Model Forward"):
        output = model(features, population, coordinates)
```

## Contributing

When contributing to this project:

1. Create a new branch for your feature or fix
2. Write tests for any new functionality
3. Update documentation to reflect your changes
4. Submit a pull request with a clear description of the changes

For any questions about extending or modifying the code, please open an issue on the repository.
