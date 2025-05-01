# Multi-Ethnic Cultural-Spatial Resonance Networks (CSRN)

## Overview

This repository contains the implementation of Multi-Ethnic Cultural-Spatial Resonance Networks (CSRN) for Population Dynamics Prediction. The model combines neural networks with physics-informed modeling through a specialized mixture of experts framework to predict the dynamics of multi-ethnic populations across spatial regions.

The core innovations include:
- Area-specific parameter learning for heterogeneous dynamics
- Specialized experts for different population regimes (colonization, jump process, decline, PDE-based)
- Learnable expert routing with inductive biases
- Multi-matrix spatial attention for complex spatial relationships
- CUDA-accelerated physics-informed neural PDE solver
- Ethnicity interaction modeling for interdependent population dynamics

## Requirements

### Python Dependencies
```
numpy>=1.19.0
pandas>=1.1.0
torch>=1.9.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
geopandas>=0.9.0
pysal>=2.3.0
esda>=2.3.1
tqdm>=4.50.0
```

### CUDA Requirements
- CUDA Toolkit ≥ 10.2
- cuDNN ≥ 7.6

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/multi-ethnic-csrn.git
cd multi-ethnic-csrn
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install CUDA extensions:
```bash
cd cuda_ext
python setup.py install
cd ..
```

## Project Structure

```
.
├── csrn/                       # Core CSRN implementation
│   ├── data_loader/            # Data loading utilities
│   │   └── spatial_data_loader.py  # Multi-ethnic data handler
│   ├── experts/                # Specialized expert models
│   ├── routing/                # Expert routing components
│   └── pde/                    # PDE solver implementation
├── cuda_ext/                   # CUDA extensions
│   ├── csrn_cuda.cpp           # C++ interface
│   ├── csrn_cuda_kernels.cu    # CUDA kernel implementations
│   └── setup.py                # Build configuration
├── scripts/                    # Utility scripts
│   ├── train.py                # Main training script
│   ├── hyperopt.py             # Hyperparameter optimization
│   └── visualize.py            # Visualization tools
├── spatial_moe_model.py        # Multi-Ethnic Spatial MoE Predictor
├── spatial_moe_trainer.py      # Trainer implementation
├── model_visualization.py      # Visualization utilities
└── ethnicity_interaction_visualization.py  # Ethnicity interaction analysis
```

## Usage

### Data Preparation

The model expects data in the following formats:
- Features: CSV files with demographic features for each Dissemination Area (DA)
- Population: CSV files with population by ethnicity for each DA
- Spatial: NPY files with walking, transit, and proximity matrices

#### Data Download

The required datasets are available on Figshare:

- **Main Dataset**: Population and features data can be downloaded from:  
  [https://doi.org/10.6084/m9.figshare.28912370](https://doi.org/10.6084/m9.figshare.28912370)

- **Spatial Data**: Walking, transit, and proximity matrices can be downloaded from:  
  [https://doi.org/10.6084/m9.figshare.28912400](https://doi.org/10.6084/m9.figshare.28912400)

After downloading, extract the files and organize them in the following structure:

```
data/
├── 2016_train_features_all.csv
├── 2016_train_population_all.csv
├── 2021_train_population_all.csv
├── 2016_val_features_all.csv
├── 2016_val_population_all.csv
├── 2021_val_population_all.csv
├── 2016_test_features_all.csv
├── 2016_test_population_all.csv
└── 2021_test_population_all.csv

spatial_data/
├── walking_matrix.npy
├── transit_matrix.npy
├── proximity_matrix.npy
├── toronto_da.csv
└── da_boundary.shp
```

### Training

To train the model with default parameters:

```bash
python train.py --data_dir ./data --spatial_dir ./spatial_data --model_path ./model/multi_ethnic_spatial_moe_predictor.pt
```

For hyperparameter optimization:

```bash
python hyperopt.py --data_dir ./data --spatial_dir ./spatial_data --n_trials 50 --output_dir ./optuna_results
```

### Model Interpretation and Analysis

To analyze a trained model:

```bash
python model_interpretation.py --model_path ./model/multi_ethnic_spatial_moe_predictor.pt --data_dir ./data --spatial_dir ./spatial_data --output_dir ./interpretations --run_ethnicity_analysis
```

For ethnicity interaction analysis:

```bash
python ethnicity_interaction_analysis.py --interpretation_dir ./interpretations --output_dir ./ethnicity_visualizations --shapefile_path ./spatial_data/da_boundary.shp --all
```

## Model Architecture

The MESMoE model consists of four specialized experts:

1. **Colonization Expert**: Handles zero-to-nonzero population transitions, modeling the initial establishment of ethnic groups in previously uninhabited areas.

2. **Jump Process Expert**: Models discontinuous jumps in population values that cannot be well-captured by continuous diffusion processes, particularly for rapid growth scenarios.

3. **Decline Expert**: Specializes in modeling population decline scenarios, with particular focus on exodus events.

4. **CSRN Diffusion Solver**: Implements a physics-informed neural PDE solver for the Cultural-Spatial Resonance Network.

The expert routing is handled by a learnable network that maps input features and current populations to expert assignment probabilities.

## CUDA Extensions

The CUDA extensions accelerate computationally intensive operations:

- Gradient computation
- Laplacian computation
- PDE terms calculation
- Matrix operations for spatial attention


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank Statistics Canada for providing the census data used in this research.