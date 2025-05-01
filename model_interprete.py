import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import geopandas as gpd

from csrn.data_loader.spatial_data_loader import MultiEthnicDataHandler
from spatial_moe_model import MultiEthnicSpatialMoEPredictor
from model_visualization import create_model_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_interpretation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_trained_model(model_path, device='cuda'):
    """
    Load a trained model from disk by examining the actual state dict.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load model info
    model_info = torch.load(model_path, map_location=device)
    
    # Extract saved state dict
    state_dict = model_info['model_state_dict']
    ethnicity_names = model_info['ethnicity_names']
    
    # Extract model configuration
    model_config = model_info.get('model_config', {})
    
    # Determine exact dimensions from state dict
    input_dim = model_config.get('input_dim', 0)
    num_ethnicities = len(ethnicity_names)
    
    # Extract hidden dims directly from model config if available
    hidden_dims = model_config.get('hidden_dims', [])
    
    # Otherwise, try to infer from state dict
    if not hidden_dims:
        # Try to get first hidden dimension from feature embedding
        if 'feature_embedding.0.weight' in state_dict:
            feature_embed = state_dict['feature_embedding.0.weight']
            first_dim = feature_embed.shape[0]
            hidden_dims.append(first_dim)
        
        # Add remaining dims by examining linear layers
        layer_idx = 0
        while f'layers.{layer_idx}.weight' in state_dict:
            weight = state_dict[f'layers.{layer_idx}.weight']
            # Add output dimension of each linear layer
            hidden_dims.append(weight.shape[0])
            # Skip to next linear layer (4 entries per block: Linear, ReLU, BatchNorm, Dropout)
            layer_idx += 4
    
    # Get coordinate embedding dimension
    coord_embed_dim = model_config.get('coord_embed_dim', 32)
    if 'coord_embedding.coord_embed.0.weight' in state_dict:
        coord_embed = state_dict['coord_embedding.coord_embed.0.weight']
        coord_embed_dim = coord_embed.shape[0]
    
    logger.info(f"Determined model dimensions: input_dim={input_dim}, coord_embed_dim={coord_embed_dim}, hidden_dims={hidden_dims}")
    
    # Create model with EXACT same dimensions
    model = MultiEthnicSpatialMoEPredictor(
        input_dim=input_dim,
        num_ethnicities=num_ethnicities,
        ethnicity_names=ethnicity_names,
        hidden_dims=hidden_dims,
        coord_embed_dim=coord_embed_dim,
        num_attention_heads=model_config.get('num_attention_heads', 4),
        num_matrices=model_config.get('num_matrices', 3),
        dropout_rate=model_config.get('dropout_rate', 0.2),
        colonization_threshold=model_config.get('colonization_threshold', 0.1),
        colonization_expert_weight=model_config.get('colonization_expert_weight', 0.8),
        jump_threshold=model_config.get('jump_threshold', 100.0),
        jump_expert_weight=model_config.get('jump_expert_weight', 0.7),
        decline_threshold=model_config.get('decline_threshold', 50.0),
        decline_expert_weight=model_config.get('decline_expert_weight', 0.8),
        pde_expert_weight=model_config.get('pde_expert_weight', 0.6),
        log_space=model_config.get('log_space', False),
        skip_dim_adjustments=True,  # Important: Skip any dimension adjustments
        device=device
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully - predicts {len(ethnicity_names)} ethnicities: {ethnicity_names}")
    return model, model_info, ethnicity_names

def load_data(args):
    """
    Load the dataset using the MultiEthnicDataHandler.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: Data handler and data loaders
    """
    logger.info(f"Loading data from {args.data_dir}")
    
    # Load DA coordinates
    try:
        da_coordinates = pd.read_csv(args.da_coordinates_path)
        logger.info(f"Loaded DA coordinates from {args.da_coordinates_path}")
    except Exception as e:
        logger.warning(f"Error loading DA coordinates: {str(e)}")
        da_coordinates = None
    
    # Create data handler
    data_handler = MultiEthnicDataHandler(args.data_dir, args.spatial_dir, scaler_type=args.scaler)
    
    # Pass DA coordinates to the data handler if loaded successfully
    if da_coordinates is not None:
        data_handler.da_coordinates = da_coordinates
    
    # Load data
    data = data_handler.load_data()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_handler.create_dataloaders(
        data, batch_size=args.batch_size
    )
    
    # Get ethnicity names
    ethnicity_names = []
    if hasattr(train_loader.dataset, 'ethnicity_columns'):
        ethnicity_names = train_loader.dataset.ethnicity_columns
    else:
        # Try to get them from the columns in the population data
        if 'train_pop_2016' in data:
            ethnicity_names = [col for col in data['train_pop_2016'].columns if col != 'DAUID']
    
    logger.info(f"Loaded data with {len(ethnicity_names)} ethnicities: {ethnicity_names}")
    
    return data_handler, train_loader, val_loader, test_loader, ethnicity_names

def generate_interpretations(model, data_loader, ethnicity_names, output_dir, device='cuda'):
    """
    Generate and save comprehensive interpretations for the model on the given dataset.
    """
    logger.info("Generating model interpretations...")
    
    # Create output directories
    interpretation_dir = os.path.join(output_dir, "interpretations")
    os.makedirs(interpretation_dir, exist_ok=True)
    
    # Storage for batch-level interpretations
    all_interpretations = []
    all_predictions = []
    all_expert_weights = []
    all_base_populations = []
    all_dauid_list = []  # Store all DAUIDs here
    
    # Get predictions for selected batches with interpretations
    batch_count = len(data_loader)
    interpret_every = max(1, batch_count // 10)  # Interpret 10 batches evenly distributed
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move data to device
            features = batch['features'].to(device)
            population_2016 = batch['population_2016'].to(device)
            coordinates = batch['coordinates'].to(device)
            
            # Get spatial matrices if available
            walking_matrix = batch.get('walking_matrix')
            transit_matrix = batch.get('transit_matrix')
            proximity_matrix = batch.get('proximity_matrix')
            
            # Extract DAUIDs if available
            batch_dauid_list = batch['dauid'].tolist() if 'dauid' in batch else None
            
            if walking_matrix is not None:
                walking_matrix = walking_matrix.to(device)
            if transit_matrix is not None:
                transit_matrix = transit_matrix.to(device)
            if proximity_matrix is not None:
                proximity_matrix = proximity_matrix.to(device)
            
            # Decide whether to store interpretations for this batch
            store_interpretations = (batch_idx % interpret_every == 0)
            
            # Forward pass with DAUIDs
            outputs, expert_weights, _ = model(
                features, population_2016, coordinates, 
                walking_matrix, transit_matrix, proximity_matrix,
                store_interpretations=store_interpretations,
                dauid_list=batch_dauid_list  # Pass DAUIDs to the model
            )
            
            # Store predictions for all batches
            all_predictions.append(outputs.cpu().numpy())
            all_base_populations.append(population_2016.cpu().numpy())
            all_expert_weights.append(expert_weights.cpu().numpy())
            
            # Store DAUIDs if available - for this batch only
            if batch_dauid_list:
                all_dauid_list.append(np.array(batch_dauid_list))
            
            # Store interpretations for selected batches
            if store_interpretations:
                # Get interpretability data
                interpretation_data = model.get_interpretability_data()
                
                # Add DAUID list to interpretation data for traceability
                interpretation_data['dauid_list'] = batch_dauid_list
                
                # Save the interpretation to JSON
                interpretation_path = os.path.join(interpretation_dir, f"interpretation_batch_{batch_idx}.json")
                
                # Convert numpy arrays to lists for JSON serialization
                def convert_numpy_to_python(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
                    elif isinstance(obj, list) or isinstance(obj, tuple):
                        return [convert_numpy_to_python(item) for item in obj]
                    else:
                        return obj
                
                # Apply the conversion function to the entire interpretation data
                serialized_data = convert_numpy_to_python(interpretation_data)
                
                with open(interpretation_path, 'w') as f:
                    json.dump(serialized_data, f, indent=2)
                
                all_interpretations.append({
                    'batch_idx': batch_idx,
                    'interpretation_path': interpretation_path,
                    'data': interpretation_data
                })
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_base_populations = np.concatenate(all_base_populations, axis=0)
    all_expert_weights = np.concatenate(all_expert_weights, axis=0)
    
    # Concatenate all DAUIDs if available
    if all_dauid_list:
        all_dauid = np.concatenate(all_dauid_list)
    else:
        all_dauid = None
    
    # Create a DataFrame with predictions
    data_dict = {}
    
    # Add DAUIDs if available
    if all_dauid is not None:
        data_dict['DAUID'] = all_dauid
    
    # Add base populations (2016) for each ethnicity
    for i, eth in enumerate(ethnicity_names):
        data_dict[f"{eth}_2016"] = all_base_populations[:, i]
    
    # Add predicted populations for each ethnicity
    for i, eth in enumerate(ethnicity_names):
        data_dict[f"{eth}_predicted"] = all_predictions[:, i]
    
    # Add expert weights for each ethnicity
    for i, eth in enumerate(ethnicity_names):
        for j, expert in enumerate(['colonization', 'jump', 'decline', 'pde']):
            data_dict[f"{eth}_{expert}_weight"] = all_expert_weights[:, i, j]
    
    # Create the DataFrame
    predictions_df = pd.DataFrame(data_dict)
    
    # Save the predictions
    predictions_path = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    return predictions_df, all_interpretations, interpretation_dir

# ------ Ethnicity Interaction Analysis Functions ------

def load_interpretation_data(interpretation_dir):
    """
    Load interpretation data from JSON files in the specified directory
    
    Args:
        interpretation_dir (str): Directory containing interpretation JSON files
        
    Returns:
        list: List of interpretation data dictionaries
    """
    interpretations = []
    
    # Find all interpretation JSON files
    json_files = [f for f in os.listdir(interpretation_dir) 
                 if f.endswith('.json') and 'interpretation' in f]
    
    if not json_files:
        logger.warning(f"No interpretation files found in {interpretation_dir}")
        return interpretations
    
    logger.info(f"Found {len(json_files)} interpretation files")
    
    # Load each file
    for json_file in json_files:
        file_path = os.path.join(interpretation_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            interpretations.append(data)
            logger.info(f"Loaded interpretation data from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    return interpretations

def load_da_coordinates(da_coordinates_path):
    """
    Load DA coordinates from CSV file
    
    Args:
        da_coordinates_path (str): Path to the CSV file with DA coordinates
        
    Returns:
        dict: Mapping from DAUID to (x,y) coordinates
    """
    if not os.path.exists(da_coordinates_path):
        logger.warning(f"DA coordinates file not found: {da_coordinates_path}")
        return {}
    
    try:
        da_coords_df = pd.read_csv(da_coordinates_path)
        logger.info(f"Loaded DA coordinates from {da_coordinates_path} with {len(da_coords_df)} rows")
        
        # Create mapping from DAUID to (X, Y) coordinates
        dauid_to_coords = {}
        for _, row in da_coords_df.iterrows():
            if 'DAUID' in row and 'X_axis' in row and 'Y_axis' in row:
                dauid_to_coords[str(int(float(row['DAUID'])))] = (row['X_axis'], row['Y_axis'])
            elif 'DAUID' in row and 'X' in row and 'Y' in row:
                dauid_to_coords[str(row['DAUID'])] = (row['X'], row['Y'])
        
        logger.info(f"Created coordinate mapping for {len(dauid_to_coords)} DAUIDs")
        return dauid_to_coords
    except Exception as e:
        logger.error(f"Error loading DA coordinates: {str(e)}")
        return {}

def prepare_interaction_data(interpretations, ethnicity_names):
    """
    Process and prepare ethnicity interaction data from multiple interpretations
    
    Args:
        interpretations (list): List of interpretation data dictionaries
        ethnicity_names (list): List of ethnicity names
        
    Returns:
        dict: Processed interaction data
    """
    # Initialize processed data
    processed_data = {
        'interaction_weights': None,
        'interaction_effect': None,
        'ethnicity_interactions_by_dauid': {},
        'interaction_effect_by_dauid': {},
        'dauid_list': []
    }
    
    # Extract ethnicity names from the first interpretation if available
    num_ethnicities = None
    for interp in interpretations:
        if 'model_architecture' in interp and 'num_ethnicities' in interp['model_architecture']:
            num_ethnicities = interp['model_architecture']['num_ethnicities']
            logger.info(f"Found {num_ethnicities} ethnicities in model architecture")
            break
    
    if num_ethnicities is None and ethnicity_names:
        num_ethnicities = len(ethnicity_names)
        logger.info(f"Using {num_ethnicities} ethnicities from data")
    
    # Process each interpretation
    for interp in interpretations:
        # Get interaction weights and effects
        interaction_weights = interp.get('interaction_weights')
        interaction_effect = interp.get('interaction_effect')
        
        # Get DAUID list
        if 'dauid_list' in interp:
            processed_data['dauid_list'].extend(interp['dauid_list'])
        
        # Get by-DAUID interaction data
        if 'ethnicity_interactions_by_dauid' in interp:
            for dauid, interactions in interp['ethnicity_interactions_by_dauid'].items():
                processed_data['ethnicity_interactions_by_dauid'][dauid] = interactions
        
        # Get by-DAUID effect data
        if 'interaction_effect_by_dauid' in interp:
            for dauid, effects in interp['interaction_effect_by_dauid'].items():
                processed_data['interaction_effect_by_dauid'][dauid] = effects
        
        # Store valid interaction data (just use the first valid one we find)
        if processed_data['interaction_weights'] is None and interaction_weights is not None:
            # Check if it's a batch
            if isinstance(interaction_weights, list) and len(interaction_weights) > 0:
                # Create a properly sized interaction matrix
                if num_ethnicities:
                    # Create a proper average interaction matrix for all ethnicities
                    batches = []
                    for batch in interaction_weights:
                        if isinstance(batch, list) and len(batch) == num_ethnicities:
                            # Valid batch - convert to numpy array
                            batch_array = np.array(batch)
                            if batch_array.shape == (num_ethnicities, num_ethnicities):
                                batches.append(batch_array)
                    
                    if batches:
                        # Average across valid batches
                        processed_data['interaction_weights'] = np.stack(batches)
                        logger.info(f"Processed interaction weights with shape {processed_data['interaction_weights'].shape}")
                else:
                    # Just use the first batch
                    processed_data['interaction_weights'] = np.array([np.array(interaction_weights[0])])
                    logger.info(f"Using first batch of interaction weights")
            else:
                # Single batch or direct matrix
                processed_data['interaction_weights'] = np.array([np.array(interaction_weights)])
                logger.info(f"Using direct interaction weights")
        
        # Store valid effect data
        if processed_data['interaction_effect'] is None and interaction_effect is not None:
            if isinstance(interaction_effect, list) and len(interaction_effect) > 0:
                processed_data['interaction_effect'] = np.array(interaction_effect)
            else:
                processed_data['interaction_effect'] = np.array([interaction_effect])
    
    # Deduplicate DAUID list
    processed_data['dauid_list'] = list(set(processed_data['dauid_list']))
    
    logger.info(f"Processed interaction data from {len(interpretations)} interpretations")
    logger.info(f"Found data for {len(processed_data['ethnicity_interactions_by_dauid'])} DAUIDs")
    
    return processed_data


def run_ethnicity_analysis(args, ethnicity_names, interpretation_dir):
    """
    Run the ethnicity interaction analysis
    
    Args:
        args: Command line arguments
        ethnicity_names: List of ethnicity names
        interpretation_dir: Directory containing interpretation JSON files
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=== Starting Ethnicity Interaction Analysis ===")
    
    # Import the EthnicityInteractionAnalyzer after checking for various file paths
    try:
        from ethnicity_interaction_visualization import EthnicityInteractionAnalyzer
        logger.info("Imported EthnicityInteractionAnalyzer from ethnicity_interaction_visualization")
    except ImportError:
        try:
            from ethnicity_interaction_visualization import EthnicityInteractionAnalyzer
            logger.info("Imported EthnicityInteractionAnalyzer from ethnicity_interaction_visualization_fix")
        except ImportError:
            try:
                # Try to find it in the same directory as the original file
                import sys
                sys.path.append(os.path.dirname(os.path.realpath(__file__)))
                from ethnicity_interaction_visualization import EthnicityInteractionAnalyzer
                logger.info("Imported EthnicityInteractionAnalyzer using absolute path")
            except ImportError:
                logger.error("Could not import EthnicityInteractionAnalyzer. Make sure the file exists in the current directory.")
                return False
    
    # Create output directory for ethnicity visualizations
    ethnicity_viz_dir = os.path.join(args.output_dir, "ethnicity_visualizations")
    os.makedirs(ethnicity_viz_dir, exist_ok=True)
    
    # Load interpretation data
    interpretations = load_interpretation_data(interpretation_dir)
    
    if not interpretations:
        logger.error("No interpretation data found. Exiting ethnicity analysis.")
        return False
    
    # Load DA coordinates if provided
    dauid_to_coords = {}
    if args.da_coordinates_path:
        dauid_to_coords = load_da_coordinates(args.da_coordinates_path)
        logger.info(f"Loaded {len(dauid_to_coords)} DA coordinates for ethnicity analysis")
    
    # Process and prepare interaction data
    processed_data = prepare_interaction_data(interpretations, ethnicity_names)
    
    # Use provided ethnicity names if available
    if ethnicity_names:
        logger.info(f"Using ethnicity names: {ethnicity_names}")
    else:
        # If still no ethnicity names, create default ones
        num_ethnicities = 5  # Default if we can't determine
        
        # Try to infer number of ethnicities from interaction data
        if processed_data['interaction_weights'] is not None:
            weights_shape = processed_data['interaction_weights'].shape
            if len(weights_shape) >= 3:
                num_ethnicities = weights_shape[1]  # [batch, ethnicity, ethnicity]
            elif len(weights_shape) == 2:
                num_ethnicities = weights_shape[0]  # [ethnicity, ethnicity]
        
        ethnicity_names = [f"Ethnicity {i}" for i in range(num_ethnicities)]
        logger.info(f"Created default ethnicity names for {num_ethnicities} ethnicities")
    
    # Create custom color palette for ethnicities
    colors = plt.cm.tab10.colors
    if args.colormap:
        try:
            colors = plt.cm.get_cmap(args.colormap).colors
            logger.info(f"Using custom colormap: {args.colormap}")
        except:
            logger.warning(f"Invalid colormap: {args.colormap}, using default")
    
    # Initialize the ethnicity interaction analyzer
    analyzer = EthnicityInteractionAnalyzer(
        ethnicity_names=ethnicity_names,
        output_dir=ethnicity_viz_dir,
        color_palette=colors,
        significant_threshold=args.threshold
    )
    
    # Set DAUID coordinates
    if dauid_to_coords:
        analyzer.set_dauid_coordinates(dauid_to_coords)
        logger.info(f"Set {len(dauid_to_coords)} DA coordinates in the analyzer")
    else:
        logger.warning("No DA coordinates available for ethnicity analysis")
    
    # Extract interaction data from prepared data
    logger.info("Extracting interaction data")
    success = analyzer.extract_interaction_data(processed_data)
    
    if not success:
        logger.error("Failed to extract interaction data. Exiting ethnicity analysis.")
        return False
    
    # Load shapefile if provided
    gdf = None
    if args.shapefile_path and os.path.exists(args.shapefile_path):
        try:
            gdf = gpd.read_file(args.shapefile_path)
            logger.info(f"Loaded shapefile with {len(gdf)} geometries")
        except Exception as e:
            logger.error(f"Error loading shapefile: {str(e)}")
    
    # Generate appropriate visualizations based on flags if specified
    if hasattr(args, 'network') and args.network:
        logger.info("Generating network visualizations")
        analyzer.visualize_interaction_network()
    
    if hasattr(args, 'flow') and args.flow:
        logger.info("Generating flow diagram visualizations")
        analyzer.visualize_flow_diagrams()
    
    if hasattr(args, 'spatial') and args.spatial and (dauid_to_coords or gdf is not None):
        logger.info("Generating spatial visualizations")
        analyzer.visualize_spatial_interactions(gdf, args.shapefile_path)
    
    if hasattr(args, 'temporal') and args.temporal:
        logger.info("Generating temporal cascade visualizations")
        analyzer.visualize_temporal_cascade(time_steps=args.time_steps)
    
    if hasattr(args, 'metrics') and args.metrics:
        logger.info("Generating interaction metrics visualizations")
        analyzer.visualize_interaction_metrics()
    
    # If no specific flags are set, run all visualizations
    if (not hasattr(args, 'network') or 
        (not args.network and not args.flow and not args.spatial 
         and not args.temporal and not args.metrics) or 
        (hasattr(args, 'all') and args.all)):
        logger.info("Generating all ethnicity visualizations")
        analyzer.run_all_visualizations(gdf, args.shapefile_path)
    
    logger.info("=== Ethnicity Interaction Analysis Completed ===")
    return True

def main(args):
    """
    Main function to load a trained model, generate interpretations,
    conduct ethnicity analysis, and create visualizations.
    
    Args:
        args: Command line arguments
    """
    logger.info("=== Starting Model Interpretation and Analysis ===")
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    data_handler, train_loader, val_loader, test_loader, ethnicity_names = load_data(args)
    
    # Load trained model
    model, model_info, model_ethnicity_names = load_trained_model(args.model_path, device)
    
    # Check if ethnicity names match between model and data
    if model_ethnicity_names and set(model_ethnicity_names) != set(ethnicity_names):
        logger.warning("Ethnicity names in model do not match data. Using model's ethnicity names.")
        ethnicity_names = model_ethnicity_names
    
    # Decide which data loader to use for interpretations
    data_loader = None
    if args.dataset == 'train':
        data_loader = train_loader
        logger.info("Using training dataset for interpretations")
    elif args.dataset == 'validation':
        data_loader = val_loader
        logger.info("Using validation dataset for interpretations")
    elif args.dataset == 'test':
        data_loader = test_loader
        logger.info("Using test dataset for interpretations")
    else:
        logger.error(f"Unknown dataset type: {args.dataset}")
        return
    
    # Generate interpretations
    predictions_df, interpretations, interpretation_dir = generate_interpretations(
        model=model, 
        data_loader=data_loader, 
        ethnicity_names=ethnicity_names,
        output_dir=args.output_dir,
        device=device
    )
    
    # Run ethnicity interaction analysis
    if args.run_ethnicity_analysis:
        # Add specific visualization flags for ethnicity analysis
        # This ensures we pass the visualization control flags to the ethnicity analyzer
        for flag in ['all', 'network', 'flow', 'spatial', 'temporal', 'metrics']:
            if not hasattr(args, flag):
                setattr(args, flag, False)
        
        # Set default to all visualizations if none specified
        if not any([args.all, args.network, args.flow, args.spatial, args.temporal, args.metrics]):
            args.all = True
            
        ethnicity_success = run_ethnicity_analysis(
            args=args,
            ethnicity_names=ethnicity_names,
            interpretation_dir=interpretation_dir
        )
        
        if ethnicity_success:
            logger.info("Successfully completed ethnicity interaction analysis")
        else:
            logger.warning("Ethnicity interaction analysis had issues or was incomplete")
    
    # Generate visualizations from interpretations if requested
    if args.create_visualizations:
        logger.info("Generating model visualizations...")
        
        # Path to the predictions CSV
        predictions_path = os.path.join(args.output_dir, "predictions.csv")
        
        # Determine the shapefile path
        shapefile_path = None
        if args.shapefile_path:
            shapefile_path = args.shapefile_path
        elif args.spatial_dir:
            potential_shapefile = os.path.join(args.spatial_dir, "da_boundary.shp")
            if os.path.exists(potential_shapefile):
                shapefile_path = potential_shapefile
                logger.info(f"Using shapefile found at: {shapefile_path}")
        
        # Call the visualization function
        success = create_model_visualizations(
            output_dir=args.output_dir,
            da_coordinates_path=args.da_coordinates_path,
            ethnicity_names=ethnicity_names,
            predictions_path=predictions_path,
            shapefile_path=shapefile_path
        )
        
        if success:
            logger.info("Successfully created model visualizations")
        else:
            logger.warning("Failed to create model visualizations")
    
    logger.info("=== Model Interpretation and Analysis Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Interpretation and Ethnicity Interaction Analysis")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="./interpretations",
                        help="Directory to save interpretations and visualizations")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--spatial_dir", type=str, default="./spatial_data",
                        help="Directory containing spatial data files")
    parser.add_argument("--da_coordinates_path", type=str, default="./data/toronto_da.csv",
                        help="Path to the CSV file containing DA coordinates")
    parser.add_argument("--shapefile_path", type=str, default="./spatial_data/da_boundary.shp",
                        help="Path to shapefile with DA boundaries for spatial visualization")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use for feature normalization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    
    # Interpretation parameters
    parser.add_argument("--dataset", type=str, default="train", choices=["train", "validation", "test"],
                        help="Which dataset to use for interpretations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation")
    
    # Visualization flags
    parser.add_argument("--create_visualizations", action="store_true",
                        help="Generate model visualizations from interpretations")
    
    # Ethnicity analysis parameters
    parser.add_argument("--run_ethnicity_analysis", action="store_true",
                        help="Run ethnicity interaction analysis")
    parser.add_argument("--colormap", type=str, default=None,
                        help="Matplotlib colormap name for ethnicity colors")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold for significant interactions")
    parser.add_argument("--time_steps", type=int, default=5,
                        help="Number of time steps for temporal cascade visualization")
    
    # Visualization type flags specifically for ethnicity analysis
    parser.add_argument("--all", action="store_true",
                        help="Generate all ethnicity visualization types")
    parser.add_argument("--network", action="store_true",
                        help="Generate network visualizations")
    parser.add_argument("--flow", action="store_true",
                        help="Generate flow diagram visualizations")
    parser.add_argument("--spatial", action="store_true",
                        help="Generate spatial visualizations")
    parser.add_argument("--temporal", action="store_true",
                        help="Generate temporal cascade visualizations")
    parser.add_argument("--metrics", action="store_true",
                        help="Generate interaction metrics visualizations")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)