import os
import logging
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse

from csrn.data_loader.spatial_data_loader import MultiEthnicDataHandler
# Add these imports at the beginning of your file
from esda.moran import Moran
from pysal.lib import weights
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# Add these functions to your file
def calculate_spatial_autocorrelation(gdf, value_column):
    """
    Calculate Moran's I spatial autocorrelation statistic
    
    Args:
        gdf (GeoDataFrame): GeoPandas dataframe with geometries and values
        value_column (str): Column name containing values to analyze
    
    Returns:
        tuple: (Moran's I statistic, p-value)
    """
    # Drop rows with NaN values
    valid_gdf = gdf.dropna(subset=[value_column])
    
    if len(valid_gdf) < 10:  # Need enough data points for meaningful statistics
        logger.warning(f"Not enough valid data for {value_column} to calculate Moran's I")
        return None, None
    
    try:
        # Create spatial weights matrix using queen contiguity
        w = weights.Queen.from_dataframe(valid_gdf)
        w.transform = 'r'  # Row-standardize the weights
        
        # Calculate Moran's I
        moran = Moran(valid_gdf[value_column], w)
        
        logger.info(f"Moran's I for {value_column}: {moran.I:.4f} (p-value: {moran.p_value:.6f})")
        return moran.I, moran.p_value
    except Exception as e:
        logger.error(f"Error calculating Moran's I for {value_column}: {str(e)}")
        return None, None

def analyze_expert_weights_spatial_patterns(gdf, dauid_column, interpretations, ethnicity_name, output_dir):
    """
    Analyze spatial patterns of expert weights for a specific ethnicity
    
    Args:
        gdf (GeoDataFrame): GeoPandas dataframe with geometries
        dauid_column (str): Name of the column with DAUIDs
        interpretations (list): List of interpretation data dictionaries
        ethnicity_name (str): Name or index of the ethnicity to analyze
        output_dir (str): Directory to save results
        
    Returns:
        dict: Spatial statistics for each expert
    """
    # Create a directory for spatial statistics
    spatial_stats_dir = os.path.join(output_dir, "spatial_statistics")
    os.makedirs(spatial_stats_dir, exist_ok=True)
    
    # Extract expert weights by DAUID
    expert_weights_by_dauid = {}
    expert_types = ['colonization', 'jump', 'decline', 'pde']
    
    # First, determine if ethnicity_name is an index or a name
    ethnicity_idx = None
    ethnicity_label = None
    
    # Try to convert the ethnicity_name to an integer index
    try:
        ethnicity_idx = int(ethnicity_name)
        ethnicity_label = f"Ethnicity {ethnicity_idx}"
        logger.info(f"Using ethnicity at index {ethnicity_idx}")
    except ValueError:
        # It's a string name, so we need to look it up
        ethnicity_label = ethnicity_name
        # We'll keep ethnicity_idx as None, which will be handled below
        
    # If no ethnicity index yet, try to find the index from interpretations
    if ethnicity_idx is None:
        for interp in interpretations:
            if 'ethnicity_names' in interp and ethnicity_name in interp['ethnicity_names']:
                ethnicity_idx = interp['ethnicity_names'].index(ethnicity_name)
                logger.info(f"Found ethnicity {ethnicity_name} at index {ethnicity_idx}")
                break
    
    # If still no index, try to extract from name pattern like "Ethnicity 0"
    if ethnicity_idx is None and ethnicity_name.startswith("Ethnicity "):
        try:
            ethnicity_idx = int(ethnicity_name.split("Ethnicity ")[1])
            logger.info(f"Extracted index {ethnicity_idx} from {ethnicity_name}")
        except (ValueError, IndexError):
            pass
            
    # If we still don't have an index, print available ethnicities and use index 0
    if ethnicity_idx is None:
        # Print available ethnicity names for debugging
        for interp in interpretations:
            if 'ethnicity_names' in interp:
                logger.info(f"Available ethnicities: {interp['ethnicity_names']}")
                break
                
        # Default to index 0 and log a warning
        ethnicity_idx = 0
        logger.warning(f"Could not find ethnicity {ethnicity_name}, defaulting to index {ethnicity_idx}")
    
    # Extract expert weights for each DAUID
    for interp in interpretations:
        if 'expert_weights' not in interp or 'dauid_list' not in interp:
            continue
        
        expert_weights = interp['expert_weights']
        dauid_list = interp['dauid_list']
        
        # Make sure the data has the right structure
        if isinstance(expert_weights, list) and len(expert_weights) == len(dauid_list):
            for i, dauid in enumerate(dauid_list):
                if i < len(expert_weights):
                    # Try to extract weights for this ethnicity by index
                    if len(expert_weights[i]) > ethnicity_idx:
                        weights_for_ethnicity = expert_weights[i][ethnicity_idx]
                        
                        if weights_for_ethnicity is not None and len(weights_for_ethnicity) >= len(expert_types):
                            expert_weights_by_dauid[str(dauid)] = {
                                expert_type: weights_for_ethnicity[j] 
                                for j, expert_type in enumerate(expert_types)
                            }
    
    # Number of DAUIDs with expert weights
    logger.info(f"Extracted expert weights for {len(expert_weights_by_dauid)} DAUIDs for ethnicity index {ethnicity_idx}")
    
    if not expert_weights_by_dauid:
        logger.warning(f"No expert weights found for ethnicity index {ethnicity_idx}")
        return {}
    
    # Add expert weights to the GeoDataFrame and calculate spatial statistics
    results = {}
    
    for expert_type in expert_types:
        weight_column = f"ethnicity_{ethnicity_idx}_{expert_type}_weight"
        
        # Add expert weights to the GeoDataFrame
        gdf[weight_column] = gdf[dauid_column].apply(
            lambda dauid: expert_weights_by_dauid.get(str(dauid), {}).get(expert_type, np.nan)
        )
        
        # Calculate Moran's I (spatial autocorrelation)
        moran_i, p_value = calculate_spatial_autocorrelation(gdf, weight_column)
        
        valid_data = gdf.dropna(subset=[weight_column])
        
        results[expert_type] = {
            'moran_i': moran_i,
            'p_value': p_value,
            'mean': valid_data[weight_column].mean() if not valid_data.empty else None,
            'max': valid_data[weight_column].max() if not valid_data.empty else None,
            'coverage': len(valid_data) / len(gdf) if len(gdf) > 0 else 0
        }
        
        # Create a map visualization of the expert weights
        if not valid_data.empty:
            fig, ax = plt.subplots(figsize=(15, 12))
            
            # Plot the base map
            gdf.plot(
                ax=ax,
                color='lightgrey',
                edgecolor='darkgrey',
                linewidth=0.5,
                alpha=0.3
            )
            
            # Plot the expert weights
            vmin = valid_data[weight_column].min()
            vmax = valid_data[weight_column].max()
            
            valid_data.plot(
                column=weight_column,
                ax=ax,
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                legend=True,
                edgecolor='black',
                linewidth=0.3,
                alpha=0.8
            )
            
            ax.set_title(f"{expert_type.capitalize()} Expert Weight for {ethnicity_label}")
            ax.set_axis_off()
            
            # Add Moran's I information if available
            if moran_i is not None and p_value is not None:
                ax.text(
                    0.05, 0.05, 
                    f"Moran's I: {moran_i:.4f} (p-value: {p_value:.6f})\n"
                    f"Spatial autocorrelation: {'Strong' if moran_i > 0.5 else 'Moderate' if moran_i > 0.3 else 'Weak'}",
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8)
                )
            
            # Add information about missing data
            missing_count = len(gdf) - len(valid_data)
            if missing_count > 0:
                ax.text(
                    0.05, 0.95, 
                    f"Missing data: {missing_count}/{len(gdf)} areas", 
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top'
                )
            
            # Save the figure
            save_path = os.path.join(spatial_stats_dir, f"{expert_type}_ethnicity_{ethnicity_idx}_weight_map.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logger.info(f"Created {expert_type} expert weight map for ethnicity index {ethnicity_idx}")
    
    # Save the results to a JSON file
    results_path = os.path.join(spatial_stats_dir, f"spatial_statistics_ethnicity_{ethnicity_idx}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create a summary table
    summary_df = pd.DataFrame({
        'Expert': expert_types,
        'Mean Weight': [results[et]['mean'] for et in expert_types],
        'Max Weight': [results[et]['max'] for et in expert_types],
        'Moran\'s I': [results[et]['moran_i'] for et in expert_types],
        'P-value': [results[et]['p_value'] for et in expert_types],
        'Coverage': [results[et]['coverage'] for et in expert_types]
    })
    
    summary_path = os.path.join(spatial_stats_dir, f"spatial_statistics_summary_ethnicity_{ethnicity_idx}.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Saved spatial statistics for ethnicity index {ethnicity_idx} to {results_path}")
    
    return results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ethnicity_interaction_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_spatial_autocorrelation(gdf, value_column):
    """
    Calculate Moran's I spatial autocorrelation statistic with improved handling of disconnected components
    
    Args:
        gdf (GeoDataFrame): GeoPandas dataframe with geometries and values
        value_column (str): Column name containing values to analyze
    
    Returns:
        tuple: (Moran's I statistic, p-value)
    """
    # Drop rows with NaN values
    valid_gdf = gdf.dropna(subset=[value_column])
    
    if len(valid_gdf) < 10:  # Need enough data points for meaningful statistics
        logger.warning(f"Not enough valid data for {value_column} to calculate Moran's I")
        return None, None
    
    try:
        # Silence warnings about islands temporarily
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Create spatial weights matrix using queen contiguity
            w = weights.Queen.from_dataframe(valid_gdf, use_index=True)
            
            # Check for islands and disconnected components
            islands = w.islands
            n_components = w.n_components
            
            if islands:
                logger.info(f"Found {len(islands)} islands in spatial weights matrix. "
                           f"There are {n_components} disconnected components.")
                
                # Option 1: Remove islands for better analysis
                if len(islands) < len(valid_gdf) * 0.5:  # If less than 50% are islands
                    # Keep only non-island indices
                    non_islands = [i for i in range(len(valid_gdf)) if i not in islands]
                    if len(non_islands) > 10:  # Still need enough data points
                        valid_gdf = valid_gdf.iloc[non_islands]
                        # Recreate weights matrix
                        w = weights.Queen.from_dataframe(valid_gdf, use_index=True)
                        logger.info(f"Removed {len(islands)} islands, continuing with {len(valid_gdf)} areas")
                
                # Option 2: Add small distance connections
                else:
                    # If too many islands, try distance-based weights instead
                    try:
                        # Get centroids for distance calculation
                        centroids = valid_gdf.geometry.centroid
                        # Create a distance-based weights matrix
                        w_dist = weights.DistanceBand.from_dataframe(valid_gdf, threshold=0.1, binary=False)
                        
                        # If this creates a better connected graph, use it
                        if w_dist.n_components < n_components:
                            w = w_dist
                            logger.info(f"Switched to distance-based weights, reducing disconnected "
                                       f"components from {n_components} to {w_dist.n_components}")
                    except Exception as e:
                        logger.warning(f"Could not create distance-based weights: {str(e)}")
            
            # Row-standardize the weights
            w.transform = 'r'
            
            # Calculate Moran's I
            moran = Moran(valid_gdf[value_column], w)
            
            # Check if p_value is available as an attribute
            # Different versions of esda might store it differently
            p_value = None
            if hasattr(moran, 'p_value'):
                p_value = moran.p_value
            elif hasattr(moran, 'p_sim'):
                p_value = moran.p_sim
            elif hasattr(moran, 'p'):
                p_value = moran.p
            else:
                # Try to calculate p-value manually if available
                try:
                    p_value = moran.significance_filter(0.05)  # Using 0.05 as default
                except:
                    logger.warning(f"Could not determine p-value for Moran's I calculation on {value_column}")
                    p_value = None
            
            logger.info(f"Moran's I for {value_column}: {moran.I:.4f} (p-value: {p_value if p_value is not None else 'N/A'})")
            return moran.I, p_value
    except Exception as e:
        logger.error(f"Error calculating Moran's I for {value_column}: {str(e)}")
        return None, None


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

def prepare_interaction_data(interpretations):
    """
    Process and prepare ethnicity interaction data from multiple interpretations
    
    Args:
        interpretations (list): List of interpretation data dictionaries
        
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
    ethnicity_names = None
    num_ethnicities = None
    for interp in interpretations:
        if 'model_architecture' in interp and 'num_ethnicities' in interp['model_architecture']:
            num_ethnicities = interp['model_architecture']['num_ethnicities']
            logger.info(f"Found {num_ethnicities} ethnicities in model architecture")
            break
    
    data_handler, train_loader, val_loader, test_loader, ethnicity_names = load_data(args)
    
    ethnicity_names = ethnicity_names
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
    
    return processed_data, ethnicity_names

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

def main(args):
    """
    Main function to generate ethnicity interaction visualizations and spatial statistics
    
    Args:
        args: Command line arguments
    """
    logger.info("=== Starting Ethnicity Interaction Analysis ===")
    logger.info(f"Arguments: {args}")
    
    # Import the EthnicityInteractionAnalyzer after checking for the file
    try:
        from ethnicity_interaction_visualization import EthnicityInteractionAnalyzer
        logger.info("Imported EthnicityInteractionAnalyzer from ethnicity_interaction_visualization_fix")
    except ImportError:
        try:
            from ethnicity_interaction_visualization import EthnicityInteractionAnalyzer
            logger.info("Imported EthnicityInteractionAnalyzer from ethnicity_interaction_visualization")
        except ImportError:
            logger.error("Could not import EthnicityInteractionAnalyzer. Make sure the file exists in the current directory.")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load interpretation data
    interpretation_dir = args.interpretation_dir
    interpretations = load_interpretation_data(interpretation_dir)
    
    if not interpretations:
        logger.error("No interpretation data found. Exiting.")
        return
    
    # Load DA coordinates if provided
    dauid_to_coords = {}
    if args.da_coordinates_path:
        dauid_to_coords = load_da_coordinates(args.da_coordinates_path)
    
    # Process and prepare interaction data (skip loading ethnicity_names for now)
    # We'll use dummy ethnicity names based on indices instead
    processed_data, _ = prepare_interaction_data(interpretations)
    
    # Determine number of ethnicities from interaction weights
    num_ethnicities = 5  # Default
    if processed_data['interaction_weights'] is not None:
        weights_shape = processed_data['interaction_weights'].shape
        if len(weights_shape) >= 3:
            num_ethnicities = weights_shape[1]  # [batch, ethnicity, ethnicity]
        elif len(weights_shape) == 2:
            num_ethnicities = weights_shape[0]  # [ethnicity, ethnicity]
    
    logger.info(f"Determined there are {num_ethnicities} ethnicities based on interaction weights")
    
    # If user provided ethnicity names/indices, use those
    ethnicity_indices = []
    if args.ethnicity_names:
        # Convert any string indices to integers
        for name in args.ethnicity_names:
            try:
                # If it can be converted to an integer, it's an index
                idx = int(name)
                if 0 <= idx < num_ethnicities:
                    ethnicity_indices.append(idx)
                else:
                    logger.warning(f"Ethnicity index {idx} is out of range (0-{num_ethnicities-1}), skipping")
            except ValueError:
                # If it's not a number, try to find its index or use a default
                eth_idx = 0  # Default to first ethnicity
                found = False
                
                # Try to find the ethnicity name in interpretations
                for interp in interpretations:
                    if 'ethnicity_names' in interp and name in interp['ethnicity_names']:
                        eth_idx = interp['ethnicity_names'].index(name)
                        found = True
                        break
                
                if found:
                    ethnicity_indices.append(eth_idx)
                    logger.info(f"Found ethnicity '{name}' at index {eth_idx}")
                else:
                    logger.warning(f"Could not find ethnicity '{name}', using default index 0")
                    ethnicity_indices.append(0)
    else:
        # If no ethnicities specified, use all indices
        ethnicity_indices = list(range(num_ethnicities))
        logger.info(f"Using all ethnicity indices from 0 to {num_ethnicities-1}")
    
    # Create display names for visualization based on indices
    ethnicity_names = [f"Ethnicity {idx}" for idx in ethnicity_indices]
    logger.info(f"Using ethnicity names for visualization: {ethnicity_names}")
    
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
        output_dir=args.output_dir,
        color_palette=colors,
        significant_threshold=args.threshold
    )
    
    # Set DAUID coordinates
    if dauid_to_coords:
        analyzer.set_dauid_coordinates(dauid_to_coords)
    
    # Extract interaction data from prepared data
    logger.info("Extracting interaction data")
    success = analyzer.extract_interaction_data(processed_data)
    
    if not success:
        logger.error("Failed to extract interaction data. Exiting.")
        return
    
    # Load shapefile if provided
    gdf = None
    if args.shapefile_path and os.path.exists(args.shapefile_path):
        try:
            gdf = gpd.read_file(args.shapefile_path)
            logger.info(f"Loaded shapefile with {len(gdf)} geometries")
            
            # Find DAUID column in shapefile
            dauid_column = None
            for column in gdf.columns:
                if column.upper() == 'DAUID' or 'DAUID' in column.upper():
                    dauid_column = column
                    break
            
            if dauid_column is None:
                # Try common ID field names in shapefiles
                for column in ['ID', 'GEOID', 'AREA_ID', 'DA_ID', 'DISSEMINATION_AREA_ID']:
                    if column in gdf.columns:
                        dauid_column = column
                        logger.info(f"Using {dauid_column} as the DA identifier")
                        break
            
            if dauid_column is not None:
                # Ensure DAUID is string type for consistent joining
                gdf[dauid_column] = gdf[dauid_column].astype(str)
                
                # Perform spatial analysis for the specified ethnicities
                spatial_results = {}
                
                # Use ethnicity indices directly for spatial analysis
                for idx in ethnicity_indices:
                    logger.info(f"Analyzing spatial patterns for ethnicity index {idx}")
                    spatial_results[str(idx)] = analyze_expert_weights_spatial_patterns(
                        gdf.copy(), dauid_column, interpretations, idx, args.output_dir
                    )
                
                # Save overall spatial statistics
                spatial_stats_path = os.path.join(args.output_dir, "spatial_statistics", "all_ethnicities_spatial_stats.json")
                os.makedirs(os.path.dirname(spatial_stats_path), exist_ok=True)
                
                with open(spatial_stats_path, 'w') as f:
                    json.dump(spatial_results, f, indent=2)
                
                logger.info(f"Saved overall spatial statistics to {spatial_stats_path}")
            else:
                logger.warning("No DAUID column found in shapefile, skipping spatial analysis")
        except Exception as e:
            logger.error(f"Error loading shapefile or performing spatial analysis: {str(e)}")
    
    # Generate all visualizations
    if args.all:
        logger.info("Generating all visualizations")
        analyzer.run_all_visualizations(gdf, args.shapefile_path)
    else:
        # Generate individual visualizations based on flags
        if args.network:
            logger.info("Generating network visualizations")
            analyzer.visualize_interaction_network()
        
        if args.flow:
            logger.info("Generating flow diagram visualizations")
            analyzer.visualize_flow_diagrams()
        
        if args.spatial and (dauid_to_coords or gdf is not None):
            logger.info("Generating spatial visualizations")
            analyzer.visualize_spatial_interactions(gdf, args.shapefile_path)
        
        if args.temporal:
            logger.info("Generating temporal cascade visualizations")
            analyzer.visualize_temporal_cascade(time_steps=args.time_steps)
        
        if args.metrics:
            logger.info("Generating interaction metrics visualizations")
            analyzer.visualize_interaction_metrics()
    
    logger.info("=== Ethnicity Interaction Analysis Completed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ethnicity Interaction Analysis")
    
    # Required parameters
    parser.add_argument("--interpretation_dir", type=str, required=True,
                        help="Directory containing interpretation JSON files")
    parser.add_argument("--output_dir", type=str, default="./ethnicity_visualizations",
                        help="Directory to save ethnicity interaction visualizations")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--spatial_dir", type=str, default="./spatial_data",
                        help="Directory containing spatial data files")
    # Add this to your argument parser
    parser.add_argument("--spatial_analysis", action="store_true",
                        help="Perform spatial autocorrelation analysis")
    # Optional parameters
    parser.add_argument("--ethnicity_names", type=str, nargs='+',
                        help="List of ethnicity names")
    parser.add_argument("--da_coordinates_path", type=str, default='./spatial_data/toronto_da.csv',
                        help="Path to the CSV file containing DA coordinates")
    parser.add_argument("--shapefile_path", type=str, default='./spatial_data/da_boundary.shp',
                        help="Path to shapefile with DA boundaries for spatial visualization")
    parser.add_argument("--colormap", type=str, default=None,
                        help="Matplotlib colormap name for ethnicity colors")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold for significant interactions")
    parser.add_argument("--time_steps", type=int, default=5,
                        help="Number of time steps for temporal cascade visualization")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use for feature normalization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    # Visualization type flags
    parser.add_argument("--all", action="store_true",
                        help="Generate all visualization types")
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
    
    # If no specific visualization is requested, enable all
    if not any([args.all, args.network, args.flow, args.spatial, args.temporal, args.metrics]):
        args.all = True
    
    main(args)