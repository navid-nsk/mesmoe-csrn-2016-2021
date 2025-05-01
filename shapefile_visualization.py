import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import logging

logger = logging.getLogger(__name__)

def visualize_spatial_patterns_with_shapefile(visualizer, interpretation_data, predictions_df, shapefile_path, batch_idx=0):
    """
    Create spatial visualizations using shapefile boundaries instead of point coordinates
    
    Args:
        visualizer: The ModelInterpretationVisualizer instance
        interpretation_data (dict): Interpretation data
        predictions_df (pd.DataFrame): Predictions DataFrame with DAUIDs
        shapefile_path (str): Path to the shapefile with DA boundaries
        batch_idx (int): Batch index for saving
        
    Returns:
        bool: Success status
    """
    try:
        # Load shapefile into GeoDataFrame
        logger.info(f"Loading shapefile from {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded shapefile with {len(gdf)} geometries")
        except Exception as e:
            logger.error(f"Error loading shapefile: {str(e)}")
            return False
        
        # Check if 'DAUID' column exists in the shapefile
        dauid_column = None
        for column in gdf.columns:
            if column.upper() == 'DAUID' or 'DAUID' in column.upper():
                dauid_column = column
                break
        
        if dauid_column is None:
            logger.warning("DAUID column not found in shapefile, trying common ID field names")
            # Try common ID field names in shapefiles
            for column in ['ID', 'GEOID', 'AREA_ID', 'DA_ID', 'DISSEMINATION_AREA_ID']:
                if column in gdf.columns:
                    dauid_column = column
                    logger.info(f"Using {dauid_column} as the DA identifier")
                    break
            
            if dauid_column is None:
                logger.error("No suitable ID column found in shapefile")
                # As a last resort, show columns to help diagnose
                logger.info(f"Available columns in shapefile: {list(gdf.columns)}")
                return False
        
        # Ensure DAUID is string type for consistent joining
        gdf[dauid_column] = gdf[dauid_column].astype(str)
        
        # Check if we have expert weights by DAUID
        dauid_data_available = False
        
        # Try to find DAUID-indexed data in different experts
        if ('colonization_expert' in interpretation_data and 
            'size_probabilities' in interpretation_data['colonization_expert'] and 
            isinstance(interpretation_data['colonization_expert']['size_probabilities'], dict)):
            dauid_data_available = True
            sample_dauid_dict = interpretation_data['colonization_expert']['size_probabilities']
        elif ('jump_expert' in interpretation_data and 
              'predictions' in interpretation_data['jump_expert'] and 
              'jump_probabilities' in interpretation_data['jump_expert']['predictions'] and
              isinstance(interpretation_data['jump_expert']['predictions']['jump_probabilities'], dict)):
            dauid_data_available = True
            sample_dauid_dict = interpretation_data['jump_expert']['predictions']['jump_probabilities']
        elif ('decline_expert' in interpretation_data and 
              'predictions' in interpretation_data['decline_expert'] and 
              'decline_probabilities' in interpretation_data['decline_expert']['predictions'] and
              isinstance(interpretation_data['decline_expert']['predictions']['decline_probabilities'], dict)):
            dauid_data_available = True
            sample_dauid_dict = interpretation_data['decline_expert']['predictions']['decline_probabilities']
        elif ('pde_expert' in interpretation_data and 
              'diffusion' in interpretation_data['pde_expert'] and 
              isinstance(interpretation_data['pde_expert']['diffusion'], dict)):
            dauid_data_available = True
            sample_dauid_dict = interpretation_data['pde_expert']['diffusion']
        
        if not dauid_data_available:
            logger.warning("No DAUID-indexed data found in interpretation data, skipping spatial visualizations")
            return False
        
        # Build expert weight visualizations if available
        if 'expert_weights' in interpretation_data and 'dauid_list' in interpretation_data:
            # Create a mapping from DAUID to index
            interp_dauid_list = interpretation_data['dauid_list']
            dauid_to_idx = {str(d): i for i, d in enumerate(interp_dauid_list)}
            
            # Convert expert_weights to numpy array if it's a list
            expert_weights = np.array(interpretation_data['expert_weights'])
            
            # Get the number of ethnicities
            num_ethnicities = expert_weights.shape[1] if hasattr(expert_weights, 'shape') else 0
            
            # If we couldn't determine number of ethnicities from expert_weights, try ethnicity_names
            if num_ethnicities == 0 and visualizer.ethnicity_names:
                num_ethnicities = len(visualizer.ethnicity_names)
                logger.info(f"Using {num_ethnicities} ethnicities from ethnicity_names")
            
            if num_ethnicities == 0:
                logger.warning("Could not determine number of ethnicities, skipping expert weight visualizations")
            else:
                # Create visualizations for a few ethnicities
                ethnicities_to_visualize = min(3, num_ethnicities)
                expert_names = ['Colonization', 'Jump', 'Decline', 'PDE']
                
                for eth_idx in range(ethnicities_to_visualize):
                    eth_name = visualizer.ethnicity_names[eth_idx] if visualizer.ethnicity_names else f'Ethnicity_{eth_idx}'
                    
                    # Create 2x2 subplots for expert weights
                    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                    axes = axes.flatten()
                    
                    for exp_idx, exp_name in enumerate(expert_names):
                        # Create a new GeoDataFrame for this visualization
                        plot_gdf = gdf.copy()
                        
                        # Add weight data to GeoDataFrame
                        weight_column = f"{exp_name}_Weight"
                        plot_gdf[weight_column] = np.nan  # Initialize with NaN
                        
                        # Add weights from interpretation data
                        for dauid, idx in dauid_to_idx.items():
                            if idx < expert_weights.shape[0]:
                                weight = expert_weights[idx, eth_idx, exp_idx]
                                
                                # Find matching rows in GeoDataFrame
                                matching_rows = plot_gdf[plot_gdf[dauid_column] == str(dauid)]
                                if not matching_rows.empty:
                                    plot_gdf.loc[matching_rows.index, weight_column] = weight
                        
                        # Plot in the appropriate subplot
                        ax = axes[exp_idx]
                        
                        # Create custom colormap for weights (from 0 to 1)
                        cmap = plt.cm.viridis
                        
                        # Plot the data
                        plot_gdf.plot(
                            column=weight_column,
                            ax=ax,
                            legend=True,
                            cmap=cmap,
                            missing_kwds={'color': 'lightgrey'},
                            edgecolor='black',
                            linewidth=0.3,
                            alpha=0.8
                        )
                        
                        # Add title and styling
                        ax.set_title(f'{exp_name} Expert Weight for {eth_name}', fontsize=14)
                        ax.set_axis_off()
                        
                        # Add a note about missing data
                        missing_count = plot_gdf[weight_column].isna().sum()
                        if missing_count > 0:
                            ax.text(
                                0.05, 0.05, 
                                f"Missing data: {missing_count}/{len(plot_gdf)} areas", 
                                transform=ax.transAxes,
                                bbox=dict(facecolor='white', alpha=0.8)
                            )
                    
                    # Add main title
                    plt.suptitle(f'Expert Weights for {eth_name}', fontsize=16, y=0.95)
                    plt.tight_layout()
                    
                    # Save the figure
                    save_path = os.path.join(visualizer.spatial_dir, 
                                           f'spatial_expert_weights_{eth_name}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                    
                    logger.info(f"Created expert weight spatial visualization for {eth_name}")
        
        # Create ethnicity interaction effect visualizations if available
        if 'interaction_effect_by_dauid' in interpretation_data:
            num_ethnicities = len(visualizer.ethnicity_names) if visualizer.ethnicity_names else 0
            if num_ethnicities == 0 and 'expert_weights' in interpretation_data:
                expert_weights = np.array(interpretation_data['expert_weights'])
                num_ethnicities = expert_weights.shape[1] if hasattr(expert_weights, 'shape') else 0
            
            # Skip if we don't know the number of ethnicities
            if num_ethnicities > 0:
                ethnicities_to_visualize = min(3, num_ethnicities)
                
                for eth_idx in range(ethnicities_to_visualize):
                    eth_name = visualizer.ethnicity_names[eth_idx] if visualizer.ethnicity_names else f'Ethnicity_{eth_idx}'
                    
                    # Create a new GeoDataFrame for this visualization
                    plot_gdf = gdf.copy()
                    
                    # Add effect data to GeoDataFrame
                    effect_column = 'Interaction_Effect'
                    plot_gdf[effect_column] = np.nan  # Initialize with NaN
                    
                    # Fill effect values from interpretation data
                    for dauid, effects in interpretation_data['interaction_effect_by_dauid'].items():
                        # Find effect for this ethnicity
                        effect = 0
                        if eth_name in effects:
                            effect = effects[eth_name]
                        elif f'ethnicity_{eth_idx}' in effects:
                            effect = effects[f'ethnicity_{eth_idx}']
                        
                        # Add to GeoDataFrame
                        matching_rows = plot_gdf[plot_gdf[dauid_column] == str(dauid)]
                        if not matching_rows.empty:
                            plot_gdf.loc[matching_rows.index, effect_column] = effect
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Create diverging colormap for positive/negative effects
                    cmap = plt.cm.coolwarm
                    
                    # Get min and max for symmetric colorbar
                    vmax = max(abs(plot_gdf[effect_column].min()), abs(plot_gdf[effect_column].max()))
                    vmin = -vmax
                    
                    # Plot the data
                    plot_gdf.plot(
                        column=effect_column,
                        ax=ax,
                        legend=True,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        missing_kwds={'color': 'lightgrey'},
                        edgecolor='black',
                        linewidth=0.3,
                        alpha=0.8
                    )
                    
                    # Add title and styling
                    ax.set_title(f'Spatial Distribution of Interaction Effects for {eth_name}', fontsize=14)
                    ax.set_axis_off()
                    
                    # Add a note about missing data
                    missing_count = plot_gdf[effect_column].isna().sum()
                    if missing_count > 0:
                        ax.text(
                            0.05, 0.05, 
                            f"Missing data: {missing_count}/{len(plot_gdf)} areas", 
                            transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.8)
                        )
                    
                    plt.tight_layout()
                    save_path = os.path.join(visualizer.spatial_dir, 
                                           f'spatial_interaction_effects_{eth_name}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                    
                    logger.info(f"Created interaction effect spatial visualization for {eth_name}")
        
        # Create visualizations from predictions DataFrame if provided
        if predictions_df is not None:
            # Ensure DAUID is string type
            predictions_df['DAUID'] = predictions_df['DAUID'].astype(str)
            
            # Identify available ethnicities in the predictions
            eth_columns = []
            for col in predictions_df.columns:
                if col.endswith('_predicted'):
                    eth_columns.append(col)
            
            # Create spatial plots for each ethnicity
            for col in eth_columns[:min(5, len(eth_columns))]:
                eth_name = col.replace('_predicted', '')
                
                # Create a new GeoDataFrame for this visualization
                plot_gdf = gdf.copy()
                
                # Create column for predicted values
                plot_gdf['predicted'] = np.nan
                
                # Join with predictions
                for idx, row in predictions_df.iterrows():
                    dauid = str(row['DAUID'])
                    pred_value = row[col]
                    
                    # Find matching rows in GeoDataFrame
                    matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                    if not matching_rows.empty:
                        plot_gdf.loc[matching_rows.index, 'predicted'] = pred_value
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Create colormap
                cmap = plt.cm.viridis
                
                # Plot the data
                plot_gdf.plot(
                    column='predicted',
                    ax=ax,
                    legend=True,
                    cmap=cmap,
                    missing_kwds={'color': 'lightgrey'},
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.8
                )
                
                # Add title and styling
                ax.set_title(f'Spatial Distribution of Predicted Population for {eth_name}', fontsize=14)
                ax.set_axis_off()
                
                # Add a note about missing data
                missing_count = plot_gdf['predicted'].isna().sum()
                if missing_count > 0:
                    ax.text(
                        0.05, 0.05, 
                        f"Missing data: {missing_count}/{len(plot_gdf)} areas", 
                        transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                
                plt.tight_layout()
                save_path = os.path.join(visualizer.spatial_dir, 
                                       f'spatial_predicted_population_{eth_name}_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
                
                # If we have both predicted and 2016 data, create change map
                base_col = f"{eth_name}_2016"
                if base_col in predictions_df.columns:
                    # Create a new GeoDataFrame for this visualization
                    plot_gdf = gdf.copy()
                    
                    # Create column for change values
                    plot_gdf['percent_change'] = np.nan
                    
                    # Join with predictions to calculate percent change
                    for idx, row in predictions_df.iterrows():
                        dauid = str(row['DAUID'])
                        pred_value = row[col]
                        base_value = row[base_col]
                        
                        # Calculate percent change with safety for zeros
                        percent_change = ((pred_value - base_value) / (base_value + 1)) * 100
                        
                        # Find matching rows in GeoDataFrame
                        matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                        if not matching_rows.empty:
                            plot_gdf.loc[matching_rows.index, 'percent_change'] = percent_change
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Create diverging colormap
                    cmap = plt.cm.coolwarm
                    
                    # Clamp extreme values for better visualization
                    vmin = -100
                    vmax = 100
                    
                    # Plot the data
                    plot_gdf.plot(
                        column='percent_change',
                        ax=ax,
                        legend=True,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        missing_kwds={'color': 'lightgrey'},
                        edgecolor='black',
                        linewidth=0.3,
                        alpha=0.8
                    )
                    
                    # Add title and styling
                    ax.set_title(f'Spatial Distribution of Population Change for {eth_name}', fontsize=14)
                    ax.set_axis_off()
                    
                    # Add a note about missing data
                    missing_count = plot_gdf['percent_change'].isna().sum()
                    if missing_count > 0:
                        ax.text(
                            0.05, 0.05, 
                            f"Missing data: {missing_count}/{len(plot_gdf)} areas", 
                            transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.8)
                        )
                    
                    plt.tight_layout()
                    save_path = os.path.join(visualizer.spatial_dir, 
                                           f'spatial_population_change_{eth_name}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                    
                    logger.info(f"Created population change spatial visualization for {eth_name}")
        
        logger.info(f"Successfully created spatial visualizations from shapefile")
        return True
        
    except Exception as e:
        logger.error(f"Error creating spatial visualizations with shapefile: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False