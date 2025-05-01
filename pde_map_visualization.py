import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import matplotlib.cm as cm
from matplotlib import patches
import logging
from collections import defaultdict
from matplotlib.cm import ScalarMappable

def create_improved_velocity_visualization(plot_gdf, eth_name, dauid_column, output_dir):
    """
    Create an improved velocity vector field visualization with properly scaled arrows.
    
    Args:
        plot_gdf (GeoDataFrame): GeoDataFrame with velocity_x, velocity_y, velocity_magnitude
        eth_name (str): Name of the ethnicity
        dauid_column (str): Name of the column with DAUIDs
        output_dir (str): Directory to save the visualization
        
    Returns:
        str: Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import os
    
    # Drop rows with missing velocity data
    vector_gdf = plot_gdf.dropna(subset=['velocity_x', 'velocity_y'])
    
    if not vector_gdf.empty:
        # Create a more effective velocity visualization
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot the base map
        plot_gdf.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.5
        )
        
        # Calculate velocities for scaling the vectors
        velocities = vector_gdf['velocity_magnitude']
        max_velocity = velocities.max()
        min_velocity = velocities.min()
        
        # Print some statistics to understand the velocity range
        print(f"Velocity magnitude statistics for {eth_name}:")
        print(f"  Min: {min_velocity:.6f}")
        print(f"  Max: {max_velocity:.6f}")
        print(f"  Mean: {velocities.mean():.6f}")
        print(f"  Median: {velocities.median():.6f}")
        print(f"  5th percentile: {np.percentile(velocities, 5):.6f}")
        print(f"  95th percentile: {np.percentile(velocities, 95):.6f}")
        
        # Use percentile-based normalization for colors (more effective than log for velocities)
        p05 = np.percentile(velocities, 5)
        p95 = np.percentile(velocities, 95)
        norm = mcolors.Normalize(vmin=p05, vmax=p95)
        
        # FIXED: Better arrow scaling approach
        # Instead of using an arbitrary scale factor, use a fixed scale value
        # This ensures arrows don't become enormous
        # A larger number here means smaller arrows
        scale_value = 20  # Start with this value and adjust if needed
        
        # Plot velocity vectors with enhanced visibility and proper scaling
        q = ax.quiver(
            vector_gdf['centroid_x'], 
            vector_gdf['centroid_y'],
            vector_gdf['velocity_x'], 
            vector_gdf['velocity_y'],
            velocities,
            cmap='viridis',
            norm=norm,
            scale=scale_value,  # Fixed scale value instead of dynamic calculation
            width=0.003,  # Thicker arrows
            headwidth=6,
            headlength=8,
            minshaft=2,
            alpha=0.9
        )
        
        # Add colorbar with percentile scale
        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label(f'Velocity Magnitude (5th-95th percentile: {p05:.6f}-{p95:.6f})')
        
        # Add title and styling
        ax.set_title(f'Enhanced Velocity Vector Field for {eth_name}', fontsize=16)
        ax.set_axis_off()
        
        # Add a note about data coverage
        missing_count = plot_gdf.shape[0] - vector_gdf.shape[0]
        if missing_count > 0:
            ax.text(
                0.05, 0.05, 
                f"Areas with velocity data: {vector_gdf.shape[0]}/{plot_gdf.shape[0]}", 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        # Add a reference arrow for scale
        # Use median velocity for reference arrow to represent typical values
        median_velocity = velocities.median()
        ax.quiverkey(q, 0.9, 0.9, median_velocity, f'{median_velocity:.2e}', 
                     labelpos='E', coordinates='figure')
        
        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'fixed_velocity_vector_{eth_name.replace(" ", "_")}.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"Created fixed velocity vector field map for {eth_name}")
        return save_path
    
    return None


def visualize_pde_parameters_with_improved_scale(plot_gdf, param_name, eth_name, dauid_column, output_dir, param_data=None):
    """
    Create improved visualizations for PDE parameters with better scaling and normalization.
    
    Args:
        plot_gdf (GeoDataFrame): GeoDataFrame with geometry and parameter data
        param_name (str): Name of the parameter to visualize ('diffusion', 'amplification', etc.)
        eth_name (str): Name of the ethnicity
        dauid_column (str): Name of the column containing DAUID values
        output_dir (str): Directory to save the visualizations
        param_data (dict, optional): Dictionary of parameter data by DAUID and ethnicity
    
    Returns:
        None
    """
    # Create a figure and axis
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Choose colormap based on parameter
    if param_name == 'diffusion':
        cmap = 'Blues'
        title_base = f'Average Diffusion Coefficient for {eth_name}'
    elif param_name == 'amplification':
        cmap = 'Oranges'
        title_base = f'Average Amplification Factor for {eth_name}'
    else:  # velocity_magnitude
        cmap = 'Reds'
        title_base = f'Average Velocity Magnitude for {eth_name}'
    
    # 1. Standard visualization (Linear Scale)
    ax = axes[0]
    plot_gdf.plot(
        column=param_name,
        ax=ax,
        legend=True,
        cmap=cmap,
        missing_kwds={'color': 'lightgrey'},
        edgecolor='black',
        linewidth=0.3,
        alpha=0.8
    )
    ax.set_title(f'{title_base}\n(Standard Linear Scale)', fontsize=12)
    ax.set_axis_off()
    
    # 2. Visualization with custom min/max scaling
    ax = axes[1]
    # Calculate a more appropriate vmin by finding the minimum non-zero value
    # and scaling it down slightly
    values = plot_gdf[param_name].dropna()
    if len(values) > 0:
        vmin = values.min() * 0.9  # Slightly lower than min
        vmax = values.max() * 1.1  # Slightly higher than max
        
        plot_gdf.plot(
            column=param_name,
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
    else:
        plot_gdf.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.8
        )
    ax.set_title(f'{title_base}\n(Adjusted Min/Max Scale)', fontsize=12)
    ax.set_axis_off()
    
    # 3. Visualization with log scale (only for positive values)
    ax = axes[2]
    # Create a copy for log visualization
    log_plot_gdf = plot_gdf.copy()
    
    # Ensure all values are positive for log scale
    min_positive = log_plot_gdf[param_name].replace(0, np.nan).dropna().min() if not log_plot_gdf[param_name].replace(0, np.nan).dropna().empty else 0.001
    epsilon = min_positive / 10  # Small value to add to avoid log(0)
    
    log_plot_gdf[param_name] = log_plot_gdf[param_name].apply(lambda x: x + epsilon if pd.notnull(x) else x)
    
    # Create a log-normalized colormap
    norm = LogNorm(vmin=min_positive, vmax=log_plot_gdf[param_name].max())
    
    log_plot_gdf.plot(
        column=param_name,
        ax=ax,
        legend=True,
        cmap=cmap,
        norm=norm,
        missing_kwds={'color': 'lightgrey'},
        edgecolor='black',
        linewidth=0.3,
        alpha=0.8
    )
    ax.set_title(f'{title_base}\n(Log Scale)', fontsize=12)
    ax.set_axis_off()
    
    # 4. Visualization with custom percentile-based colormap
    ax = axes[3]
    values = plot_gdf[param_name].dropna()
    
    if len(values) > 0:
        # Calculate percentiles for more balanced color distribution
        p05 = np.percentile(values, 5)
        p95 = np.percentile(values, 95)
        
        # Create a custom normalization that focuses on the central 90% of the data
        norm = Normalize(vmin=p05, vmax=p95)
        
        plot_gdf.plot(
            column=param_name,
            ax=ax,
            legend=True,
            cmap=cmap,
            norm=norm,
            missing_kwds={'color': 'lightgrey'},
            edgecolor='black',
            linewidth=0.3,
            alpha=0.8
        )
        
        # Add note about percentile range
        ax.text(
            0.05, 0.95, 
            f"Color range: 5th to 95th percentile\n({p05:.6f} to {p95:.6f})", 
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top'
        )
    else:
        plot_gdf.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.8
        )
    
    ax.set_title(f'{title_base}\n(Percentile-Based Scale)', fontsize=12)
    ax.set_axis_off()
    
    # Add a main title
    plt.suptitle(f'Different Scaling Methods for {param_name.capitalize()} Visualization - {eth_name}', 
                 fontsize=16, y=0.98)
    
    # Add a note about missing data
    missing_count = plot_gdf[param_name].isna().sum()
    if missing_count > 0:
        fig.text(
            0.5, 0.01, 
            f"Missing data: {missing_count}/{len(plot_gdf)} areas", 
            ha='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'improved_pde_{param_name}_{eth_name.replace(" ", "_")}.png')
    plt.savefig(save_path)
    plt.close()
    
    # 5. Create a separate high-contrast visualization
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Create a custom high-contrast colormap
    if param_name == 'diffusion':
        # Create a colormap that goes from white to dark blue
        colors = [(1, 1, 1), (0, 0, 0.8)]  # White to dark blue
        high_contrast_cmap = LinearSegmentedColormap.from_list('high_contrast_blues', colors)
    elif param_name == 'amplification':
        # Create a colormap that goes from white to dark orange
        colors = [(1, 1, 1), (0.8, 0.4, 0)]  # White to dark orange
        high_contrast_cmap = LinearSegmentedColormap.from_list('high_contrast_oranges', colors)
    else:  # velocity_magnitude
        # Create a colormap that goes from white to dark red
        colors = [(1, 1, 1), (0.8, 0, 0)]  # White to dark red
        high_contrast_cmap = LinearSegmentedColormap.from_list('high_contrast_reds', colors)
    
    # Get data values
    values = plot_gdf[param_name].dropna()
    
    if len(values) > 0:
        # Use min-max normalization for full range visibility
        vmin = values.min()
        vmax = values.max()
        
        # Plot with high contrast
        plot_gdf.plot(
            column=param_name,
            ax=ax,
            legend=True,
            cmap=high_contrast_cmap,
            vmin=vmin,
            vmax=vmax,
            missing_kwds={'color': 'lightgrey'},
            edgecolor='black',
            linewidth=0.3,
            alpha=1.0  # Full opacity for higher contrast
        )
        
        # Add value range
        ax.text(
            0.05, 0.05, 
            f"Value range: {vmin:.6f} to {vmax:.6f}", 
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    else:
        plot_gdf.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.8
        )
    
    # Add title
    ax.set_title(f'{title_base}\n(High Contrast Visualization)', fontsize=16)
    ax.set_axis_off()
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'high_contrast_pde_{param_name}_{eth_name.replace(" ", "_")}.png')
    plt.savefig(save_path)
    plt.close()
    
    # 6. Create a 3D visualization to better show differences in small values
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Only create 3D visualization if we have centroid coordinates
        if 'centroid_x' not in plot_gdf.columns or 'centroid_y' not in plot_gdf.columns:
            plot_gdf['centroid_x'] = plot_gdf.geometry.centroid.x
            plot_gdf['centroid_y'] = plot_gdf.geometry.centroid.y
        
        # Drop rows with missing data
        plot_3d = plot_gdf.dropna(subset=[param_name, 'centroid_x', 'centroid_y'])
        
        if not plot_3d.empty:
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Scale the coordinates to fit nicely in the plot
            x = plot_3d['centroid_x']
            y = plot_3d['centroid_y']
            z = plot_3d[param_name]
            
            # Normalize x and y to [0, 1] range for better visualization
            x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
            y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y
            
            # Create scatter plot with color based on parameter value
            scatter = ax.scatter(
                x_norm, y_norm, z,
                c=z,
                cmap=cmap,
                s=50,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.2
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(param_name.capitalize())
            
            # Add gridlines for better depth perception
            ax.grid(True)
            
            # Set labels
            ax.set_xlabel('Normalized X (East-West)')
            ax.set_ylabel('Normalized Y (North-South)')
            ax.set_zlabel(param_name.capitalize())
            
            # Set title
            ax.set_title(f'3D Visualization of {param_name.capitalize()} for {eth_name}', fontsize=16)
            
            # Save the figure
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'3d_pde_{param_name}_{eth_name.replace(" ", "_")}.png')
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Could not create 3D visualization: {str(e)}")
    
    return

def visualize_pde_parameters_percentile_focused(plot_gdf, param_name, eth_name, dauid_column, output_dir, param_data=None):
    """
    Create visualizations for PDE parameters focusing on percentile-based scaling.
    
    Args:
        plot_gdf (GeoDataFrame): GeoDataFrame with geometry and parameter data
        param_name (str): Name of the parameter to visualize ('diffusion', 'amplification', etc.)
        eth_name (str): Name of the ethnicity
        dauid_column (str): Name of the column containing DAUID values
        output_dir (str): Directory to save the visualizations
        param_data (dict, optional): Dictionary of parameter data by DAUID and ethnicity
    
    Returns:
        str: Path to the saved visualization
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Choose colormap based on parameter
    if param_name == 'diffusion':
        cmap = 'Blues'
        title_base = f'Diffusion Coefficient for {eth_name}'
    elif param_name == 'amplification':
        cmap = 'Oranges'
        title_base = f'Amplification Factor for {eth_name}'
    else:  # velocity_magnitude
        cmap = 'Reds'
        title_base = f'Velocity Magnitude for {eth_name}'
    
    # Get non-missing values
    values = plot_gdf[param_name].dropna()
    
    if len(values) > 0:
        # Calculate more nuanced percentiles for visualization
        percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        percentile_values = [np.percentile(values, p) for p in percentiles]
        
        # Print the percentile distribution for reference
        print(f"\nPercentile distribution for {param_name} - {eth_name}:")
        for p, val in zip(percentiles, percentile_values):
            print(f"  {p}th percentile: {val:.6f}")
        
        # Focus on the central 5-95% range for better visualization
        p05 = np.percentile(values, 5)
        p95 = np.percentile(values, 95)
        
        # Create a normalization that focuses on the central 90% of the data
        norm = matplotlib.colors.Normalize(vmin=p05, vmax=p95)
        
        # Plot with percentile-based scaling
        plot_gdf.plot(
            column=param_name,
            ax=ax,
            legend=True,
            cmap=cmap,
            norm=norm,
            missing_kwds={'color': 'lightgrey'},
            edgecolor='black',
            linewidth=0.3,
            alpha=0.8
        )
        
        # Add a customized colorbar with percentile markers
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{param_name.capitalize()} Value', fontsize=12)
        
        # Add percentile markers to the colorbar
        percentile_markers = [5, 25, 50, 75, 95]
        percentile_values = [np.percentile(values, p) for p in percentile_markers]
        
        # Only add markers that fall within the colorbar range
        visible_markers = []
        visible_values = []
        visible_labels = []
        
        for p, val in zip(percentile_markers, percentile_values):
            if p05 <= val <= p95:
                visible_markers.append((val - p05) / (p95 - p05))  # Normalize to 0-1 range
                visible_values.append(val)
                visible_labels.append(f"{p}%: {val:.4f}")
        
        cbar.ax.set_yticks(visible_markers)
        cbar.ax.set_yticklabels(visible_labels)
        
        # Add note about data distribution and full range
        ax.text(
            0.05, 0.05, 
            f"Full range: {values.min():.6f} to {values.max():.6f}\n"
            f"Showing 5th to 95th percentile for better visibility", 
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=10
        )
    else:
        # No data available, just plot base map
        plot_gdf.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.5
        )
        
        # Add note about missing data
        ax.text(
            0.5, 0.5,
            "No data available for this parameter and ethnicity",
            transform=ax.transAxes,
            fontsize=14,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Add title and style
    ax.set_title(f'{title_base}\n(Percentile-Based Scale)', fontsize=16)
    ax.set_axis_off()
    
    # Add a note about missing data
    missing_count = plot_gdf[param_name].isna().sum()
    total_count = len(plot_gdf)
    if missing_count > 0:
        ax.text(
            0.05, 0.95, 
            f"Data coverage: {total_count - missing_count}/{total_count} areas ({(total_count - missing_count)/total_count*100:.1f}%)", 
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top'
        )
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'percentile_pde_{param_name}_{eth_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Create an additional visualization with multiple percentile bands
    fig, ax = plt.subplots(figsize=(15, 12))
    
    if len(values) > 0:
        # Create a custom colormap with distinct bands for percentile ranges
        import matplotlib.colors as mcolors
        
        # Calculate percentile thresholds for visualization
        thresholds = [
            values.min(),  # 0th percentile
            np.percentile(values, 10),  # 10th percentile
            np.percentile(values, 25),  # 25th percentile
            np.percentile(values, 50),  # 50th percentile
            np.percentile(values, 75),  # 75th percentile
            np.percentile(values, 90),  # 90th percentile
            np.percentile(values, 95),  # 95th percentile
            np.percentile(values, 99),  # 99th percentile
            values.max()   # 100th percentile
        ]
        
        # Create a boundary norm for discrete color bands
        norm = mcolors.BoundaryNorm(thresholds, 256)
        
        # Plot with percentile-based coloring
        plot_gdf.plot(
            column=param_name,
            ax=ax,
            cmap=cmap,
            norm=norm,
            missing_kwds={'color': 'lightgrey'},
            edgecolor='black',
            linewidth=0.3,
            alpha=0.8
        )
        
        # Create a custom colorbar with percentile labels
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, ticks=thresholds)
        cbar.set_label(f'{param_name.capitalize()} Value (Percentile Bands)', fontsize=12)
        
        # Create percentile labels
        percentile_labels = [
            "0% (min)",
            "10%",
            "25%",
            "50% (median)",
            "75%",
            "90%",
            "95%",
            "99%",
            "100% (max)"
        ]
        
        # Format the values to be more compact based on their magnitude
        formatted_labels = []
        for label, value in zip(percentile_labels, thresholds):
            if abs(value) < 0.01:
                # Use scientific notation for very small values
                formatted_labels.append(f"{label}: {value:.2e}")
            else:
                # Use regular notation with appropriate precision
                formatted_labels.append(f"{label}: {value:.4f}")
        
        cbar.ax.set_yticklabels(formatted_labels, fontsize=9)
        
        # Add a title explaining the bands
        ax.set_title(f'{title_base}\n(Percentile Band Visualization)', fontsize=16)
    else:
        # No data case
        plot_gdf.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.5
        )
        
        ax.text(
            0.5, 0.5,
            "No data available for this parameter and ethnicity",
            transform=ax.transAxes,
            fontsize=14,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Set axis off
    ax.set_axis_off()
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'percentile_bands_pde_{param_name}_{eth_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def update_visualize_pde_parameters_on_map(visualizer, interpretation_dir, shapefile_path, output_dir):
    """
    Update the map-based visualizations for PDE parameters to prioritize percentile-based scaling.
    
    Args:
        visualizer: The ModelInterpretationVisualizer instance
        interpretation_dir (str): Directory containing interpretation JSON files
        shapefile_path (str): Path to the shapefile with DA boundaries
        output_dir (str): Directory to save the visualizations
        
    Returns:
        bool: Success status
    """
    try:
        # Create output directory for percentile-based PDE map visualizations
        percentile_pde_dir = os.path.join(output_dir, "percentile_based_pde_maps")
        os.makedirs(percentile_pde_dir, exist_ok=True)
        
        # Load shapefile
        print(f"Loading shapefile from {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            print(f"Loaded shapefile with {len(gdf)} geometries")
        except Exception as e:
            print(f"Error loading shapefile: {str(e)}")
            return False
        
        # Find DAUID column in shapefile
        dauid_column = None
        for column in gdf.columns:
            if column.upper() == 'DAUID' or 'DAUID' in column.upper():
                dauid_column = column
                break
        
        if dauid_column is None:
            print("DAUID column not found in shapefile, trying common ID field names")
            # Try common ID field names in shapefiles
            for column in ['ID', 'GEOID', 'AREA_ID', 'DA_ID', 'DISSEMINATION_AREA_ID']:
                if column in gdf.columns:
                    dauid_column = column
                    print(f"Using {dauid_column} as the DA identifier")
                    break
            
            if dauid_column is None:
                print("No suitable ID column found in shapefile")
                print(f"Available columns in shapefile: {list(gdf.columns)}")
                return False
        
        # Ensure DAUID is string type for consistent joining
        gdf[dauid_column] = gdf[dauid_column].astype(str)
        
        # Find all interpretation JSON files
        json_files = [f for f in os.listdir(interpretation_dir) 
                    if f.endswith('.json') and 'interpretation' in f]
        
        if not json_files:
            print(f"No interpretation files found in {interpretation_dir}")
            return False
        
        print(f"Found {len(json_files)} interpretation files")
        
        # Initialize data structures to store aggregated PDE parameters by DAUID and ethnicity
        pde_params = {
            'diffusion': defaultdict(lambda: defaultdict(list)),
            'amplification': defaultdict(lambda: defaultdict(list)),
            'velocity_x': defaultdict(lambda: defaultdict(list)),
            'velocity_y': defaultdict(lambda: defaultdict(list)),
            'velocity_magnitude': defaultdict(lambda: defaultdict(list))
        }
        
        # Maximum number of ethnicities encountered
        max_ethnicities = 0
        ethnicity_names = visualizer.ethnicity_names if hasattr(visualizer, 'ethnicity_names') and visualizer.ethnicity_names else None
        
        # Process each interpretation file
        for json_file in json_files:
            file_path = os.path.join(interpretation_dir, json_file)
            
            # Load interpretation data
            interpretation_data = visualizer.load_interpretation_data(file_path)
            if interpretation_data is None or 'pde_expert' not in interpretation_data:
                print(f"No PDE expert data found in {json_file}, skipping")
                continue
            
            pde_data = interpretation_data['pde_expert']
            
            # Process PDE parameters for each DAUID and ethnicity
            if 'diffusion' in pde_data and isinstance(pde_data['diffusion'], dict):
                for dauid, values in pde_data['diffusion'].items():
                    for eth_idx, value in enumerate(values):
                        pde_params['diffusion'][dauid][eth_idx].append(float(value))
                        if eth_idx + 1 > max_ethnicities:
                            max_ethnicities = eth_idx + 1
            
            if 'amplification' in pde_data and isinstance(pde_data['amplification'], dict):
                for dauid, values in pde_data['amplification'].items():
                    for eth_idx, value in enumerate(values):
                        pde_params['amplification'][dauid][eth_idx].append(float(value))
                        if eth_idx + 1 > max_ethnicities:
                            max_ethnicities = eth_idx + 1
            
            if 'velocity' in pde_data and isinstance(pde_data['velocity'], dict):
                for dauid, values in pde_data['velocity'].items():
                    for eth_idx, velocity in enumerate(values):
                        if len(velocity) >= 2:  # Ensure we have x and y components
                            vx = float(velocity[0])
                            vy = float(velocity[1])
                            magnitude = np.sqrt(vx**2 + vy**2)
                            
                            pde_params['velocity_x'][dauid][eth_idx].append(vx)
                            pde_params['velocity_y'][dauid][eth_idx].append(vy)
                            pde_params['velocity_magnitude'][dauid][eth_idx].append(magnitude)
                            
                            if eth_idx + 1 > max_ethnicities:
                                max_ethnicities = eth_idx + 1
        
        # If no ethnicity names were provided, create default ones
        if ethnicity_names is None or len(ethnicity_names) < max_ethnicities:
            ethnicity_names = [f'Ethnicity_{i}' for i in range(max_ethnicities)]
        
        # Calculate averages for each parameter by DAUID and ethnicity
        avg_params = {
            'diffusion': defaultdict(dict),
            'amplification': defaultdict(dict),
            'velocity_x': defaultdict(dict),
            'velocity_y': defaultdict(dict),
            'velocity_magnitude': defaultdict(dict)
        }
        
        for param_name in avg_params.keys():
            for dauid in pde_params[param_name]:
                for eth_idx in pde_params[param_name][dauid]:
                    if pde_params[param_name][dauid][eth_idx]:  # Only calculate if there's data
                        avg_params[param_name][dauid][eth_idx] = np.mean(pde_params[param_name][dauid][eth_idx])
        
        # Print some statistics about the data
        for param_name in avg_params.keys():
            all_values = []
            for dauid in avg_params[param_name]:
                for eth_idx in avg_params[param_name][dauid]:
                    all_values.append(avg_params[param_name][dauid][eth_idx])
            
            if all_values:
                print(f"\n{param_name.capitalize()} statistics:")
                print(f"  Number of values: {len(all_values)}")
                print(f"  Min: {min(all_values):.6f}")
                print(f"  Max: {max(all_values):.6f}")
                print(f"  Mean: {np.mean(all_values):.6f}")
                print(f"  Median: {np.median(all_values):.6f}")
                print(f"  5th percentile: {np.percentile(all_values, 5):.6f}")
                print(f"  95th percentile: {np.percentile(all_values, 95):.6f}")
        
        # Create visualizations for each ethnicity and parameter
        for eth_idx in range(max_ethnicities):
            eth_name = ethnicity_names[eth_idx] if eth_idx < len(ethnicity_names) else f'Ethnicity_{eth_idx}'
            
            # Create visualizations for each parameter (focusing on diffusion and amplification)
            for param_name in ['diffusion', 'amplification', 'velocity_magnitude']:
                # Create a copy of the GeoDataFrame for this visualization
                plot_gdf = gdf.copy()
                
                # Add parameter values to GeoDataFrame
                plot_gdf[param_name] = np.nan
                
                # Fill in values from our aggregated data
                for dauid in avg_params[param_name]:
                    if eth_idx in avg_params[param_name][dauid]:
                        value = avg_params[param_name][dauid][eth_idx]
                        
                        # Find matching rows in GeoDataFrame
                        matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                        if not matching_rows.empty:
                            plot_gdf.loc[matching_rows.index, param_name] = value
                
                # Create the percentile-focused visualizations (using our new function)
                visualize_pde_parameters_percentile_focused(
                    plot_gdf=plot_gdf,
                    param_name=param_name,
                    eth_name=eth_name,
                    dauid_column=dauid_column,
                    output_dir=percentile_pde_dir,
                    param_data=avg_params[param_name]
                )
                
                print(f"Created percentile-based {param_name} visualizations for {eth_name}")
                
            # Create velocity vector field visualization with improvements
            plot_gdf = gdf.copy()
            
            # Add centroid coordinates for plotting vectors
            plot_gdf['centroid_x'] = plot_gdf.geometry.centroid.x
            plot_gdf['centroid_y'] = plot_gdf.geometry.centroid.y
            
            # Add velocity components
            plot_gdf['velocity_x'] = np.nan
            plot_gdf['velocity_y'] = np.nan
            plot_gdf['velocity_magnitude'] = np.nan
            
            # Fill in values from our aggregated data
            for dauid in avg_params['velocity_x']:
                if eth_idx in avg_params['velocity_x'][dauid] and eth_idx in avg_params['velocity_y'][dauid]:
                    vx = avg_params['velocity_x'][dauid][eth_idx]
                    vy = avg_params['velocity_y'][dauid][eth_idx]
                    magnitude = avg_params['velocity_magnitude'][dauid][eth_idx]
                    
                    # Find matching rows in GeoDataFrame
                    matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                    if not matching_rows.empty:
                        plot_gdf.loc[matching_rows.index, 'velocity_x'] = vx
                        plot_gdf.loc[matching_rows.index, 'velocity_y'] = vy
                        plot_gdf.loc[matching_rows.index, 'velocity_magnitude'] = magnitude
            
            # Create improved velocity visualization
            create_improved_velocity_visualization(
                plot_gdf=plot_gdf,
                eth_name=eth_name,
                dauid_column=dauid_column,
                output_dir=percentile_pde_dir
            )
            
            # Create interactive HTML visualizations if folium is available
            try:
                import folium
                for param_name in ['diffusion', 'amplification']:
                    html_path = create_interactive_visualization(
                        avg_params=avg_params,
                        param_name=param_name,
                        eth_idx=eth_idx,
                        eth_name=eth_name,
                        gdf=gdf,
                        dauid_column=dauid_column,
                        output_dir=percentile_pde_dir
                    )
                    if html_path:
                        print(f"Created interactive {param_name} visualization for {eth_name}: {html_path}")
            except ImportError:
                print("Could not create interactive visualizations: folium package not available")
        
        print(f"Successfully created percentile-based PDE parameter map visualizations in {percentile_pde_dir}")
        return True
        
    except Exception as e:
        print(f"Error creating percentile-based PDE parameter map visualizations: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def visualize_pde_parameters_on_map(visualizer, interpretation_dir, shapefile_path, output_dir):
    """
    Create map-based visualizations for PDE parameters with improved scaling and normalization.
    This is an improved version of the original visualize_pde_parameters_on_map function.
    
    Args:
        visualizer: The ModelInterpretationVisualizer instance
        interpretation_dir (str): Directory containing interpretation JSON files
        shapefile_path (str): Path to the shapefile with DA boundaries
        output_dir (str): Directory to save the visualizations
        
    Returns:
        bool: Success status
    """
    try:
        # Create output directory for improved PDE map visualizations
        pde_map_dir = os.path.join(output_dir, "improved_pde_map_visualizations")
        os.makedirs(pde_map_dir, exist_ok=True)
        
        # Load shapefile
        print(f"Loading shapefile from {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            print(f"Loaded shapefile with {len(gdf)} geometries")
        except Exception as e:
            print(f"Error loading shapefile: {str(e)}")
            return False
        
        # Find DAUID column in shapefile
        dauid_column = None
        for column in gdf.columns:
            if column.upper() == 'DAUID' or 'DAUID' in column.upper():
                dauid_column = column
                break
        
        if dauid_column is None:
            print("DAUID column not found in shapefile, trying common ID field names")
            # Try common ID field names in shapefiles
            for column in ['ID', 'GEOID', 'AREA_ID', 'DA_ID', 'DISSEMINATION_AREA_ID']:
                if column in gdf.columns:
                    dauid_column = column
                    print(f"Using {dauid_column} as the DA identifier")
                    break
            
            if dauid_column is None:
                print("No suitable ID column found in shapefile")
                print(f"Available columns in shapefile: {list(gdf.columns)}")
                return False
        
        # Ensure DAUID is string type for consistent joining
        gdf[dauid_column] = gdf[dauid_column].astype(str)
        
        # Find all interpretation JSON files
        json_files = [f for f in os.listdir(interpretation_dir) 
                    if f.endswith('.json') and 'interpretation' in f]
        
        if not json_files:
            print(f"No interpretation files found in {interpretation_dir}")
            return False
        
        print(f"Found {len(json_files)} interpretation files")
        
        # Initialize data structures to store aggregated PDE parameters by DAUID and ethnicity
        pde_params = {
            'diffusion': defaultdict(lambda: defaultdict(list)),
            'amplification': defaultdict(lambda: defaultdict(list)),
            'velocity_x': defaultdict(lambda: defaultdict(list)),
            'velocity_y': defaultdict(lambda: defaultdict(list)),
            'velocity_magnitude': defaultdict(lambda: defaultdict(list))
        }
        
        # Maximum number of ethnicities encountered
        max_ethnicities = 0
        ethnicity_names = visualizer.ethnicity_names if hasattr(visualizer, 'ethnicity_names') and visualizer.ethnicity_names else None
        
        # Process each interpretation file
        for json_file in json_files:
            file_path = os.path.join(interpretation_dir, json_file)
            
            # Load interpretation data
            interpretation_data = visualizer.load_interpretation_data(file_path)
            if interpretation_data is None or 'pde_expert' not in interpretation_data:
                print(f"No PDE expert data found in {json_file}, skipping")
                continue
            
            pde_data = interpretation_data['pde_expert']
            
            # Process PDE parameters for each DAUID and ethnicity
            if 'diffusion' in pde_data and isinstance(pde_data['diffusion'], dict):
                for dauid, values in pde_data['diffusion'].items():
                    for eth_idx, value in enumerate(values):
                        pde_params['diffusion'][dauid][eth_idx].append(float(value))
                        if eth_idx + 1 > max_ethnicities:
                            max_ethnicities = eth_idx + 1
            
            if 'amplification' in pde_data and isinstance(pde_data['amplification'], dict):
                for dauid, values in pde_data['amplification'].items():
                    for eth_idx, value in enumerate(values):
                        pde_params['amplification'][dauid][eth_idx].append(float(value))
                        if eth_idx + 1 > max_ethnicities:
                            max_ethnicities = eth_idx + 1
            
            if 'velocity' in pde_data and isinstance(pde_data['velocity'], dict):
                for dauid, values in pde_data['velocity'].items():
                    for eth_idx, velocity in enumerate(values):
                        if len(velocity) >= 2:  # Ensure we have x and y components
                            vx = float(velocity[0])
                            vy = float(velocity[1])
                            magnitude = np.sqrt(vx**2 + vy**2)
                            
                            pde_params['velocity_x'][dauid][eth_idx].append(vx)
                            pde_params['velocity_y'][dauid][eth_idx].append(vy)
                            pde_params['velocity_magnitude'][dauid][eth_idx].append(magnitude)
                            
                            if eth_idx + 1 > max_ethnicities:
                                max_ethnicities = eth_idx + 1
        
        # If no ethnicity names were provided, create default ones
        if ethnicity_names is None or len(ethnicity_names) < max_ethnicities:
            ethnicity_names = [f'Ethnicity_{i}' for i in range(max_ethnicities)]
        
        # Calculate averages for each parameter by DAUID and ethnicity
        avg_params = {
            'diffusion': defaultdict(dict),
            'amplification': defaultdict(dict),
            'velocity_x': defaultdict(dict),
            'velocity_y': defaultdict(dict),
            'velocity_magnitude': defaultdict(dict)
        }
        
        for param_name in avg_params.keys():
            for dauid in pde_params[param_name]:
                for eth_idx in pde_params[param_name][dauid]:
                    if pde_params[param_name][dauid][eth_idx]:  # Only calculate if there's data
                        avg_params[param_name][dauid][eth_idx] = np.mean(pde_params[param_name][dauid][eth_idx])
        
        # Print some statistics about the data
        for param_name in avg_params.keys():
            all_values = []
            for dauid in avg_params[param_name]:
                for eth_idx in avg_params[param_name][dauid]:
                    all_values.append(avg_params[param_name][dauid][eth_idx])
            
            if all_values:
                print(f"\n{param_name.capitalize()} statistics:")
                print(f"  Number of values: {len(all_values)}")
                print(f"  Min: {min(all_values):.6f}")
                print(f"  Max: {max(all_values):.6f}")
                print(f"  Mean: {np.mean(all_values):.6f}")
                print(f"  Median: {np.median(all_values):.6f}")
                print(f"  5th percentile: {np.percentile(all_values, 5):.6f}")
                print(f"  95th percentile: {np.percentile(all_values, 95):.6f}")
        
        # Create visualizations for each ethnicity and parameter
        for eth_idx in range(max_ethnicities):
            eth_name = ethnicity_names[eth_idx] if eth_idx < len(ethnicity_names) else f'Ethnicity_{eth_idx}'
            
            # Create visualizations for each parameter
            for param_name in ['diffusion', 'amplification', 'velocity_magnitude']:
                # Create a copy of the GeoDataFrame for this visualization
                plot_gdf = gdf.copy()
                
                # Add parameter values to GeoDataFrame
                plot_gdf[param_name] = np.nan
                
                # Fill in values from our aggregated data
                for dauid in avg_params[param_name]:
                    if eth_idx in avg_params[param_name][dauid]:
                        value = avg_params[param_name][dauid][eth_idx]
                        
                        # Find matching rows in GeoDataFrame
                        matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                        if not matching_rows.empty:
                            plot_gdf.loc[matching_rows.index, param_name] = value
                
                # Create the improved visualizations
                visualize_pde_parameters_with_improved_scale(
                    plot_gdf=plot_gdf,
                    param_name=param_name,
                    eth_name=eth_name,
                    dauid_column=dauid_column,
                    output_dir=pde_map_dir,
                    param_data=avg_params[param_name]
                )
                
                print(f"Created improved {param_name} visualizations for {eth_name}")
                
            # Create velocity vector field visualization with improvements
            plot_gdf = gdf.copy()
            
            # Add centroid coordinates for plotting vectors
            plot_gdf['centroid_x'] = plot_gdf.geometry.centroid.x
            plot_gdf['centroid_y'] = plot_gdf.geometry.centroid.y
            
            # Add velocity components
            plot_gdf['velocity_x'] = np.nan
            plot_gdf['velocity_y'] = np.nan
            plot_gdf['velocity_magnitude'] = np.nan
            
            # Fill in values from our aggregated data
            for dauid in avg_params['velocity_x']:
                if eth_idx in avg_params['velocity_x'][dauid] and eth_idx in avg_params['velocity_y'][dauid]:
                    vx = avg_params['velocity_x'][dauid][eth_idx]
                    vy = avg_params['velocity_y'][dauid][eth_idx]
                    magnitude = avg_params['velocity_magnitude'][dauid][eth_idx]
                    
                    # Find matching rows in GeoDataFrame
                    matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                    if not matching_rows.empty:
                        plot_gdf.loc[matching_rows.index, 'velocity_x'] = vx
                        plot_gdf.loc[matching_rows.index, 'velocity_y'] = vy
                        plot_gdf.loc[matching_rows.index, 'velocity_magnitude'] = magnitude
            
            # Drop rows with missing velocity data
            vector_gdf = plot_gdf.dropna(subset=['velocity_x', 'velocity_y'])
            
            # if not vector_gdf.empty:
               
                
            create_improved_velocity_visualization(
                plot_gdf=plot_gdf,  # Your GeoDataFrame with velocity_x, velocity_y columns
                eth_name=eth_name,
                dauid_column=dauid_column,
                output_dir=pde_map_dir
            )
        
        print(f"Successfully created improved PDE parameter map visualizations in {pde_map_dir}")
        return True
        
    except Exception as e:
        print(f"Error creating improved PDE parameter map visualizations: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

# Function to create interactive HTML visualizations
def create_interactive_visualization(avg_params, param_name, eth_idx, eth_name, gdf, dauid_column, output_dir):
    """
    Create an interactive HTML visualization for a specific parameter and ethnicity.
    
    Args:
        avg_params (dict): Dictionary with average parameter values by DAUID and ethnicity
        param_name (str): Name of the parameter to visualize
        eth_idx (int): Index of the ethnicity
        eth_name (str): Name of the ethnicity
        gdf (GeoDataFrame): GeoDataFrame with geometry
        dauid_column (str): Name of the column containing DAUID values
        output_dir (str): Directory to save the visualization
        
    Returns:
        str: Path to the saved HTML file
    """
    # Make a copy of the GeoDataFrame to avoid modifying the original
    plot_gdf = gdf.copy()
    
    # Add parameter values to GeoDataFrame
    plot_gdf[param_name] = np.nan
    
    # Fill in values from our aggregated data
    for dauid in avg_params[param_name]:
        if eth_idx in avg_params[param_name][dauid]:
            value = avg_params[param_name][dauid][eth_idx]
            
            # Find matching rows in GeoDataFrame
            matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
            if not matching_rows.empty:
                plot_gdf.loc[matching_rows.index, param_name] = value
    
    # Create the interactive map
    try:
        import folium
        from folium.plugins import FloatImage
        from folium.branca.colormap import LinearColormap # type: ignore
        import branca.colormap as cm
        import io
        import base64
        
        # Create a centroid point for the map center
        center_lat = plot_gdf.geometry.centroid.y.mean()
        center_lon = plot_gdf.geometry.centroid.x.mean()
        
        # Create a folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Get the parameter values
        values = plot_gdf[param_name].dropna()
        
        if not values.empty:
            # Create a custom colormap
            if param_name == 'diffusion':
                colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
                cmap_name = 'Blues'
            elif param_name == 'amplification':
                colors = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
                cmap_name = 'Oranges'
            else:  # velocity_magnitude
                colors = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
                cmap_name = 'Reds'
            
            # Get min and max values for colormap
            vmin = values.min()
            vmax = values.max()
            
            # Create a colormap with percentile-based scaling for better visibility
            p05 = np.percentile(values, 5)
            p95 = np.percentile(values, 95)
            
            # Create a colormap
            colormap = cm.LinearColormap(
                colors=colors,
                vmin=p05,
                vmax=p95,
                caption=f'{param_name.capitalize()} (5th-95th percentile: {p05:.6f}-{p95:.6f})'
            )
            
            # Add the colormap to the map
            colormap.add_to(m)
            
            # Define style function for GeoJSON
            def style_function(feature):
                value = feature['properties'][param_name]
                if pd.isna(value):
                    return {
                        'fillColor': 'lightgrey',
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.5
                    }
                else:
                    return {
                        'fillColor': colormap(value),
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.8
                    }
            
            # Define tooltip function
            def tooltip_function(feature):
                dauid = feature['properties'][dauid_column]
                value = feature['properties'][param_name]
                if pd.isna(value):
                    return folium.Tooltip(f"DAUID: {dauid}<br>Value: No data")
                else:
                    return folium.Tooltip(f"DAUID: {dauid}<br>{param_name.capitalize()}: {value:.6f}")
            
            # Add GeoJSON data to the map
            folium.GeoJson(
                plot_gdf,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=[dauid_column, param_name],
                    aliases=["DAUID:", f"{param_name.capitalize()}:"],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
                    localize=True,
                    sticky=True,
                    labels=True,
                    max_width=300,
                ),
                name=f"{param_name.capitalize()} for {eth_name}"
            ).add_to(m)
            
            # Add a title
            title_html = f'<h3 align="center" style="font-size:16px"><b>{param_name.capitalize()} for {eth_name}</b></h3>'
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save the map
            save_path = os.path.join(output_dir, f'interactive_pde_{param_name}_{eth_name.replace(" ", "_")}.html')
            m.save(save_path)
            
            return save_path
    except ImportError:
        print("Could not create interactive visualization, folium package is required")
        return None
    except Exception as e:
        print(f"Error creating interactive visualization: {str(e)}")
        return None