import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import logging

from pde_map_visualization import update_visualize_pde_parameters_on_map, visualize_pde_parameters_on_map
from shapefile_visualization import visualize_spatial_patterns_with_shapefile

logger = logging.getLogger(__name__)

class ModelInterpretationVisualizer:
    """
    Class to create visualizations for model interpretations across different experts
    """
    def __init__(self, output_dir, ethnicity_names=None, map_dauid_to_coords=None):
        """
        Initialize the visualizer with output directory and optional ethnicity names
        
        Args:
            output_dir (str): Directory to save visualizations
            ethnicity_names (list, optional): List of ethnicity names
            map_dauid_to_coords (dict, optional): Mapping from DAUID to spatial coordinates
        """
        self.output_dir = output_dir
        self.ethnicity_names = ethnicity_names
        self.map_dauid_to_coords = map_dauid_to_coords
        
        # Create directories for different visualization types
        self.expert_dir = os.path.join(output_dir, "expert_visualizations")
        self.comparison_dir = os.path.join(output_dir, "comparison_visualizations")
        self.spatial_dir = os.path.join(output_dir, "spatial_visualizations")
        self.feature_dir = os.path.join(output_dir, "feature_visualizations")
        
        os.makedirs(self.expert_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        os.makedirs(self.spatial_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # Set default plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.set_plot_style()
        
        # Define color palettes for different expert types
        self.expert_colors = {
            'colonization': '#4CAF50',  # Green
            'jump': '#2196F3',          # Blue
            'decline': '#F44336',       # Red
            'pde': '#9C27B0'            # Purple
        }
        
        # Define basic color palette for ethnicities
        self.ethnicity_colors = plt.cm.tab10.colors
        
    def set_plot_style(self):
        """Set consistent style for all visualizations"""
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.figsize': (12, 8),
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,
        })

    def load_interpretation_data(self, interpretation_path):
        """
        Load interpretation data from JSON file
        
        Args:
            interpretation_path (str): Path to the interpretation JSON file
            
        Returns:
            dict: Loaded interpretation data
        """
        try:
            with open(interpretation_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded interpretation data from {interpretation_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading interpretation data: {str(e)}")
            return None

    def visualize_colonization_expert(self, interpretation_data, batch_idx=0):
        """
        Create visualizations for the colonization expert
        
        Args:
            interpretation_data (dict): Colonization expert interpretation data
            batch_idx (int): Batch index for saving
            
        Returns:
            bool: Success status
        """
        try:
            if 'colonization_expert' not in interpretation_data:
                logger.warning("No colonization expert data found in interpretation data")
                return False
            
            colonization_data = interpretation_data['colonization_expert']
            
            # 1. Feature importance visualization
            if 'feature_importance' in colonization_data:
                feature_importance = colonization_data['feature_importance']
                
                self.visualize_feature_importance(
                    feature_importance=feature_importance,
                    expert_name='colonization',
                    batch_idx=batch_idx,
                    top_n=20,
                    category_mapping=None  # Add category mapping if available
                )
                
            # 2. Size probabilities and predictions by ethnicity
            if 'size_probabilities' in colonization_data and isinstance(colonization_data['size_probabilities'], dict):
                # Extract data for a few sample DAUIDs
                dauid_keys = list(colonization_data['size_probabilities'].keys())
                sample_keys = dauid_keys[:min(5, len(dauid_keys))]
                
                # For each sample, create visualization
                for dauid in sample_keys:
                    # Get data for this DAUID
                    size_probs = np.array(colonization_data['size_probabilities'][dauid])
                    eth_importance = np.array(colonization_data['ethnicity_importance'][dauid])
                    
                    # Create subplot grid
                    fig = plt.figure(figsize=(15, 10))
                    gs = GridSpec(2, 2, figure=fig)
                    
                    # Size probabilities by ethnicity
                    ax1 = fig.add_subplot(gs[0, 0])
                    ethnicity_indices = np.arange(size_probs.shape[0])
                    
                    # Use ethnicity names if available
                    x_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in ethnicity_indices]
                    
                    for i, size_name in enumerate(['Small', 'Medium', 'Large', 'Very Large']):
                        bottom = np.sum(size_probs[:, :i], axis=1) if i > 0 else np.zeros(size_probs.shape[0])
                        ax1.bar(ethnicity_indices, size_probs[:, i], bottom=bottom, 
                                label=size_name, alpha=0.7)
                    
                    ax1.set_xticks(ethnicity_indices)
                    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax1.set_ylabel('Probability')
                    ax1.set_title(f'Colonization Size Probabilities by Ethnicity (DAUID: {dauid})')
                    ax1.legend()
                    
                    # Ethnicity importance
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.bar(ethnicity_indices, eth_importance, 
                            color=[self.ethnicity_colors[i % len(self.ethnicity_colors)] 
                                   for i in range(len(eth_importance))], alpha=0.7)
                    ax2.set_xticks(ethnicity_indices)
                    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax2.set_ylabel('Importance Score')
                    ax2.set_title(f'Ethnicity Importance for Colonization (DAUID: {dauid})')
                    
                    # Predicted colonization values by size
                    ax3 = fig.add_subplot(gs[1, 0:])
                    
                    # Get prediction data
                    small_preds = np.array(colonization_data['predictor_outputs']['small'][dauid])
                    medium_preds = np.array(colonization_data['predictor_outputs']['medium'][dauid])
                    large_preds = np.array(colonization_data['predictor_outputs']['large'][dauid])
                    very_large_preds = np.array(colonization_data['predictor_outputs']['very_large'][dauid])
                    
                    # Create the grouped bar chart
                    x = np.arange(len(small_preds))
                    width = 0.2
                    
                    ax3.bar(x - 1.5*width, small_preds, width, label='Small', color='skyblue')
                    ax3.bar(x - 0.5*width, medium_preds, width, label='Medium', color='lightgreen')
                    ax3.bar(x + 0.5*width, large_preds, width, label='Large', color='salmon')
                    ax3.bar(x + 1.5*width, very_large_preds, width, label='Very Large', color='purple')
                    
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax3.set_ylabel('Predicted Population')
                    ax3.set_title(f'Colonization Predictions by Size Category (DAUID: {dauid})')
                    ax3.legend()
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.expert_dir, 
                                           f'colonization_detailed_dauid_{dauid}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                
            # 3. Parameters visualization
            if 'scales' in colonization_data:
                scales_data = colonization_data['scales']
                
                # Create a bar chart for all scale parameters
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                # Small scale
                small_scale = scales_data['small_scale']
                axes[0].bar(np.arange(len(small_scale)), small_scale, color='skyblue')
                axes[0].set_title('Small Colonization Scale by Ethnicity')
                if self.ethnicity_names:
                    axes[0].set_xticks(np.arange(len(small_scale)))
                    axes[0].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                # Medium scale
                medium_scale = scales_data['medium_scale']
                axes[1].bar(np.arange(len(medium_scale)), medium_scale, color='lightgreen')
                axes[1].set_title('Medium Colonization Scale by Ethnicity')
                if self.ethnicity_names:
                    axes[1].set_xticks(np.arange(len(medium_scale)))
                    axes[1].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                # Large scale
                large_scale = scales_data['large_scale']
                axes[2].bar(np.arange(len(large_scale)), large_scale, color='salmon')
                axes[2].set_title('Large Colonization Scale by Ethnicity')
                if self.ethnicity_names:
                    axes[2].set_xticks(np.arange(len(large_scale)))
                    axes[2].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                # Very large scale
                very_large_scale = scales_data['very_large_scale']
                axes[3].bar(np.arange(len(very_large_scale)), very_large_scale, color='purple')
                axes[3].set_title('Very Large Colonization Scale by Ethnicity')
                if self.ethnicity_names:
                    axes[3].set_xticks(np.arange(len(very_large_scale)))
                    axes[3].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                plt.tight_layout()
                save_path = os.path.join(self.expert_dir, f'colonization_scales_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
                
                # Create a bar chart for all bias parameters
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                # Small bias
                small_bias = scales_data['small_bias']
                axes[0].bar(np.arange(len(small_bias)), small_bias, color='skyblue')
                axes[0].set_title('Small Colonization Bias by Ethnicity')
                if self.ethnicity_names:
                    axes[0].set_xticks(np.arange(len(small_bias)))
                    axes[0].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                # Medium bias
                medium_bias = scales_data['medium_bias']
                axes[1].bar(np.arange(len(medium_bias)), medium_bias, color='lightgreen')
                axes[1].set_title('Medium Colonization Bias by Ethnicity')
                if self.ethnicity_names:
                    axes[1].set_xticks(np.arange(len(medium_bias)))
                    axes[1].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                # Large bias
                large_bias = scales_data['large_bias']
                axes[2].bar(np.arange(len(large_bias)), large_bias, color='salmon')
                axes[2].set_title('Large Colonization Bias by Ethnicity')
                if self.ethnicity_names:
                    axes[2].set_xticks(np.arange(len(large_bias)))
                    axes[2].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                # Very large bias
                very_large_bias = scales_data['very_large_bias']
                axes[3].bar(np.arange(len(very_large_bias)), very_large_bias, color='purple')
                axes[3].set_title('Very Large Colonization Bias by Ethnicity')
                if self.ethnicity_names:
                    axes[3].set_xticks(np.arange(len(very_large_bias)))
                    axes[3].set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                plt.tight_layout()
                save_path = os.path.join(self.expert_dir, f'colonization_biases_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
            
            logger.info(f"Created colonization expert visualizations for batch {batch_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating colonization expert visualizations: {str(e)}")
            return False

    def visualize_jump_expert(self, interpretation_data, batch_idx=0):
        """
        Create visualizations for the jump expert
        
        Args:
            interpretation_data (dict): Jump expert interpretation data
            batch_idx (int): Batch index for saving
            
        Returns:
            bool: Success status
        """
        try:
            if 'jump_expert' not in interpretation_data:
                logger.warning("No jump expert data found in interpretation data")
                return False
            
            jump_data = interpretation_data['jump_expert']
            
            # 1. Feature importance visualization
            if 'feature_importance' in jump_data:
                feature_importance = jump_data['feature_importance']
                
                self.visualize_feature_importance(
                    feature_importance=feature_importance,
                    expert_name='jump',
                    batch_idx=batch_idx,
                    top_n=20,
                    category_mapping=None  # Add category mapping if available
                )
            
            # 2. Jump parameters visualization
            if 'parameters' in jump_data:
                params = jump_data['parameters']
                
                # Plot ethnicity tendency (some ethnicities more likely to increase/decrease)
                if 'ethnicity_tendency' in params:
                    ethnicity_tendency = params['ethnicity_tendency']
                    
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(np.arange(len(ethnicity_tendency)), ethnicity_tendency, 
                                   color=self.expert_colors['jump'], alpha=0.7)
                    
                    # Color positive and negative values differently
                    for i, v in enumerate(ethnicity_tendency):
                        if v < 0:
                            bars[i].set_color('red')
                        else:
                            bars[i].set_color('green')
                    
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Set labels
                    x_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in range(len(ethnicity_tendency))]
                    plt.xticks(np.arange(len(ethnicity_tendency)), x_labels, rotation=45, ha='right')
                    plt.ylabel('Tendency Score')
                    plt.title('Jump Expert - Ethnicity Jump Tendency (Positive = Grow, Negative = Decline)')
                    plt.tight_layout()
                    
                    save_path = os.path.join(self.expert_dir, f'jump_ethnicity_tendency_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                
                # Plot extreme jump base scaler
                if 'extreme_jump_base_scaler' in params:
                    extreme_scaler = params['extreme_jump_base_scaler']
                    
                    plt.figure(figsize=(12, 6))
                    plt.bar(np.arange(len(extreme_scaler)), extreme_scaler, color='purple', alpha=0.7)
                    x_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in range(len(extreme_scaler))]
                    plt.xticks(np.arange(len(extreme_scaler)), x_labels, rotation=45, ha='right')
                    plt.ylabel('Scale Factor')
                    plt.title('Jump Expert - Extreme Jump Base Scaler by Ethnicity')
                    plt.tight_layout()
                    
                    save_path = os.path.join(self.expert_dir, f'jump_extreme_scaler_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
            
            # 3. Per-DAUID predictions
            if 'predictions' in jump_data and isinstance(jump_data['predictions']['jump_probabilities'], dict):
                # Extract data for a few sample DAUIDs
                dauid_keys = list(jump_data['predictions']['jump_probabilities'].keys())
                sample_keys = dauid_keys[:min(5, len(dauid_keys))]
                
                # For each sample, create visualization
                for dauid in sample_keys:
                    # Get data for this DAUID
                    jump_probs = np.array(jump_data['predictions']['jump_probabilities'][dauid])
                    jump_mags = np.array(jump_data['predictions']['jump_magnitudes'][dauid])
                    ext_probs = np.array(jump_data['predictions']['extreme_jump_probabilities'][dauid])
                    ext_mags = np.array(jump_data['predictions']['extreme_jump_magnitudes'][dauid])
                    final_preds = np.array(jump_data['predictions']['final_predictions'][dauid])
                    
                    # Ethnicity interaction data
                    eth_interactions = np.array(jump_data['predictions']['ethnicity_interactions'][dauid])
                    
                    # Create subplots
                    fig = plt.figure(figsize=(15, 18))
                    gs = GridSpec(4, 2, figure=fig)
                    
                    # Jump probabilities
                    ax1 = fig.add_subplot(gs[0, 0])
                    x = np.arange(len(jump_probs))
                    ax1.bar(x, jump_probs, color=self.expert_colors['jump'], alpha=0.7)
                    x_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in x]
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax1.set_ylabel('Probability')
                    ax1.set_title(f'Jump Probabilities by Ethnicity (DAUID: {dauid})')
                    
                    # Jump magnitudes
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.bar(x, jump_mags, color=self.expert_colors['jump'], alpha=0.7)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax2.set_ylabel('Magnitude')
                    ax2.set_title(f'Jump Magnitudes by Ethnicity (DAUID: {dauid})')
                    
                    # Extreme jump probabilities
                    ax3 = fig.add_subplot(gs[1, 0])
                    ax3.bar(x, ext_probs, color='purple', alpha=0.7)
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax3.set_ylabel('Probability')
                    ax3.set_title(f'Extreme Jump Probabilities by Ethnicity (DAUID: {dauid})')
                    
                    # Extreme jump magnitudes
                    ax4 = fig.add_subplot(gs[1, 1])
                    ax4.bar(x, ext_mags, color='purple', alpha=0.7)
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax4.set_ylabel('Magnitude')
                    ax4.set_title(f'Extreme Jump Magnitudes by Ethnicity (DAUID: {dauid})')
                    
                    # Final predictions
                    ax5 = fig.add_subplot(gs[2, :])
                    ax5.bar(x, final_preds, color='green', alpha=0.7)
                    ax5.set_xticks(x)
                    ax5.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax5.set_ylabel('Predicted Population')
                    ax5.set_title(f'Final Jump Expert Predictions by Ethnicity (DAUID: {dauid})')
                    
                    # Ethnicity interactions as heatmap
                    ax6 = fig.add_subplot(gs[3, :])
                    im = ax6.imshow(eth_interactions, cmap='coolwarm', vmin=0, vmax=1)
                    ax6.set_xlabel('Source Ethnicity')
                    ax6.set_ylabel('Target Ethnicity')
                    ax6.set_title(f'Ethnicity Interaction Weights (DAUID: {dauid})')
                    ax6.set_xticks(np.arange(len(x_labels)))
                    ax6.set_yticks(np.arange(len(x_labels)))
                    ax6.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax6.set_yticklabels(x_labels)
                    plt.colorbar(im, ax=ax6, label='Interaction Strength')
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.expert_dir, 
                                           f'jump_detailed_dauid_{dauid}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                    
                    # Feature-based jump classifications (separate plot)
                    if 'feature_jump_classes' in jump_data['predictions']:
                        feat_classes = np.array(jump_data['predictions']['feature_jump_classes'][dauid])
                        feat_scales = np.array(jump_data['predictions']['feature_jump_scales'][dauid])
                        
                        # Create plot
                        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
                        
                        # Jump classes as stacked bar
                        bottom = np.zeros(feat_classes.shape[0])
                        for i, class_name in enumerate(['No Jump', 'Moderate Jump', 'Extreme Jump']):
                            axes[0].bar(x, feat_classes[:, i], bottom=bottom, 
                                      label=class_name, alpha=0.7)
                            bottom += feat_classes[:, i]
                        
                        axes[0].set_xticks(x)
                        axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
                        axes[0].set_ylabel('Probability')
                        axes[0].set_title(f'Feature-Based Jump Classification by Ethnicity (DAUID: {dauid})')
                        axes[0].legend()
                        
                        # Jump scales as grouped bar
                        width = 0.25
                        for i, scale_name in enumerate(['No Jump Scale', 'Moderate Jump Scale', 'Extreme Jump Scale']):
                            axes[1].bar(x + (i-1)*width, feat_scales[:, i], width, 
                                      label=scale_name, alpha=0.7)
                        
                        axes[1].set_xticks(x)
                        axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
                        axes[1].set_ylabel('Scale Factor')
                        axes[1].set_title(f'Feature-Based Jump Scaling by Ethnicity (DAUID: {dauid})')
                        axes[1].legend()
                        
                        plt.tight_layout()
                        save_path = os.path.join(self.expert_dir, 
                                               f'jump_feature_classes_dauid_{dauid}_batch_{batch_idx}.png')
                        plt.savefig(save_path)
                        plt.close()
            
            logger.info(f"Created jump expert visualizations for batch {batch_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating jump expert visualizations: {str(e)}")
            return False

    def visualize_decline_expert(self, interpretation_data, batch_idx=0):
        """
        Create visualizations for the decline expert
        
        Args:
            interpretation_data (dict): Decline expert interpretation data
            batch_idx (int): Batch index for saving
            
        Returns:
            bool: Success status
        """
        try:
            if 'decline_expert' not in interpretation_data:
                logger.warning("No decline expert data found in interpretation data")
                return False
            
            decline_data = interpretation_data['decline_expert']
            
            # 1. Feature importance visualization
            if 'feature_importance' in decline_data:
                feature_importance = decline_data['feature_importance']
                
                self.visualize_feature_importance(
                    feature_importance=feature_importance,
                    expert_name='decline',
                    batch_idx=batch_idx,
                    top_n=20,
                    category_mapping=None  # Add category mapping if available
                )
            
            # 2. Decline parameters visualization
            if 'parameters' in decline_data:
                params = decline_data['parameters']
                
                # Plot ethnicity decline tendency
                if 'ethnicity_decline_tendency' in params:
                    decline_tendency = params['ethnicity_decline_tendency']
                    
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(np.arange(len(decline_tendency)), decline_tendency, 
                                  color=self.expert_colors['decline'], alpha=0.7)
                    
                    # Set labels
                    x_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in range(len(decline_tendency))]
                    plt.xticks(np.arange(len(decline_tendency)), x_labels, rotation=45, ha='right')
                    plt.ylabel('Decline Tendency')
                    plt.title('Decline Expert - Ethnicity Decline Tendency')
                    plt.tight_layout()
                    
                    save_path = os.path.join(self.expert_dir, f'decline_ethnicity_tendency_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
            
            # 3. Per-DAUID predictions
            if 'predictions' in decline_data and isinstance(decline_data['predictions']['decline_probabilities'], dict):
                # Extract data for a few sample DAUIDs
                dauid_keys = list(decline_data['predictions']['decline_probabilities'].keys())
                sample_keys = dauid_keys[:min(5, len(dauid_keys))]
                
                # For each sample, create visualization
                for dauid in sample_keys:
                    # Get data for this DAUID
                    decline_probs = np.array(decline_data['predictions']['decline_probabilities'][dauid])
                    decline_factors = np.array(decline_data['predictions']['decline_factors'][dauid])
                    feature_risk = np.array(decline_data['predictions']['feature_risk'][dauid])
                    feature_severity = np.array(decline_data['predictions']['feature_severity'][dauid])
                    extreme_prob = np.array(decline_data['predictions']['extreme_decline_probability'][dauid])
                    sensitivity = np.array(decline_data['predictions']['population_sensitivity'][dauid])
                    final_preds = np.array(decline_data['predictions']['final_predictions'][dauid])
                    
                    # Ethnicity interaction data
                    eth_interactions = np.array(decline_data['predictions']['ethnicity_interaction_weights'][dauid])
                    
                    # Create subplots
                    fig = plt.figure(figsize=(15, 18))
                    gs = GridSpec(4, 2, figure=fig)
                    
                    # Decline probabilities
                    ax1 = fig.add_subplot(gs[0, 0])
                    x = np.arange(len(decline_probs))
                    ax1.bar(x, decline_probs, color=self.expert_colors['decline'], alpha=0.7)
                    x_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in x]
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax1.set_ylabel('Probability')
                    ax1.set_title(f'Decline Probabilities by Ethnicity (DAUID: {dauid})')
                    
                    # Decline factors (how much is retained, 0-1)
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.bar(x, decline_factors, color=self.expert_colors['decline'], alpha=0.7)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax2.set_ylabel('Retention Factor')
                    ax2.set_title(f'Decline Retention Factors by Ethnicity (DAUID: {dauid})')
                    
                    # Feature-based decline risk
                    ax3 = fig.add_subplot(gs[1, 0])
                    ax3.bar(x, feature_risk, color='darkred', alpha=0.7)
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax3.set_ylabel('Risk Score')
                    ax3.set_title(f'Feature-Based Decline Risk by Ethnicity (DAUID: {dauid})')
                    
                    # Feature-based decline severity (0 = total, 1 = none)
                    ax4 = fig.add_subplot(gs[1, 1])
                    ax4.bar(x, feature_severity, color='darkred', alpha=0.7)
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax4.set_ylabel('Severity (0 = worst)')
                    ax4.set_title(f'Feature-Based Decline Severity by Ethnicity (DAUID: {dauid})')
                    
                    # Extreme decline probability and population sensitivity
                    ax5 = fig.add_subplot(gs[2, 0])
                    width = 0.4
                    ax5.bar(x - width/2, extreme_prob, width, color='black', alpha=0.7, label='Extreme Decline Probability')
                    ax5.bar(x + width/2, sensitivity, width, color='blue', alpha=0.7, label='Population Sensitivity')
                    ax5.set_xticks(x)
                    ax5.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax5.set_ylabel('Probability')
                    ax5.set_title(f'Extreme Decline Probability & Sensitivity (DAUID: {dauid})')
                    ax5.legend()
                    
                    # Final predictions
                    ax6 = fig.add_subplot(gs[2, 1])
                    ax6.bar(x, final_preds, color='green', alpha=0.7)
                    ax6.set_xticks(x)
                    ax6.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax6.set_ylabel('Predicted Population')
                    ax6.set_title(f'Final Decline Expert Predictions by Ethnicity (DAUID: {dauid})')
                    
                    # Ethnicity interactions as heatmap
                    ax7 = fig.add_subplot(gs[3, :])
                    im = ax7.imshow(eth_interactions, cmap='Reds', vmin=0, vmax=1)
                    ax7.set_xlabel('Source Ethnicity')
                    ax7.set_ylabel('Target Ethnicity')
                    ax7.set_title(f'Ethnicity Interaction Weights (DAUID: {dauid})')
                    ax7.set_xticks(np.arange(len(x_labels)))
                    ax7.set_yticks(np.arange(len(x_labels)))
                    ax7.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax7.set_yticklabels(x_labels)
                    plt.colorbar(im, ax=ax7, label='Interaction Strength')
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.expert_dir, 
                                           f'decline_detailed_dauid_{dauid}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
            
            logger.info(f"Created decline expert visualizations for batch {batch_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating decline expert visualizations: {str(e)}")
            return False

    def visualize_pde_expert(self, interpretation_data, batch_idx=0):
        """
        Create visualizations for the PDE expert
        
        Args:
            interpretation_data (dict): PDE expert interpretation data
            batch_idx (int): Batch index for saving
            
        Returns:
            bool: Success status
        """
        try:
            if 'pde_expert' not in interpretation_data:
                logger.warning("No PDE expert data found in interpretation data")
                return False
            
            pde_data = interpretation_data['pde_expert']
            
            # 1. Global parameters visualization
            global_params = {
                'K': pde_data.get('K', 0),
                'sigma_squared': pde_data.get('sigma_squared', 0),
                'alpha_0': pde_data.get('alpha_0', 0),
                'beta_0': pde_data.get('beta_0', 0),
                'mass_conservation_error': pde_data.get('mass_conservation_error', 0),
                'energy_conservation_error': pde_data.get('energy_conservation_error', 0)
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Key PDE parameters
            param_names = ['K', 'sigma_squared', 'alpha_0', 'beta_0']
            param_values = [global_params[p] for p in param_names]
            
            ax1.bar(np.arange(len(param_names)), param_values, color=self.expert_colors['pde'], alpha=0.7)
            ax1.set_xticks(np.arange(len(param_names)))
            ax1.set_xticklabels(param_names)
            ax1.set_ylabel('Parameter Value')
            ax1.set_title('PDE Expert - Global Parameters')
            
            # Conservation errors
            error_names = ['Mass Conservation Error', 'Energy Conservation Error']
            error_values = [global_params['mass_conservation_error'], global_params['energy_conservation_error']]
            
            ax2.bar(np.arange(len(error_names)), error_values, color='orange', alpha=0.7)
            ax2.set_xticks(np.arange(len(error_names)))
            ax2.set_xticklabels(error_names, rotation=15, ha='right')
            ax2.set_ylabel('Error Magnitude')
            ax2.set_title('PDE Expert - Conservation Errors')
            
            plt.tight_layout()
            save_path = os.path.join(self.expert_dir, f'pde_global_parameters_batch_{batch_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 2. Per-DAUID visualizations
            if (('diffusion' in pde_data and isinstance(pde_data['diffusion'], dict)) or 
                ('amplification' in pde_data and isinstance(pde_data['amplification'], dict))):
                
                # Extract data for a few sample DAUIDs
                if 'diffusion' in pde_data and isinstance(pde_data['diffusion'], dict):
                    dauid_keys = list(pde_data['diffusion'].keys())
                elif 'amplification' in pde_data and isinstance(pde_data['amplification'], dict):
                    dauid_keys = list(pde_data['amplification'].keys())
                else:
                    dauid_keys = []
                
                sample_keys = dauid_keys[:min(5, len(dauid_keys))]
                
                # For each sample, create visualization
                for dauid in sample_keys:
                    # Get data for this DAUID
                    diffusion = np.array(pde_data['diffusion'][dauid]) if 'diffusion' in pde_data else None
                    amplification = np.array(pde_data['amplification'][dauid]) if 'amplification' in pde_data else None
                    velocity = np.array(pde_data['velocity'][dauid]) if 'velocity' in pde_data else None
                    alpha = np.array(pde_data['alpha'][dauid]) if 'alpha' in pde_data else None
                    beta = np.array(pde_data['beta'][dauid]) if 'beta' in pde_data else None
                    gamma = np.array(pde_data['gamma'][dauid]) if 'gamma' in pde_data else None
                    
                    # Create subplots
                    fig = plt.figure(figsize=(15, 18))
                    gs = GridSpec(3, 2, figure=fig)
                    
                    # Use ethnicity names if available
                    num_ethnicities = diffusion.shape[0] if diffusion is not None else (
                        amplification.shape[0] if amplification is not None else 0)
                    
                    x = np.arange(num_ethnicities)
                    x_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in x]
                    
                    # Diffusion coefficients
                    ax1 = fig.add_subplot(gs[0, 0])
                    if diffusion is not None:
                        ax1.bar(x, diffusion, color=self.expert_colors['pde'], alpha=0.7)
                        ax1.set_xticks(x)
                        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
                        ax1.set_ylabel('Diffusion Coefficient')
                        ax1.set_title(f'PDE Diffusion Coefficients by Ethnicity (DAUID: {dauid})')
                    
                    # Amplification parameters
                    ax2 = fig.add_subplot(gs[0, 1])
                    if amplification is not None:
                        ax2.bar(x, amplification, color=self.expert_colors['pde'], alpha=0.7)
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
                        ax2.set_ylabel('Amplification Factor')
                        ax2.set_title(f'PDE Amplification Factors by Ethnicity (DAUID: {dauid})')
                    
                    # Alpha, Beta and Gamma parameters
                    ax3 = fig.add_subplot(gs[1, 0])
                    if alpha is not None and beta is not None and gamma is not None:
                        width = 0.25
                        ax3.bar(x - width, alpha, width, color='red', alpha=0.7, label='Alpha')
                        ax3.bar(x, beta, width, color='green', alpha=0.7, label='Beta')
                        ax3.bar(x + width, gamma, width, color='blue', alpha=0.7, label='Gamma')
                        ax3.set_xticks(x)
                        ax3.set_xticklabels(x_labels, rotation=45, ha='right')
                        ax3.set_ylabel('Parameter Value')
                        ax3.set_title(f'PDE Parameters by Ethnicity (DAUID: {dauid})')
                        ax3.legend()
                    
                    # Velocity vectors
                    ax4 = fig.add_subplot(gs[1, 1])
                    if velocity is not None:
                        # Plot velocity as vectors
                        vx = velocity[:, 0]
                        vy = velocity[:, 1]
                        
                        magnitudes = np.sqrt(vx**2 + vy**2)
                        
                        # Create a scatter plot with custom colors based on magnitude
                        sc = ax4.scatter(x, np.zeros_like(x), c=magnitudes, cmap='viridis', s=100, alpha=0.7)
                        
                        # Add velocity vectors
                        for i, (x_i, v_x, v_y) in enumerate(zip(x, vx, vy)):
                            ax4.arrow(x_i, 0, v_x, v_y, width=0.02, head_width=0.08, 
                                     head_length=0.1, fc=plt.cm.viridis(magnitudes[i]/max(magnitudes)), 
                                     ec=plt.cm.viridis(magnitudes[i]/max(magnitudes)), alpha=0.7)
                        
                        ax4.set_xticks(x)
                        ax4.set_xticklabels(x_labels, rotation=45, ha='right')
                        ax4.set_ylabel('Velocity Y-Component')
                        ax4.set_xlabel('Velocity X-Component')
                        ax4.set_title(f'PDE Velocity Vectors by Ethnicity (DAUID: {dauid})')
                        plt.colorbar(sc, ax=ax4, label='Velocity Magnitude')
                    
                    # Add a text summary
                    ax5 = fig.add_subplot(gs[2, :])
                    ax5.axis('off')
                    
                    # Create summary text
                    summary_text = f"PDE Expert Summary for DAUID: {dauid}\n\n"
                    
                    # Add diffusion summary
                    if diffusion is not None:
                        avg_diffusion = np.mean(diffusion)
                        max_eth = x_labels[np.argmax(diffusion)]
                        summary_text += f"Diffusion: Avg = {avg_diffusion:.4f}, Highest for {max_eth}\n"
                    
                    # Add amplification summary
                    if amplification is not None:
                        avg_amplification = np.mean(amplification)
                        max_eth = x_labels[np.argmax(amplification)]
                        summary_text += f"Amplification: Avg = {avg_amplification:.4f}, Highest for {max_eth}\n"
                    
                    # Add parameter summaries
                    if alpha is not None:
                        avg_alpha = np.mean(alpha)
                        summary_text += f"Alpha (mean reversion): Avg = {avg_alpha:.4f}\n"
                    
                    if beta is not None:
                        avg_beta = np.mean(beta)
                        summary_text += f"Beta (gradient penalty): Avg = {avg_beta:.4f}\n"
                    
                    if gamma is not None:
                        avg_gamma = np.mean(gamma)
                        summary_text += f"Gamma (advection stabilization): Avg = {avg_gamma:.4f}\n"
                    
                    # Add velocity summary
                    if velocity is not None:
                        avg_magnitude = np.mean(magnitudes)
                        max_eth = x_labels[np.argmax(magnitudes)]
                        summary_text += f"Velocity: Avg Magnitude = {avg_magnitude:.4f}, Highest for {max_eth}\n"
                        
                        # Add directional tendency
                        avg_vx = np.mean(vx)
                        avg_vy = np.mean(vy)
                        direction = ""
                        if avg_vx > 0 and avg_vy > 0:
                            direction = "northeast"
                        elif avg_vx > 0 and avg_vy < 0:
                            direction = "southeast"
                        elif avg_vx < 0 and avg_vy > 0:
                            direction = "northwest"
                        elif avg_vx < 0 and avg_vy < 0:
                            direction = "southwest"
                        elif avg_vx > 0:
                            direction = "east"
                        elif avg_vx < 0:
                            direction = "west"
                        elif avg_vy > 0:
                            direction = "north"
                        elif avg_vy < 0:
                            direction = "south"
                        
                        if direction:
                            summary_text += f"Overall flow direction: {direction}\n"
                    
                    # Add global parameters
                    summary_text += f"\nGlobal Parameters:\n"
                    summary_text += f"K (gradient threshold) = {global_params['K']:.4f}\n"
                    summary_text += f"Sigma squared = {global_params['sigma_squared']:.4f}\n"
                    summary_text += f"Alpha base = {global_params['alpha_0']:.4f}\n"
                    summary_text += f"Beta base = {global_params['beta_0']:.4f}\n"
                    summary_text += f"Mass conservation error = {global_params['mass_conservation_error']:.6f}\n"
                    summary_text += f"Energy conservation error = {global_params['energy_conservation_error']:.6f}\n"
                    
                    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                            verticalalignment='top', horizontalalignment='left',
                            fontsize=10, fontfamily='monospace')
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.expert_dir, 
                                           f'pde_detailed_dauid_{dauid}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
            
            logger.info(f"Created PDE expert visualizations for batch {batch_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating PDE expert visualizations: {str(e)}")
            return False


    def visualize_expert_comparison(self, interpretation_data, batch_idx=0):
        """
        Create visualizations comparing the outputs of all experts
        
        Args:
            interpretation_data (dict): Complete interpretation data
            batch_idx (int): Batch index for saving
            
        Returns:
            bool: Success status
        """
        try:
            # Check if we have expert outputs and weights
            if ('expert_predictions' not in interpretation_data or 
                'expert_weights' not in interpretation_data):
                logger.warning("No expert comparison data found in interpretation data")
                return False
            
            # Ensure we're working with numpy arrays, not lists
            import numpy as np
            expert_outputs = interpretation_data['expert_predictions']
            
            # Convert expert_weights to numpy array if it's a list
            if isinstance(interpretation_data['expert_weights'], list):
                expert_weights = np.array(interpretation_data['expert_weights'])
            else:
                expert_weights = interpretation_data['expert_weights']
                
            # Check if main_model_output is available and convert if needed
            if 'main_model_output' in interpretation_data and interpretation_data['main_model_output'] is not None:
                if isinstance(interpretation_data['main_model_output'], list):
                    main_model_output = np.array(interpretation_data['main_model_output'])
                else:
                    main_model_output = interpretation_data['main_model_output']
            else:
                main_model_output = None
            
            # Get the number of ethnicities from expert weights shape
            if hasattr(expert_weights, 'shape') and len(expert_weights.shape) >= 2:
                num_ethnicities = expert_weights.shape[1]
            elif self.ethnicity_names:
                # Fall back to ethnicity_names if available
                num_ethnicities = len(self.ethnicity_names)
                logger.info(f"Using ethnicity_names to determine num_ethnicities = {num_ethnicities}")
            else:
                logger.warning("Could not determine number of ethnicities from expert_weights or ethnicity_names")
                num_ethnicities = 1  # Default to 1 ethnicity as fallback
            
            # 1. Expert Weight Comparison
            # Calculate average weight across the batch for each expert and ethnicity
            if hasattr(expert_weights, 'shape') and len(expert_weights.shape) >= 3:
                avg_weights = np.mean(expert_weights, axis=0)  # [num_ethnicities, num_experts]
            else:
                # Handle the case where expert_weights doesn't have the expected shape
                logger.warning("expert_weights doesn't have expected shape for averaging, creating dummy data")
                # Create dummy data with 4 experts
                avg_weights = np.ones((num_ethnicities, 4)) * 0.25  # Equal weights for 4 experts
            
            # Create a grouped bar chart comparing expert weights
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            x = np.arange(num_ethnicities)
            width = 0.2
            
            # Set ethnicity labels
            ethn_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in range(num_ethnicities)]
            
            # Create bars for each expert
            expert_names = ['Colonization', 'Jump', 'Decline', 'PDE']
            
            # Make sure avg_weights has the right shape
            if avg_weights.shape[1] < len(expert_names):
                # Pad with zeros if necessary
                padding = np.zeros((avg_weights.shape[0], len(expert_names) - avg_weights.shape[1]))
                avg_weights = np.hstack((avg_weights, padding))
            
            for i, expert in enumerate(expert_names):
                ax.bar(x + (i - 1.5) * width, avg_weights[:, i], width, 
                    label=expert, color=self.expert_colors[expert.lower()], alpha=0.7)
            
            ax.set_xlabel('Ethnicity')
            ax.set_ylabel('Average Expert Weight')
            ax.set_title('Average Expert Weights by Ethnicity')
            ax.set_xticks(x)
            ax.set_xticklabels(ethn_labels, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            save_path = os.path.join(self.comparison_dir, f'expert_weights_comparison_batch_{batch_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 2. Expert Output Comparison
            # Check if all required expert outputs are available
            required_keys = ['colonization', 'jump', 'decline', 'pde']
            if not all(key in expert_outputs for key in required_keys):
                logger.warning("Missing one or more expert outputs, skipping output comparison")
                return True  # Return True since we at least created the weights visualization
            
            # Get outputs from all experts
            try:
                # Convert to numpy arrays if they're lists
                colonization_outputs = np.array(expert_outputs['colonization'][0]) if isinstance(expert_outputs['colonization'], list) else expert_outputs['colonization'][0]
                jump_outputs = np.array(expert_outputs['jump'][0]) if isinstance(expert_outputs['jump'], list) else expert_outputs['jump'][0]
                decline_outputs = np.array(expert_outputs['decline'][0]) if isinstance(expert_outputs['decline'], list) else expert_outputs['decline'][0]
                pde_outputs = np.array(expert_outputs['pde'][0]) if isinstance(expert_outputs['pde'], list) else expert_outputs['pde'][0]
                
                if main_model_output is not None:
                    main_outputs = np.array(main_model_output[0]) if isinstance(main_model_output, list) else main_model_output[0]
                else:
                    main_outputs = None
                
                # Validate output shapes
                output_shapes = {
                    'colonization': colonization_outputs.shape if hasattr(colonization_outputs, 'shape') else (num_ethnicities,),
                    'jump': jump_outputs.shape if hasattr(jump_outputs, 'shape') else (num_ethnicities,),
                    'decline': decline_outputs.shape if hasattr(decline_outputs, 'shape') else (num_ethnicities,),
                    'pde': pde_outputs.shape if hasattr(pde_outputs, 'shape') else (num_ethnicities,)
                }
                
                # Check if shapes are compatible
                if len(set(str(shape) for shape in output_shapes.values())) > 1:
                    logger.warning(f"Expert outputs have incompatible shapes: {output_shapes}")
                    # Try to reshape to match num_ethnicities
                    if not hasattr(colonization_outputs, 'shape') or colonization_outputs.shape[0] != num_ethnicities:
                        colonization_outputs = np.zeros(num_ethnicities)
                    if not hasattr(jump_outputs, 'shape') or jump_outputs.shape[0] != num_ethnicities:
                        jump_outputs = np.zeros(num_ethnicities)
                    if not hasattr(decline_outputs, 'shape') or decline_outputs.shape[0] != num_ethnicities:
                        decline_outputs = np.zeros(num_ethnicities)
                    if not hasattr(pde_outputs, 'shape') or pde_outputs.shape[0] != num_ethnicities:
                        pde_outputs = np.zeros(num_ethnicities)
                    if main_outputs is not None and (not hasattr(main_outputs, 'shape') or main_outputs.shape[0] != num_ethnicities):
                        main_outputs = np.zeros(num_ethnicities)
                
                # Create visualization comparing outputs
                fig, axes = plt.subplots(num_ethnicities, 1, figsize=(10, num_ethnicities * 3))
                if num_ethnicities == 1:
                    axes = [axes]
                    
                # Create comparison for each ethnicity
                for eth_idx in range(num_ethnicities):
                    ax = axes[eth_idx]
                    
                    # Try to safely get outputs for this ethnicity
                    try:
                        c_out = float(colonization_outputs[eth_idx]) if eth_idx < len(colonization_outputs) else 0
                        j_out = float(jump_outputs[eth_idx]) if eth_idx < len(jump_outputs) else 0
                        d_out = float(decline_outputs[eth_idx]) if eth_idx < len(decline_outputs) else 0
                        p_out = float(pde_outputs[eth_idx]) if eth_idx < len(pde_outputs) else 0
                        
                        # Collect outputs for this ethnicity
                        outputs = {
                            'Colonization': c_out,
                            'Jump': j_out,
                            'Decline': d_out,
                            'PDE': p_out
                        }
                        
                        if main_outputs is not None and eth_idx < len(main_outputs):
                            outputs['Main Model'] = float(main_outputs[eth_idx])
                        
                        # Create bar chart
                        x_pos = np.arange(len(outputs))
                        colors = [self.expert_colors.get(name.lower(), 'gray') for name in outputs.keys()]
                        
                        ax.bar(x_pos, list(outputs.values()), color=colors, alpha=0.7)
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(list(outputs.keys()), rotation=45, ha='right')
                        
                        eth_name = self.ethnicity_names[eth_idx] if self.ethnicity_names else f'Ethnicity {eth_idx}'
                        ax.set_title(f'Expert Outputs for {eth_name}')
                        ax.set_ylabel('Predicted Population')
                        
                    except Exception as inner_e:
                        logger.warning(f"Error plotting expert outputs for ethnicity {eth_idx}: {str(inner_e)}")
                        ax.text(0.5, 0.5, f"Error plotting expert outputs for ethnicity {eth_idx}", 
                            ha='center', va='center', transform=ax.transAxes)
                
                plt.tight_layout()
                save_path = os.path.join(self.comparison_dir, f'expert_outputs_comparison_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
                
                # 3. Expert weight vs output influence visualization
                # Only create if the data shapes are compatible
                try:
                    # Select a few samples to visualize (first 3 samples)
                    if hasattr(expert_weights, 'shape') and len(expert_weights.shape) >= 3:
                        num_samples = min(3, expert_weights.shape[0])
                        
                        for sample_idx in range(num_samples):
                            # Get expert weights for this sample
                            sample_weights = expert_weights[sample_idx]  # [num_ethnicities, num_experts]
                            
                            # Get expert outputs for this sample
                            sample_outputs = {}
                            try:
                                sample_outputs['colonization'] = np.array(expert_outputs['colonization'][sample_idx]) if isinstance(expert_outputs['colonization'], list) else expert_outputs['colonization'][sample_idx]
                                sample_outputs['jump'] = np.array(expert_outputs['jump'][sample_idx]) if isinstance(expert_outputs['jump'], list) else expert_outputs['jump'][sample_idx]
                                sample_outputs['decline'] = np.array(expert_outputs['decline'][sample_idx]) if isinstance(expert_outputs['decline'], list) else expert_outputs['decline'][sample_idx]
                                sample_outputs['pde'] = np.array(expert_outputs['pde'][sample_idx]) if isinstance(expert_outputs['pde'], list) else expert_outputs['pde'][sample_idx]
                            except (IndexError, TypeError) as e:
                                logger.warning(f"Error getting expert outputs for sample {sample_idx}: {str(e)}")
                                continue
                                
                            sample_main_output = None
                            if main_model_output is not None:
                                try:
                                    sample_main_output = np.array(main_model_output[sample_idx]) if isinstance(main_model_output, list) else main_model_output[sample_idx]
                                except (IndexError, TypeError) as e:
                                    logger.warning(f"Error getting main model output for sample {sample_idx}: {str(e)}")
                            
                            # Create visualization for a few ethnicities
                            ethnicities_to_show = min(3, num_ethnicities)
                            
                            fig, axes = plt.subplots(ethnicities_to_show, 1, figsize=(12, ethnicities_to_show * 5))
                            if ethnicities_to_show == 1:
                                axes = [axes]
                            
                            for i in range(ethnicities_to_show):
                                ax = axes[i]
                                
                                # Get weights and outputs for this ethnicity
                                eth_weights = sample_weights[i] if i < len(sample_weights) else np.zeros(len(expert_names))
                                
                                # Ensure eth_weights has 4 elements (one per expert)
                                if len(eth_weights) < 4:
                                    eth_weights = np.pad(eth_weights, (0, 4 - len(eth_weights)), 'constant')
                                
                                # Safely get expert outputs for this ethnicity
                                eth_outputs = []
                                for expert_key in ['colonization', 'jump', 'decline', 'pde']:
                                    try:
                                        if i < len(sample_outputs[expert_key]):
                                            eth_outputs.append(float(sample_outputs[expert_key][i]))
                                        else:
                                            eth_outputs.append(0.0)
                                    except (IndexError, TypeError):
                                        eth_outputs.append(0.0)
                                
                                # Calculate weighted output contribution
                                weighted_outputs = [w * o for w, o in zip(eth_weights, eth_outputs)]
                                
                                # Create grouped bar chart
                                x = np.arange(len(expert_names))
                                width = 0.35
                                
                                ax.bar(x - width/2, eth_outputs, width, color='lightblue', label='Raw Output')
                                ax.bar(x + width/2, weighted_outputs, width, color='orange', label='Weighted Output')
                                
                                # Add weight labels above bars
                                for j, w in enumerate(eth_weights):
                                    ax.text(j, max(eth_outputs[j], weighted_outputs[j]) + 5, 
                                        f'w={w:.2f}', ha='center')
                                
                                ax.set_xticks(x)
                                ax.set_xticklabels(expert_names)
                                
                                eth_name = self.ethnicity_names[i] if self.ethnicity_names and i < len(self.ethnicity_names) else f'Ethnicity {i}'
                                ax.set_title(f'Expert Raw & Weighted Outputs for {eth_name} (Sample {sample_idx})')
                                ax.set_ylabel('Population Value')
                                ax.legend()
                                
                                # Add the main model output and weighted sum as text
                                if sample_main_output is not None and i < len(sample_main_output):
                                    ax.text(0.98, 0.98, f'Main Model: {float(sample_main_output[i]):.2f}', 
                                        transform=ax.transAxes, ha='right', va='top', 
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                                
                                weighted_sum = sum(weighted_outputs)
                                ax.text(0.98, 0.9, f'Weighted Sum: {weighted_sum:.2f}', 
                                    transform=ax.transAxes, ha='right', va='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                            
                            plt.tight_layout()
                            save_path = os.path.join(self.comparison_dir, 
                                                f'expert_weight_influence_sample_{sample_idx}_batch_{batch_idx}.png')
                            plt.savefig(save_path)
                            plt.close()
                except Exception as inner_e:
                    logger.warning(f"Error creating expert weight influence visualizations: {str(inner_e)}")
            
            except Exception as output_e:
                logger.warning(f"Error processing expert outputs: {str(output_e)}")
            
            logger.info(f"Created expert comparison visualizations for batch {batch_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating expert comparison visualizations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
            
    def visualize_ethnicity_interactions(self, interpretation_data, batch_idx=0):
        """
        Create visualizations for ethnicity interactions
        
        Args:
            interpretation_data (dict): Interpretation data
            batch_idx (int): Batch index for saving
            
        Returns:
            bool: Success status
        """
        try:
            if 'interaction_weights' not in interpretation_data:
                logger.warning("No ethnicity interaction data found in interpretation data")
                return False
            
            interaction_weights = interpretation_data['interaction_weights']  # [batch_size, num_ethnicities, num_ethnicities]
            interaction_effect = interpretation_data.get('interaction_effect')  # [batch_size, num_ethnicities]
            
            # Average interaction weights across the batch
            avg_interactions = np.mean(interaction_weights, axis=0)  # [num_ethnicities, num_ethnicities]
            
            # Create heatmap of average interaction weights
            plt.figure(figsize=(12, 10))
            
            # Set ethnicity labels
            num_ethnicities = avg_interactions.shape[0]
            ethn_labels = self.ethnicity_names if self.ethnicity_names else [f'Ethnicity {i}' for i in range(num_ethnicities)]
            
            sns.heatmap(avg_interactions, annot=True, cmap='viridis', 
                      xticklabels=ethn_labels, yticklabels=ethn_labels)
            plt.title('Average Ethnicity Interaction Weights')
            plt.xlabel('Source Ethnicity')
            plt.ylabel('Target Ethnicity')
            
            plt.tight_layout()
            save_path = os.path.join(self.comparison_dir, f'ethnicity_interactions_heatmap_batch_{batch_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # Visualize specific DAUIDs if they're available
            if 'ethnicity_interactions_by_dauid' in interpretation_data:
                interactions_by_dauid = interpretation_data['ethnicity_interactions_by_dauid']
                
                # Select a few sample DAUIDs
                sample_dauid_keys = list(interactions_by_dauid.keys())[:min(5, len(interactions_by_dauid))]
                
                for dauid in sample_dauid_keys:
                    # Extract interaction data
                    dauid_interactions = interactions_by_dauid[dauid]
                    
                    # Reshape interaction data to a matrix
                    interaction_matrix = np.zeros((num_ethnicities, num_ethnicities))
                    
                    for key, value in dauid_interactions.items():
                        # Parse the key format "ethnicity_i_to_ethnicity_j"
                        if '_to_' in key:
                            source, target = key.split('_to_')
                            
                            # Extract indices from the source and target strings
                            # Find all ethnicities from the ethnicity_names list
                            source_idx = -1
                            target_idx = -1
                            
                            if self.ethnicity_names:
                                for idx, name in enumerate(self.ethnicity_names):
                                    if name in source:
                                        source_idx = idx
                                    if name in target:
                                        target_idx = idx
                            
                            # If we couldn't find the ethnicities by name, try to extract indices
                            if source_idx == -1 or target_idx == -1:
                                try:
                                    # Try to extract indices using regex or simple parsing
                                    # This assumes format like "ethnicity_0_to_ethnicity_1"
                                    source_parts = source.split('_')
                                    target_parts = target.split('_')
                                    
                                    if len(source_parts) > 1 and source_parts[-1].isdigit():
                                        source_idx = int(source_parts[-1])
                                    if len(target_parts) > 1 and target_parts[-1].isdigit():
                                        target_idx = int(target_parts[-1])
                                except:
                                    # If parsing fails, skip this entry
                                    continue
                            
                            # If we have valid indices, update the matrix
                            if 0 <= source_idx < num_ethnicities and 0 <= target_idx < num_ethnicities:
                                interaction_matrix[target_idx, source_idx] = value
                    
                    # Create heatmap for this DAUID
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(interaction_matrix, annot=True, cmap='viridis', 
                              xticklabels=ethn_labels, yticklabels=ethn_labels)
                    plt.title(f'Ethnicity Interaction Weights for DAUID {dauid}')
                    plt.xlabel('Source Ethnicity')
                    plt.ylabel('Target Ethnicity')
                    
                    # Add interaction effect visualization if available
                    if 'interaction_effect_by_dauid' in interpretation_data:
                        effect_by_dauid = interpretation_data['interaction_effect_by_dauid']
                        if dauid in effect_by_dauid:
                            effect_data = effect_by_dauid[dauid]
                            
                            # Create a separate plot for interaction effects
                            plt.figure(figsize=(10, 6))
                            
                            # Extract effect values
                            effect_values = []
                            for i in range(num_ethnicities):
                                eth_name = ethn_labels[i]
                                # Try different ways to find the ethnicity in the dict
                                if eth_name in effect_data:
                                    effect_values.append(effect_data[eth_name])
                                elif f'ethnicity_{i}' in effect_data:
                                    effect_values.append(effect_data[f'ethnicity_{i}'])
                                else:
                                    effect_values.append(0)  # Default if not found
                            
                            # Create bar chart
                            bars = plt.bar(np.arange(len(effect_values)), effect_values, alpha=0.7)
                            
                            # Color bars based on positive/negative effects
                            for i, v in enumerate(effect_values):
                                if v < 0:
                                    bars[i].set_color('red')
                                else:
                                    bars[i].set_color('green')
                            
                            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                            plt.xticks(np.arange(len(effect_values)), ethn_labels, rotation=45, ha='right')
                            plt.ylabel('Interaction Effect')
                            plt.title(f'Ethnicity Interaction Effects for DAUID {dauid}')
                            
                            plt.tight_layout()
                            save_path = os.path.join(self.comparison_dir, 
                                                   f'ethnicity_interaction_effect_dauid_{dauid}_batch_{batch_idx}.png')
                            plt.savefig(save_path)
                            plt.close()
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.comparison_dir, 
                                           f'ethnicity_interactions_dauid_{dauid}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
            
            # If we have interaction effect data, create aggregate visualization
            if interaction_effect is not None:
                # Average interaction effects across the batch
                avg_effects = np.mean(interaction_effect, axis=0)  # [num_ethnicities]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(np.arange(len(avg_effects)), avg_effects, alpha=0.7)
                
                # Color bars based on positive/negative effects
                for i, v in enumerate(avg_effects):
                    if v < 0:
                        bars[i].set_color('red')
                    else:
                        bars[i].set_color('green')
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.xticks(np.arange(len(avg_effects)), ethn_labels, rotation=45, ha='right')
                plt.ylabel('Average Interaction Effect')
                plt.title('Average Ethnicity Interaction Effects')
                
                plt.tight_layout()
                save_path = os.path.join(self.comparison_dir, f'ethnicity_interaction_effects_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
            
            logger.info(f"Created ethnicity interaction visualizations for batch {batch_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ethnicity interaction visualizations: {str(e)}")
            return False

    def visualize_spatial_patterns(self, interpretation_data, predictions_df=None, batch_idx=0):
        """
        Create spatial visualizations if coordinates are available
        
        Args:
            interpretation_data (dict): Interpretation data
            predictions_df (pd.DataFrame, optional): Predictions DataFrame with DAUIDs
            batch_idx (int): Batch index for saving
            
        Returns:
            bool: Success status
        """
        try:
            # Check if we have spatial mapping
            if self.map_dauid_to_coords is None:
                logger.warning("No DAUID to coordinates mapping available, skipping spatial visualizations")
                return False
            
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
            
            # Get DAUIDs from available data
            dauid_list = list(sample_dauid_dict.keys())
            
            # Create spatial visualizations for different metrics
            
            # 1. Expert weights spatial visualization
            if 'ethnicity_interactions_by_dauid' in interpretation_data:
                # Create a new DataFrame for spatial plotting
                spatial_df = pd.DataFrame({'DAUID': dauid_list})
                
                # Add coordinates
                x_coords = []
                y_coords = []
                for dauid in dauid_list:
                    if dauid in self.map_dauid_to_coords:
                        x, y = self.map_dauid_to_coords[dauid]
                        x_coords.append(x)
                        y_coords.append(y)
                    else:
                        # Use dummy coordinates if not found
                        x_coords.append(0)
                        y_coords.append(0)
                
                spatial_df['X'] = x_coords
                spatial_df['Y'] = y_coords
                
                # Filter out entries with dummy coordinates
                spatial_df = spatial_df[(spatial_df['X'] != 0) | (spatial_df['Y'] != 0)]
                
                if len(spatial_df) == 0:
                    logger.warning("No valid coordinates found for any DAUID, skipping spatial visualizations")
                    return False
                
                # Add expert weights if available in interpretation data
                if 'expert_weights' in interpretation_data:
                    # We need to map the DAUIDs to the correct indices in expert_weights
                    if 'dauid_list' in interpretation_data:
                        # If dauid_list is provided in interpretation data, use it for mapping
                        interp_dauid_list = interpretation_data['dauid_list']
                        
                        # Create a mapping from DAUID to index
                        dauid_to_idx = {str(d): i for i, d in enumerate(interp_dauid_list)}
                        
                        # Add columns for expert weights
                        expert_names = ['Colonization', 'Jump', 'Decline', 'PDE']
                        
                        for eth_idx in range(interpretation_data['expert_weights'].shape[1]):
                            eth_name = self.ethnicity_names[eth_idx] if self.ethnicity_names else f'Ethnicity_{eth_idx}'
                            
                            for exp_idx, exp_name in enumerate(expert_names):
                                weight_col = f"{eth_name}_{exp_name}_Weight"
                                weight_values = []
                                
                                for dauid in spatial_df['DAUID']:
                                    if dauid in dauid_to_idx:
                                        # Get the expert weight for this DAUID, ethnicity, and expert
                                        idx = dauid_to_idx[dauid]
                                        if idx < interpretation_data['expert_weights'].shape[0]:
                                            weight = interpretation_data['expert_weights'][idx, eth_idx, exp_idx]
                                            weight_values.append(weight)
                                        else:
                                            weight_values.append(0)
                                    else:
                                        weight_values.append(0)
                                
                                spatial_df[weight_col] = weight_values
                
                # For a sample of ethnicities, create spatial plots of expert weights
                num_ethnicities = interpretation_data['expert_weights'].shape[1]
                ethnicities_to_visualize = min(3, num_ethnicities)  # Visualize up to 3 ethnicities
                
                for eth_idx in range(ethnicities_to_visualize):
                    eth_name = self.ethnicity_names[eth_idx] if self.ethnicity_names else f'Ethnicity_{eth_idx}'
                    
                    # Create a 2x2 grid for the 4 experts
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    axes = axes.flatten()
                    
                    for exp_idx, exp_name in enumerate(['Colonization', 'Jump', 'Decline', 'PDE']):
                        ax = axes[exp_idx]
                        weight_col = f"{eth_name}_{exp_name}_Weight"
                        
                        if weight_col in spatial_df.columns:
                            # Create scatter plot where point size and color reflect the weight
                            scatter = ax.scatter(
                                spatial_df['X'], 
                                spatial_df['Y'],
                                c=spatial_df[weight_col], 
                                s=spatial_df[weight_col] * 100,  # Scale size
                                cmap='viridis',
                                alpha=0.7,
                                edgecolors='k'
                            )
                            
                            # Add colorbar
                            plt.colorbar(scatter, ax=ax, label=f'{exp_name} Weight')
                            
                            ax.set_title(f'{exp_name} Expert Weight for {eth_name}')
                            ax.set_xlabel('X Coordinate')
                            ax.set_ylabel('Y Coordinate')
                            ax.grid(True, alpha=0.3)
                        else:
                            ax.text(0.5, 0.5, f"No data for {weight_col}", 
                                   ha='center', va='center', transform=ax.transAxes)
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.spatial_dir, 
                                           f'spatial_expert_weights_{eth_name}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                
                # 2. Create spatial plots for ethnicity interactions
                for eth_idx in range(ethnicities_to_visualize):
                    eth_name = self.ethnicity_names[eth_idx] if self.ethnicity_names else f'Ethnicity_{eth_idx}'
                    
                    # Add interaction effect data if available
                    if 'interaction_effect_by_dauid' in interpretation_data:
                        effect_values = []
                        
                        for dauid in spatial_df['DAUID']:
                            if dauid in interpretation_data['interaction_effect_by_dauid']:
                                dauid_effects = interpretation_data['interaction_effect_by_dauid'][dauid]
                                
                                # Find the effect for this ethnicity
                                effect = 0
                                if eth_name in dauid_effects:
                                    effect = dauid_effects[eth_name]
                                elif f'ethnicity_{eth_idx}' in dauid_effects:
                                    effect = dauid_effects[f'ethnicity_{eth_idx}']
                                
                                effect_values.append(effect)
                            else:
                                effect_values.append(0)
                        
                        spatial_df[f'{eth_name}_Effect'] = effect_values
                    
                        # Create spatial plot of interaction effects
                        plt.figure(figsize=(12, 10))
                        
                        # Use diverging colormap for positive/negative effects
                        scatter = plt.scatter(
                            spatial_df['X'], 
                            spatial_df['Y'],
                            c=spatial_df[f'{eth_name}_Effect'], 
                            s=np.abs(spatial_df[f'{eth_name}_Effect']) * 100 + 20,  # Scale size
                            cmap='coolwarm',
                            vmin=-np.abs(spatial_df[f'{eth_name}_Effect']).max(),
                            vmax=np.abs(spatial_df[f'{eth_name}_Effect']).max(),
                            alpha=0.7,
                            edgecolors='k'
                        )
                        
                        # Add colorbar
                        plt.colorbar(scatter, label=f'Interaction Effect for {eth_name}')
                        
                        plt.title(f'Spatial Distribution of Interaction Effects for {eth_name}')
                        plt.xlabel('X Coordinate')
                        plt.ylabel('Y Coordinate')
                        plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        save_path = os.path.join(self.spatial_dir, 
                                               f'spatial_interaction_effects_{eth_name}_batch_{batch_idx}.png')
                        plt.savefig(save_path)
                        plt.close()
            
            # 3. Create spatial plots from predictions DataFrame if provided
            if predictions_df is not None:
                # Set up predictions DataFrame for visualization
                pred_df = predictions_df.copy()
                
                # Add coordinates
                x_coords = []
                y_coords = []
                
                for dauid in pred_df['DAUID']:
                    if dauid in self.map_dauid_to_coords:
                        x, y = self.map_dauid_to_coords[dauid]
                        x_coords.append(x)
                        y_coords.append(y)
                    else:
                        # Use dummy coordinates if not found
                        x_coords.append(0)
                        y_coords.append(0)
                
                pred_df['X'] = x_coords
                pred_df['Y'] = y_coords
                
                # Filter out entries with dummy coordinates
                pred_df = pred_df[(pred_df['X'] != 0) | (pred_df['Y'] != 0)]
                
                if len(pred_df) == 0:
                    logger.warning("No valid coordinates found for predictions, skipping prediction visualizations")
                    return False
                
                # Identify available ethnicities in the predictions
                eth_columns = []
                for col in pred_df.columns:
                    if col.endswith('_predicted'):
                        eth_columns.append(col)
                
                # Create spatial plots for each ethnicity
                for col in eth_columns[:min(5, len(eth_columns))]:
                    eth_name = col.replace('_predicted', '')
                    
                    # Create scatter plot where point size and color reflect the predicted population
                    plt.figure(figsize=(12, 10))
                    
                    scatter = plt.scatter(
                        pred_df['X'], 
                        pred_df['Y'],
                        c=pred_df[col], 
                        s=np.log1p(pred_df[col]) * 5,  # Scale size using log for better visualization
                        cmap='viridis',
                        alpha=0.7,
                        edgecolors='k'
                    )
                    
                    # Add colorbar
                    plt.colorbar(scatter, label=f'Predicted Population')
                    
                    plt.title(f'Spatial Distribution of Predicted Population for {eth_name}')
                    plt.xlabel('X Coordinate')
                    plt.ylabel('Y Coordinate')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.spatial_dir, 
                                           f'spatial_predicted_population_{eth_name}_batch_{batch_idx}.png')
                    plt.savefig(save_path)
                    plt.close()
                    
                    # If we have both predicted and 2016 data, create change map
                    base_col = f"{eth_name}_2016"
                    if base_col in pred_df.columns:
                        # Calculate and visualize the percent change
                        pred_df[f'{eth_name}_percent_change'] = (
                            (pred_df[col] - pred_df[base_col]) / (pred_df[base_col] + 1)) * 100
                        
                        plt.figure(figsize=(12, 10))
                        
                        # Use diverging colormap for positive/negative changes
                        scatter = plt.scatter(
                            pred_df['X'], 
                            pred_df['Y'],
                            c=pred_df[f'{eth_name}_percent_change'], 
                            s=np.abs(pred_df[f'{eth_name}_percent_change']),  # Scale size
                            cmap='coolwarm',
                            vmin=-100,  # Limit to ±100% for better visualization
                            vmax=100,
                            alpha=0.7,
                            edgecolors='k'
                        )
                        
                        # Add colorbar
                        plt.colorbar(scatter, label=f'Percent Change (%)')
                        
                        plt.title(f'Spatial Distribution of Population Change for {eth_name}')
                        plt.xlabel('X Coordinate')
                        plt.ylabel('Y Coordinate')
                        plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        save_path = os.path.join(self.spatial_dir, 
                                               f'spatial_population_change_{eth_name}_batch_{batch_idx}.png')
                        plt.savefig(save_path)
                        plt.close()
            
            logger.info(f"Created spatial visualizations for batch {batch_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating spatial visualizations: {str(e)}")
            return False

    def visualize_batch_interpretation(self, interpretation_path, predictions_df=None, batch_idx=0):
        """
        Create all visualizations for a batch's interpretation data
        
        Args:
            interpretation_path (str): Path to interpretation JSON file
            predictions_df (pd.DataFrame, optional): Predictions DataFrame with DAUIDs
            batch_idx (int): Batch index for reference
            
        Returns:
            bool: Success status
        """
        try:
            # Load interpretation data
            interpretation_data = self.load_interpretation_data(interpretation_path)
            if interpretation_data is None:
                return False
            
            # Create visualizations for each expert
            success_colonization = self.visualize_colonization_expert(interpretation_data, batch_idx)
            success_jump = self.visualize_jump_expert(interpretation_data, batch_idx)
            success_decline = self.visualize_decline_expert(interpretation_data, batch_idx)
            success_pde = self.visualize_pde_expert(interpretation_data, batch_idx)
            
            # Create comparison visualizations
            success_comparison = self.visualize_expert_comparison(interpretation_data, batch_idx)
            
            # Create ethnicity interaction visualizations
            success_interactions = self.visualize_ethnicity_interactions(interpretation_data, batch_idx)
            
            # Create spatial visualizations if coordinates mapping is available
            success_spatial = self.visualize_spatial_patterns(interpretation_data, predictions_df, batch_idx)
            
            logger.info(f"Completed visualizations for batch {batch_idx}")
            
            # Return overall success status
            return (success_colonization or success_jump or success_decline or 
                   success_pde or success_comparison or success_interactions or 
                   success_spatial)
                   
        except Exception as e:
            logger.error(f"Error generating visualizations for batch {batch_idx}: {str(e)}")
            return False
    
    def visualize_feature_importance(self, feature_importance, expert_name, batch_idx=0, top_n=20, category_mapping=None):
        """
        Create improved visualizations for feature importance
        
        Args:
            feature_importance (list): Feature importance scores
            expert_name (str): Name of the expert (colonization, jump, decline, pde)
            batch_idx (int): Batch index for saving
            top_n (int): Number of top features to show
            category_mapping (dict, optional): Mapping from feature indices to categories
            
        Returns:
            bool: Success status
        """
        try:
            # Create an array of feature names or indices
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
            
            # Convert to numpy array for easier manipulation
            import numpy as np
            feat_importance = np.array(feature_importance)
            
            # Create a DataFrame for easier manipulation
            import pandas as pd
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feat_importance
            })
            
            # Sort by importance (descending)
            df = df.sort_values('Importance', ascending=False)
            
            # 1. Show only top N features
            plt.figure(figsize=(12, 8))
            top_df = df.head(top_n)
            plt.barh(np.arange(len(top_df)), top_df['Importance'].values, 
                    color=self.expert_colors[expert_name], alpha=0.7)
            plt.yticks(np.arange(len(top_df)), top_df['Feature'].values)
            plt.xlabel('Importance Score')
            plt.title(f'{expert_name.capitalize()} Expert - Top {top_n} Feature Importance')
            plt.tight_layout()
            
            save_path = os.path.join(self.feature_dir, f'{expert_name}_top{top_n}_feature_importance_batch_{batch_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 2. Create horizontal bar chart with grid lines and better formatting
            plt.figure(figsize=(14, max(8, top_n * 0.3)))  # Dynamic height based on number of features
            
            # Create the horizontal bar chart
            bars = plt.barh(np.arange(len(top_df)), top_df['Importance'].values, 
                    color=self.expert_colors[expert_name], alpha=0.8)
            
            # Add importance values as text next to the bars
            for i, v in enumerate(top_df['Importance'].values):
                plt.text(v + 0.01, i, f'{v:.4f}', va='center')
            
            # Add grid lines for better readability
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Improve the appearance
            plt.yticks(np.arange(len(top_df)), top_df['Feature'].values)
            plt.xlabel('Importance Score')
            plt.title(f'{expert_name.capitalize()} Expert - Top {top_n} Feature Importance', fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(self.feature_dir, f'{expert_name}_formatted_feature_importance_batch_{batch_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 3. Create a heatmap of top N features
            plt.figure(figsize=(10, max(8, top_n * 0.3)))
            import seaborn as sns
            sns.heatmap(top_df['Importance'].values.reshape(-1, 1), 
                    annot=True, fmt='.4f', cmap='viridis',
                    yticklabels=top_df['Feature'].values,
                    xticklabels=['Importance'])
            plt.title(f'{expert_name.capitalize()} Expert - Feature Importance Heatmap', fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(self.feature_dir, f'{expert_name}_heatmap_feature_importance_batch_{batch_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 4. If category mapping is provided, visualize importance by category
            if category_mapping is not None:
                # Add category to DataFrame
                df['Category'] = df['Feature'].apply(lambda x: category_mapping.get(x, 'Other'))
                
                # Calculate total importance by category
                category_importance = df.groupby('Category')['Importance'].sum().reset_index()
                category_importance = category_importance.sort_values('Importance', ascending=False)
                
                # Create bar chart of importance by category
                plt.figure(figsize=(12, 8))
                plt.barh(np.arange(len(category_importance)), category_importance['Importance'].values, 
                        color=plt.cm.Set3.colors[:len(category_importance)], alpha=0.8)
                plt.yticks(np.arange(len(category_importance)), category_importance['Category'].values)
                plt.xlabel('Total Importance Score')
                plt.title(f'{expert_name.capitalize()} Expert - Feature Importance by Category', fontsize=14)
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                save_path = os.path.join(self.feature_dir, f'{expert_name}_category_importance_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
                
                # Create heatmap of top features within each of the top categories
                top_categories = category_importance.head(5)['Category'].values
                plt.figure(figsize=(15, 12))
                
                # Create subplots for each category
                fig, axes = plt.subplots(len(top_categories), 1, figsize=(12, 4 * len(top_categories)))
                if len(top_categories) == 1:
                    axes = [axes]
                    
                for i, category in enumerate(top_categories):
                    # Get top features for this category
                    cat_df = df[df['Category'] == category].head(10)
                    
                    # Create barh plot
                    ax = axes[i]
                    ax.barh(np.arange(len(cat_df)), cat_df['Importance'].values, 
                        color=plt.cm.Set3.colors[i % len(plt.cm.Set3.colors)], alpha=0.8)
                    ax.set_yticks(np.arange(len(cat_df)))
                    ax.set_yticklabels(cat_df['Feature'].values)
                    ax.set_title(f'Top Features in {category} Category')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                plt.tight_layout()
                save_path = os.path.join(self.feature_dir, f'{expert_name}_category_breakdown_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
            
            # 5. Create a treemap visualization (requires squarify)
            try:
                import squarify
                
                plt.figure(figsize=(12, 8))
                # Normalize sizes to be between 0 and 1
                sizes = top_df['Importance'].values
                norm_sizes = sizes / sizes.sum()
                
                # Create color scale based on importance
                cmap = plt.cm.get_cmap('viridis')
                colors = [cmap(i) for i in np.linspace(0, 0.8, len(norm_sizes))]
                
                # Create treemap
                squarify.plot(sizes=norm_sizes, label=top_df['Feature'].values, alpha=0.8, color=colors)
                plt.axis('off')
                plt.title(f'{expert_name.capitalize()} Expert - Feature Importance Treemap', fontsize=14)
                
                save_path = os.path.join(self.feature_dir, f'{expert_name}_treemap_feature_importance_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
            except ImportError:
                logger.warning("squarify not installed, skipping treemap visualization")
            
            # 6. Try to create a word cloud of feature importance (requires wordcloud)
            try:
                from wordcloud import WordCloud
                
                # Create a dictionary of feature name -> importance
                feature_importance_dict = dict(zip(top_df['Feature'].values, top_df['Importance'].values))
                
                # Create word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                colormap='viridis', max_words=100).generate_from_frequencies(feature_importance_dict)
                
                plt.figure(figsize=(12, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'{expert_name.capitalize()} Expert - Feature Importance Word Cloud', fontsize=14)
                
                save_path = os.path.join(self.feature_dir, f'{expert_name}_wordcloud_feature_importance_batch_{batch_idx}.png')
                plt.savefig(save_path)
                plt.close()
            except ImportError:
                logger.warning("wordcloud not installed, skipping word cloud visualization")
                
            return True
        except Exception as e:
            logger.error(f"Error creating feature importance visualizations: {str(e)}")
            return False
    
    def generate_all_visualizations(self, interpretation_dir, predictions_path=None):
        """
        Generate all visualizations for all interpretation files in a directory
        
        Args:
            interpretation_dir (str): Directory containing interpretation JSON files
            predictions_path (str, optional): Path to CSV file with predictions
            
        Returns:
            bool: Success status
        """
        try:
            # Load predictions if available
            predictions_df = None
            if predictions_path and os.path.exists(predictions_path):
                predictions_df = pd.read_csv(predictions_path)
                logger.info(f"Loaded predictions from {predictions_path} with {len(predictions_df)} rows")
            
            # Find all interpretation JSON files
            json_files = [f for f in os.listdir(interpretation_dir) 
                          if f.endswith('.json') and 'interpretation' in f]
            
            logger.info(f"Found {len(json_files)} interpretation files")
            
            # Process each file
            success_count = 0
            for i, json_file in enumerate(json_files):
                file_path = os.path.join(interpretation_dir, json_file)
                
                # Extract batch index from filename
                batch_idx = i
                try:
                    # Try to parse batch index from filename (e.g., interpretation_batch_5.json)
                    if 'batch_' in json_file:
                        batch_str = json_file.split('batch_')[1].split('.')[0]
                        batch_idx = int(batch_str)
                except:
                    pass
                
                logger.info(f"Processing interpretation file {json_file} as batch {batch_idx}")
                
                # Generate visualizations for this batch
                success = self.visualize_batch_interpretation(file_path, predictions_df, batch_idx)
                if success:
                    success_count += 1
            
            logger.info(f"Successfully generated visualizations for {success_count}/{len(json_files)} interpretation files")
            
            # Generate summary visualizations across all batches
            # (could be implemented if needed)
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error generating all visualizations: {str(e)}")
            return False

# Main function to be called from the model_interpretation.py script
# Update the create_model_visualizations function to include PDE map visualization
def create_model_visualizations(output_dir, da_coordinates_path=None, ethnicity_names=None, 
                                predictions_path=None, shapefile_path=None):
    """
    Main function to create visualizations from model interpretations
    
    Args:
        output_dir (str): Directory containing interpretation data and where to save visualizations
        da_coordinates_path (str, optional): Path to the CSV file with DA coordinates
        ethnicity_names (list, optional): List of ethnicity names
        predictions_path (str, optional): Path to the predictions CSV file
        shapefile_path (str, optional): Path to shapefile with DA boundaries
        
    Returns:
        bool: Success status
    """
    try:
        logger.info("Initializing model interpretation visualizer...")
        
        # Parse DA coordinates if provided
        map_dauid_to_coords = None
        if da_coordinates_path and os.path.exists(da_coordinates_path):
            try:
                da_coords_df = pd.read_csv(da_coordinates_path)
                logger.info(f"Loaded DA coordinates from {da_coordinates_path} with {len(da_coords_df)} rows")
                
                # Create mapping from DAUID to (X, Y) coordinates
                map_dauid_to_coords = {}
                for _, row in da_coords_df.iterrows():
                    if 'DAUID' in row and 'X_axis' in row and 'Y_axis' in row:
                        map_dauid_to_coords[str(row['DAUID'])] = (row['X_axis'], row['Y_axis'])
                    elif 'DAUID' in row and 'X' in row and 'Y' in row:
                        map_dauid_to_coords[str(row['DAUID'])] = (row['X'], row['Y'])
                
                logger.info(f"Created coordinate mapping for {len(map_dauid_to_coords)} DAUIDs")
            except Exception as e:
                logger.warning(f"Error processing DA coordinates: {str(e)}")
        
        # Initialize visualizer
        visualizer = ModelInterpretationVisualizer(
            output_dir=output_dir,
            ethnicity_names=ethnicity_names,
            map_dauid_to_coords=map_dauid_to_coords
        )
        
        # Find interpretation directory
        interpretation_dir = os.path.join(output_dir, "interpretations")
        if not os.path.exists(interpretation_dir):
            interpretation_dir = output_dir  # Use output dir directly if no interpretations subdirectory
        
        # Check if shapefile is provided for spatial visualizations
        use_shapefile = False
        if shapefile_path and os.path.exists(shapefile_path):
            logger.info(f"Shapefile provided at {shapefile_path}")
            use_shapefile = True
            
            # Check if we have the geopandas library
            try:
                import geopandas
                logger.info("GeoPandas library is available for shapefile processing")
            except ImportError:
                logger.warning("GeoPandas library not available. Please install with 'pip install geopandas' for shapefile support.")
                use_shapefile = False
        
        # Generate all visualizations
        success = False
        
        # Generate standard visualizations
        standard_success = visualizer.generate_all_visualizations(
            interpretation_dir=interpretation_dir,
            predictions_path=predictions_path
        )
        
        # Add shapefile-based visualizations if available
        shapefile_success = False
        if use_shapefile:
            # Load predictions if available for shapefile visualizations
            predictions_df = None
            if predictions_path and os.path.exists(predictions_path):
                predictions_df = pd.read_csv(predictions_path)
                logger.info(f"Loaded predictions from {predictions_path} with {len(predictions_df)} rows")
            
            # Find interpretation files
            json_files = [f for f in os.listdir(interpretation_dir) 
                         if f.endswith('.json') and 'interpretation' in f]
            
            logger.info(f"Found {len(json_files)} interpretation files")
            
            # Process each file with shapefile visualization
            for i, json_file in enumerate(json_files):
                file_path = os.path.join(interpretation_dir, json_file)
                
                # Extract batch index from filename
                batch_idx = i
                try:
                    if 'batch_' in json_file:
                        batch_str = json_file.split('batch_')[1].split('.')[0]
                        batch_idx = int(batch_str)
                except:
                    pass
                
                logger.info(f"Processing interpretation file {json_file} with shapefile visualization")
                
                # Load interpretation data
                interpretation_data = visualizer.load_interpretation_data(file_path)
                if interpretation_data is None:
                    continue
                
                # Run regular visualizations
                visualizer.visualize_batch_interpretation(file_path, predictions_df, batch_idx)
                
                # Add shapefile-based spatial visualizations
                shapefile_batch_success = visualize_spatial_patterns_with_shapefile(
                    visualizer=visualizer,
                    interpretation_data=interpretation_data,
                    predictions_df=predictions_df,
                    shapefile_path=shapefile_path,
                    batch_idx=batch_idx
                )
                
                if shapefile_batch_success:
                    logger.info(f"Successfully created shapefile-based visualizations for batch {batch_idx}")
                    shapefile_success = True
            
            # Generate PDE map-based visualizations that aggregate across all batches and DAUIDs
            pde_success = update_visualize_pde_parameters_on_map(
                visualizer=visualizer,
                interpretation_dir=interpretation_dir, 
                shapefile_path=shapefile_path,
                output_dir=output_dir
            )
            
            
            if pde_success:
                logger.info("Successfully created PDE parameter map visualizations")
                shapefile_success = True
        
        success =  shapefile_success
        
        if success:
            logger.info("Successfully created model interpretation visualizations")
        else:
            logger.warning("Failed to create model interpretation visualizations")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in create_model_visualizations: {str(e)}")
        return False

if __name__ == "__main__":
    # When run as a script, parse arguments and create visualizations
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations from model interpretations")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing interpretation data and where to save visualizations")
    parser.add_argument("--da_coordinates_path", type=str, default=None,
                        help="Path to the CSV file with DA coordinates")
    parser.add_argument("--ethnicity_names", type=str, default=None, nargs='+',
                        help="List of ethnicity names")
    parser.add_argument("--predictions_path", type=str, default=None,
                        help="Path to the predictions CSV file")
    parser.add_argument("--shapefile_path", type=str, default=None,
                        help="Path to shapefile with DA boundaries for spatial visualization")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "visualizations.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create visualizations
    create_model_visualizations(
        output_dir=args.output_dir,
        da_coordinates_path=args.da_coordinates_path,
        ethnicity_names=args.ethnicity_names,
        predictions_path=args.predictions_path,
        shapefile_path=args.shapefile_path
    )