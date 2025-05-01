import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
import networkx as nx
import logging
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import cm
from collections import defaultdict

logger = logging.getLogger(__name__)

class EthnicityInteractionAnalyzer:
    """
    Class for advanced analysis and visualization of ethnicity interactions
    in the multi-ethnic spatial model.
    """
    def __init__(self, 
                 ethnicity_names=None, 
                 output_dir=None, 
                 color_palette='tab10',
                 significant_threshold=0.05):
        """
        Initialize ethnicity interaction analyzer
        
        Args:
            ethnicity_names (list): List of ethnicity names
            output_dir (str): Output directory for visualizations
            color_palette (str): Matplotlib color palette name
            significant_threshold (float): Threshold for significant interactions
        """
        self.ethnicity_names = ethnicity_names
        self.output_dir = output_dir
        
        # Create output directory if needed
        if output_dir:
            self.interaction_dir = os.path.join(output_dir)
            os.makedirs(self.interaction_dir, exist_ok=True)
        
        # Set color palette for ethnicities
        if isinstance(color_palette, str):
            self.ethnicity_colors = plt.cm.get_cmap(color_palette).colors
        else:
            self.ethnicity_colors = color_palette
            
        # Set threshold for significant interactions
        self.significant_threshold = significant_threshold
        
        # Store extracted interaction data
        self.interaction_matrix = None
        self.interaction_effect = None
        self.interaction_by_dauid = None
        self.effect_by_dauid = None
        self.dauid_coordinates = None
        self.ethnicity_flow_data = None
        self.secondary_effects = None
        self.dominant_ethnicities = None
        self.avg_interaction_matrix = None
        self.net_influence = None

    def extract_interaction_data(self, interpretation_data, batch_idx=0):
        """
        Extract and process interaction data from model interpretation
        
        Args:
            interpretation_data (dict): Model interpretation data
            batch_idx (int): Batch index
            
        Returns:
            bool: Success status
        """
        try:
            # Print out the keys in interpretation_data for debugging
            logger.info(f"Keys in interpretation_data: {list(interpretation_data.keys())}")
            
            # Get interaction weights (inter-ethnicity influence)
            if 'interaction_weights' not in interpretation_data:
                logger.warning("No interaction weights found in interpretation data")
                return False
            
            # Get interaction effect (how much each ethnicity is affected)
            if 'interaction_effect' not in interpretation_data:
                logger.warning("No interaction effect found in interpretation data")
                return False
            
            # Extract interaction weights and effects
            interaction_weights = interpretation_data['interaction_weights']
            interaction_effect = interpretation_data['interaction_effect']
            
            # Debug: Check types and shapes
            logger.info(f"interaction_weights type: {type(interaction_weights)}")
            if isinstance(interaction_weights, np.ndarray):
                logger.info(f"interaction_weights shape: {interaction_weights.shape}")
            elif isinstance(interaction_weights, list):
                logger.info(f"interaction_weights length: {len(interaction_weights)}")
                if len(interaction_weights) > 0:
                    logger.info(f"First element type: {type(interaction_weights[0])}")
                    if isinstance(interaction_weights[0], np.ndarray):
                        logger.info(f"First element shape: {interaction_weights[0].shape}")
                    elif isinstance(interaction_weights[0], list):
                        logger.info(f"First element length: {len(interaction_weights[0])}")
            
            # Convert lists to numpy arrays if needed
            if isinstance(interaction_weights, list):
                try:
                    # Try to create a 3D array (batch, ethnicity, ethnicity)
                    self.interaction_matrix = np.array(interaction_weights)
                except Exception as e:
                    logger.warning(f"Error converting interaction weights to array: {str(e)}")
                    # Try a different approach - just use the first batch
                    if len(interaction_weights) > 0:
                        self.interaction_matrix = np.array(interaction_weights[0])
                    else:
                        logger.error("Empty interaction weights list")
                        return False
            else:
                self.interaction_matrix = interaction_weights
            
            # Handle interaction effect
            if isinstance(interaction_effect, list):
                try:
                    self.interaction_effect = np.array(interaction_effect)
                except Exception as e:
                    logger.warning(f"Error converting interaction effect to array: {str(e)}")
                    if len(interaction_effect) > 0:
                        self.interaction_effect = np.array(interaction_effect[0])
                    else:
                        logger.error("Empty interaction effect list")
                        return False
            else:
                self.interaction_effect = interaction_effect
            
            # Get by-DAUID data if available
            if 'ethnicity_interactions_by_dauid' in interpretation_data:
                self.interaction_by_dauid = {}
                original_data = interpretation_data['ethnicity_interactions_by_dauid']
                
                # Transform the data into the expected format
                for dauid, interactions in original_data.items():
                    # Create a properly formatted entry for this DAUID
                    self.interaction_by_dauid[dauid] = {}
                    
                    # Process each interaction key-value pair
                    for key, value in interactions.items():
                        # Extract ethnicity names from keys like "ethnicity_0_to_ethnicity_1"
                        # and store in the expected format
                        if '_to_' in key:
                            parts = key.split('_to_')
                            source_eth = parts[0]
                            target_eth = parts[1]
                            self.interaction_by_dauid[dauid][key] = value
                        else:
                            # Handle other formats as needed
                            self.interaction_by_dauid[dauid][key] = value
                            
                # Log a sample of the transformed data for debugging
                if self.interaction_by_dauid:
                    sample_dauid = next(iter(self.interaction_by_dauid))
                    logger.info(f"Transformed interaction data for DAUID {sample_dauid}: {self.interaction_by_dauid[sample_dauid]}")
            
            if 'interaction_effect_by_dauid' in interpretation_data:
                self.effect_by_dauid = interpretation_data['interaction_effect_by_dauid']
            
            # Get DAUID list if available
            self.dauid_list = interpretation_data.get('dauid_list', None)
            
            # If no ethnicity names provided, create generic ones
            if not self.ethnicity_names:
                # Infer number of ethnicities from interaction matrix
                if len(self.interaction_matrix.shape) >= 3:
                    num_ethnicities = self.interaction_matrix.shape[1]
                else:
                    num_ethnicities = self.interaction_matrix.shape[0]
                
                self.ethnicity_names = [f"Ethnicity {i}" for i in range(num_ethnicities)]
                logger.info(f"Created generic ethnicity names for {num_ethnicities} ethnicities")
            
            # Calculate derived metrics for use in visualizations
            self._calculate_derived_metrics()
            
            return True
        
        except Exception as e:
            logger.error(f"Error extracting interaction data: {str(e)}", exc_info=True)
            return False
    
    def set_dauid_coordinates(self, dauid_to_coords):
        """
        Set mapping from DAUID to spatial coordinates
        
        Args:
            dauid_to_coords (dict): Mapping from DAUID to (x,y) coordinates
        """
        self.dauid_coordinates = dauid_to_coords
    
    def _calculate_derived_metrics(self):
        """
        Calculate derived metrics from interaction data for visualization
        """
        if self.interaction_matrix is None:
            logger.warning("No interaction matrix available for calculating derived metrics")
            return
        
        logger.info(f"Interaction matrix shape: {self.interaction_matrix.shape}")
        
        # Average interaction weights across the batch if needed
        if len(self.interaction_matrix.shape) >= 3:
            # We have a batch dimension
            avg_interactions = np.mean(self.interaction_matrix, axis=0)
            logger.info(f"Averaging interactions across batch dimension. Result shape: {avg_interactions.shape}")
        else:
            # Already a 2D matrix
            avg_interactions = self.interaction_matrix
            logger.info(f"Using interaction matrix directly. Shape: {avg_interactions.shape}")
        
        self.avg_interaction_matrix = avg_interactions
        
        # Calculate net effect (outgoing - incoming) for each ethnicity
        # This shows whether an ethnicity is more influential or more influenced
        n = avg_interactions.shape[0]  # Number of ethnicities
        net_effect = np.zeros(n)
        
        for i in range(n):
            # Sum of outgoing effects (how much this ethnicity influences others)
            outgoing = np.sum(avg_interactions[:, i]) - avg_interactions[i, i]
            
            # Sum of incoming effects (how much others influence this ethnicity)
            incoming = np.sum(avg_interactions[i, :]) - avg_interactions[i, i]
            
            # Calculate net effect as a scalar value
            net_effect[i] = float(outgoing - incoming)
        
        self.net_influence = net_effect
        logger.info(f"Calculated net influence: {self.net_influence}")
        
        # Calculate ethnicity dominance (which ethnicity has the strongest influence on each other)
        dominant_ethnicities = np.zeros(n, dtype=int)
        
        for i in range(n):
            # Find ethnicity with max influence on this one (excluding self)
            influences = avg_interactions[i, :].copy()
            influences[i] = -1  # Exclude self-influence
            dominant_ethnicities[i] = np.argmax(influences)
        
        self.dominant_ethnicities = dominant_ethnicities
        
        # Calculate cascade effects (secondary influence through intermediary ethnicities)
        # This represents multi-step influence paths (A→B→C)
        # We do this by multiplying the interaction matrix with itself
        secondary_effects = np.matmul(avg_interactions, avg_interactions)
        
        # Zero out the diagonal for clarity
        np.fill_diagonal(secondary_effects, 0)
        
        self.secondary_effects = secondary_effects
        
        # Calculate ethnicity flow data for spatial visualization if coordinates are available
        if self.interaction_by_dauid and self.dauid_coordinates:
            logger.info("Calculating spatial flow data")
            try:
                self.ethnicity_flow_data = self._calculate_spatial_flow()
            except Exception as e:
                logger.error(f"Error calculating spatial flow: {str(e)}", exc_info=True)
                self.ethnicity_flow_data = None

    def _calculate_spatial_flow(self):
        """
        Calculate spatial visualization data for ethnicity interactions within each DAUID
        """
        if not self.interaction_by_dauid or not self.dauid_coordinates:
            logger.warning("No interaction_by_dauid or dauid_coordinates available")
            return None
        
        # Debug the data structure
        logger.info(f"Number of DAUIDs with interaction data: {len(self.interaction_by_dauid)}")
        logger.info(f"Number of DAUIDs with coordinates: {len(self.dauid_coordinates)}")
        
        # Check overlap
        common_dauids = set(self.interaction_by_dauid.keys()) & set(self.dauid_coordinates.keys())
        logger.info(f"Number of DAUIDs with both interaction data and coordinates: {len(common_dauids)}")
        if not common_dauids:
            logger.warning("No common DAUIDs between interaction data and coordinates!")
            logger.info(f"Sample DAUIDs from interaction data: {list(self.interaction_by_dauid.keys())[:5]}")
            logger.info(f"Sample DAUIDs from coordinates: {list(self.dauid_coordinates.keys())[:5]}")
            return None  # No common DAUIDs, can't create spatial visualization
        
        # Instead of organizing by ethnicity, organize by DAUID
        flow_data = {}
        
        # Initialize data structure
        for eth_name in self.ethnicity_names:
            flow_data[eth_name] = {
                'dauid': [],
                'x': [],
                'y': [],
                'interactions': []  # Will store dict of interactions for this ethnicity at each DAUID
            }
        
        # Process the interaction data for each DAUID
        for dauid in common_dauids:
            interactions = self.interaction_by_dauid[dauid]
            
            # Debug the first DAUID's data
            if dauid == list(common_dauids)[0]:
                logger.info(f"Example interactions for DAUID {dauid}: {interactions}")
            
            # Skip if DAUID doesn't have coordinates
            if dauid not in self.dauid_coordinates:
                continue
                
            # Get coordinates for this DAUID
            x, y = self.dauid_coordinates[dauid]
            
            # Process each ethnicity
            for eth_idx, eth_name in enumerate(self.ethnicity_names):
                # Extract all interactions for this ethnicity
                eth_interactions = {}
                significant_interaction_found = False
                
                # Debug the ethnicity keys we're looking for
                if eth_idx == 0 and dauid == list(common_dauids)[0]:
                    example_keys = [f"{eth_name}_to_{other_eth}" for other_eth in self.ethnicity_names]
                    logger.info(f"Looking for keys like: {example_keys}")
                
                for key, value in interactions.items():
                    # Check if this key represents an interaction from this ethnicity
                    if key.startswith(f"{eth_name}_to_"):
                        target_eth = key.split("_to_")[1]
                        eth_interactions[target_eth] = value
                        
                        # Debug the value and threshold comparison
                        if eth_idx == 0 and dauid == list(common_dauids)[0]:
                            logger.info(f"Found key {key} with value {value}, threshold is {self.significant_threshold}")
                        
                        # Check if any interaction exceeds the threshold
                        if value > self.significant_threshold:
                            significant_interaction_found = True
                            if eth_idx == 0 and dauid == list(common_dauids)[0]:
                                logger.info(f"Significant interaction found: {key} = {value}")
                
                # Only add to flow data if we found at least one significant interaction
                if significant_interaction_found:
                    flow_data[eth_name]['dauid'].append(dauid)
                    flow_data[eth_name]['x'].append(x)
                    flow_data[eth_name]['y'].append(y)
                    flow_data[eth_name]['interactions'].append(eth_interactions)
                    logger.info(f"Added significant interactions for {eth_name} at DAUID {dauid}")
                else:
                    logger.info(f"No significant interactions found for {eth_name} at DAUID {dauid}")
        
        # Check if we found any data
        empty_ethnicities = []
        for eth_name in self.ethnicity_names:
            if not flow_data[eth_name]['dauid']:
                empty_ethnicities.append(eth_name)
        
        if empty_ethnicities:
            logger.warning(f"No flow data for ethnicities: {empty_ethnicities}")
        
        return flow_data

    def visualize_interaction_network(self, output_path=None):
        """
        Create a network visualization of ethnicity interactions
        
        Args:
            output_path (str, optional): Custom output path
            
        Returns:
            bool: Success status
        """
        try:
            if self.avg_interaction_matrix is None:
                logger.warning("No interaction matrix available for network visualization")
                return False
            
            logger.info("Creating network visualization")
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes for each ethnicity
            for i, name in enumerate(self.ethnicity_names):
                # Make sure we don't go out of bounds
                if i < len(self.net_influence):
                    # Calculate node size based on net influence
                    node_size = 1000 + abs(self.net_influence[i]) * 5000
                    
                    # Determine node color
                    if self.net_influence[i] > 0:
                        # Positive influence (influencer)
                        color = 'blue'
                    else:
                        # Negative influence (influenced)
                        color = 'red'
                    
                    # Add node with attributes
                    G.add_node(i, name=name, size=node_size, color=color, 
                              net_influence=self.net_influence[i])
                else:
                    # Default values for nodes beyond net_influence length
                    G.add_node(i, name=name, size=1000, color='gray', 
                              net_influence=0.0)
            
            # Add edges for significant interactions
            for i in range(self.avg_interaction_matrix.shape[0]):
                for j in range(self.avg_interaction_matrix.shape[1]):
                    if i != j:  # Skip self-loops
                        weight = self.avg_interaction_matrix[i, j]
                        
                        # Only add significant edges
                        if weight > self.significant_threshold:
                            G.add_edge(j, i, weight=weight)
            
            # Check if graph has any edges
            if len(G.edges()) == 0:
                logger.warning("No significant interactions found for network visualization")
                # Add a fallback - use the top 10% of interactions
                threshold = np.percentile(self.avg_interaction_matrix.flatten(), 90)
                logger.info(f"Using fallback threshold: {threshold}")
                
                for i in range(self.avg_interaction_matrix.shape[0]):
                    for j in range(self.avg_interaction_matrix.shape[1]):
                        if i != j and self.avg_interaction_matrix[i, j] > threshold:
                            G.add_edge(j, i, weight=self.avg_interaction_matrix[i, j])
            
            # Create visualization
            plt.figure(figsize=(14, 12))
            
            # Set positions using a spring layout
            pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)
            
            # Draw nodes with varying sizes and colors
            node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
            node_colors = [G.nodes[node]['color'] for node in G.nodes()]
            
            # Draw network components
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                  node_color=node_colors, alpha=0.8)
            
            # Draw edges if there are any
            if len(G.edges()) > 0:
                edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
                
                # Use a colormap for edge weights
                edge_colors = [plt.cm.viridis(w/max(edge_weights)) if max(edge_weights) > 0 else 'gray' 
                              for w in edge_weights]
                
                nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors,
                                      alpha=0.7, arrowsize=20, arrowstyle='->')
            
            # Add labels
            labels = {node: G.nodes[node]['name'] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, 
                                   font_weight='bold')
            
            # Add a title and legend
            plt.title("Ethnicity Interaction Network", fontsize=16)
            
            # Add legend manually
            legend_elements = [
                Line2D([0], [0], color='blue', marker='o', markersize=10, 
                      label='Net Influencer (outgoing > incoming)', linestyle=''),
                Line2D([0], [0], color='red', marker='o', markersize=10, 
                      label='Net Influenced (incoming > outgoing)', linestyle=''),
                Line2D([0], [0], color='green', markersize=0, 
                      label='Node size = influence magnitude'),
                Line2D([0], [0], color='black', linewidth=2, 
                      label='Edge width = interaction strength')
            ]
            
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save the visualization
            if output_path:
                plt.savefig(output_path, dpi=300)
            else:
                plt.savefig(os.path.join(self.interaction_dir, 'ethnicity_interaction_network.png'), dpi=300)
            
            plt.close()
            
            # Proceed with creating other visualizations if possible
            try:
                self._visualize_influence_chains()
            except Exception as e:
                logger.warning(f"Could not create influence chains: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating ethnicity interaction network: {str(e)}", exc_info=True)
            return False
    
    def _visualize_influence_chains(self):
        """
        Create a visualization showing primary influence chains between ethnicities
        """
        if self.avg_interaction_matrix is None or self.dominant_ethnicities is None:
            return
        
        logger.info("Creating influence chain visualization")
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each ethnicity
        for i, name in enumerate(self.ethnicity_names):
            G.add_node(i, name=name)
        
        # Add edges for dominant influences only
        for i, dominant in enumerate(self.dominant_ethnicities):
            if dominant < self.avg_interaction_matrix.shape[1]:  # Make sure we don't go out of bounds
                weight = self.avg_interaction_matrix[i, dominant]
                G.add_edge(dominant, i, weight=weight)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Use a hierarchical layout to show influence chains - fall back to spring layout if it fails
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_colors = [self.ethnicity_colors[i % len(self.ethnicity_colors)] for i in range(len(self.ethnicity_names))]
        nx.draw_networkx_nodes(G, pos, node_size=800, 
                              node_color=node_colors, alpha=0.8)
        
        # Draw edges if there are any
        if len(G.edges()) > 0:
            edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, 
                                  edge_color='gray', alpha=0.7, 
                                  arrowsize=20, arrowstyle='->')
        
        # Add labels
        labels = {node: G.nodes[node]['name'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, 
                               font_weight='bold')
        
        # Add a title
        plt.title("Dominant Ethnicity Influence Chains", fontsize=16)
        
        # Add legend using ethnicity colors
        legend_elements = []
        for i, name in enumerate(self.ethnicity_names):
            if i < len(self.ethnicity_colors):
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=self.ethnicity_colors[i],
                          markersize=10, label=name)
                )
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.interaction_dir, 'ethnicity_dominant_influences.png'), dpi=300)
        plt.close()

    def visualize_flow_diagrams(self):
        """
        Create a comprehensive set of influence flow diagrams among ethnicities
        
        Returns:
            bool: Success status
        """
        try:
            if self.avg_interaction_matrix is None:
                logger.warning("No interaction matrix available for flow diagrams")
                return False
            
            logger.info("Creating flow diagram visualizations")
            
            # Create annotated heatmap - this is the most reliable visualization
            self._create_annotated_heatmap()
            
            # Try to create other visualizations with fallbacks
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import plotly.io as pio
                
                # Create Sankey diagram
                self._create_plotly_sankey()
            except ImportError:
                logger.warning("Plotly not installed, skipping Sankey diagram")
            except Exception as e:
                logger.warning(f"Error creating Sankey diagram: {str(e)}")
            
            # Try to create chord diagram
            try:
                self._create_chord_diagram()
            except Exception as e:
                logger.warning(f"Error creating chord diagram: {str(e)}")
            
            # Create cascade visualization
            try:
                self._visualize_influence_cascade()
            except Exception as e:
                logger.warning(f"Error creating influence cascade visualization: {str(e)}")
            
            logger.info("Successfully created flow diagram visualizations")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ethnicity flow diagrams: {str(e)}", exc_info=True)
            return False
    
    def _create_plotly_sankey(self):
        """Create a Sankey diagram using Plotly"""
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # Prepare data for Sankey diagram
        sources = []
        targets = []
        values = []
        colors = []
        
        # Add significant interactions to the diagram
        for i in range(self.avg_interaction_matrix.shape[0]):
            for j in range(self.avg_interaction_matrix.shape[1]):
                if i != j:  # Skip self-loops
                    weight = self.avg_interaction_matrix[i, j]
                    
                    # Only add significant interactions
                    if weight > self.significant_threshold:
                        sources.append(j)
                        targets.append(i)
                        values.append(weight)
                        if j < len(self.ethnicity_colors):
                            colors.append(f'rgba({int(self.ethnicity_colors[j][0]*255)}, '
                                        f'{int(self.ethnicity_colors[j][1]*255)}, '
                                        f'{int(self.ethnicity_colors[j][2]*255)}, 0.8)')
                        else:
                            colors.append('rgba(150, 150, 150, 0.8)')
        
        # If no significant interactions, use top 10%
        if not values:
            logger.warning("No significant interactions for Sankey diagram, using top 10%")
            threshold = np.percentile(self.avg_interaction_matrix.flatten(), 90)
            
            for i in range(self.avg_interaction_matrix.shape[0]):
                for j in range(self.avg_interaction_matrix.shape[1]):
                    if i != j and self.avg_interaction_matrix[i, j] > threshold:
                        sources.append(j)
                        targets.append(i)
                        values.append(self.avg_interaction_matrix[i, j])
                        if j < len(self.ethnicity_colors):
                            colors.append(f'rgba({int(self.ethnicity_colors[j][0]*255)}, '
                                        f'{int(self.ethnicity_colors[j][1]*255)}, '
                                        f'{int(self.ethnicity_colors[j][2]*255)}, 0.8)')
                        else:
                            colors.append('rgba(150, 150, 150, 0.8)')
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=self.ethnicity_names,
                color=[f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.8)' 
                      for c in self.ethnicity_colors[:len(self.ethnicity_names)]]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )])
        
        # Update layout
        fig.update_layout(
            title_text="Ethnicity Interaction Flows",
            font_size=12,
            height=800
        )
        
        # Save as PNG
        pio.write_image(fig, os.path.join(self.interaction_dir, 'ethnicity_sankey_diagram.png'), 
                       width=1200, height=800, scale=2)
        
        # Save as interactive HTML
        pio.write_html(fig, os.path.join(self.interaction_dir, 'ethnicity_sankey_diagram_interactive.html'))
        
        logger.info("Created Sankey diagram for ethnicity interactions")
    
    def _create_chord_diagram(self):
        """
        Create a chord diagram showing ethnicity interactions
        """
        try:
            # Check if we have matplotlib's chord diagram capabilities
            from matplotlib.path import Path
            import matplotlib.patches as patches
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
            
            # Number of ethnicities
            n = len(self.ethnicity_names)
            
            # Get the interaction matrix data
            interaction_matrix = self.avg_interaction_matrix.copy()
            
            # Zero out insignificant interactions for clarity
            interaction_matrix[interaction_matrix < self.significant_threshold] = 0
            
            # Set up the angles for each ethnicity (evenly spaced around the circle)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            
            # Width of each ethnicity segment on the circle
            width = 2*np.pi / n * 0.95  # slightly less than the full segment
            
            # Draw the outer segments for each ethnicity
            for i in range(n):
                # Calculate total outgoing influence for scaling
                total_outgoing = np.sum(interaction_matrix[:, i])
                
                # Draw segment
                theta = angles[i]
                radii = 0.8  # Outer ring radius
                
                # Create a wedge for this ethnicity
                wedge = patches.Wedge(
                    center=(0, 0), 
                    r=radii, 
                    theta1=np.degrees(theta), 
                    theta2=np.degrees(theta + width),
                    width=0.2,
                    facecolor=self.ethnicity_colors[i % len(self.ethnicity_colors)],
                    alpha=0.7,
                    edgecolor='white',
                    linewidth=1
                )
                
                ax.add_patch(wedge)
                
                # Add label
                label_angle = theta + width/2
                
                ax.text(
                    label_angle, radii + 0.22,
                    self.ethnicity_names[i],
                    ha='center', va='center',
                    rotation=np.degrees(label_angle) - 90 if np.cos(label_angle) < 0 else np.degrees(label_angle) + 90,
                    fontsize=10, fontweight='bold',
                    rotation_mode='anchor'
                )
            
            # Draw links between ethnicities
            max_influence = np.max(interaction_matrix)
            
            for i in range(n):
                for j in range(n):
                    if i != j and interaction_matrix[i, j] > self.significant_threshold:
                        # Calculate positions
                        start_angle = angles[j] + width/2
                        end_angle = angles[i] + width/2
                        
                        # Scale width by influence strength
                        link_width = 0.1 * (interaction_matrix[i, j] / max_influence)
                        
                        # Create a curved path from source to target
                        verts = [
                            # Move to the start point
                            (start_angle, 0.8),
                            # Control points for the curve
                            (start_angle, 0.3),
                            (end_angle, 0.3),
                            # End point
                            (end_angle, 0.8)
                        ]
                        
                        codes = [
                            Path.MOVETO,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4
                        ]
                        
                        path = Path(verts, codes)
                        
                        # Create a patch from the path
                        patch = patches.PathPatch(
                            path,
                            facecolor='none',
                            edgecolor=self.ethnicity_colors[j % len(self.ethnicity_colors)],
                            alpha=0.6,
                            linewidth=link_width * 10
                        )
                        
                        ax.add_patch(patch)
            
            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove axis spines
            ax.spines['polar'].set_visible(False)
            
            # Add a title
            plt.title('Ethnicity Interaction Chord Diagram', fontsize=16, y=1.1)
            
            # Add a legend for interaction strength
            legend_elements = [
                Line2D([0], [0], color='gray', linewidth=1, 
                      label=f'Low influence (>{self.significant_threshold:.2f})'),
                Line2D([0], [0], color='gray', linewidth=3, 
                      label=f'Medium influence (>{self.significant_threshold*2:.2f})'),
                Line2D([0], [0], color='gray', linewidth=5, 
                      label=f'Strong influence (>{self.significant_threshold*3:.2f})')
            ]
            
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05))
            
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(os.path.join(self.interaction_dir, 'ethnicity_chord_diagram.png'), dpi=300)
            plt.close()
            
            logger.info("Created chord diagram for ethnicity interactions")
            
        except Exception as e:
            raise Exception(f"Error creating chord diagram: {str(e)}")
    
    def _create_annotated_heatmap(self):
        """
        Create an enhanced heatmap of the interaction matrix with annotations
        """
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Number of ethnicities
        n = len(self.ethnicity_names)
        
        # Create heatmap
        # Create mask for diagonal if needed, but don't use if n is small
        mask = np.eye(self.avg_interaction_matrix.shape[0], dtype=bool) if n > 3 else None
        
        # Using seaborn for a nicer heatmap
        ax = sns.heatmap(
            self.avg_interaction_matrix,
            mask=mask,
            cmap='viridis',
            annot=True,
            fmt='.3f',
            linewidths=0.5,
            square=True,
            cbar_kws={'label': 'Interaction Strength'},
            xticklabels=self.ethnicity_names,
            yticklabels=self.ethnicity_names
        )
        
        # Try to add enhancements if dominant ethnicities are available
        try:
            if self.dominant_ethnicities is not None:
                # Add colored edges to show dominant influences
                for i in range(n):
                    if i < len(self.dominant_ethnicities):
                        dominant = self.dominant_ethnicities[i]
                        if i != dominant and dominant < n:  # Skip self-dominance and out of bounds
                            # Find the cell in the heatmap
                            cell_coords = (i, dominant)
                            
                            # Add a colored border to the cell
                            rect = plt.Rectangle(
                                (cell_coords[1], cell_coords[0]),
                                1, 1,
                                fill=False,
                                edgecolor='red',
                                linewidth=3,
                                clip_on=False
                            )
                            
                            ax.add_patch(rect)
        except Exception as e:
            logger.warning(f"Could not add dominant influences to heatmap: {str(e)}")
        
        # Try to add more annotations if secondary effects are available
        try:
            if self.secondary_effects is not None:
                # Highlight cells based on significance
                for i in range(min(n, self.avg_interaction_matrix.shape[0])):
                    for j in range(min(n, self.avg_interaction_matrix.shape[1])):
                        if i != j:  # Skip diagonal
                            val = self.avg_interaction_matrix[i, j]
                            
                            # Add a star to significant interactions
                            if val > self.significant_threshold * 2:
                                ax.text(j + 0.5, i + 0.7, '★', 
                                      ha='center', va='center', color='yellow', fontsize=15)
                            
                            # Add a circle to interactions that are part of a cascade
                            if i < self.secondary_effects.shape[0] and j < self.secondary_effects.shape[1]:
                                if self.secondary_effects[i, j] > self.significant_threshold * 2:
                                    ax.text(j + 0.5, i + 0.3, '●', 
                                          ha='center', va='center', color='cyan', fontsize=15)
        except Exception as e:
            logger.warning(f"Could not add annotations to heatmap: {str(e)}")
        
        # Add title and labels
        plt.title('Ethnicity Interaction Heatmap', fontsize=16)
        plt.xlabel('Source Ethnicity (Influencer)', fontsize=12)
        plt.ylabel('Target Ethnicity (Influenced)', fontsize=12)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add a legend for the annotations
        legend_elements = [
            Patch(facecolor='white', edgecolor='red', label='Dominant influence'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', 
                  markersize=10, label='Highly significant interaction'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                  markersize=10, label='Part of a significant cascade')
        ]
        
        plt.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.interaction_dir, 'ethnicity_annotated_heatmap.png'), dpi=300)
        plt.close()
        
        logger.info("Created annotated heatmap for ethnicity interactions")
    
    def _visualize_influence_cascade(self):
        """
        Create a visualization showing influence cascades
        """
        # Create a figure
        fig = plt.figure(figsize=(14, 10))
        
        # Number of ethnicities
        n = len(self.ethnicity_names)
        
        # Compute total influence on each ethnicity
        total_influenced = np.sum(self.avg_interaction_matrix, axis=1)
        total_influencer = np.sum(self.avg_interaction_matrix, axis=0)
        
        # Create a 2x2 grid for different plots
        gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 2])
        
        # 1. Top-left: Primary influence arrows
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Define ethnicity positions
        x_pos = np.arange(n)
        y_pos = total_influenced  # Position by how much they're influenced
        
        # Plot ethnicities as circles
        for i in range(n):
            ax1.scatter(x_pos[i], y_pos[i], 
                       s=1000 * (total_influencer[i] / np.max(total_influencer)) if np.max(total_influencer) > 0 else 1000,
                       color=self.ethnicity_colors[i % len(self.ethnicity_colors)],
                       alpha=0.8, edgecolor='black', linewidth=1)
            
            ax1.text(x_pos[i], y_pos[i], self.ethnicity_names[i], 
                    ha='center', va='center', fontweight='bold')
        
        # Draw arrows for primary influences
        significant_count = 0
        for i in range(n):
            for j in range(n):
                if i != j and self.avg_interaction_matrix[i, j] > self.significant_threshold:
                    ax1.annotate(
                        '',
                        xy=(x_pos[i], y_pos[i]),  # End point
                        xytext=(x_pos[j], y_pos[j]),  # Start point
                        arrowprops=dict(
                            arrowstyle='->',
                            lw=self.avg_interaction_matrix[i, j] * 5,
                            color=self.ethnicity_colors[j % len(self.ethnicity_colors)],
                            alpha=0.6,
                            connectionstyle='arc3,rad=0.2'
                        )
                    )
                    significant_count += 1
        
        # If no significant interactions, add some anyway
        if significant_count == 0:
            logger.warning("No significant primary influences found, adding top interactions")
            # Add top 3 interactions per ethnicity
            for i in range(n):
                influences = self.avg_interaction_matrix[i, :]
                top_indices = np.argsort(influences)[-3:]  # Top 3 influences
                
                for j in top_indices:
                    if i != j and influences[j] > 0:
                        ax1.annotate(
                            '',
                            xy=(x_pos[i], y_pos[i]),  # End point
                            xytext=(x_pos[j], y_pos[j]),  # Start point
                            arrowprops=dict(
                                arrowstyle='->',
                                lw=influences[j] * 10,  # Make lines more visible
                                color=self.ethnicity_colors[j % len(self.ethnicity_colors)],
                                alpha=0.6,
                                connectionstyle='arc3,rad=0.2'
                            )
                        )
        
        # Set title and labels
        ax1.set_title('Primary Ethnicity Influences', fontsize=14)
        ax1.set_xlabel('Ethnicity Index')
        ax1.set_ylabel('Total Influence Received')
        
        # Set x-ticks to ethnicity indices
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([str(i) for i in range(n)])
        
        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Top-right: Net influence bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate net influence (outgoing - incoming)
        net_influence = total_influencer - total_influenced
        
        # Create bar chart
        bars = ax2.barh(x_pos, net_influence, 
                       color=[self.ethnicity_colors[i % len(self.ethnicity_colors)] for i in range(n)],
                       alpha=0.7)
        
        # Change color for negative values
        for i, bar in enumerate(bars):
            if net_influence[i] < 0:
                bar.set_color('gray')
                bar.set_alpha(0.5)
        
        # Add a zero line
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set title and labels
        ax2.set_title('Net Influence\n(Outgoing - Incoming)', fontsize=14)
        ax2.set_xlabel('Net Influence Score')
        
        # Set y-ticks to match ethnicity indices
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels([str(i) for i in range(n)])
        
        # 3. Bottom-left: Secondary effects matrix if available
        ax3 = fig.add_subplot(gs[1, 0])
        
        if self.secondary_effects is not None:
            # Create heatmap of secondary effects
            im = ax3.imshow(self.secondary_effects, cmap='plasma', interpolation='nearest')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Secondary Influence Strength\n(via intermediaries)')
            
            # Set title and labels
            ax3.set_title('Secondary Ethnicity Influences (A→B→C)', fontsize=14)
            ax3.set_xlabel('Source Ethnicity')
            ax3.set_ylabel('Target Ethnicity')
            
            # Set ticks to ethnicity names
            ax3.set_xticks(np.arange(min(n, self.secondary_effects.shape[1])))
            ax3.set_yticks(np.arange(min(n, self.secondary_effects.shape[0])))
            ax3.set_xticklabels(self.ethnicity_names[:min(n, self.secondary_effects.shape[1])], 
                              rotation=45, ha='right')
            ax3.set_yticklabels(self.ethnicity_names[:min(n, self.secondary_effects.shape[0])])
            
            # Add text annotations
            for i in range(min(n, self.secondary_effects.shape[0])):
                for j in range(min(n, self.secondary_effects.shape[1])):
                    if i != j:  # Skip diagonal
                        text = ax3.text(j, i, f'{self.secondary_effects[i, j]:.3f}',
                                       ha="center", va="center", 
                                       color="white" if self.secondary_effects[i, j] > 0.1 else "black",
                                       fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'Secondary effects not available', 
                    ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4. Bottom-right: Simple cascade examples
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create a simple cascade visualization with artificial examples
        ax4.text(0.5, 0.9, "Example Influence Cascades", ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        # Draw a sample cascade
        for i in range(min(3, n)):
            # Position vertically
            y_top = 0.7 - i * 0.25
            y_mid = y_top - 0.08
            y_bot = y_top - 0.16
            
            # Create a sample cascade
            source_idx = i
            mid_idx = (i + 1) % n
            target_idx = (i + 2) % n
            
            # Draw nodes
            ax4.scatter([0.2], [y_top], s=300, 
                       color=self.ethnicity_colors[source_idx % len(self.ethnicity_colors)],
                       edgecolor='black', linewidth=1)
            
            ax4.scatter([0.5], [y_mid], s=300, 
                       color=self.ethnicity_colors[mid_idx % len(self.ethnicity_colors)],
                       edgecolor='black', linewidth=1)
            
            ax4.scatter([0.8], [y_bot], s=300, 
                       color=self.ethnicity_colors[target_idx % len(self.ethnicity_colors)],
                       edgecolor='black', linewidth=1)
            
            # Add ethnicity names
            ax4.text(0.2, y_top, f' {self.ethnicity_names[source_idx]}', 
                    ha='center', va='center', fontweight='bold')
            
            ax4.text(0.5, y_mid, f' {self.ethnicity_names[mid_idx]}', 
                    ha='center', va='center', fontweight='bold')
            
            ax4.text(0.8, y_bot, f' {self.ethnicity_names[target_idx]}', 
                    ha='center', va='center', fontweight='bold')
            
            # Draw arrows
            arrow_props = dict(
                arrowstyle='->', lw=2, color='gray', alpha=0.8,
                connectionstyle='arc3,rad=-0.2'
            )
            
            ax4.annotate('', xy=(0.5, y_mid), xytext=(0.2, y_top), arrowprops=arrow_props)
            ax4.annotate('', xy=(0.8, y_bot), xytext=(0.5, y_mid), arrowprops=arrow_props)
            
            # Add explanation
            ax4.text(0.5, y_bot - 0.05, 
                    f'When {self.ethnicity_names[source_idx]} influences {self.ethnicity_names[mid_idx]},\n' +
                    f'which influences {self.ethnicity_names[target_idx]}', 
                    ha='center', va='top', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # Turn off axis
        ax4.axis('off')
        
        # Add overall title
        fig.suptitle('Ethnicity Influence Cascade Analysis', fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.interaction_dir, 'ethnicity_cascade_analysis.png'), dpi=300)
        plt.close()
        
        logger.info("Created influence cascade visualization")

    def visualize_spatial_interactions(self, gdf=None, shapefile_path=None):
        """
        Create spatial visualizations of ethnicity interactions
        
        Args:
            gdf (GeoDataFrame, optional): GeoPandas dataframe with geometries
            shapefile_path (str, optional): Path to shapefile
            
        Returns:
            bool: Success status
        """
        try:
            # Check if we have spatial data available
            if not self.dauid_coordinates and not gdf and not shapefile_path:
                logger.warning("No spatial data available for spatial interaction visualization")
                return False
            
            logger.info("Creating spatial interaction visualizations")
            
            # Load shapefile if provided
            if gdf is None and shapefile_path is not None:
                try:
                    gdf = gpd.read_file(shapefile_path)
                    logger.info(f"Loaded shapefile with {len(gdf)} geometries")
                except Exception as e:
                    logger.error(f"Error loading shapefile: {str(e)}")
                    gdf = None
            
            # Find DAUID column in shapefile
            dauid_column = None
            if gdf is not None:
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
                
                # If still no DAUID column found, add a warning
                if dauid_column is None:
                    logger.warning("No DAUID column found in shapefile")
                else:
                    # Ensure DAUID is string type for consistent joining
                    gdf[dauid_column] = gdf[dauid_column].astype(str)
            
            # Create a comprehensive interaction visualization showing all ethnicities together
            if self.ethnicity_flow_data and self.dauid_coordinates and gdf is not None:
                # Create a figure
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Draw geometries from shapefile as base map
                gdf.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.5, alpha=0.3)
                
                # Create a mapping from DAUID to centroids in the shapefile
                dauid_to_centroid = {}
                if dauid_column is not None:
                    for idx, row in gdf.iterrows():
                        dauid_to_centroid[str(row[dauid_column])] = (row.geometry.centroid.x, row.geometry.centroid.y)
                
                # Get all DAUIDs with ethnicity interaction data
                all_dauids = set()
                for eth_name in self.ethnicity_flow_data:
                    all_dauids.update(self.ethnicity_flow_data[eth_name]['dauid'])
                
                # Plot interactions for all DAUIDs
                for dauid in all_dauids:
                    if dauid in dauid_to_centroid:
                        # Use shapefile centroid coordinates
                        x, y = dauid_to_centroid[dauid]
                        
                        # Find all ethnicities with interactions at this DAUID
                        ethnicities_with_interactions = []
                        for eth_idx, eth_name in enumerate(self.ethnicity_names):
                            if (eth_name in self.ethnicity_flow_data and 
                                dauid in self.ethnicity_flow_data[eth_name]['dauid']):
                                ethnicities_with_interactions.append((eth_idx, eth_name))
                        
                        if ethnicities_with_interactions:
                            # Draw a marker for each ethnicity with interactions
                            for i, (eth_idx, eth_name) in enumerate(ethnicities_with_interactions):
                                # Get the index where this DAUID appears in the flow data
                                dauid_idx = self.ethnicity_flow_data[eth_name]['dauid'].index(dauid)
                                interactions = self.ethnicity_flow_data[eth_name]['interactions'][dauid_idx]
                                
                                # Calculate the total interaction strength for sizing
                                total_strength = sum(v for k, v in interactions.items() 
                                                if v > self.significant_threshold)
                                
                                # Plot a marker sized by total interaction strength
                                marker_size = 100 + 500 * total_strength
                                ax.scatter(x, y, s=marker_size, 
                                        color=self.ethnicity_colors[eth_idx % len(self.ethnicity_colors)],
                                        alpha=0.7, edgecolor='black', linewidth=0.5, 
                                        label=eth_name if i == 0 else "")
                
                # Add a legend for ethnicities
                legend_elements = []
                for i, name in enumerate(self.ethnicity_names):
                    if i < len(self.ethnicity_colors):
                        legend_elements.append(
                            Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=self.ethnicity_colors[i % len(self.ethnicity_colors)],
                                markersize=10, label=name)
                        )
                
                # Add legend for marker sizes
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                        markersize=8, label='Weaker Interactions')
                )
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                        markersize=14, label='Stronger Interactions')
                )
                
                ax.legend(handles=legend_elements, loc='upper right')
                
                # Set title
                ax.set_title("Spatial Distribution of Ethnicity Interactions", fontsize=16)
                
                # Turn off axis
                ax.set_axis_off()
                
                plt.tight_layout()
                
                # Save the visualization
                plt.savefig(os.path.join(self.interaction_dir, 'spatial_ethnicity_interactions_combined.png'), dpi=300)
                plt.close()
                
                # Create an interactive plot showing influence networks
                self._create_comprehensive_interaction_map(gdf, dauid_column)
            
            # Individual ethnicity visualizations (modified to use shapefile coordinates)
            if self.ethnicity_flow_data and self.dauid_coordinates:
                # If we have flow data and coordinates, create individual ethnicity flows
                for eth_idx, eth_name in enumerate(self.ethnicity_names):
                    try:
                        self._create_improved_ethnicity_flow(eth_idx, eth_name, gdf, dauid_column)
                    except Exception as e:
                        logger.warning(f"Error creating spatial flow for {eth_name}: {str(e)}")
            
            logger.info("Successfully created spatial ethnicity interaction visualizations")
            return True
            
        except Exception as e:
            logger.error(f"Error creating spatial ethnicity interaction visualizations: {str(e)}", exc_info=True)
            return False
    
    def _create_improved_ethnicity_flow(self, eth_idx, eth_name, gdf=None, dauid_column=None):
        """
        Create an improved spatial visualization showing ethnicity interactions
        
        Args:
            eth_idx (int): Ethnicity index
            eth_name (str): Ethnicity name
            gdf (GeoDataFrame): GeoPandas dataframe with geometries
            dauid_column (str): DAUID column name in GeoDataFrame
        """
        if eth_name not in self.ethnicity_flow_data or not self.ethnicity_flow_data[eth_name]['dauid']:
            logger.warning(f"No flow data available for {eth_name}")
            return
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Draw geometries if available
        if gdf is not None and dauid_column is not None:
            gdf.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.5, alpha=0.3)
        
        # Create a mapping from DAUID to centroids in the shapefile
        dauid_to_centroid = {}
        if gdf is not None and dauid_column is not None:
            for idx, row in gdf.iterrows():
                dauid_to_centroid[str(row[dauid_column])] = (row.geometry.centroid.x, row.geometry.centroid.y)
        
        # Get data for this ethnicity
        flow_data = self.ethnicity_flow_data[eth_name]
        dauids = flow_data['dauid']
        interactions_list = flow_data['interactions']
        
        # Draw interaction networks at each significant location
        for i, (dauid, interactions) in enumerate(zip(dauids, interactions_list)):
            if dauid in dauid_to_centroid:
                # Use shapefile centroid coordinates
                x, y = dauid_to_centroid[dauid]
                
                # Calculate total interaction strength for sizing
                total_strength = sum(v for k, v in interactions.items() if v > self.significant_threshold)
                
                # Draw the source ethnicity point (sized by total interaction strength)
                marker_size = 100 + 500 * total_strength
                ax.scatter(x, y, s=marker_size, 
                        color=self.ethnicity_colors[eth_idx % len(self.ethnicity_colors)],
                        alpha=0.8, edgecolor='black', linewidth=1, zorder=10)
                
                # Calculate a suitable radius for the interaction network based on the map scale
                # This needs adjustment based on your specific map coordinates
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                radius = x_range * 0.01  # Use 1% of x-axis range as radius
                
                # Draw interactions with other ethnicities
                for target_idx, target_eth in enumerate(self.ethnicity_names):
                    if target_eth == eth_name:
                        continue  # Skip self-interaction
                    
                    if target_eth in interactions:
                        strength = interactions[target_eth]
                        
                        if strength > self.significant_threshold:
                            # Calculate position for target ethnicity (in a circle around the source)
                            angle = 2 * np.pi * target_idx / len(self.ethnicity_names)
                            target_x = x + radius * np.cos(angle)
                            target_y = y + radius * np.sin(angle)
                            
                            # Size the target point by interaction strength
                            target_size = 50 + 200 * strength
                            
                            # Draw the target ethnicity point
                            ax.scatter(target_x, target_y, s=target_size, 
                                    color=self.ethnicity_colors[target_idx % len(self.ethnicity_colors)],
                                    alpha=0.7, edgecolor='black', linewidth=0.5, zorder=5)
                            
                            # Draw the connection line with width based on strength
                            line_width = 1 + 5 * strength
                            ax.plot([x, target_x], [y, target_y], 
                                linewidth=line_width, 
                                color=self.ethnicity_colors[target_idx % len(self.ethnicity_colors)],
                                alpha=0.6, zorder=3)
        
        # Set title and legend
        ax.set_title(f"Spatial Distribution of {eth_name} Interactions", fontsize=16)
        
        # Add a legend
        legend_elements = []
        for i, name in enumerate(self.ethnicity_names):
            if i < len(self.ethnicity_colors):
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=self.ethnicity_colors[i % len(self.ethnicity_colors)],
                        markersize=10, label=name)
                )
        
        # Add legend for interaction strength
        legend_elements.append(
            Line2D([0], [0], linewidth=1, color='gray', label='Weak Interaction')
        )
        legend_elements.append(
            Line2D([0], [0], linewidth=3, color='gray', label='Medium Interaction')
        )
        legend_elements.append(
            Line2D([0], [0], linewidth=5, color='gray', label='Strong Interaction')
        )
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Turn off axis
        ax.set_axis_off()
        
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.interaction_dir, 
                            f'spatial_interaction_map_{eth_name.replace(" ", "_")}.png'), 
                dpi=300)
        plt.close()
    
    def _create_comprehensive_interaction_map(self, gdf, dauid_column):
        """
        Create a comprehensive map showing all ethnicity interactions together
        
        Args:
            gdf (GeoDataFrame): GeoPandas dataframe with geometries
            dauid_column (str): DAUID column name in GeoDataFrame
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Draw geometries from shapefile as base map
        gdf.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.5, alpha=0.3)
        
        # Create a mapping from DAUID to centroids in the shapefile
        dauid_to_centroid = {}
        if dauid_column is not None:
            for idx, row in gdf.iterrows():
                dauid_to_centroid[str(row[dauid_column])] = (row.geometry.centroid.x, row.geometry.centroid.y)
        
        # Calculate a suitable radius for the interaction network based on the map scale
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        radius = x_range * 0.005  # Use 0.5% of x-axis range as radius
        
        # Track the strongest interactions for highlighting
        strongest_interactions = []
        
        # First pass: Find the strongest interactions overall
        max_interaction = 0
        for eth_idx, eth_name in enumerate(self.ethnicity_names):
            if eth_name not in self.ethnicity_flow_data:
                continue
                
            flow_data = self.ethnicity_flow_data[eth_name]
            for i, (dauid, interactions) in enumerate(zip(flow_data['dauid'], flow_data['interactions'])):
                for target_idx, target_eth in enumerate(self.ethnicity_names):
                    if target_eth == eth_name or target_eth not in interactions:
                        continue
                    
                    strength = interactions[target_eth]
                    max_interaction = max(max_interaction, strength)
        
        # Set a threshold for "strong" interactions (e.g., top 25%)
        strong_threshold = max_interaction * 0.75
        
        # Second pass: Plot all interactions
        for eth_idx, eth_name in enumerate(self.ethnicity_names):
            if eth_name not in self.ethnicity_flow_data:
                continue
                
            flow_data = self.ethnicity_flow_data[eth_name]
            for i, (dauid, interactions) in enumerate(zip(flow_data['dauid'], flow_data['interactions'])):
                if dauid not in dauid_to_centroid:
                    continue
                    
                # Use shapefile centroid coordinates
                center_x, center_y = dauid_to_centroid[dauid]
                
                # Draw interactions with other ethnicities
                for target_idx, target_eth in enumerate(self.ethnicity_names):
                    if target_eth == eth_name or target_eth not in interactions:
                        continue
                    
                    strength = interactions[target_eth]
                    if strength <= self.significant_threshold:
                        continue
                    
                    # Calculate positions for ethnicity markers
                    eth_angle = 2 * np.pi * eth_idx / len(self.ethnicity_names)
                    target_angle = 2 * np.pi * target_idx / len(self.ethnicity_names)
                    
                    # Position the markers on opposite sides of the center point
                    eth_x = center_x + radius * np.cos(eth_angle)
                    eth_y = center_y + radius * np.sin(eth_angle)
                    target_x = center_x + radius * np.cos(target_angle)
                    target_y = center_y + radius * np.sin(target_angle)
                    
                    # Draw the ethnicity markers
                    ax.scatter(eth_x, eth_y, s=50, 
                            color=self.ethnicity_colors[eth_idx % len(self.ethnicity_colors)],
                            alpha=0.7, edgecolor='black', linewidth=0.5, zorder=5)
                    
                    ax.scatter(target_x, target_y, s=50, 
                            color=self.ethnicity_colors[target_idx % len(self.ethnicity_colors)],
                            alpha=0.7, edgecolor='black', linewidth=0.5, zorder=5)
                    
                    # Draw the connection line with width and alpha based on interaction strength
                    # Scale for visibility
                    line_width = 0.5 + 4 * (strength / max_interaction)
                    line_alpha = 0.3 + 0.6 * (strength / max_interaction)
                    
                    # For strong interactions, use a more visible style
                    if strength >= strong_threshold:
                        line_style = '-'
                        strongest_interactions.append({
                            'source': eth_name,
                            'target': target_eth,
                            'strength': strength,
                            'coords': [(eth_x, eth_y), (target_x, target_y)],
                            'source_idx': eth_idx,
                            'target_idx': target_idx
                        })
                    else:
                        line_style = ':'
                    
                    # Draw connection line
                    ax.plot([eth_x, target_x], [eth_y, target_y], 
                        linestyle=line_style,
                        linewidth=line_width, 
                        color='gray',
                        alpha=line_alpha, 
                        zorder=2)
        
        # Third pass: Highlight the strongest interactions
        for interaction in strongest_interactions:
            (eth_x, eth_y), (target_x, target_y) = interaction['coords']
            eth_idx = interaction['source_idx']
            target_idx = interaction['target_idx']
            
            # Draw thicker, more visible lines for strongest interactions
            ax.plot([eth_x, target_x], [eth_y, target_y], 
                linestyle='-',
                linewidth=3, 
                color=self.ethnicity_colors[eth_idx % len(self.ethnicity_colors)],
                alpha=0.8, 
                zorder=4)
            
            # Add a small annotation showing the strength
            mid_x = (eth_x + target_x) / 2
            mid_y = (eth_y + target_y) / 2
            ax.annotate(f"{interaction['strength']:.2f}", 
                    (mid_x, mid_y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # Add a legend for ethnicities
        legend_elements = []
        for i, name in enumerate(self.ethnicity_names):
            if i < len(self.ethnicity_colors):
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=self.ethnicity_colors[i % len(self.ethnicity_colors)],
                        markersize=10, label=name)
                )
        
        # Add legend for interaction strength
        legend_elements.append(
            Line2D([0], [0], linestyle=':', linewidth=1, color='gray', label='Weak Interaction')
        )
        legend_elements.append(
            Line2D([0], [0], linestyle='-', linewidth=2, color='gray', label='Strong Interaction')
        )
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title
        ax.set_title("Comprehensive Ethnicity Interaction Network", fontsize=16)
        
        # Turn off axis
        ax.set_axis_off()
        
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.interaction_dir, 'spatial_comprehensive_interaction_network.png'), dpi=300)
        plt.close()
    
    
    def _create_spatial_ethnicity_flow(self, eth_idx, eth_name, gdf=None, dauid_column=None):
        """
        Create a spatial visualization showing ethnicity interactions at each DAUID
        
        Args:
            eth_idx (int): Ethnicity index
            eth_name (str): Ethnicity name
            gdf (GeoDataFrame, optional): GeoPandas dataframe with geometries
            dauid_column (str, optional): DAUID column name in GeoDataFrame
        """
        if eth_name not in self.ethnicity_flow_data or not self.ethnicity_flow_data[eth_name]['dauid']:
            logger.warning(f"No flow data available for {eth_name}")
            return
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Draw geometries from shapefile as the base map
        if gdf is not None and dauid_column is not None:
            gdf.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.5, alpha=0.3)
        
        # Get data for this ethnicity
        flow_data = self.ethnicity_flow_data[eth_name]
        locations_x = flow_data['x']
        locations_y = flow_data['y']
        interactions_list = flow_data['interactions']
        dauids = flow_data['dauid']
        
        # Create a mapping from DAUID to locations in shapefile (if applicable)
        dauid_to_centroid = {}
        if gdf is not None and dauid_column is not None:
            for idx, row in gdf.iterrows():
                dauid = str(row[dauid_column])
                if dauid in dauids:
                    # Store centroid coordinates of the shapefile geometry
                    dauid_to_centroid[dauid] = (row.geometry.centroid.x, row.geometry.centroid.y)
        
        # Draw interaction networks at each significant location
        for i, (x, y, interactions, dauid) in enumerate(zip(locations_x, locations_y, interactions_list, dauids)):
            # Use shapefile centroid if available, otherwise use custom coordinates
            if dauid in dauid_to_centroid:
                x, y = dauid_to_centroid[dauid]
            
            # Draw the source ethnicity point (larger)
            ax.scatter(x, y, s=180, 
                    color=self.ethnicity_colors[eth_idx % len(self.ethnicity_colors)],
                    alpha=0.8, edgecolor='black', linewidth=1, zorder=10)
            
            # Draw small circular network around this point showing interactions with other ethnicities
            # Adjust radius based on the coordinate system (larger if shapefile coords)
            radius = 0.02 if not dauid_to_centroid else 0.001  # Dynamically adjust based on coordinate system
            
            # Draw connections to other ethnicities
            for target_idx, target_eth in enumerate(self.ethnicity_names):
                if target_eth == eth_name:
                    continue  # Skip self-interaction
                
                if target_eth in interactions:
                    strength = interactions[target_eth]
                    
                    if strength > self.significant_threshold:
                        # Calculate position for target ethnicity (in a circle around the source)
                        angle = 2 * np.pi * target_idx / len(self.ethnicity_names)
                        target_x = x + radius * np.cos(angle)
                        target_y = y + radius * np.sin(angle)
                        
                        # Draw the target ethnicity point
                        ax.scatter(target_x, target_y, s=100, 
                                color=self.ethnicity_colors[target_idx % len(self.ethnicity_colors)],
                                alpha=0.7, edgecolor='black', linewidth=0.5, zorder=5)
                        
                        # Draw the connection line with width based on strength
                        line_width = 1 + 5 * strength / self.significant_threshold
                        ax.plot([x, target_x], [y, target_y], 
                            linewidth=line_width, 
                            color=self.ethnicity_colors[eth_idx % len(self.ethnicity_colors)],
                            alpha=0.6, zorder=3)
        
        # Set title and legend
        ax.set_title(f"Spatial Distribution of {eth_name} Interactions", fontsize=16)
        
        # Add a legend
        legend_elements = []
        for i, name in enumerate(self.ethnicity_names):
            if i < len(self.ethnicity_colors):
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=self.ethnicity_colors[i % len(self.ethnicity_colors)],
                        markersize=10, label=name)
                )
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Turn off axis
        ax.set_axis_off()
        
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(self.interaction_dir, 
                            f'spatial_interaction_map_{eth_name.replace(" ", "_")}.png'), 
                dpi=300)
        plt.close()

    def visualize_temporal_cascade(self, time_steps=5, amplification_factor=1.2):
        """
        Create a visualization showing the cascade of ethnicity interactions over time
        
        Args:
            time_steps (int): Number of time steps to simulate
            amplification_factor (float): Factor for amplifying the cascade effects
            
        Returns:
            bool: Success status
        """
        try:
            if self.avg_interaction_matrix is None:
                logger.warning("No interaction matrix available for temporal cascade visualization")
                return False
            
            logger.info(f"Creating temporal cascade visualization with {time_steps} steps")
            
            # Create a figure with subplots for each time step
            fig, axes = plt.subplots(1, time_steps, figsize=(5*time_steps, 8), sharey=True)
            
            # Handle case where time_steps=1
            if time_steps == 1:
                axes = [axes]
            
            # Number of ethnicities
            n = len(self.ethnicity_names)
            
            # Initialize population values
            # Start with equal populations for simplicity
            base_population = np.ones(n) * 100
            
            # Track population changes over time
            populations = [base_population.copy()]
            
            # Initialize influence matrices for each time step
            influence_matrices = [np.zeros((n, n))]
            
            # Simulate interaction cascade over time
            for t in range(1, time_steps):
                # Calculate influence from previous step
                prev_pop = populations[t-1]
                
                # Calculate change based on interactions
                change = np.zeros(n)
                influence_mat = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        if i != j:  # Skip self-influence
                            # Calculate influence from ethnicity j to i
                            influence = self.avg_interaction_matrix[i, j] * prev_pop[j] * amplification_factor
                            
                            # Add to change
                            change[i] += influence
                            
                            # Record in influence matrix
                            influence_mat[i, j] = influence
                
                # Update population
                new_pop = prev_pop + change
                
                # Ensure non-negative population
                new_pop = np.maximum(new_pop, 0)
                
                # Record results
                populations.append(new_pop)
                influence_matrices.append(influence_mat)
            
            # Create visualizations for each time step
            for t in range(time_steps):
                ax = axes[t]
                
                # Bar chart of populations
                bars = ax.bar(np.arange(n), populations[t], 
                             color=[self.ethnicity_colors[i % len(self.ethnicity_colors)] 
                                   for i in range(n)],
                             alpha=0.7)
                
                # Add labels and title
                ax.set_title(f"Time Step {t}")
                ax.set_xlabel("Ethnicity")
                if t == 0:
                    ax.set_ylabel("Population")
                
                # Set x-ticks
                ax.set_xticks(np.arange(n))
                ax.set_xticklabels(self.ethnicity_names, rotation=45, ha='right', fontsize=8)
                
                # For steps after the first, show incoming influences
                if t > 0:
                    influence_mat = influence_matrices[t]
                    
                    # Add arrows for strongest influences
                    for i in range(n):
                        # Find strongest influence
                        influences = influence_mat[i, :]
                        if np.sum(influences) > 0:  # Only if there are influences
                            strongest_idx = np.argmax(influences)
                            strongest_val = influences[strongest_idx]
                            
                            # Only show if significant
                            if strongest_val > 5:  # Arbitrary threshold
                                # Add an arrow
                                try:
                                    bar_i = bars[i]
                                    bar_j = bars[strongest_idx]
                                    
                                    # Calculate arrow positions
                                    x_i = bar_i.get_x() + bar_i.get_width() / 2
                                    y_i = bar_i.get_y() + bar_i.get_height() / 2
                                    x_j = bar_j.get_x() + bar_j.get_width() / 2
                                    y_j = bar_j.get_y() + bar_j.get_height() / 2
                                    
                                    # Draw arrow
                                    ax.annotate(
                                        '',
                                        xy=(x_i, y_i),  # End point
                                        xytext=(x_j, y_j),  # Start point
                                        arrowprops=dict(
                                            arrowstyle='->',
                                            lw=1 + strongest_val / 20,
                                            color=self.ethnicity_colors[strongest_idx % len(self.ethnicity_colors)],
                                            alpha=0.6,
                                            connectionstyle='arc3,rad=0.3'
                                        )
                                    )
                                except Exception as e:
                                    logger.warning(f"Could not draw arrow: {str(e)}")
            
            # Set overall title
            fig.suptitle("Ethnicity Interaction Cascade Over Time", fontsize=16, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the visualization
            plt.savefig(os.path.join(self.interaction_dir, 'ethnicity_temporal_cascade.png'), dpi=300)
            plt.close()
            
            # Try to create a basic animation if possible
            try:
                import matplotlib.animation as animation
                
                # Create a new figure for animation
                fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
                
                # Initialize bars
                bars_anim = ax_anim.bar(
                    np.arange(n), 
                    populations[0],
                    color=[self.ethnicity_colors[i % len(self.ethnicity_colors)] for i in range(n)],
                    alpha=0.7
                )
                
                # Set axis limits
                ax_anim.set_ylim(0, max([max(pop) for pop in populations]) * 1.2)
                
                # Set labels and title
                ax_anim.set_xlabel("Ethnicity")
                ax_anim.set_ylabel("Population")
                title = ax_anim.set_title("Time Step: 0")
                
                # Set x-ticks
                ax_anim.set_xticks(np.arange(n))
                ax_anim.set_xticklabels(self.ethnicity_names, rotation=45, ha='right')
                
                # Update function for animation
                def update(frame):
                    # Update bar heights
                    for i, bar in enumerate(bars_anim):
                        bar.set_height(populations[frame][i])
                    
                    # Update title
                    title.set_text(f"Time Step: {frame}")
                    
                    return bars_anim
                
                # Create animation
                anim = animation.FuncAnimation(
                    fig_anim, update, frames=time_steps, 
                    interval=1000, blit=True, repeat=True
                )
                
                # Save animation
                try:
                    anim.save(os.path.join(self.interaction_dir, 'ethnicity_cascade_animation.gif'), 
                             writer='pillow', fps=1, dpi=100)
                except Exception as e:
                    logger.warning(f"Could not save animation: {str(e)}")
                
                plt.close(fig_anim)
                
            except ImportError:
                logger.warning("Animation module not available, skipping animated visualization")
            
            logger.info("Successfully created temporal cascade visualization")
            return True
            
        except Exception as e:
            logger.error(f"Error creating temporal cascade visualization: {str(e)}", exc_info=True)
            return False

    def visualize_interaction_metrics(self):
        """
        Create visualizations for various interaction metrics across ethnicities
        
        Returns:
            bool: Success status
        """
        try:
            if self.avg_interaction_matrix is None:
                logger.warning("No interaction matrix available for metrics visualization")
                return False
            
            logger.info("Creating interaction metrics visualization")
            
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(2, 2, figure=fig)
            
            # Number of ethnicities
            n = len(self.ethnicity_names)
            
            # 1. Top-left: Influence given vs. received
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Calculate total influence given and received
            influence_given = np.sum(self.avg_interaction_matrix, axis=0)  # Sum across rows
            influence_received = np.sum(self.avg_interaction_matrix, axis=1)  # Sum across columns
            
            # Create scatter plot
            scatter = ax1.scatter(
                influence_given, influence_received,
                s=200, alpha=0.7,
                c=range(n),
                cmap=plt.cm.viridis,
                edgecolor='black', linewidth=1
            )
            
            # Add ethnicity labels
            for i in range(n):
                ax1.annotate(
                    self.ethnicity_names[i],
                    (influence_given[i], influence_received[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha='left'
                )
            
            # Add diagonal line (balanced influence)
            max_val = max(np.max(influence_given), np.max(influence_received))
            ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            # Add labels and title
            ax1.set_xlabel('Total Influence Given')
            ax1.set_ylabel('Total Influence Received')
            ax1.set_title('Ethnicity Influence Balance')
            
            # Add annotations
            ax1.annotate(
                'Net Influencers',
                xy=(0.75, 0.25),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
            )
            
            ax1.annotate(
                'Net Influenced',
                xy=(0.25, 0.75),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3)
            )
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # 2. Top-right: Influence distribution
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Calculate influence diversity using entropy-like measure
            influence_diversity = np.zeros(n)
            for i in range(n):
                # Get outgoing influences
                outgoing = self.avg_interaction_matrix[:, i].copy()
                # Remove self-influence
                outgoing[i] = 0
                # Normalize
                total = np.sum(outgoing)
                if total > 0:
                    normalized = outgoing / total
                    # Calculate entropy (higher = more diverse influence)
                    entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
                    influence_diversity[i] = entropy
            
            # Create bar chart
            bars = ax2.bar(
                np.arange(n), influence_diversity,
                color=[self.ethnicity_colors[i % len(self.ethnicity_colors)] for i in range(n)],
                alpha=0.7
            )
            
            # Add line showing maximum possible diversity
            max_entropy = -np.log2(1/(n-1)) if n > 1 else 0
            ax2.axhline(y=max_entropy, color='r', linestyle='--', alpha=0.5, 
                       label=f'Max diversity ({max_entropy:.2f})')
            
            # Add labels and title
            ax2.set_xlabel('Ethnicity')
            ax2.set_ylabel('Influence Diversity (entropy)')
            ax2.set_title('How Broadly Each Ethnicity Influences Others')
            
            # Set x-ticks
            ax2.set_xticks(np.arange(n))
            ax2.set_xticklabels(self.ethnicity_names, rotation=45, ha='right', fontsize=8)
            
            # Add legend
            ax2.legend()
            
            # 3. Bottom-left: Relative influence strength
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Calculate relative strength (how strong each ethnicity's influence is)
            relative_strength = np.zeros(n)
            
            # Sum of all influences
            total_influence = np.sum(self.avg_interaction_matrix)
            
            for i in range(n):
                # Sum of this ethnicity's influence on others
                eth_influence = np.sum(self.avg_interaction_matrix[:, i])
                
                # Calculate relative strength
                if total_influence > 0:
                    relative_strength[i] = eth_influence / total_influence * n  # Scaled by n for better visualization
            
            # Create horizontal bar chart
            bars = ax3.barh(
                np.arange(n), relative_strength,
                color=[self.ethnicity_colors[i % len(self.ethnicity_colors)] for i in range(n)],
                alpha=0.7
            )
            
            # Add labels and title
            ax3.set_xlabel('Relative Influence Strength')
            ax3.set_ylabel('Ethnicity')
            ax3.set_title('Ethnicity Relative Influence Strength')
            
            # Set y-ticks
            ax3.set_yticks(np.arange(n))
            ax3.set_yticklabels(self.ethnicity_names)
            
            # Add reference line for "average" influence
            ax3.axvline(x=1.0, color='black', linestyle='--', alpha=0.5,
                      label='Average influence level')
            ax3.legend()
            
            # 4. Bottom-right: Influence network metrics
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Calculate simple network metrics
            metrics = {
                'Outgoing Connections': [],
                'Incoming Connections': [],
                'Net Connections': []
            }
            
            # Use threshold to determine connections
            threshold = np.percentile(self.avg_interaction_matrix.flatten(), 75)  # Top 25% are connections
            
            for i in range(n):
                # Count outgoing connections
                outgoing = len(np.where(self.avg_interaction_matrix[:, i] > threshold)[0])
                
                # Count incoming connections
                incoming = len(np.where(self.avg_interaction_matrix[i, :] > threshold)[0])
                
                # Store metrics
                metrics['Outgoing Connections'].append(outgoing)
                metrics['Incoming Connections'].append(incoming)
                metrics['Net Connections'].append(outgoing - incoming)
            
            # Create bar chart for net connections
            bars = ax4.bar(
                np.arange(n), metrics['Net Connections'],
                color=[self.ethnicity_colors[i % len(self.ethnicity_colors)] for i in range(n)],
                alpha=0.7
            )
            
            # Color negative bars differently
            for i, val in enumerate(metrics['Net Connections']):
                if val < 0:
                    bars[i].set_color('gray')
                    bars[i].set_alpha(0.5)
            
            # Add a zero line
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add labels and title
            ax4.set_xlabel('Ethnicity')
            ax4.set_ylabel('Net Connections\n(Outgoing - Incoming)')
            ax4.set_title('Ethnicity Network Connectivity Balance')
            
            # Set x-ticks
            ax4.set_xticks(np.arange(n))
            ax4.set_xticklabels(self.ethnicity_names, rotation=45, ha='right', fontsize=8)
            
            # Add grid
            ax4.grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle('Ethnicity Interaction Metrics Analysis', fontsize=16, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the visualization
            plt.savefig(os.path.join(self.interaction_dir, 'ethnicity_interaction_metrics.png'), dpi=300)
            plt.close()
            
            logger.info("Successfully created interaction metrics visualization")
            return True
            
        except Exception as e:
            logger.error(f"Error creating interaction metrics visualization: {str(e)}", exc_info=True)
            return False
            
    def run_all_visualizations(self, gdf=None, shapefile_path=None):
        """
        Run all visualization methods to create a comprehensive set of visualizations
        
        Args:
            gdf (GeoDataFrame, optional): GeoPandas dataframe with geometries
            shapefile_path (str, optional): Path to shapefile
            
        Returns:
            bool: Success status
        """
        success = True
        
        # Network visualization
        logger.info("Running network visualization")
        network_success = self.visualize_interaction_network()
        success = success and network_success
        
        # Flow diagrams
        logger.info("Running flow diagram visualization")
        flow_success = self.visualize_flow_diagrams()
        success = success and flow_success
        
        # Spatial interactions if coordinates available
        if self.dauid_coordinates:
            logger.info("Running spatial interaction visualization")
            spatial_success = self.visualize_spatial_interactions(gdf, shapefile_path)
            success = success and spatial_success
        
        # Temporal cascade
        logger.info("Running temporal cascade visualization")
        cascade_success = self.visualize_temporal_cascade()
        success = success and cascade_success
        
        # Interaction metrics
        logger.info("Running interaction metrics visualization")
        metrics_success = self.visualize_interaction_metrics()
        success = success and metrics_success
        
        return success