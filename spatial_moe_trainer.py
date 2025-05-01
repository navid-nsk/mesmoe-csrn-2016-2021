import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class MultiEthnicMoEModelTrainer:
    """
    Enhanced trainer class for training and evaluating multi-ethnic spatial population prediction models 
    with Mixture of Experts - now supporting colonization, jump, decline, and PDE experts
    for multiple ethnicity predictions.
    """
    def __init__(self, model, ethnicity_names, colonization_threshold=0.1, jump_threshold=100.0, 
                 decline_threshold=50.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer
        
        Args:
            model (nn.Module): The neural network model
            ethnicity_names (list): List of ethnicity names for reporting
            colonization_threshold (float): Threshold below which population is considered zero
            jump_threshold (float): Threshold for identifying potential jump cases
            decline_threshold (float): Threshold for identifying potential decline cases
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.ethnicity_names = ethnicity_names
        self.num_ethnicities = len(ethnicity_names)
        self.device = device
        self.colonization_threshold = colonization_threshold
        self.jump_threshold = jump_threshold
        self.decline_threshold = decline_threshold
        self.model.to(self.device)
        
        logger.info(f"Multi-Ethnic Spatial MoE model initialized on {self.device}")
        logger.info(f"Model will predict {self.num_ethnicities} ethnicities: {self.ethnicity_names}")
        # logger.info(f"Model architecture: {model}")
    
    def train(self, train_loader, val_loader=None, epochs=100, lr=0.0005, weight_decay=1e-6,
            colonization_weight=5.0, jump_weight=3.0, decline_weight=4.0, pde_weight=2.0,
            aux_loss_weight=0.2, patience=10, early_stopping_patience=20, 
            save_train_predictions_at_epoch=60, predictions_path="training_predictions_epoch60.csv",
            ethnicity_weights=None, pruning_callback=None):
        """
        Train the model with learnable expert routing and auxiliary loss for multi-ethnic predictions
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            epochs (int): Maximum number of epochs
            lr (float): Learning rate
            weight_decay (float): L2 regularization
            colonization_weight (float): Extra weight for colonization loss cases
            jump_weight (float): Extra weight for jump loss cases
            decline_weight (float): Extra weight for decline loss cases
            pde_weight (float): Extra weight for PDE loss cases
            aux_loss_weight (float): Weight for auxiliary loss
            patience (int): Patience for lr scheduler
            early_stopping_patience (int): Patience for early stopping
            save_train_predictions_at_epoch (int): Epoch at which to save training predictions
            predictions_path (str): Path to save training predictions
            ethnicity_weights (list, optional): Weights for each ethnicity in loss calculation
            
        Returns:
            dict: Training history
        """
        logger.info("Starting multi-ethnic spatial MoE model training with learnable expert router...")
        logger.info(f"Using colonization_weight={colonization_weight}, jump_weight={jump_weight}, "
                    f"decline_weight={decline_weight}, pde_weight={pde_weight}")
        
        # Set default ethnicity weights if not provided
        if ethnicity_weights is None:
            ethnicity_weights = torch.ones(self.num_ethnicities, device=self.device)
        else:
            ethnicity_weights = torch.tensor(ethnicity_weights, device=self.device)
            logger.info(f"Using custom ethnicity weights: {ethnicity_weights}")
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
        
        # Base loss function
        base_criterion = nn.MSELoss(reduction='none')  # No reduction to allow per-sample weighting
        classification_criterion = nn.BCELoss()  # For auxiliary expert classification loss
        
        # Training history (add tracking per ethnicity)
        history = {
            'train_loss': [],
            'train_aux_loss': [],
            'train_colonization_loss': [],
            'train_jump_loss': [],
            'train_decline_loss': [],
            'train_pde_loss': [],
            'train_regular_loss': [],
            'val_loss': [],
            'val_aux_loss': [],
            'val_colonization_loss': [],
            'val_jump_loss': [],
            'val_decline_loss': [],
            'val_pde_loss': [],
            'val_regular_loss': [],
            'val_mae': [],
            'val_r2': [],
            'ethnicity_loss': {eth: [] for eth in self.ethnicity_names},  # Track loss per ethnicity
            'ethnicity_mae': {eth: [] for eth in self.ethnicity_names},   # Track MAE per ethnicity
            'ethnicity_r2': {eth: [] for eth in self.ethnicity_names}     # Track R2 per ethnicity
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        no_improve_epochs = 0
        best_model_state = None
        # torch.autograd.set_detect_anomaly(True)
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_aux_loss = 0.0
            train_steps = 0
            train_colonization_loss = 0.0
            train_colonization_steps = 0
            train_jump_loss = 0.0
            train_jump_steps = 0
            train_decline_loss = 0.0
            train_decline_steps = 0
            train_pde_loss = 0.0
            train_pde_steps = 0
            train_regular_loss = 0.0
            train_regular_steps = 0
            
            # Track losses per ethnicity
            eth_losses = {eth: 0.0 for eth in self.ethnicity_names}
            eth_steps = {eth: 0 for eth in self.ethnicity_names}
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move data to device
                features = batch['features'].to(self.device)
                population_2016 = batch['population_2016'].to(self.device)
                coordinates = batch['coordinates'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Calculate population changes for each ethnicity
                population_change = targets - population_2016  # [batch_size, num_ethnicities]
                
                # Create case type masks for each ethnicity
                # Shapes will be [batch_size, num_ethnicities]
                is_colonization = (population_2016 <= self.colonization_threshold)
                is_decline = (~is_colonization) & (population_2016 <= self.decline_threshold) & (population_change <= 0)
                is_jump = (~is_colonization) & (~is_decline) & (population_change > self.jump_threshold)
                
                # Define which cases should use PDE expert
                is_pde = (~is_colonization) & (~is_jump) & (~is_decline) & (
                    (population_change.abs() <= self.jump_threshold) | 
                    ((population_change < 0) & (population_change > -self.decline_threshold))
                )
                
                is_regular = (~is_colonization) & (~is_jump) & (~is_decline) & (~is_pde)
                
                # One-hot encode the case types for each ethnicity
                # Shape: [batch_size, num_ethnicities, 4]
                expert_ground_truth = torch.zeros(
                    (features.size(0), self.num_ethnicities, 4), 
                    device=self.device
                )
                expert_ground_truth[:, :, 0] = is_colonization.float()
                expert_ground_truth[:, :, 1] = is_jump.float()
                expert_ground_truth[:, :, 2] = is_decline.float()
                expert_ground_truth[:, :, 3] = is_pde.float()
                
                # Get spatial matrices if available
                walking_matrix = batch.get('walking_matrix')
                transit_matrix = batch.get('transit_matrix')
                proximity_matrix = batch.get('proximity_matrix')
                
                if walking_matrix is not None:
                    walking_matrix = walking_matrix.to(self.device)
                if transit_matrix is not None:
                    transit_matrix = transit_matrix.to(self.device)
                if proximity_matrix is not None:
                    proximity_matrix = proximity_matrix.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - now returns predictions, expert weights, and expert class probabilities
                # outputs shape: [batch_size, num_ethnicities]
                # expert_weights shape: [batch_size, num_ethnicities, num_experts]
                # expert_classes shape: [batch_size, num_ethnicities, num_experts]
                outputs, expert_weights, expert_classes = self.model(
                    features, population_2016, coordinates, 
                    walking_matrix, transit_matrix, proximity_matrix
                )
                
                # Calculate MSE loss for each batch item and ethnicity
                # element_losses shape: [batch_size, num_ethnicities]
                element_losses = base_criterion(outputs, targets)
                
                if is_jump.any():
                    # Extract jump values
                    jump_outputs = outputs[is_jump]
                    jump_targets = targets[is_jump]
                    
                    # Apply log transformation (add 1 to avoid log(0))
                    log_outputs = torch.log(jump_outputs + 1.0)
                    log_targets = torch.log(jump_targets + 1.0)
                    
                    # Calculate loss in log space
                    log_space_losses = base_criterion(log_outputs, log_targets) * 100.0  # Scale factor
                    
                    # Replace original losses with log-space losses
                    element_losses[is_jump] = log_space_losses
                
                # Create weight tensor for different case types
                # weights shape: [batch_size, num_ethnicities]
                weights = torch.ones_like(element_losses)
                
                # MAX_LOSS_VALUE = 2500.0  # Cap individual losses
                # element_losses = torch.clamp(element_losses, max=MAX_LOSS_VALUE)
                
                # Apply different weights for different case types
                weights[is_colonization] = colonization_weight
                weights[is_jump] = jump_weight
                weights[is_decline] = decline_weight
                weights[is_pde] = pde_weight
                
                # Apply ethnicity weights - multiply each ethnicity column by its weight
                for i, eth_weight in enumerate(ethnicity_weights):
                    weights[:, i] = weights[:, i] * eth_weight
                
                # Apply weights and take mean for MSE loss
                weighted_losses = element_losses * weights
                mse_loss = weighted_losses.mean()
                
                # Calculate auxiliary loss - how well the model is predicting the right expert
                # Reshape expert_classes and expert_ground_truth to match
                # [batch_size*num_ethnicities, num_experts]
                flattened_classes = expert_classes.reshape(-1, 4)
                flattened_ground_truth = expert_ground_truth.reshape(-1, 4)
                
                aux_loss = classification_criterion(flattened_classes, flattened_ground_truth)
                
                # Combine losses
                total_loss = mse_loss + aux_loss_weight * aux_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                # Update weights
                optimizer.step()
                
                # Monitor parameter norm
                # with torch.no_grad():
                #     param_norm = 0
                #     for p in self.model.parameters():
                #         param_norm += p.norm(2).item() ** 2
                #     param_norm = param_norm ** 0.5
                #     if param_norm > 2400:  # Choose an appropriate threshold
                #         logger.warning(f"Parameter norm is large: {param_norm:.2f}")
                
                # Update statistics
                with torch.no_grad():
                    # Overall loss (unweighted for fair tracking)
                    batch_loss = element_losses.mean().item()
                    train_loss += batch_loss
                    train_aux_loss += aux_loss.item()
                    train_steps += 1
                    
                    # Track loss per ethnicity
                    for i, eth in enumerate(self.ethnicity_names):
                        eth_loss = element_losses[:, i].mean().item()
                        eth_losses[eth] += eth_loss
                        eth_steps[eth] += 1
                    
                    # Separate losses for different case types
                    # Process flattened losses
                    flat_losses = element_losses.view(-1)
                    flat_is_colonization = is_colonization.view(-1)
                    flat_is_jump = is_jump.view(-1)
                    flat_is_decline = is_decline.view(-1)
                    flat_is_pde = is_pde.view(-1)
                    flat_is_regular = is_regular.view(-1)
                    
                    # Colonization loss
                    colonization_indices = torch.nonzero(flat_is_colonization).squeeze(-1)
                    if len(colonization_indices) > 0 and colonization_indices.numel() > 0:
                        col_loss = flat_losses[colonization_indices].mean().item()
                        train_colonization_loss += col_loss
                        train_colonization_steps += 1
                    
                    # Jump loss
                    jump_indices = torch.nonzero(flat_is_jump).squeeze(-1)
                    if len(jump_indices) > 0 and jump_indices.numel() > 0:
                        jump_loss = flat_losses[jump_indices].mean().item()
                        train_jump_loss += jump_loss
                        train_jump_steps += 1
                    
                    # Decline loss
                    decline_indices = torch.nonzero(flat_is_decline).squeeze(-1)
                    if len(decline_indices) > 0 and decline_indices.numel() > 0:
                        decline_loss = flat_losses[decline_indices].mean().item()
                        train_decline_loss += decline_loss
                        train_decline_steps += 1
                    
                    # PDE loss
                    pde_indices = torch.nonzero(flat_is_pde).squeeze(-1)
                    if len(pde_indices) > 0 and pde_indices.numel() > 0:
                        pde_loss = flat_losses[pde_indices].mean().item()
                        train_pde_loss += pde_loss
                        train_pde_steps += 1
                    
                    # Regular loss
                    regular_indices = torch.nonzero(flat_is_regular).squeeze(-1)
                    if len(regular_indices) > 0 and regular_indices.numel() > 0:
                        reg_loss = flat_losses[regular_indices].mean().item()
                        train_regular_loss += reg_loss
                        train_regular_steps += 1
            
            # Calculate average training losses for this epoch
            avg_train_loss = train_loss / max(1, train_steps)
            avg_train_aux_loss = train_aux_loss / max(1, train_steps)
            avg_colonization_loss = train_colonization_loss / max(1, train_colonization_steps)
            avg_jump_loss = train_jump_loss / max(1, train_jump_steps)
            avg_decline_loss = train_decline_loss / max(1, train_decline_steps)
            avg_pde_loss = train_pde_loss / max(1, train_pde_steps)
            avg_regular_loss = train_regular_loss / max(1, train_regular_steps)
            
            # Calculate average loss per ethnicity
            avg_eth_losses = {eth: eth_losses[eth] / max(1, eth_steps[eth]) for eth in self.ethnicity_names}
            
            # Record training history
            history['train_loss'].append(avg_train_loss)
            history['train_aux_loss'].append(avg_train_aux_loss)
            history['train_colonization_loss'].append(avg_colonization_loss)
            history['train_jump_loss'].append(avg_jump_loss)
            history['train_decline_loss'].append(avg_decline_loss)
            history['train_pde_loss'].append(avg_pde_loss)
            history['train_regular_loss'].append(avg_regular_loss)
            
            # Record ethnicity losses
            for eth in self.ethnicity_names:
                history['ethnicity_loss'][eth].append(avg_eth_losses[eth])
                
            # Validation phase (if validation data is provided)
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['mse']
                val_mae = val_metrics['mae']
                val_r2 = val_metrics['r2']
                val_colonization_loss = val_metrics.get('colonization_mse', 0.0)
                val_jump_loss = val_metrics.get('jump_mse', 0.0)
                val_decline_loss = val_metrics.get('decline_mse', 0.0)
                val_pde_loss = val_metrics.get('pde_mse', 0.0)
                val_regular_loss = val_metrics.get('regular_mse', 0.0)
                
                # Record validation metrics
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
                history['val_r2'].append(val_r2)
                history['val_colonization_loss'].append(val_colonization_loss)
                history['val_jump_loss'].append(val_jump_loss)
                history['val_decline_loss'].append(val_decline_loss)
                history['val_pde_loss'].append(val_pde_loss)
                history['val_regular_loss'].append(val_regular_loss)
                
                # Record per-ethnicity metrics
                for eth in self.ethnicity_names:
                    eth_key = f"{eth}_mse"
                    if eth_key in val_metrics:
                        history['ethnicity_loss'][eth].append(val_metrics[eth_key])
                    
                    eth_mae_key = f"{eth}_mae"
                    if eth_mae_key in val_metrics:
                        history['ethnicity_mae'][eth].append(val_metrics[eth_mae_key])
                    
                    eth_r2_key = f"{eth}_r2"
                    if eth_r2_key in val_metrics:
                        history['ethnicity_r2'][eth].append(val_metrics[eth_r2_key])
                
                if pruning_callback is not None:
                    # Create metrics dict to pass to callback
                    current_metrics = {
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                        'val_r2': val_r2,
                        'val_colonization_loss': val_colonization_loss,
                        'val_jump_loss': val_jump_loss,
                        'val_decline_loss': val_decline_loss,
                        'val_pde_loss': val_pde_loss,
                        'val_regular_loss': val_regular_loss
                    }
                    # Call pruning callback with current epoch and metrics
                    try:
                        pruning_callback(epoch, current_metrics)
                    except optuna.TrialPruned:
                        logger.info(f"Trial pruned at epoch {epoch+1}")
                        return history  # Return history so far and exit training loop
                    
                # Update learning rate based on validation loss
                scheduler.step(val_loss)
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                    # Save the best model
                    best_model_state = self.model.state_dict().copy()
                else:
                    no_improve_epochs += 1
                
                # Log progress
                # Create ethnicity loss message
                eth_loss_msg = ", ".join([f"{eth}: {avg_eth_losses[eth]:.4f}" for eth in self.ethnicity_names[:3]])
                if len(self.ethnicity_names) > 3:
                    eth_loss_msg += "... (more ethnicities)"
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} "
                          f"(Col: {avg_colonization_loss:.4f}, Jump: {avg_jump_loss:.4f}, "
                          f"Decline: {avg_decline_loss:.4f}, PDE: {avg_pde_loss:.4f}, Reg: {avg_regular_loss:.4f}), "
                          f"Eth Losses: {eth_loss_msg}, "
                          f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}")
                
                # Early stopping
                if no_improve_epochs >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                # Log progress without validation metrics
                eth_loss_msg = ", ".join([f"{eth}: {avg_eth_losses[eth]:.4f}" for eth in self.ethnicity_names[:3]])
                if len(self.ethnicity_names) > 3:
                    eth_loss_msg += "... (more ethnicities)"
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} "
                          f"(Col: {avg_colonization_loss:.4f}, Jump: {avg_jump_loss:.4f}, "
                          f"Decline: {avg_decline_loss:.4f}, PDE: {avg_pde_loss:.4f}, Reg: {avg_regular_loss:.4f}), "
                          f"Eth Losses: {eth_loss_msg}")
        
        # If we completed all epochs without early stopping, make sure we have the best model
        if val_loader and best_model_state and no_improve_epochs < early_stopping_patience:
            self.model.load_state_dict(best_model_state)
            
        return history

    def get_predictions(self, data_loader):
        """
        Get predictions for a dataset without calculating metrics
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            DataFrame: DataFrame with predictions for all ethnicities
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_base_populations = []
        all_expert_weights = []
        dauid_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                features = batch['features'].to(self.device)
                population_2016 = batch['population_2016'].to(self.device)
                coordinates = batch['coordinates'].to(self.device)
                
                # Get spatial matrices if available
                walking_matrix = batch.get('walking_matrix')
                transit_matrix = batch.get('transit_matrix')
                proximity_matrix = batch.get('proximity_matrix')
                
                if walking_matrix is not None:
                    walking_matrix = walking_matrix.to(self.device)
                if transit_matrix is not None:
                    transit_matrix = transit_matrix.to(self.device)
                if proximity_matrix is not None:
                    proximity_matrix = proximity_matrix.to(self.device)
                
                # Forward pass - now returns predictions, expert weights, and expert classes
                outputs, expert_weights, _ = self.model(
                    features, population_2016, coordinates, 
                    walking_matrix, transit_matrix, proximity_matrix
                )
                
                # Store predictions - shape [batch_size, num_ethnicities]
                all_predictions.append(outputs.cpu().numpy())
                
                # Store base populations - shape [batch_size, num_ethnicities]
                all_base_populations.append(population_2016.cpu().numpy())
                
                # Store expert weights - shape [batch_size, num_ethnicities, num_experts]
                all_expert_weights.append(expert_weights.cpu().numpy())
                
                # If targets are available
                if 'target' in batch:
                    targets = batch['target']
                    all_targets.append(targets.numpy())
                
                # Store DAUIDs
                if 'dauid' in batch:
                    dauid_list.extend(batch['dauid'].numpy())
        
        # Concatenate all predictions and populations
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_base_populations = np.concatenate(all_base_populations, axis=0)
        all_expert_weights = np.concatenate(all_expert_weights, axis=0)
        
        if all_targets:
            all_targets = np.concatenate(all_targets, axis=0)
        
        # Create a dictionary that will hold all columns for our DataFrame
        data_dict = {'DAUID': dauid_list}
        
        # Add base populations (2016) for each ethnicity
        for i, eth in enumerate(self.ethnicity_names):
            data_dict[f"{eth}_2016"] = all_base_populations[:, i]
        
        # Add predicted populations for each ethnicity
        for i, eth in enumerate(self.ethnicity_names):
            data_dict[f"{eth}_predicted"] = all_predictions[:, i]
        
        # Add expert weights for each ethnicity
        for i, eth in enumerate(self.ethnicity_names):
            for j, expert in enumerate(['colonization', 'jump', 'decline', 'pde']):
                data_dict[f"{eth}_{expert}_weight"] = all_expert_weights[:, i, j]
        
        # Add actual values and calculate flags if targets are available
        if len(all_targets) > 0:
            # First add all actual values
            for i, eth in enumerate(self.ethnicity_names):
                data_dict[f"{eth}_actual"] = all_targets[:, i]
                data_dict[f"{eth}_error"] = all_targets[:, i] - all_predictions[:, i]
            
            # Create all flags at once without modifying the DataFrame repeatedly
            for i, eth in enumerate(self.ethnicity_names):
                eth_2016 = all_base_populations[:, i]
                eth_actual = all_targets[:, i]
                
                # Calculate all flags
                is_colonization = (eth_2016 <= self.colonization_threshold)
                is_decline = (~is_colonization) & (eth_2016 <= self.decline_threshold)
                
                pop_change = eth_actual - eth_2016
                is_jump = (~is_colonization) & (~is_decline) & (pop_change > self.jump_threshold)
                
                is_pde = (~is_colonization) & (~is_jump) & (~is_decline) & (
                    (np.abs(pop_change) <= self.jump_threshold) | 
                    ((pop_change < 0) & (pop_change > -self.decline_threshold))
                )
                
                # Add all flags to the dictionary
                data_dict[f"{eth}_is_colonization"] = is_colonization
                data_dict[f"{eth}_is_decline"] = is_decline
                data_dict[f"{eth}_is_jump"] = is_jump 
                data_dict[f"{eth}_is_pde"] = is_pde
        
        # Create the DataFrame all at once
        results_df = pd.DataFrame(data_dict)
        
        return results_df
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset
        
        Args:
            data_loader (DataLoader): Data loader for evaluation
            
        Returns:
            dict: Dictionary with evaluation metrics for all ethnicities
        """
        self.model.eval()
        
        # Get predictions DataFrame
        results_df = self.get_predictions(data_loader)
        
        # Check if targets are available for evaluation
        if not any(f"{eth}_actual" in results_df.columns for eth in self.ethnicity_names):
            return results_df  # Return predictions only if no targets
        
        # Initialize metrics dictionary
        metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'r2': 0.0
        }
        
        # Calculate metrics across all ethnicities combined
        all_predictions = []
        all_actuals = []
        
        # Collect predictions and actuals
        for eth in self.ethnicity_names:
            all_predictions.extend(results_df[f"{eth}_predicted"].values)
            all_actuals.extend(results_df[f"{eth}_actual"].values)
        
        # Calculate overall metrics
        metrics['mse'] = mean_squared_error(all_actuals, all_predictions)
        metrics['mae'] = mean_absolute_error(all_actuals, all_predictions)
        metrics['r2'] = r2_score(all_actuals, all_predictions)
        
        # Calculate metrics per ethnicity
        for eth in self.ethnicity_names:
            eth_predictions = results_df[f"{eth}_predicted"].values
            eth_actuals = results_df[f"{eth}_actual"].values
            
            metrics[f"{eth}_mse"] = mean_squared_error(eth_actuals, eth_predictions)
            metrics[f"{eth}_mae"] = mean_absolute_error(eth_actuals, eth_predictions)
            metrics[f"{eth}_r2"] = r2_score(eth_actuals, eth_predictions)
        
        # Calculate metrics for specific case types
        # First for all ethnicities combined
        for case_type in ['colonization', 'decline', 'jump', 'pde']:
            case_predictions = []
            case_actuals = []
            case_count = 0
            
            # Collect case-specific data across all ethnicities
            for eth in self.ethnicity_names:
                case_mask = results_df[f"{eth}_is_{case_type}"]
                if case_mask.sum() > 0:
                    case_predictions.extend(results_df.loc[case_mask, f"{eth}_predicted"].values)
                    case_actuals.extend(results_df.loc[case_mask, f"{eth}_actual"].values)
                    case_count += case_mask.sum()
            
            # Calculate case-specific metrics if there are any cases
            if case_count > 0:
                metrics[f"{case_type}_mse"] = mean_squared_error(case_actuals, case_predictions)
                metrics[f"{case_type}_mae"] = mean_absolute_error(case_actuals, case_predictions)
                metrics[f"{case_type}_count"] = case_count
        
        # Calculate regular case metrics
        regular_predictions = []
        regular_actuals = []
        regular_count = 0
        
        for eth in self.ethnicity_names:
            # Define regular as not being any of the special cases
            regular_mask = ~(
                results_df[f"{eth}_is_colonization"] | 
                results_df[f"{eth}_is_decline"] |
                results_df[f"{eth}_is_jump"] |
                results_df[f"{eth}_is_pde"]
            )
            
            if regular_mask.sum() > 0:
                regular_predictions.extend(results_df.loc[regular_mask, f"{eth}_predicted"].values)
                regular_actuals.extend(results_df.loc[regular_mask, f"{eth}_actual"].values)
                regular_count += regular_mask.sum()
        
        # Calculate regular case metrics if there are any regular cases
        if regular_count > 0:
            metrics['regular_mse'] = mean_squared_error(regular_actuals, regular_predictions)
            metrics['regular_mae'] = mean_absolute_error(regular_actuals, regular_predictions)
            metrics['regular_count'] = regular_count
        
        return metrics
    
    def save_model(self, path):
        """
        Save model to file with complete architecture information
        """
        # Get all hidden dimensions, including the first one
        hidden_dims = [
            self.model.feature_embedding[0].out_features  # First hidden dim
        ]
        
        # Add remaining hidden dimensions
        for module in self.model.layers:
            if isinstance(module, nn.Linear):
                hidden_dims.append(module.out_features)
        
        # Save model state and architecture parameters
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'ethnicity_names': self.ethnicity_names,
            'model_config': {
                'input_dim': self.model.input_dim,
                'num_ethnicities': self.model.num_ethnicities,
                'hidden_dims': hidden_dims,  # Complete hidden dims
                'coord_embed_dim': self.model.coord_embedding.coord_embed[0].out_features,  # Actual embedding dim
                'dropout_rate': next(module.p for module in self.model.layers if isinstance(module, nn.Dropout)),
                'num_attention_heads': self.model.spatial_attention.attentions[0].num_heads,
                'num_matrices': len(self.model.spatial_attention.attentions),
                'colonization_threshold': self.model.colonization_threshold,
                'colonization_expert_weight': self.model.colonization_expert_weight,
                'jump_threshold': self.model.jump_threshold,
                'jump_expert_weight': self.model.jump_expert_weight,
                'decline_threshold': self.model.decline_threshold,
                'decline_expert_weight': self.model.decline_expert_weight,
                'pde_expert_weight': self.model.pde_expert_weight,
                'log_space': self.model.log_space
            }
        }
        
        torch.save(model_info, path)
        logger.info(f"Multi-Ethnic Spatial MoE model saved to {path}")
    
    def visualize_results(self, history, save_path='./viz_plot'):
        """
        Visualize training history with separate plots for different case types and ethnicities
        
        Args:
            history (dict): Training history
            save_path (str, optional): Path to save the plots
        """
        # First create overall plots for general metrics
        fig1, axs1 = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot training and validation loss
        axs1[0, 0].plot(history['train_loss'], label='Training Loss')
        axs1[0, 0].plot(history['val_loss'], label='Validation Loss')
        axs1[0, 0].set_title('Overall MSE Loss')
        axs1[0, 0].set_xlabel('Epoch')
        axs1[0, 0].set_ylabel('Loss')
        axs1[0, 0].legend()
        axs1[0, 0].grid(True)
        
        # Plot case-specific losses
        axs1[0, 1].plot(history['train_colonization_loss'], label='Colonization')
        axs1[0, 1].plot(history['train_jump_loss'], label='Jump')
        axs1[0, 1].plot(history['train_decline_loss'], label='Decline')
        axs1[0, 1].plot(history['train_pde_loss'], label='PDE')
        axs1[0, 1].set_title('Case-Specific Training Losses')
        axs1[0, 1].set_xlabel('Epoch')
        axs1[0, 1].set_ylabel('Loss')
        axs1[0, 1].legend()
        axs1[0, 1].grid(True)
        
        # Plot validation metrics
        axs1[1, 0].plot(history['val_mae'], color='orange', label='MAE')
        axs1[1, 0].set_title('Validation MAE')
        axs1[1, 0].set_xlabel('Epoch')
        axs1[1, 0].set_ylabel('MAE')
        axs1[1, 0].grid(True)
        
        axs1[1, 1].plot(history['val_r2'], color='green', label='R²')
        axs1[1, 1].set_title('Validation R²')
        axs1[1, 1].set_xlabel('Epoch')
        axs1[1, 1].set_ylabel('R²')
        axs1[1, 1].grid(True)
        
        # Summary information
        axs1[2, 0].axis('off')
        info_text = (
            "Multi-Ethnic Spatial MoE Model Training Summary:\n"
            f"Final Training Loss: {history['train_loss'][-1]:.4f}\n"
            f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n"
            f"Final Validation MAE: {history['val_mae'][-1]:.4f}\n"
            f"Final Validation R²: {history['val_r2'][-1]:.4f}\n\n"
            f"Best Validation Loss: {min(history['val_loss']):.4f}\n"
            f"Best Validation R²: {max(history['val_r2']):.4f}"
        )
        axs1[2, 0].text(0.1, 0.5, info_text, fontsize=12)
        
        # Case type summary
        axs1[2, 1].axis('off')
        case_info_text = (
            "Case-Specific Metrics:\n"
            f"Colonization Loss: {history['val_colonization_loss'][-1]:.4f}\n"
            f"Jump Loss: {history['val_jump_loss'][-1]:.4f}\n"
            f"Decline Loss: {history['val_decline_loss'][-1]:.4f}\n"
            f"PDE Loss: {history['val_pde_loss'][-1]:.4f}\n"
            f"Regular Loss: {history['val_regular_loss'][-1]:.4f}\n\n"
            f"Best Colonization Loss: {min(history['val_colonization_loss']):.4f}\n"
            f"Best Jump Loss: {min(history['val_jump_loss']):.4f}\n"
            f"Best Decline Loss: {min(history['val_decline_loss']):.4f}\n"
            f"Best PDE Loss: {min(history['val_pde_loss']):.4f}\n"
            f"Best Regular Loss: {min(history['val_regular_loss']):.4f}"
        )
        axs1[2, 1].text(0.1, 0.5, case_info_text, fontsize=12)
        
        plt.tight_layout()
        
        # Create ethnicity-specific plots
        max_ethnicities_per_fig = 8
        total_ethnicities = len(self.ethnicity_names)
        num_ethnicity_figures = (total_ethnicities + max_ethnicities_per_fig - 1) // max_ethnicities_per_fig
        
        ethnicity_figures = []
        
        for fig_idx in range(num_ethnicity_figures):
            start_idx = fig_idx * max_ethnicities_per_fig
            end_idx = min(start_idx + max_ethnicities_per_fig, total_ethnicities)
            ethnicities_to_plot = self.ethnicity_names[start_idx:end_idx]
            n_ethnicities = len(ethnicities_to_plot)
            
            # Create figure with enough rows for all ethnicities in this batch
            n_rows = min(n_ethnicities, 4)  # Maximum 4 ethnicities per row
            n_cols = (n_ethnicities + n_rows - 1) // n_rows
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
            
            # Handle the case where axs is a single axis
            if n_rows == 1 and n_cols == 1:
                axs = np.array([[axs]])
            elif n_rows == 1:
                axs = np.array([axs])
            elif n_cols == 1:
                axs = np.array([[ax] for ax in axs])
            
            # Plot ethnicity-specific metrics
            for i, eth in enumerate(ethnicities_to_plot):
                row = i // n_cols
                col = i % n_cols
                
                # Plot loss for this ethnicity
                axs[row, col].plot(history['ethnicity_loss'][eth], label='Loss')
                if 'ethnicity_mae' in history and eth in history['ethnicity_mae']:
                    axs[row, col].plot(history['ethnicity_mae'][eth], label='MAE', linestyle='--')
                
                axs[row, col].set_title(f'{eth} Metrics')
                axs[row, col].set_xlabel('Epoch')
                axs[row, col].set_ylabel('Value')
                axs[row, col].legend()
                axs[row, col].grid(True)
            
            # Hide empty subplots
            for i in range(len(ethnicities_to_plot), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axs[row, col].axis('off')
            
            plt.tight_layout()
            ethnicity_figures.append(fig)
        
        # Save plots if path is provided
        if save_path:
            base_path = save_path.rsplit('.', 1)[0]
            ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
            
            # Save main figure
            main_path = f"{base_path}_main.{ext}"
            fig1.savefig(main_path)
            logger.info(f"Main plot saved to {main_path}")
            
            # Save ethnicity figures
            for i, fig in enumerate(ethnicity_figures):
                eth_path = f"{base_path}_ethnicities_{i}.{ext}"
                fig.savefig(eth_path)
                logger.info(f"Ethnicity plot {i+1} saved to {eth_path}")
        
        return fig1, ethnicity_figures
    
    def save_metrics_to_csv(self, history, output_dir='./metrics'):
        """
        Save training, validation, and test metrics to CSV files
        
        Args:
            history (dict): Training history dictionary
            output_dir (str): Directory to save CSV files
        """
        import os
        import pandas as pd
        import numpy as np
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving metrics to {output_dir}")
        
        # --- Fix for training metrics --- 
        # Get the maximum length of epoch-related metrics
        train_history_length = len(history['train_loss'])
        
        # Initialize the training metrics dict with the epoch column
        train_metrics = {
            'epoch': list(range(1, train_history_length + 1))
        }
        
        # Add all scalar training metrics, ensuring they're of the right length
        scalar_metrics = [
            'train_loss', 'train_aux_loss', 'train_colonization_loss', 
            'train_jump_loss', 'train_decline_loss', 'train_pde_loss', 
            'train_regular_loss'
        ]
        
        for metric in scalar_metrics:
            metric_values = history[metric]
            # Ensure all metrics have the same length by padding with NaN if necessary
            if len(metric_values) < train_history_length:
                metric_values = metric_values + [np.nan] * (train_history_length - len(metric_values))
            elif len(metric_values) > train_history_length:
                metric_values = metric_values[:train_history_length]
            
            # Use the short name (without 'train_' prefix) for the column
            short_name = metric.replace('train_', '')
            train_metrics[short_name] = metric_values
        
        # Add ethnicity-specific training losses with length checking
        for eth in self.ethnicity_names:
            if eth in history['ethnicity_loss']:
                eth_values = history['ethnicity_loss'][eth]
                # Ensure the length matches
                if len(eth_values) < train_history_length:
                    eth_values = eth_values + [np.nan] * (train_history_length - len(eth_values))
                elif len(eth_values) > train_history_length:
                    eth_values = eth_values[:train_history_length]
                    
                train_metrics[f'{eth}_loss'] = eth_values
        
        # Create and save training DataFrame
        train_df = pd.DataFrame(train_metrics)
        train_path = os.path.join(output_dir, 'training_metrics.csv')
        train_df.to_csv(train_path, index=False)
        logger.info(f"Training metrics saved to {train_path}")
        
        # --- Fix for validation metrics ---
        # Save validation metrics if available
        if 'val_loss' in history and len(history['val_loss']) > 0:
            # Get the maximum length of validation metrics
            val_history_length = len(history['val_loss'])
            
            # Initialize with the epoch column
            val_metrics = {
                'epoch': list(range(1, val_history_length + 1))
            }
            
            # Add all scalar validation metrics
            scalar_val_metrics = [
                'val_loss', 'val_mae', 'val_r2', 
                'val_colonization_loss', 'val_jump_loss', 
                'val_decline_loss', 'val_pde_loss', 
                'val_regular_loss'
            ]
            
            for metric in scalar_val_metrics:
                if metric in history and len(history[metric]) > 0:
                    metric_values = history[metric]
                    # Ensure all metrics have the same length
                    if len(metric_values) < val_history_length:
                        metric_values = metric_values + [np.nan] * (val_history_length - len(metric_values))
                    elif len(metric_values) > val_history_length:
                        metric_values = metric_values[:val_history_length]
                    
                    # Use the short name (without 'val_' prefix) for the column
                    short_name = metric.replace('val_', '')
                    val_metrics[short_name] = metric_values
            
            # Add ethnicity-specific validation metrics
            for eth in self.ethnicity_names:
                # Add ethnicity-specific loss
                if eth in history['ethnicity_loss'] and len(history['ethnicity_loss'][eth]) > 0:
                    eth_values = history['ethnicity_loss'][eth]
                    # Ensure the length matches
                    if len(eth_values) < val_history_length:
                        eth_values = eth_values + [np.nan] * (val_history_length - len(eth_values))
                    elif len(eth_values) > val_history_length:
                        eth_values = eth_values[:val_history_length]
                        
                    val_metrics[f'{eth}_loss'] = eth_values
                
                # Add ethnicity-specific MAE if available
                if 'ethnicity_mae' in history and eth in history['ethnicity_mae'] and len(history['ethnicity_mae'][eth]) > 0:
                    mae_values = history['ethnicity_mae'][eth]
                    # Ensure the length matches
                    if len(mae_values) < val_history_length:
                        mae_values = mae_values + [np.nan] * (val_history_length - len(mae_values))
                    elif len(mae_values) > val_history_length:
                        mae_values = mae_values[:val_history_length]
                        
                    val_metrics[f'{eth}_mae'] = mae_values
                
                # Add ethnicity-specific R2 if available
                if 'ethnicity_r2' in history and eth in history['ethnicity_r2'] and len(history['ethnicity_r2'][eth]) > 0:
                    r2_values = history['ethnicity_r2'][eth]
                    # Ensure the length matches
                    if len(r2_values) < val_history_length:
                        r2_values = r2_values + [np.nan] * (val_history_length - len(r2_values))
                    elif len(r2_values) > val_history_length:
                        r2_values = r2_values[:val_history_length]
                        
                    val_metrics[f'{eth}_r2'] = r2_values
            
            # Create and save validation DataFrame
            val_df = pd.DataFrame(val_metrics)
            val_path = os.path.join(output_dir, 'validation_metrics.csv')
            val_df.to_csv(val_path, index=False)
            logger.info(f"Validation metrics saved to {val_path}")

    def save_test_metrics_to_csv(self, test_metrics, output_dir='./metrics'):
        """
        Save test metrics to CSV file
        
        Args:
            test_metrics (dict): Dictionary with test evaluation metrics
            output_dir (str): Directory to save CSV files
        """
        import os
        import pandas as pd
        import numpy as np
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Transform the test metrics dictionary into a suitable format for CSV
        # First create overall metrics
        overall_metrics = {
            'metric': ['mse', 'mae', 'r2'],
            'value': [test_metrics.get('mse', np.nan), 
                    test_metrics.get('mae', np.nan), 
                    test_metrics.get('r2', np.nan)]
        }
        
        # Create DataFrame and save
        test_overall_df = pd.DataFrame(overall_metrics)
        test_overall_path = os.path.join(output_dir, 'test_overall_metrics.csv')
        test_overall_df.to_csv(test_overall_path, index=False)
        logger.info(f"Test overall metrics saved to {test_overall_path}")
        
        # Create ethnicity-specific test metrics
        eth_data = []
        for eth in self.ethnicity_names:
            eth_mse_key = f"{eth}_mse"
            eth_mae_key = f"{eth}_mae"
            eth_r2_key = f"{eth}_r2"
            
            if eth_mse_key in test_metrics:
                eth_data.append({
                    'ethnicity': eth,
                    'mse': test_metrics[eth_mse_key],
                    'mae': test_metrics[eth_mae_key],
                    'r2': test_metrics[eth_r2_key]
                })
        
        if eth_data:
            eth_df = pd.DataFrame(eth_data)
            eth_path = os.path.join(output_dir, 'test_ethnicity_metrics.csv')
            eth_df.to_csv(eth_path, index=False)
            logger.info(f"Test ethnicity metrics saved to {eth_path}")
        
        # Create case-specific test metrics
        case_data = []
        for case_type in ['colonization', 'jump', 'decline', 'pde', 'regular']:
            mse_key = f"{case_type}_mse"
            mae_key = f"{case_type}_mae"
            count_key = f"{case_type}_count"
            
            if mse_key in test_metrics:
                case_data.append({
                    'case_type': case_type,
                    'mse': test_metrics[mse_key],
                    'mae': test_metrics[mae_key],
                    'count': test_metrics.get(count_key, 0)
                })
        
        if case_data:
            case_df = pd.DataFrame(case_data)
            case_path = os.path.join(output_dir, 'test_case_metrics.csv')
            case_df.to_csv(case_path, index=False)
            logger.info(f"Test case metrics saved to {case_path}")