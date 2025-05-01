import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from datetime import datetime
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Import your model components
from csrn.data_loader.spatial_data_loader import MultiEthnicDataHandler
from spatial_moe_model import MultiEthnicSpatialMoEPredictor
from spatial_moe_trainer import MultiEthnicMoEModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variable to store data handler and data
global_data = None
global_ethnicity_names = None

def load_data(args):
    """
    Load data once to be reused across trials
    """
    global global_data, global_ethnicity_names
    
    # Load global parameters for PDE expert
    global_params = load_global_parameters(args.global_params_path)
    
    # Load DA coordinates
    da_coordinates = load_da_coordinates(args.da_coordinates_path)
    
    # Create data handler
    data_handler = MultiEthnicDataHandler(args.data_dir, args.spatial_dir, scaler_type=args.scaler)
    
    # Pass DA coordinates to the data handler if loaded successfully
    if da_coordinates is not None:
        data_handler.da_coordinates = da_coordinates
    
    # Load data with multiple ethnicities
    data = data_handler.load_data()
    
    # Create dataloaders
    train_loader, val_loader, _ = data_handler.create_dataloaders(
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
    
    logger.info(f"Loaded data with {len(ethnicity_names)} ethnicities")
    
    # Store data for reuse
    global_data = {
        'data': data,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'data_handler': data_handler,
        'global_params': global_params
    }
    
    global_ethnicity_names = ethnicity_names
    
    return global_data, global_ethnicity_names

def load_global_parameters(json_path):
    """
    Load global parameters for the PDE expert from a JSON file
    
    Args:
        json_path (str): Path to the JSON file containing global parameters
        
    Returns:
        dict: Dictionary containing global parameters
    """
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded global parameters from {json_path}")
        return params
    except Exception as e:
        logger.warning(f"Error loading global parameters: {str(e)}")
        # Return default parameters
        default_params = {
            "K": 0.046942830571135886,
            "sigma_squared": 3.5911050665737987,
            "alpha_0": 0.5667213663794652,
            "beta_0": 3.3373053296446606
        }
        logger.info(f"Using default global parameters: {default_params}")
        return default_params
    
def load_da_coordinates(da_file_path):
    """
    Load DA coordinates from CSV file
    
    Args:
        da_file_path (str): Path to the CSV file containing DA coordinates
        
    Returns:
        pd.DataFrame: DataFrame containing DA coordinates
    """
    try:
        da_df = pd.read_csv(da_file_path)
        logger.info(f"Loaded DA coordinates from {da_file_path}")
        return da_df
    except Exception as e:
        logger.error(f"Error loading DA coordinates: {str(e)}")
        return None

def objective(trial, args, gpu_id):
    """
    Objective function for Optuna
    
    Args:
        trial: Optuna trial object
        args: Command line arguments
        gpu_id: GPU ID to use for this trial
    
    Returns:
        float: Validation loss
    """
    # Set device
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running trial {trial.number} on device {device}")
    
    # Get global data
    global global_data, global_ethnicity_names
    
    if global_data is None:
        global_data, global_ethnicity_names = load_data(args)
    
    train_loader = global_data['train_loader']
    val_loader = global_data['val_loader']
    global_params = global_data['global_params']
    
    # Get input dimension
    sample_batch = next(iter(train_loader))
    input_features_dim = sample_batch['features'].shape[1]
    
    # Define hyperparameters to optimize
    # First choose number of attention heads - this will constrain other parameters
    attention_heads = trial.suggest_int('attention_heads', 2, 6)
    
    # Now ensure hidden_dims are divisible by attention_heads
    hidden_dim_1_base = trial.suggest_int('hidden_dim_1_base', 8, 32)
    hidden_dim_2_base = trial.suggest_int('hidden_dim_2_base', 4, 16)
    hidden_dim_3_base = trial.suggest_int('hidden_dim_3_base', 2, 8)
    
    # Multiply by attention_heads to ensure divisibility
    hidden_dims = [
        hidden_dim_1_base * attention_heads,
        hidden_dim_2_base * attention_heads,
        hidden_dim_3_base * attention_heads
    ]
    
    # Make coord_embed_dim divisible by attention_heads
    coord_embed_base = trial.suggest_int('coord_embed_base', 2, 8)
    coord_embed_dim = coord_embed_base * attention_heads
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    colonization_threshold = trial.suggest_float('colonization_threshold', 0.05, 0.2)
    colonization_expert_weight = trial.suggest_float('colonization_expert_weight', 0.5, 0.9)
    colonization_weight = trial.suggest_float('colonization_weight', 1.0, 10.0)
    
    jump_threshold = trial.suggest_float('jump_threshold', 50.0, 200.0)
    jump_expert_weight = trial.suggest_float('jump_expert_weight', 0.5, 0.9)
    jump_weight = trial.suggest_float('jump_weight', 1.0, 10.0)
    
    decline_threshold = trial.suggest_float('decline_threshold', 20.0, 100.0)
    decline_expert_weight = trial.suggest_float('decline_expert_weight', 0.5, 0.9)
    decline_weight = trial.suggest_float('decline_weight', 1.0, 10.0)
    
    pde_expert_weight = trial.suggest_float('pde_expert_weight', 0.4, 0.8)
    pde_weight = trial.suggest_float('pde_weight', 1.0, 10.0)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
    
    # Create trial output directory
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    metrics_dir = os.path.join(trial_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    model_path = os.path.join(trial_dir, "model.pt")
    
    # Save trial parameters
    trial_params = {
        'trial_number': trial.number,
        'hidden_dims': hidden_dims,
        'coord_embed_dim': coord_embed_dim,
        'attention_heads': attention_heads,
        'dropout_rate': dropout_rate,
        'colonization_threshold': colonization_threshold,
        'colonization_expert_weight': colonization_expert_weight,
        'colonization_weight': colonization_weight,
        'jump_threshold': jump_threshold,
        'jump_expert_weight': jump_expert_weight,
        'jump_weight': jump_weight,
        'decline_threshold': decline_threshold,
        'decline_expert_weight': decline_expert_weight,
        'decline_weight': decline_weight,
        'pde_expert_weight': pde_expert_weight,
        'pde_weight': pde_weight,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device': device,
    }
    
    # Save trial parameters to JSON
    with open(os.path.join(trial_dir, "params.json"), 'w') as f:
        json.dump(trial_params, f, indent=4)
    
    # Create model
    try:
        # Make sure hidden dimensions are consistent with each other and attention heads
        # Fix 1: Ensure coord_embed_dim is divisible by attention_heads
        coord_embed_dim = (coord_embed_dim // attention_heads) * attention_heads
        
        # Fix 2: Ensure hidden_dims[0] is divisible by attention_heads
        hidden_dims[0] = (hidden_dims[0] // attention_heads) * attention_heads
        
        # Fix 3: Ensure layers have matching dimensions - this is critical
        # Update trial parameters with corrected dimensions
        trial_params['hidden_dims'] = hidden_dims
        trial_params['coord_embed_dim'] = coord_embed_dim
        
        logger.info(f"Using adjusted dimensions - hidden_dims: {hidden_dims}, coord_embed_dim: {coord_embed_dim}")
        
        model = MultiEthnicSpatialMoEPredictor(
            input_dim=input_features_dim,
            num_ethnicities=len(global_ethnicity_names),
            hidden_dims=hidden_dims,
            coord_embed_dim=coord_embed_dim,
            num_attention_heads=attention_heads,
            num_matrices=3,  # Walking, transit, proximity
            dropout_rate=dropout_rate,
            colonization_threshold=colonization_threshold,
            colonization_expert_weight=colonization_expert_weight,
            jump_threshold=jump_threshold,
            jump_expert_weight=jump_expert_weight,
            decline_threshold=decline_threshold,
            decline_expert_weight=decline_expert_weight,
            pde_expert_weight=pde_expert_weight,
            global_params=global_params,
            log_space=args.log_space,
            device=device
        )
        
        # Create trainer
        trainer = MultiEthnicMoEModelTrainer(
            model=model,
            ethnicity_names=global_ethnicity_names,
            colonization_threshold=colonization_threshold,
            jump_threshold=jump_threshold,
            decline_threshold=decline_threshold,
            device=device
        )
        
        # Define pruning callback
        def pruning_callback(epoch, metrics):
            # Report metrics to Optuna for potential pruning
            trial.report(metrics['val_loss'], epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Try model forward pass with a sample batch to catch any shape issues
        try:
            sample_batch = next(iter(train_loader))
            features = sample_batch['features'].to(device)
            population_2016 = sample_batch['population_2016'].to(device)
            coordinates = sample_batch['coordinates'].to(device)
            
            # Get spatial matrices if available
            walking_matrix = sample_batch.get('walking_matrix')
            transit_matrix = sample_batch.get('transit_matrix')
            proximity_matrix = sample_batch.get('proximity_matrix')
            
            if walking_matrix is not None:
                walking_matrix = walking_matrix.to(device)
            if transit_matrix is not None:
                transit_matrix = transit_matrix.to(device)
            if proximity_matrix is not None:
                proximity_matrix = proximity_matrix.to(device)
            
            # Test forward pass
            logger.info("Testing model forward pass before training...")
            with torch.no_grad():
                outputs, expert_weights, expert_classes = model(
                    features, population_2016, coordinates, 
                    walking_matrix, transit_matrix, proximity_matrix
                )
            logger.info(f"Test forward pass successful. Output shape: {outputs.shape}")
        except Exception as e:
            logger.error(f"Test forward pass failed: {str(e)}")
            return float('inf')
            
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
            colonization_weight=colonization_weight,
            jump_weight=jump_weight,
            decline_weight=decline_weight,
            pde_weight=pde_weight,
            patience=args.patience,
            early_stopping_patience=args.early_stopping,
            ethnicity_weights=None,  # Use equal weights for all ethnicities
            pruning_callback=pruning_callback
        )
        
        # Save training metrics
        trainer.save_metrics_to_csv(history, output_dir=metrics_dir)
        
        # Save model
        trainer.save_model(model_path)
        
        # Get best validation loss
        best_val_loss = min(history['val_loss'])
        best_val_r2 = max(history['val_r2'])
        
        # Update trial params with results
        trial_params['best_val_loss'] = best_val_loss
        trial_params['best_val_r2'] = best_val_r2
        trial_params['epochs_trained'] = len(history['val_loss'])
        
        # Save updated trial parameters
        with open(os.path.join(trial_dir, "params.json"), 'w') as f:
            json.dump(trial_params, f, indent=4)
        
        # Return the best validation loss for optimization
        return best_val_loss
    
    except optuna.TrialPruned:
        # Trial was pruned by Optuna
        logger.info(f"Trial {trial.number} pruned")
        raise
    except Exception as e:
        # Log other errors but don't raise them to Optuna
        logger.exception(f"Error in trial {trial.number}: {str(e)}")
        return float('inf')

def trial_worker(args, gpu_id, trials_queue, results_queue):
    """
    Worker process for running trials on a specific GPU
    
    Args:
        args: Command line arguments
        gpu_id: GPU ID to use
        trials_queue: Queue for getting trials
        results_queue: Queue for returning results
    """
    try:
        while True:
            # Get trial from queue or exit if None
            trial_item = trials_queue.get()
            if trial_item is None:
                break
                
            trial, study = trial_item
            
            # Run objective function
            value = objective(trial, args, gpu_id)
            
            # Return result
            results_queue.put((trial, value))
    except Exception as e:
        logger.exception(f"Error in worker {gpu_id}: {str(e)}")

def run_optimization(args):
    """
    Run hyperparameter optimization using Optuna distributed across multiple GPUs
    
    Args:
        args: Command line arguments
    """
    # Set up output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up study
    study_name = f"multi_ethnic_moe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_storage = f"sqlite:///{os.path.join(args.output_dir, 'study.db')}"
    
    logger.info(f"Creating Optuna study: {study_name}")
    
    # Set up sampler and pruner
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=study_storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True
    )
    
    # Set multiprocessing method
    mp.set_start_method('spawn', force=True)
    
    # Set number of GPUs to use
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    logger.info(f"Using {num_gpus} GPUs for hyperparameter optimization")
    
    # Use multiprocessing to distribute trials across GPUs
    trials_queue = mp.Queue()
    results_queue = mp.Queue()
    workers = []
    
    # Create worker processes
    for gpu_id in range(num_gpus):
        worker = mp.Process(
            target=trial_worker,
            args=(args, gpu_id, trials_queue, results_queue)
        )
        worker.start()
        workers.append(worker)
    
    # Create and queue all trials
    trials = []
    for i in range(args.n_trials):
        trial = study.ask()
        trials.append(trial)
        trials_queue.put((trial, study))
    
    # Add termination signals
    for _ in range(num_gpus):
        trials_queue.put(None)
    
    # Collect results
    for i in range(len(trials)):
        trial, value = results_queue.get()
        study.tell(trial, value)
        
        logger.info(f"Trial {trial.number} completed with value: {value:.6f}")
        
        # Save intermediate study results
        df = study.trials_dataframe()
        df.to_csv(os.path.join(args.output_dir, "study_results.csv"), index=False)
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join()
    
    # Check if there are any successful trials
    successful_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value != float('inf')]
    
    if successful_trials:
        # Get the best trial among successful ones
        best_trial = min(successful_trials, key=lambda t: t.value)
        
        logger.info("Hyperparameter optimization completed")
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_trial.value:.6f}")
        logger.info(f"Best parameters: {best_trial.params}")
        
        # Save best parameters
        best_params = {
            'trial_number': best_trial.number,
            'best_value': best_trial.value,
            'parameters': best_trial.params
        }
    else:
        logger.warning("No successful trials found. All trials failed or were pruned.")
        
        # Create empty best params
        best_params = {
            'trial_number': None,
            'best_value': None,
            'parameters': {},
            'note': "No successful trials were completed."
        }
    
    # Save best parameters to JSON
    best_params = {
        'trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'parameters': study.best_params
    }
    
    with open(os.path.join(args.output_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Export study results to CSV
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(args.output_dir, "all_trials.csv"), index=False)
    
    # Create summary CSV with key metrics
    summary_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
            params_file = os.path.join(trial_dir, "params.json")
            
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = json.load(f)
                
                summary_data.append({
                    'trial': trial.number,
                    'value': trial.value,
                    'best_val_loss': params.get('best_val_loss', float('nan')),
                    'best_val_r2': params.get('best_val_r2', float('nan')),
                    'epochs_trained': params.get('epochs_trained', 0),
                    **{f'param_{k}': v for k, v in trial.params.items()}
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Multi-Ethnic Spatial MoE Model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--spatial_dir", type=str, default="./spatial_data",
                        help="Directory containing spatial data files")
    parser.add_argument("--global_params_path", type=str, default="./data/global_parameters.json",
                        help="Path to the global parameters JSON file for PDE expert")
    parser.add_argument("--da_coordinates_path", type=str, default="./data/toronto_da.csv",
                        help="Path to the CSV file containing DA coordinates")
    
    # Optimization parameters
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of trials to run")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs to use")
    parser.add_argument("--output_dir", type=str, default="./optuna_results",
                        help="Directory to save optimization results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs per trial")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for learning rate scheduler")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use for feature normalization")
    parser.add_argument("--log_space", action="store_true",
                        help="Whether to operate in log space for population values")
    
    args = parser.parse_args()
    
    # Run optimization
    run_optimization(args)