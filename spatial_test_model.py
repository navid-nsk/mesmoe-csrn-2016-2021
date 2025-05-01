import json
import os
import logging
import argparse
import pandas as pd
import torch
import numpy as np
from csrn.data_loader.spatial_data_loader import MultiEthnicDataHandler
from spatial_moe_model import MultiEthnicSpatialMoEPredictor
from spatial_moe_trainer import MultiEthnicMoEModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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
        logger.info(f"Loaded global parameters from {json_path}: {params}")
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

def main(args):
    """
    Main function to evaluate the pretrained model on test data
    
    Args:
        args: Command line arguments
    """
    logger.info("=== Starting Multi-Ethnic Spatial MoE Model Evaluation ===")
    logger.info(f"Arguments: {args}")
    
    # Load global parameters for PDE expert
    global_params = load_global_parameters(args.global_params_path)
    
    # Load DA coordinates for PDE expert
    da_coordinates = load_da_coordinates(args.da_coordinates_path)
    
    # Create data handler and load data
    data_handler = MultiEthnicDataHandler(args.data_dir, args.spatial_dir, scaler_type=args.scaler)
    
    # Pass DA coordinates to the data handler if loaded successfully
    if da_coordinates is not None:
        data_handler.da_coordinates = da_coordinates
    
    # Load data with multiple ethnicities
    data = data_handler.load_data()
    
    # Create dataloaders - we only need the test loader for evaluation
    _, _, test_loader = data_handler.create_dataloaders(
        data, batch_size=args.batch_size
    )
    
    # Get ethnicity names
    ethnicity_names = []
    if hasattr(test_loader.dataset, 'ethnicity_columns'):
        ethnicity_names = test_loader.dataset.ethnicity_columns
    else:
        # Try to get them from the columns in the population data
        if 'test_pop_2016' in data:
            ethnicity_names = [col for col in data['test_pop_2016'].columns if col != 'DAUID']
    
    num_ethnicities = len(ethnicity_names)
    logger.info(f"Processing {num_ethnicities} ethnicities: {ethnicity_names}")
    
    try:
        # Load the pretrained model
        logger.info(f"Loading pretrained model from {args.model_path}")
        model_info = torch.load(args.model_path, map_location=args.device)
        
        # Check if the model was saved with full information
        if 'model_state_dict' in model_info and 'model_config' in model_info:
            # Extract model configuration and state dict
            model_config = model_info['model_config']
            model_state_dict = model_info['model_state_dict']
            
            # Create a new model with the same architecture
            logger.info("Creating model with the same architecture")
            model = MultiEthnicSpatialMoEPredictor(
                input_dim=model_config['input_dim'],
                num_ethnicities=model_config['num_ethnicities'],
                hidden_dims=model_config['hidden_dims'],
                coord_embed_dim=model_config['coord_embed_dim'],
                num_attention_heads=model_config['num_attention_heads'],
                num_matrices=model_config['num_matrices'],
                dropout_rate=model_config['dropout_rate'],
                colonization_threshold=model_config['colonization_threshold'],
                colonization_expert_weight=model_config['colonization_expert_weight'],
                jump_threshold=model_config['jump_threshold'],
                jump_expert_weight=model_config['jump_expert_weight'],
                decline_threshold=model_config['decline_threshold'],
                decline_expert_weight=model_config['decline_expert_weight'],
                pde_expert_weight=model_config['pde_expert_weight'],
                global_params=global_params,
                log_space=model_config['log_space'],
                device=args.device
            )
            
            # Load the weights
            model.load_state_dict(model_state_dict)
            
            # Get thresholds from the model config
            colonization_threshold = model_config['colonization_threshold']
            jump_threshold = model_config['jump_threshold']
            decline_threshold = model_config['decline_threshold']
            
        else:
            # If the model was saved with just the state dict
            logger.warning("Model file does not contain full configuration. Using default parameters.")
            
            # Get sample batch to determine input dimension
            sample_batch = next(iter(test_loader))
            input_features_dim = sample_batch['features'].shape[1]
            
            # Create hidden_dims from the optimized parameters
            hidden_dims = [
                args.hidden_dim_1_base * args.attention_heads,
                args.hidden_dim_2_base * args.attention_heads,
                args.hidden_dim_3_base * args.attention_heads
            ]
            
            # Calculate coord_embed_dim from the optimized parameters
            coord_embed_dim = args.coord_embed_base * args.attention_heads
            
            # Create a new model with default parameters
            model = MultiEthnicSpatialMoEPredictor(
                input_dim=input_features_dim,
                num_ethnicities=num_ethnicities,
                hidden_dims=hidden_dims,
                coord_embed_dim=coord_embed_dim,
                num_attention_heads=args.attention_heads,
                num_matrices=3,  # Walking, transit, proximity
                dropout_rate=args.dropout_rate,
                colonization_threshold=args.colonization_threshold,
                colonization_expert_weight=args.colonization_expert_weight,
                jump_threshold=args.jump_threshold,
                jump_expert_weight=args.jump_expert_weight,
                decline_threshold=args.decline_threshold,
                decline_expert_weight=args.decline_expert_weight,
                pde_expert_weight=args.pde_expert_weight,
                global_params=global_params,
                log_space=args.log_space,
                device=args.device
            )
            
            # Load state dict directly
            model.load_state_dict(model_info)
            
            # Use thresholds from args
            colonization_threshold = args.colonization_threshold
            jump_threshold = args.jump_threshold
            decline_threshold = args.decline_threshold
        
        # Move model to device
        model.to(args.device)
        model.eval()
        
        # Create trainer for evaluation
        trainer = MultiEthnicMoEModelTrainer(
            model=model,
            ethnicity_names=ethnicity_names,
            colonization_threshold=colonization_threshold,
            jump_threshold=jump_threshold,
            decline_threshold=decline_threshold,
            device=args.device
        )
        
        # Check if test data has ground truth (2021 population)
        has_ground_truth = any(f"{k}_2021" in data for k in ['test_pop', 'test_pop_2021'])
        
        if has_ground_truth:
            # Evaluate the model
            logger.info("Evaluating model on test set...")
            test_metrics = trainer.evaluate(test_loader)
            
            # Log overall metrics
            logger.info(f"Test MSE: {test_metrics['mse']:.4f}")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            logger.info(f"Test R²: {test_metrics['r2']:.4f}")
            
            # Log ethnicity-specific metrics
            logger.info("Ethnicity-specific metrics:")
            for eth in ethnicity_names:
                if f"{eth}_mse" in test_metrics:
                    logger.info(f"  {eth} - MSE: {test_metrics[f'{eth}_mse']:.4f}, "
                                f"MAE: {test_metrics[f'{eth}_mae']:.4f}, "
                                f"R²: {test_metrics[f'{eth}_r2']:.4f}")
            
            # Log special case metrics
            for case_type in ['colonization', 'jump', 'decline', 'pde', 'regular']:
                if f"{case_type}_mse" in test_metrics:
                    logger.info(f"{case_type.capitalize()} - MSE: {test_metrics[f'{case_type}_mse']:.4f}, "
                                f"MAE: {test_metrics[f'{case_type}_mae']:.4f}, "
                                f"Count: {test_metrics[f'{case_type}_count']}")
            
            # Save test metrics to CSV
            if args.metrics_dir:
                os.makedirs(args.metrics_dir, exist_ok=True)
                trainer.save_test_metrics_to_csv(test_metrics, output_dir=args.metrics_dir)
                logger.info(f"Test metrics saved to {args.metrics_dir}")
        
        # Generate predictions
        logger.info("Generating predictions for test set...")
        predictions_df = trainer.get_predictions(test_loader)
        
        # Save predictions
        if args.predictions_path:
            os.makedirs(os.path.dirname(args.predictions_path), exist_ok=True)
            try:
                predictions_df.to_csv(args.predictions_path, index=False)
                logger.info(f"Predictions saved to {args.predictions_path}")
            except Exception as e:
                logger.error(f"Error saving to {args.predictions_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("=== Multi-Ethnic Spatial MoE Model Evaluation Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Ethnic Toronto Spatial Population MoE Model Evaluation")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--spatial_dir", type=str, default="./spatial_data",
                        help="Directory containing spatial data files")
    parser.add_argument("--global_params_path", type=str, default="./data/global_parameters.json",
                        help="Path to the global parameters JSON file for PDE expert")
    parser.add_argument("--da_coordinates_path", type=str, default="./data/toronto_da.csv",
                        help="Path to the CSV file containing DA coordinates")
    
    # Model parameters (only used if the model file doesn't contain full configuration)
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use for feature normalization")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--attention_heads", type=int, default=3,
                        help="Number of attention heads (if needed)")
    parser.add_argument("--hidden_dim_1_base", type=int, default=28,
                        help="Base size for first hidden dimension (if needed)")
    parser.add_argument("--hidden_dim_2_base", type=int, default=9,
                        help="Base size for second hidden dimension (if needed)")
    parser.add_argument("--hidden_dim_3_base", type=int, default=8,
                        help="Base size for third hidden dimension (if needed)")
    parser.add_argument("--coord_embed_base", type=int, default=7,
                        help="Base size for coordinate embedding dimension (if needed)")
    parser.add_argument("--dropout_rate", type=float, default=0.1930287204884957,
                        help="Dropout rate (if needed)")
    parser.add_argument("--colonization_threshold", type=float, default=0.13357705254672683,
                        help="Threshold below which population is considered zero (if needed)")
    parser.add_argument("--colonization_expert_weight", type=float, default=0.7798480148640908,
                        help="Base weight for colonization expert influence (if needed)")
    parser.add_argument("--jump_threshold", type=float, default=142.22567841569256,
                        help="Threshold for identifying potential jump cases (if needed)")
    parser.add_argument("--jump_expert_weight", type=float, default=0.7157969232653937,
                        help="Base weight for jump expert influence (if needed)")
    parser.add_argument("--decline_threshold", type=float, default=73.78197294263202,
                        help="Threshold for identifying potential decline cases (if needed)")
    parser.add_argument("--decline_expert_weight", type=float, default=0.7192671362481913,
                        help="Base weight for decline expert influence (if needed)")
    parser.add_argument("--pde_expert_weight", type=float, default=0.680966412263434,
                        help="Base weight for PDE expert influence (if needed)")
    parser.add_argument("--log_space", action="store_true",
                        help="Whether to operate in log space for population values")
    
    # Model loading and output parameters
    parser.add_argument("--model_path", type=str, default="./model/model.pt",
                        help="Path to the pretrained model")
    parser.add_argument("--predictions_path", type=str, default="./predictions/model_evaluation_predictions.csv",
                        help="Path to save test predictions")
    parser.add_argument("--metrics_dir", type=str, default="./metrics/evaluation",
                        help="Directory to save test metrics as CSV files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    main(args)