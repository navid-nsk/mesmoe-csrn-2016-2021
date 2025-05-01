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
        logging.FileHandler("multi_ethnic_moe_training.log"),
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
    Main function to run the enhanced spatial population prediction with Mixture of Experts
    for multiple ethnicities
    
    Args:
        args: Command line arguments
    """
    logger.info("=== Starting Multi-Ethnic Spatial Population MoE Prediction Pipeline ===")
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
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_handler.create_dataloaders(
        data, batch_size=args.batch_size
    )
    
    # Get ethnicity names from the data handler
    ethnicity_names = []
    if hasattr(train_loader.dataset, 'ethnicity_columns'):
        ethnicity_names = train_loader.dataset.ethnicity_columns
    else:
        # Try to get them from the columns in the population data
        if 'train_pop_2016' in data:
            ethnicity_names = [col for col in data['train_pop_2016'].columns if col != 'DAUID']
    
    num_ethnicities = len(ethnicity_names)
    logger.info(f"Processing {num_ethnicities} ethnicities: {ethnicity_names}")
    
    # Determine input dimension (features only, no population included)
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    input_features_dim = sample_batch['features'].shape[1]
    
    logger.info(f"Input features dimension: {input_features_dim}")
    logger.info(f"Population vectors dimension: {sample_batch['population_2016'].shape[1]}")
    
    # Create hidden_dims from the optimized parameters
    hidden_dims = [
        args.hidden_dim_1_base * args.attention_heads,
        args.hidden_dim_2_base * args.attention_heads,
        args.hidden_dim_3_base * args.attention_heads
    ]
    
    # Calculate coord_embed_dim from the optimized parameters
    coord_embed_dim = args.coord_embed_base * args.attention_heads
    
    logger.info(f"Using optimized hidden_dims: {hidden_dims}")
    logger.info(f"Using optimized coord_embed_dim: {coord_embed_dim}")
    
    # Create Multi-Ethnic Enhanced MoE model with optimized hyperparameters
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
    
    # Log model parameters count
    total_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params_count:,} trainable parameters")
    
    # Create Enhanced Multi-Ethnic MoE trainer
    trainer = MultiEthnicMoEModelTrainer(
        model=model,
        ethnicity_names=ethnicity_names,
        colonization_threshold=args.colonization_threshold,
        jump_threshold=args.jump_threshold,
        decline_threshold=args.decline_threshold,
        device=args.device
    )
    
    # Optional: Create ethnicity weights
    ethnicity_weights = None
    if args.use_ethnicity_weights:
        # Here you can implement a weighting strategy for ethnicities
        # For example, weighting by inverse frequency or population size
        # For now, we'll use equal weights
        ethnicity_weights = torch.ones(num_ethnicities, device=args.device)
        logger.info(f"Using equal weights for all ethnicities")
    
    try:
        # Train model with optimized hyperparameters
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            colonization_weight=args.colonization_weight,
            jump_weight=args.jump_weight,
            decline_weight=args.decline_weight,
            pde_weight=args.pde_weight,
            patience=args.patience,
            early_stopping_patience=args.early_stopping,
            save_train_predictions_at_epoch=args.save_predictions_epoch,
            predictions_path=args.training_predictions_path,
            ethnicity_weights=ethnicity_weights
        )
        
        # Save model
        trainer.save_model(args.model_path)
        
        # Visualize training history
        if args.plot:
            trainer.visualize_results(history, save_path=args.plot_path)
        
        # Save training and validation metrics to CSV
        if args.metrics_dir:
            trainer.save_metrics_to_csv(history, output_dir=args.metrics_dir)
            logger.info(f"Training and validation metrics saved to {args.metrics_dir}")
        
        # Evaluate on test set if available
        if test_loader and any(f"{k}_2021" in data for k in ['test_pop', 'test_pop_2021']):
            logger.info("Evaluating on test set...")
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
                trainer.save_test_metrics_to_csv(test_metrics, output_dir=args.metrics_dir)
                logger.info(f"Test metrics saved to {args.metrics_dir}")
        
        # Generate predictions if test data is available but without 2021 population
        if test_loader and not any(f"{k}_2021" in data for k in ['test_pop', 'test_pop_2021']):
            logger.info("Generating predictions for test set...")
            predictions_df = trainer.get_predictions(test_loader)
            
            # Save predictions
            if args.predictions_path:
                try:
                    predictions_df.to_csv(args.predictions_path, index=False)
                    logger.info(f"Predictions saved to {args.predictions_path}")
                except Exception as e:
                    logger.error(f"Error saving to {args.predictions_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("=== Multi-Ethnic Spatial Population MoE Prediction Pipeline Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Ethnic Toronto Spatial Population MoE Predictor")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--spatial_dir", type=str, default="./spatial_data",
                        help="Directory containing spatial data files")
    parser.add_argument("--global_params_path", type=str, default="./data/global_parameters.json",
                        help="Path to the global parameters JSON file for PDE expert")
    parser.add_argument("--da_coordinates_path", type=str, default="./data/toronto_da.csv",
                        help="Path to the CSV file containing DA coordinates")
    
    # Optimized model parameters - UPDATED WITH BEST HYPERPARAMETERS
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use for feature normalization")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--attention_heads", type=int, default=3,  # Updated
                        help="Number of attention heads (optimized)")
    parser.add_argument("--hidden_dim_1_base", type=int, default=28,  # Updated
                        help="Base size for first hidden dimension (optimized)")
    parser.add_argument("--hidden_dim_2_base", type=int, default=9,  # Updated
                        help="Base size for second hidden dimension (optimized)")
    parser.add_argument("--hidden_dim_3_base", type=int, default=8,  # Updated
                        help="Base size for third hidden dimension (optimized)")
    parser.add_argument("--coord_embed_base", type=int, default=7,  # Updated
                        help="Base size for coordinate embedding dimension (optimized)")
    parser.add_argument("--dropout_rate", type=float, default=0.1930287204884957,  # Updated
                        help="Dropout rate (optimized)")
    parser.add_argument("--colonization_threshold", type=float, default=0.13357705254672683,  # Updated
                        help="Threshold below which population is considered zero (optimized)")
    parser.add_argument("--colonization_expert_weight", type=float, default=0.7798480148640908,  # Updated
                        help="Base weight for colonization expert influence (optimized)")
    parser.add_argument("--colonization_weight", type=float, default=7.082495126193512,  # Updated
                        help="Weight multiplier for colonization loss (optimized)")
    parser.add_argument("--jump_threshold", type=float, default=142.22567841569256,  # Updated
                        help="Threshold for identifying potential jump cases (optimized)")
    parser.add_argument("--jump_expert_weight", type=float, default=0.7157969232653937,  # Updated
                        help="Base weight for jump expert influence (optimized)")
    parser.add_argument("--jump_weight", type=float, default=8.692760655134665,  # Updated
                        help="Weight multiplier for jump loss (optimized)")
    parser.add_argument("--decline_threshold", type=float, default=73.78197294263202,  # Updated
                        help="Threshold for identifying potential decline cases (optimized)")
    parser.add_argument("--decline_expert_weight", type=float, default=0.7192671362481913,  # Updated
                        help="Base weight for decline expert influence (optimized)")
    parser.add_argument("--decline_weight", type=float, default=6.965965704778356,  # Updated
                        help="Weight multiplier for decline loss (optimized)")
    parser.add_argument("--pde_expert_weight", type=float, default=0.680966412263434,  # Updated
                        help="Base weight for PDE expert influence (optimized)")
    parser.add_argument("--pde_weight", type=float, default=6.014537808959515,  # Updated
                        help="Weight multiplier for PDE loss cases (optimized)")
    parser.add_argument("--log_space", action="store_true",
                        help="Whether to operate in log space for population values")
    parser.add_argument("--use_ethnicity_weights", action="store_true",
                        help="Whether to use custom weights for different ethnicities")
    
    # Training parameters - UPDATED WITH BEST HYPERPARAMETERS
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0006516998803138413,  # Updated
                        help="Learning rate (optimized)")
    parser.add_argument("--weight_decay", type=float, default=3.861170208669015e-05,  # Updated
                        help="Weight decay (L2 regularization) (optimized)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for learning rate scheduler")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--save_predictions_epoch", type=int, default=30,
                        help="Epoch at which to save training predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    
    # Output parameters
    parser.add_argument("--model_path", type=str, default="./model/multi_ethnic_spatial_moe_predictor.pt",
                        help="Path to save the trained model")
    parser.add_argument("--plot", action="store_true",
                        help="Generate training plots")
    parser.add_argument("--plot_path", type=str, default="./plots/multi_ethnic_spatial_moe_training.png",
                        help="Path to save the training plots")
    parser.add_argument("--predictions_path", type=str, default="./predictions/multi_ethnic_spatial_moe_test_predictions.csv",
                        help="Path to save test predictions")
    parser.add_argument("--training_predictions_path", type=str, default="./predictions/multi_ethnic_training_predictions.csv",
                        help="Path to save training predictions at specified epoch")
    parser.add_argument("--metrics_dir", type=str, default="./metrics",
                    help="Directory to save training, validation, and test metrics as CSV files")
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
    if args.predictions_path:
        os.makedirs(os.path.dirname(args.predictions_path), exist_ok=True)
    if args.training_predictions_path:
        os.makedirs(os.path.dirname(args.training_predictions_path), exist_ok=True)
    if args.metrics_dir:
        os.makedirs(args.metrics_dir, exist_ok=True)
    
    main(args)