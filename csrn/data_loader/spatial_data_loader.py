import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class SpatialMatrixHandler:
    """
    Class for handling spatial matrices and coordinates
    """
    def __init__(self, data_dir):
        """
        Initialize spatial data handler
        
        Args:
            data_dir (str): Directory containing spatial data files
        """
        self.data_dir = data_dir
        self.walking_matrix = None
        self.transit_matrix = None
        self.proximity_matrix = None
        self.da_info = None
        self.dauid_to_idx = {}  # Mapping from DAUID to matrix indices
        
    def load_spatial_data(self):
        """
        Load spatial matrices and DA information
        
        Returns:
            bool: True if loading was successful
        """
        try:
            # Load spatial matrices
            walking_path = os.path.join(self.data_dir, 'walking_matrix.npy')
            transit_path = os.path.join(self.data_dir, 'transit_matrix.npy')
            proximity_path = os.path.join(self.data_dir, 'proximity_matrix.npy')
            da_info_path = os.path.join(self.data_dir, 'toronto_da.csv')
            
            if os.path.exists(walking_path):
                self.walking_matrix = np.load(walking_path)
                logger.info(f"Loaded walking matrix with shape {self.walking_matrix.shape}")
            else:
                logger.warning(f"Walking matrix not found at {walking_path}")
            
            if os.path.exists(transit_path):
                self.transit_matrix = np.load(transit_path)
                logger.info(f"Loaded transit matrix with shape {self.transit_matrix.shape}")
            else:
                logger.warning(f"Transit matrix not found at {transit_path}")
            
            if os.path.exists(proximity_path):
                self.proximity_matrix = np.load(proximity_path)
                logger.info(f"Loaded proximity matrix with shape {self.proximity_matrix.shape}")
            else:
                logger.warning(f"Proximity matrix not found at {proximity_path}")
            
            if os.path.exists(da_info_path):
                self.da_info = pd.read_csv(da_info_path)
                logger.info(f"Loaded DA info with {len(self.da_info)} entries")
                
                # Create mapping from DAUID to indices
                for idx, dauid in enumerate(self.da_info['DAUID']):
                    self.dauid_to_idx[dauid] = idx
            else:
                logger.warning(f"DA info not found at {da_info_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading spatial data: {str(e)}")
            return False
    
    def get_matrices_for_batch(self, dauid_list):
        """
        Get submatrices for a batch of DAUIDs
        
        Args:
            dauid_list (list): List of DAUIDs in the batch
            
        Returns:
            tuple: Tuple of submatrices (walking, transit, proximity)
        """
        # Get indices for the DAUIDs in the batch
        indices = []
        for dauid in dauid_list:
            idx = self.dauid_to_idx.get(dauid, -1)
            if idx != -1:
                indices.append(idx)
            else:
                # If DAUID not found in mapping, add a placeholder
                # This ensures the batch order is maintained
                indices.append(None)
        
        # Count valid indices
        valid_count = sum(1 for idx in indices if idx is not None)
        if valid_count == 0:
            logger.warning("No valid indices found for batch")
            return None, None, None
        
        # Create filters for full matrices
        batch_size = len(dauid_list)
        
        # Extract submatrices for the valid indices
        walking_submatrix = None
        transit_submatrix = None
        proximity_submatrix = None
        
        # Filter out None values for matrix extraction
        valid_indices = [idx for idx in indices if idx is not None]
        
        if self.walking_matrix is not None:
            # Extract the submatrix for valid indices
            submatrix = self.walking_matrix[valid_indices][:, valid_indices]
            # Process the matrix to be batch_size x batch_size
            walking_submatrix = np.zeros((batch_size, batch_size))
            
            # Fill in the submatrix at the correct positions
            row_idx = 0
            for i, idx_i in enumerate(indices):
                if idx_i is None:
                    continue
                col_idx = 0
                for j, idx_j in enumerate(indices):
                    if idx_j is None:
                        continue
                    # Find the indices in the submatrix
                    sub_i = valid_indices.index(idx_i)
                    sub_j = valid_indices.index(idx_j)
                    walking_submatrix[i, j] = submatrix[sub_i, sub_j]
                    col_idx += 1
                row_idx += 1
        
        if self.transit_matrix is not None:
            # Extract the submatrix for valid indices
            submatrix = self.transit_matrix[valid_indices][:, valid_indices]
            # Process the matrix to be batch_size x batch_size
            transit_submatrix = np.zeros((batch_size, batch_size))
            
            # Fill in the submatrix at the correct positions
            row_idx = 0
            for i, idx_i in enumerate(indices):
                if idx_i is None:
                    continue
                col_idx = 0
                for j, idx_j in enumerate(indices):
                    if idx_j is None:
                        continue
                    # Find the indices in the submatrix
                    sub_i = valid_indices.index(idx_i)
                    sub_j = valid_indices.index(idx_j)
                    transit_submatrix[i, j] = submatrix[sub_i, sub_j]
                    col_idx += 1
                row_idx += 1
        
        if self.proximity_matrix is not None:
            # Extract the submatrix for valid indices
            submatrix = self.proximity_matrix[valid_indices][:, valid_indices]
            # Process the matrix to be batch_size x batch_size
            proximity_submatrix = np.zeros((batch_size, batch_size))
            
            # Fill in the submatrix at the correct positions
            row_idx = 0
            for i, idx_i in enumerate(indices):
                if idx_i is None:
                    continue
                col_idx = 0
                for j, idx_j in enumerate(indices):
                    if idx_j is None:
                        continue
                    # Find the indices in the submatrix
                    sub_i = valid_indices.index(idx_i)
                    sub_j = valid_indices.index(idx_j)
                    proximity_submatrix[i, j] = submatrix[sub_i, sub_j]
                    col_idx += 1
                row_idx += 1
        
        return walking_submatrix, transit_submatrix, proximity_submatrix
    
    def get_coordinates(self, dauid_list):
        """
        Get coordinates for a batch of DAUIDs
        
        Args:
            dauid_list (list): List of DAUIDs in the batch
            
        Returns:
            np.ndarray: Array of coordinates (X, Y)
        """
        if self.da_info is None:
            logger.warning("DA info not loaded")
            return None
        
        # Filter DA info for the DAUIDs in the batch
        coordinates = []
        for dauid in dauid_list:
            coord_row = self.da_info[self.da_info['DAUID'] == dauid]
            if len(coord_row) > 0:
                x = coord_row['X_axis'].values[0]
                y = coord_row['Y_axis'].values[0]
                coordinates.append([x, y])
            else:
                # Default coordinates if not found
                coordinates.append([0, 0])
        
        return np.array(coordinates)


class MultiEthnicPopulationDataset(Dataset):
    """
    Dataset class for loading multi-ethnic population data along with features and spatial information
    """
    def __init__(self, features_df, population_df_2016, population_df_2021=None, 
                 spatial_handler=None, transform=None):
        """
        Initialize the dataset
        
        Args:
            features_df (pd.DataFrame): DataFrame containing the features
            population_df_2016 (pd.DataFrame): DataFrame containing 2016 multi-ethnic population data
            population_df_2021 (pd.DataFrame, optional): DataFrame containing 2021 multi-ethnic population data
            spatial_handler (SpatialMatrixHandler, optional): Handler for spatial data
            transform (callable, optional): Optional transform to be applied on features
        """
        # Merge features with 2016 population data on DAUID
        self.data = pd.merge(features_df, population_df_2016, on='DAUID', how='inner')
        
        # Store ethnicities columns (all except DAUID)
        self.ethnicity_columns = [col for col in population_df_2016.columns if col != 'DAUID']
        self.num_ethnicities = len(self.ethnicity_columns)
        logger.info(f"Loaded data with {self.num_ethnicities} ethnicities: {self.ethnicity_columns}")
        
        # If 2021 population data is provided (for training), merge it as well
        if population_df_2021 is not None:
            # Use suffixes to distinguish between 2016 and 2021 population columns
            self.data = pd.merge(self.data, population_df_2021, on='DAUID', 
                               how='inner', suffixes=('_2016', '_2021'))
        
        self.transform = transform
        self.spatial_handler = spatial_handler
        
        # Store DAUIDs for reference
        self.dauid = self.data['DAUID'].values
        
        # Extract features (all columns except DAUID and all population columns)
        feature_columns = [col for col in features_df.columns if col != 'DAUID']
        self.features = self.data[feature_columns].values
        
        # Extract 2016 population vectors
        self.population_2016 = self.data[[f"{col}_2016" if population_df_2021 is not None else col 
                                        for col in self.ethnicity_columns]].values
        
        # Extract 2021 population vectors if available
        if population_df_2021 is not None:
            self.population_2021 = self.data[[f"{col}_2021" for col in self.ethnicity_columns]].values
        else:
            self.population_2021 = None
        
        # Initialize coordinates
        self.coordinates = np.zeros((len(self.dauid), 2))
        
        # Get coordinates if spatial handler is provided
        if spatial_handler is not None and hasattr(spatial_handler, 'da_info') and spatial_handler.da_info is not None:
            # Get coordinates for all DAUIDs
            self.coordinates = spatial_handler.get_coordinates(self.dauid)
            logger.info(f"Loaded coordinates for {len(self.coordinates)} DAs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        pop_2016 = self.population_2016[idx]
        dauid = self.dauid[idx]
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        
        # Get coordinates
        coords = self.coordinates[idx]
        
        # If training (2021 data available), return features and target
        if self.population_2021 is not None:
            return {
                'dauid': dauid,
                'features': torch.tensor(features, dtype=torch.float32),
                'population_2016': torch.tensor(pop_2016, dtype=torch.float32),
                'coordinates': torch.tensor(coords, dtype=torch.float32),
                'target': torch.tensor(self.population_2021[idx], dtype=torch.float32)
            }
        # If inference, just return features
        else:
            return {
                'dauid': dauid,
                'features': torch.tensor(features, dtype=torch.float32),
                'population_2016': torch.tensor(pop_2016, dtype=torch.float32),
                'coordinates': torch.tensor(coords, dtype=torch.float32)
            }


class MultiEthnicDataLoader(DataLoader):
    """
    Custom DataLoader with spatial awareness for multi-ethnic population data
    """
    def __init__(self, dataset, batch_size=32, shuffle=False, **kwargs):
        """
        Initialize the DataLoader
        
        Args:
            dataset (MultiEthnicPopulationDataset): Dataset to load
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the data
            **kwargs: Additional arguments for DataLoader
        """
        super(MultiEthnicDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.spatial_handler = dataset.spatial_handler
    
    def __iter__(self):
        """
        Custom iterator that adds spatial matrices to each batch
        """
        base_iterator = super(MultiEthnicDataLoader, self).__iter__()
    
        for batch in base_iterator:
            # Extract DAUIDs from the batch
            dauid_list = batch['dauid'].tolist()
            
            # If spatial handler is available, get spatial matrices
            if self.spatial_handler is not None:
                walking, transit, proximity = self.spatial_handler.get_matrices_for_batch(dauid_list)
                
                # Add matrices to the batch if available
                if walking is not None:
                    batch['walking_matrix'] = torch.tensor(walking, dtype=torch.float32)
                if transit is not None:
                    batch['transit_matrix'] = torch.tensor(transit, dtype=torch.float32)
                if proximity is not None:
                    batch['proximity_matrix'] = torch.tensor(proximity, dtype=torch.float32)
            
            yield batch


class MultiEthnicDataHandler:
    """
    Enhanced data handler with spatial awareness for multi-ethnic population data
    """
    def __init__(self, data_dir, spatial_dir=None, scaler_type='standard'):
        """
        Initialize data handler
        
        Args:
            data_dir (str): Directory containing the data files
            spatial_dir (str, optional): Directory containing spatial data files
            scaler_type (str): Type of scaler to use ('standard' or 'minmax')
        """
        self.data_dir = data_dir
        self.spatial_dir = spatial_dir if spatial_dir is not None else data_dir
        self.scaler_type = scaler_type
        self.feature_scaler = None  # Will be fitted during preprocessing
        self.spatial_handler = None
        
        # Initialize spatial handler if spatial directory is provided
        if spatial_dir is not None:
            try:
                self.spatial_handler = SpatialMatrixHandler(spatial_dir)
                success = self.spatial_handler.load_spatial_data()
                if not success:
                    logger.warning("Spatial data loading failed, continuing without spatial features")
            except Exception as e:
                logger.warning(f"Error initializing spatial handler: {str(e)}")
                self.spatial_handler = None
        
    def load_data(self):
        """
        Load all required data files for multi-ethnic analysis
        
        Returns:
            dict: Dictionary containing DataFrames for each dataset
        """
        logger.info("Loading multi-ethnic data files...")
        
        # Define file paths for multi-ethnic data (using _all suffix)
        files = {
            'train_features': os.path.join(self.data_dir, '2016_train_features_all.csv'),
            'train_pop_2016': os.path.join(self.data_dir, '2016_train_population_all.csv'),
            'train_pop_2021': os.path.join(self.data_dir, '2021_train_population_all.csv'),
            'val_features': os.path.join(self.data_dir, '2016_val_features_all.csv'),
            'val_pop_2016': os.path.join(self.data_dir, '2016_val_population_all.csv'),
            'val_pop_2021': os.path.join(self.data_dir, '2021_val_population_all.csv'),
            'test_features': os.path.join(self.data_dir, '2016_test_features_all.csv'),
            'test_pop_2016': os.path.join(self.data_dir, '2016_test_population_all.csv'),
            'test_pop_2021': os.path.join(self.data_dir, '2021_test_population_all.csv')
        }
        
        # Load each file into a DataFrame
        data = {}
        for key, file_path in files.items():
            if os.path.exists(file_path):
                data[key] = pd.read_csv(file_path)
                logger.info(f"Loaded {key}: {len(data[key])} rows")
                
                # Log the ethnicity columns for population data
                if 'pop' in key:
                    ethnicity_cols = [col for col in data[key].columns if col != 'DAUID']
                    logger.info(f"  - Contains {len(ethnicity_cols)} ethnicities: {ethnicity_cols}")
            else:
                logger.warning(f"File not found: {file_path}")
                if 'test_pop_2021' in key:  # This might be intentionally missing for final testing
                    logger.info(f"Note: {key} is optional for inference.")
        
        return data
    
    def preprocess_features(self, train_features, val_features=None, test_features=None):
        """
        Preprocess features by scaling them
        
        Args:
            train_features (pd.DataFrame): Training features DataFrame
            val_features (pd.DataFrame, optional): Validation features DataFrame
            test_features (pd.DataFrame, optional): Test features DataFrame
            
        Returns:
            tuple: Processed feature DataFrames (train, val, test)
        """
        logger.info("Preprocessing features...")
        
        # Create a copy of the DataFrames to avoid modifying the originals
        train_df = train_features.copy()
        val_df = val_features.copy() if val_features is not None else None
        test_df = test_features.copy() if test_features is not None else None
        
        # Extract DAUIDs to add them back later
        train_dauid = train_df['DAUID']
        val_dauid = val_df['DAUID'] if val_df is not None else None
        test_dauid = test_df['DAUID'] if test_df is not None else None
        
        # Get features only (exclude DAUID column)
        train_features_only = train_df.drop(columns=['DAUID'])
        val_features_only = val_df.drop(columns=['DAUID']) if val_df is not None else None
        test_features_only = test_df.drop(columns=['DAUID']) if test_df is not None else None
        
        # Apply scaling
        if self.scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
        else:  # minmax scaler
            self.feature_scaler = MinMaxScaler()
        
        # Fit scaler on training data and transform
        train_features_scaled = self.feature_scaler.fit_transform(train_features_only)
        
        # Transform validation and test data if available
        val_features_scaled = self.feature_scaler.transform(val_features_only) if val_features_only is not None else None
        test_features_scaled = self.feature_scaler.transform(test_features_only) if test_features_only is not None else None
        
        # Convert back to DataFrames with DAUID
        train_df_scaled = pd.DataFrame(train_features_scaled, columns=train_features_only.columns)
        train_df_scaled['DAUID'] = train_dauid.values
        
        if val_features_scaled is not None:
            val_df_scaled = pd.DataFrame(val_features_scaled, columns=val_features_only.columns)
            val_df_scaled['DAUID'] = val_dauid.values
        else:
            val_df_scaled = None
            
        if test_features_scaled is not None:
            test_df_scaled = pd.DataFrame(test_features_scaled, columns=test_features_only.columns)
            test_df_scaled['DAUID'] = test_dauid.values
        else:
            test_df_scaled = None
        
        logger.info("Feature preprocessing complete.")
        return train_df_scaled, val_df_scaled, test_df_scaled
    
    def create_dataloaders(self, data, batch_size=32):
        """
        Create spatially-aware DataLoaders for multi-ethnic training, validation, and testing
        
        Args:
            data (dict): Dictionary containing data DataFrames
            batch_size (int): Batch size for DataLoaders
            
        Returns:
            tuple: DataLoaders for training, validation, and testing
        """
        logger.info("Creating multi-ethnic spatially-aware DataLoaders...")
        
        # Preprocess features
        train_features_scaled, val_features_scaled, test_features_scaled = self.preprocess_features(
            data['train_features'], data.get('val_features'), data.get('test_features')
        )
        
        # Create datasets with spatial awareness
        train_dataset = MultiEthnicPopulationDataset(
            train_features_scaled,
            data['train_pop_2016'],
            data['train_pop_2021'],
            spatial_handler=self.spatial_handler
        )
        
        val_dataset = None
        if 'val_features' in data and 'val_pop_2016' in data and 'val_pop_2021' in data:
            val_dataset = MultiEthnicPopulationDataset(
                val_features_scaled,
                data['val_pop_2016'],
                data['val_pop_2021'],
                spatial_handler=self.spatial_handler
            )
        
        test_dataset = None
        if 'test_features' in data and 'test_pop_2016' in data:
            # For test dataset, we may or may not have 2021 population data
            test_pop_2021 = data.get('test_pop_2021')
            test_dataset = MultiEthnicPopulationDataset(
                test_features_scaled,
                data['test_pop_2016'],
                test_pop_2021,
                spatial_handler=self.spatial_handler
            )
        
        # Create spatially-aware DataLoaders
        train_loader = MultiEthnicDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = MultiEthnicDataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        test_loader = MultiEthnicDataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
        
        logger.info(f"Created DataLoaders: Train size={len(train_loader.dataset)}, " + 
                   f"Val size={len(val_loader.dataset) if val_loader else 0}, " +
                   f"Test size={len(test_loader.dataset) if test_loader else 0}")
        
        # Store number of ethnicities for model configuration
        self.num_ethnicities = train_dataset.num_ethnicities
        logger.info(f"Number of ethnicities: {self.num_ethnicities}")
        
        return train_loader, val_loader, test_loader