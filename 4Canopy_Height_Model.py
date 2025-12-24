"""
Canopy Height Prediction using Random Forest
Complete pipeline in a single file

Features:
- Data preprocessing with CHM filtering
- Feature selection based on correlation
- Random Forest training
- Hyperparameter optimization (Random/Grid Search)
- Block-wise raster prediction
- Comprehensive evaluation and reporting

Author: elainekalyn-10
"""

import os
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import joblib
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings('ignore')


# ===========================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR PROJECT
# ===========================================================================

class Config:
    """Configuration parameters for the entire pipeline"""
    
    # ==================== DATA PATHS - MODIFY THESE ====================
    RAW_DATA_PATH = "path/to/your/merged_chm_data.csv"
    OUTPUT_DIR = "path/to/your/output"
    INPUT_RASTER_DIR = "path/to/your/input/rasters"
    REFERENCE_RASTER = "path/to/your/reference/raster.tif"
    
    # Derived paths (usually don't need to change)
    MODEL_PATH = os.path.join(OUTPUT_DIR, "canopy_height_rf_model.pkl")
    FEATURE_LIST_PATH = os.path.join(OUTPUT_DIR, "model_features.txt")
    OUTPUT_RASTER = os.path.join(OUTPUT_DIR, "predicted_canopy_height.tif")
    
    # ==================== DATA PROCESSING PARAMETERS ====================
    TARGET_COLUMN = "CHM_predicted"
    CHM_MIN = 0.0
    CHM_MAX = 50.0
    TRAIN_FRACTION = 0.7
    RANDOM_SEED = 42
    N_BINS = 5
    
    # ==================== FEATURE SELECTION ====================
    EXCLUDE_COLUMNS = ['CHM_predicted', 'system:index', 'lat', 'lon', '.geo']
    N_TOP_FEATURES = 25
    
    # ==================== MODEL PARAMETERS ====================
    RF_PARAMS = {
        'n_estimators': 200,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # ==================== HYPERPARAMETER OPTIMIZATION ====================
    OPTIMIZE_HYPERPARAMETERS = True
    OPTIMIZATION_METHOD = 'random'
    
    PARAM_GRID = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    N_ITER_RANDOM_SEARCH = 20
    CV_FOLDS = 5
    
    # ==================== PREDICTION PARAMETERS ====================
    BLOCK_SIZE = 1024
    NODATA_VALUE = -9999
    OUTPUT_DTYPE = 'float32'
    OUTPUT_COMPRESSION = 'lzw'
    
    # ==================== FEATURE NAME MAPPING ====================
    FEATURE_MAPPING = {
        'your features name'
    }
    
    @classmethod
    def create_output_dir(cls):
        """Create output directory if it doesn't exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# DATA PREPROCESSING
# ===========================================================================

class DataPreprocessor:
    """Data preprocessing pipeline for canopy height data"""
    
    def __init__(self, config):
        self.config = config
        self.cleaning_report = {}
    
    def load_data(self, file_path):
        """Load raw data from CSV file"""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Comprehensive data cleaning including CHM range filtering"""
        print("\n" + "="*60)
        print("Starting data cleaning...")
        print("="*60)
        
        initial_shape = df.shape
        target_col = self.config.TARGET_COLUMN
        
        self.cleaning_report = {
            'initial_samples': initial_shape[0],
            'initial_features': initial_shape[1],
            'removed_samples': {},
            'final_samples': 0
        }
        
        print(f"Initial shape: {initial_shape}")
        print(f"Target column: {target_col}")
        print(f"CHM range: {self.config.CHM_MIN}-{self.config.CHM_MAX} meters")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Remove non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"\nRemoving non-numeric columns: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)
            self.cleaning_report['removed_samples']['non_numeric_columns'] = non_numeric_cols
        
        # Handle infinite values
        print("\nChecking for infinite values...")
        inf_mask = df.isin([np.inf, -np.inf]).any(axis=1)
        inf_count = inf_mask.sum()
        print(f"Rows with infinite values: {inf_count} ({inf_count/len(df)*100:.2f}%)")
        
        if inf_count > 0:
            df = df[~inf_mask]
            self.cleaning_report['removed_samples']['infinite_values'] = inf_count
            print(f"Shape after removing infinite values: {df.shape}")
        
        # Handle missing values
        print("\nChecking for missing values...")
        missing_rows = df.isnull().any(axis=1).sum()
        print(f"Rows with missing values: {missing_rows} ({missing_rows/len(df)*100:.2f}%)")
        
        if missing_rows > 0:
            df = df.dropna()
            self.cleaning_report['removed_samples']['missing_values'] = missing_rows
            print(f"Shape after removing missing values: {df.shape}")
        
        # CHM range filtering
        print(f"\nApplying CHM range filter ({self.config.CHM_MIN}-{self.config.CHM_MAX} meters)...")
        target_data = df[target_col]
        
        print(f"Original CHM statistics:")
        print(f"  Min: {target_data.min():.2f}m")
        print(f"  Max: {target_data.max():.2f}m")
        print(f"  Mean: {target_data.mean():.2f}m")
        print(f"  Std: {target_data.std():.2f}m")
        
        range_mask = (target_data >= self.config.CHM_MIN) & (target_data <= self.config.CHM_MAX)
        out_of_range_count = (~range_mask).sum()
        
        if out_of_range_count > 0:
            print(f"\nRemoving {out_of_range_count} samples outside CHM range")
            df = df[range_mask]
            self.cleaning_report['removed_samples']['chm_out_of_range'] = out_of_range_count
            print(f"Shape after CHM filtering: {df.shape}")
            
            target_data_filtered = df[target_col]
            print(f"\nFiltered CHM statistics:")
            print(f"  Min: {target_data_filtered.min():.2f}m")
            print(f"  Max: {target_data_filtered.max():.2f}m")
            print(f"  Mean: {target_data_filtered.mean():.2f}m")
            print(f"  Std: {target_data_filtered.std():.2f}m")
        
        # Remove zero-variance features
        print("\nChecking for zero-variance features...")
        feature_cols = [col for col in df.columns if col != target_col]
        zero_var_features = [col for col in feature_cols if df[col].var() == 0]
        
        if zero_var_features:
            print(f"Removing zero-variance features: {zero_var_features}")
            df = df.drop(columns=zero_var_features)
            self.cleaning_report['removed_samples']['zero_variance_features'] = zero_var_features
        
        # Final validation
        final_shape = df.shape
        assert df.isnull().sum().sum() == 0, "Missing values still present"
        assert not df.isin([np.inf, -np.inf]).any().any(), "Infinite values still present"
        
        final_target = df[target_col]
        assert final_target.min() >= self.config.CHM_MIN, f"CHM values below minimum"
        assert final_target.max() <= self.config.CHM_MAX, f"CHM values above maximum"
        
        print(f"\nCleaning complete! Final shape: {final_shape}")
        print(f"Samples removed: {initial_shape[0] - final_shape[0]:,}")
        print(f"Retention rate: {final_shape[0]/initial_shape[0]*100:.2f}%")
        
        self.cleaning_report['final_samples'] = final_shape[0]
        
        return df, self.cleaning_report
    
    def split_data(self, df, method='stratified'):
        """Split data into train and test sets"""
        print("\n" + "="*60)
        print(f"Splitting data using {method} method")
        print("="*60)
        
        target_col = self.config.TARGET_COLUMN
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if method == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=self.config.TRAIN_FRACTION,
                random_state=self.config.RANDOM_SEED
            )
            
        elif method == 'stratified':
            discretizer = KBinsDiscretizer(
                n_bins=self.config.N_BINS,
                encode='ordinal',
                strategy='quantile'
            )
            y_binned = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            print(f"Created {self.config.N_BINS} bins for stratification")
            
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=self.config.TRAIN_FRACTION,
                random_state=self.config.RANDOM_SEED
            )
            train_idx, test_idx = next(sss.split(X, y_binned))
            
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        stats = {
            'train_size': len(y_train),
            'test_size': len(y_test),
            'train_mean': y_train.mean(),
            'test_mean': y_test.mean(),
            'train_std': y_train.std(),
            'test_std': y_test.std(),
            'mean_diff': abs(y_train.mean() - y_test.mean()),
            'std_diff': abs(y_train.std() - y_test.std())
        }
        
        print(f"\nSplit statistics:")
        print(f"  Train size: {stats['train_size']:,}")
        print(f"  Test size: {stats['test_size']:,}")
        print(f"  Train mean: {stats['train_mean']:.3f}")
        print(f"  Test mean: {stats['test_mean']:.3f}")
        print(f"  Mean difference: {stats['mean_diff']:.4f}")
        print(f"  Std difference: {stats['std_diff']:.4f}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'stats': stats
        }
    
    def save_split_data(self, split_results, output_dir, method='stratified'):
        """Save train and test data to CSV files"""
        print(f"\nSaving split data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        train_data = pd.concat([
            split_results['X_train'],
            split_results['y_train']
        ], axis=1)
        
        test_data = pd.concat([
            split_results['X_test'],
            split_results['y_test']
        ], axis=1)
        
        train_path = os.path.join(output_dir, f"train_data_{method}.csv")
        test_path = os.path.join(output_dir, f"test_data_{method}.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"Train data saved: {train_path}")
        print(f"Test data saved: {test_path}")
        print(f"Train shape: {train_data.shape}")
        print(f"Test shape: {test_data.shape}")
        
        return train_path, test_path


# ===========================================================================
# MODEL TRAINING
# ===========================================================================

class ModelTrainer:
    """Random Forest model trainer with hyperparameter optimization"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.selected_features = []
        self.feature_importances = None
    
    def load_data(self, train_path, test_path):
        """Load training and testing data"""
        print("Loading training and testing data...")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        target_col = self.config.TARGET_COLUMN
        
        all_features = [col for col in train_data.columns 
                       if col not in self.config.EXCLUDE_COLUMNS]
        
        X_train = train_data[all_features]
        X_test = test_data[all_features]
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        
        return X_train, X_test, y_train, y_test, all_features
    
    def select_features(self, X_train, y_train, feature_names, n_top=None):
        """Select top features based on correlation with target"""
        if n_top is None:
            n_top = self.config.N_TOP_FEATURES
        
        print(f"\nSelecting top {n_top} features based on correlation...")
        
        feature_correlations = []
        for feature in feature_names:
            try:
                corr, p_val = pearsonr(X_train[feature], y_train)
                feature_correlations.append((feature, abs(corr), corr, p_val))
            except Exception:
                continue
        
        feature_correlations.sort(key=lambda x: x[1], reverse=True)
        
        self.selected_features = [item[0] for item in feature_correlations[:n_top]]
        
        print(f"\nTop 10 features by correlation:")
        for i, (feat, abs_corr, corr, p_val) in enumerate(feature_correlations[:10], 1):
            print(f"  {i:2d}. {feat:<30s} : {corr:7.4f} (p={p_val:.4e})")
        
        return self.selected_features
    
    def train_basic_model(self, X_train, y_train):
        """Train basic Random Forest model"""
        print("\nTraining basic Random Forest model...")
        
        model = RandomForestRegressor(**self.config.RF_PARAMS)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return model
    
    def optimize_hyperparameters(self, X_train, y_train, method='random'):
        """Optimize model hyperparameters"""
        print(f"\n{'='*60}")
        print(f"Hyperparameter Optimization ({method} search)")
        print(f"{'='*60}")
        
        base_model = RandomForestRegressor(
            random_state=self.config.RANDOM_SEED,
            n_jobs=self.config.RF_PARAMS.get('n_jobs', -1)
        )
        
        if method == 'random':
            print(f"Performing randomized search with {self.config.N_ITER_RANDOM_SEARCH} iterations...")
            search = RandomizedSearchCV(
                base_model,
                param_distributions=self.config.PARAM_GRID,
                n_iter=self.config.N_ITER_RANDOM_SEARCH,
                cv=self.config.CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                random_state=self.config.RANDOM_SEED,
                verbose=1
            )
        elif method == 'grid':
            print(f"Performing grid search...")
            grid_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [20, 30, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            search = GridSearchCV(
                base_model,
                param_grid=grid_params,
                cv=self.config.CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        start_time = time.time()
        search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"\nBest parameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest cross-validation R² score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        print("\nTraining Set Performance:")
        print(f"  R² Score:  {metrics['train_r2']:.4f}")
        print(f"  RMSE:      {metrics['train_rmse']:.4f} m")
        print(f"  MAE:       {metrics['train_mae']:.4f} m")
        
        print("\nTest Set Performance:")
        print(f"  R² Score:  {metrics['test_r2']:.4f}")
        print(f"  RMSE:      {metrics['test_rmse']:.4f} m")
        print(f"  MAE:       {metrics['test_mae']:.4f} m")
        
        r2_diff = metrics['train_r2'] - metrics['test_r2']
        if r2_diff > 0.1:
            print(f"\nWarning: Potential overfitting detected (R² difference: {r2_diff:.4f})")
        
        return metrics
    
    def get_feature_importances(self, model, feature_names):
        """Extract and display feature importances"""
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importances = importances
        
        print("\nTop 15 Feature Importances:")
        for idx, row in importances.head(15).iterrows():
            print(f"  {row['feature']:<30s} : {row['importance']:.4f}")
        
        return importances
    
    def save_model(self, model, feature_names):
        """Save trained model and feature list"""
        print(f"\nSaving model and features...")
        
        joblib.dump(model, self.config.MODEL_PATH)
        print(f"Model saved: {self.config.MODEL_PATH}")
        
        with open(self.config.FEATURE_LIST_PATH, 'w', encoding='utf-8') as f:
            f.write("Model Features\n")
            f.write("="*40 + "\n")
            for i, feat in enumerate(feature_names, 1):
                f.write(f"{i}. {feat}\n")
        
        print(f"Feature list saved: {self.config.FEATURE_LIST_PATH}")
    
    def train(self, train_path, test_path):
        """Complete training pipeline"""
        print("="*60)
        print("Random Forest Canopy Height Model Training")
        print("="*60)
        
        X_train, X_test, y_train, y_test, all_features = self.load_data(
            train_path, test_path
        )
        
        selected_features = self.select_features(
            X_train, y_train, all_features
        )
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        print(f"\nUsing {len(selected_features)} selected features for training")
        
        if self.config.OPTIMIZE_HYPERPARAMETERS:
            self.model = self.optimize_hyperparameters(
                X_train_selected, y_train,
                method=self.config.OPTIMIZATION_METHOD
            )
        else:
            self.model = self.train_basic_model(X_train_selected, y_train)
        
        metrics = self.evaluate_model(
            self.model, X_train_selected, X_test_selected, y_train, y_test
        )
        
        feature_importances = self.get_feature_importances(
            self.model, selected_features
        )
        
        self.save_model(self.model, selected_features)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Final Test R² Score: {metrics['test_r2']:.4f}")
        print(f"Number of features: {len(selected_features)}")
        
        return {
            'model': self.model,
            'metrics': metrics,
            'feature_importances': feature_importances,
            'selected_features': selected_features
        }


# ===========================================================================
# MODEL PREDICTION
# ===========================================================================

class CanopyHeightPredictor:
    """Random Forest predictor for canopy height estimation"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.features = []
        self.reference_profile = None
        self.feature_paths = {}
    
    def load_model_and_features(self):
        """Load trained model and feature list"""
        print("Loading model and features...")
        print("="*60)
        
        if not os.path.exists(self.config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {self.config.MODEL_PATH}")
        
        self.model = joblib.load(self.config.MODEL_PATH)
        print(f"Model loaded: {type(self.model).__name__}")
        
        if hasattr(self.model, 'n_estimators'):
            print(f"  n_estimators: {self.model.n_estimators}")
        
        if not os.path.exists(self.config.FEATURE_LIST_PATH):
            raise FileNotFoundError(f"Feature file not found: {self.config.FEATURE_LIST_PATH}")
        
        self.features = self._load_features_from_file()
        print(f"Loaded {len(self.features)} features")
        
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop 10 Feature Importances:")
            feature_importance = list(zip(self.features, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:10], 1):
                print(f"  {i:2d}. {feature:<30s} : {importance:.4f}")
    
    def _load_features_from_file(self):
        """Load feature names from file"""
        features = []
        try:
            with open(self.config.FEATURE_LIST_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Model') and not line.startswith('='):
                    if '. ' in line:
                        feature_name = line.split('. ', 1)[1].strip()
                        features.append(feature_name)
            
            if not features:
                raise ValueError("No features found in feature file")
            
            return features
        
        except Exception as e:
            raise Exception(f"Failed to load features: {e}")
    
    def load_reference_info(self):
        """Load spatial reference information"""
        print(f"\nLoading reference raster information...")
        
        if not os.path.exists(self.config.REFERENCE_RASTER):
            raise FileNotFoundError(f"Reference raster not found: {self.config.REFERENCE_RASTER}")
        
        with rasterio.open(self.config.REFERENCE_RASTER) as src:
            self.reference_profile = src.profile.copy()
            print(f"Raster dimensions: {src.width} × {src.height}")
            print(f"CRS: {src.crs}")
            print(f"Resolution: {src.res}")
    
    def find_feature_files(self):
        """Find and validate feature raster files"""
        print(f"\nSearching for feature files...")
        print("="*60)
        
        self.feature_paths = {}
        found_features = []
        missing_features = []
        
        for feature in self.features:
            found = False
            
            for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                file_path = os.path.join(self.config.INPUT_RASTER_DIR, f"{feature}{ext}")
                if os.path.exists(file_path):
                    self.feature_paths[feature] = file_path
                    found_features.append(feature)
                    found = True
                    break
            
            if not found and feature in self.config.FEATURE_MAPPING:
                for alt_name in self.config.FEATURE_MAPPING[feature]:
                    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                        alt_path = os.path.join(self.config.INPUT_RASTER_DIR, f"{alt_name}{ext}")
                        if os.path.exists(alt_path):
                            self.feature_paths[feature] = alt_path
                            found_features.append(feature)
                            print(f"✓ {feature:<30s} → {alt_name}")
                            found = True
                            break
                    if found:
                        break
            
            if found and feature not in self.config.FEATURE_MAPPING:
                print(f"✓ {feature}")
            elif not found:
                missing_features.append(feature)
                print(f"✗ {feature} - NOT FOUND")
        
        print(f"\nValidating raster dimensions...")
        size_mismatched = []
        for feature in found_features:
            try:
                with rasterio.open(self.feature_paths[feature]) as src:
                    if (src.width, src.height) != (
                        self.reference_profile['width'],
                        self.reference_profile['height']
                    ):
                        size_mismatched.append(feature)
                        print(f"⚠ {feature:<30s} : Size mismatch")
                    else:
                        print(f"✓ {feature:<30s} : ({src.width}, {src.height})")
            except Exception:
                print(f"✗ {feature:<30s} : Read error")
                size_mismatched.append(feature)
        
        available_features = [f for f in found_features if f not in size_mismatched]
        self.features = available_features
        
        print(f"\n{'='*60}")
        print(f"Feature File Summary:")
        print(f"  Files found: {len(found_features)}")
        print(f"  Size matched: {len(available_features)}")
        print(f"  Missing files: {len(missing_features)}")
        print(f"  Size mismatched: {len(size_mismatched)}")
        
        if len(available_features) == 0:
            raise FileNotFoundError("No valid feature files found!")
        
        return len(available_features)
    
    def predict_by_blocks(self, block_size=None):
        """Predict canopy height in blocks"""
        if block_size is None:
            block_size = self.config.BLOCK_SIZE
        
        print(f"\nStarting block-wise prediction (block size: {block_size}×{block_size})")
        print("="*60)
        
        height = self.reference_profile['height']
        width = self.reference_profile['width']
        
        n_blocks_y = (height + block_size - 1) // block_size
        n_blocks_x = (width + block_size - 1) // block_size
        total_blocks = n_blocks_y * n_blocks_x
        
        print(f"Total blocks to process: {total_blocks} ({n_blocks_y}×{n_blocks_x})")
        
        output_profile = self.reference_profile.copy()
        output_profile.update({
            'dtype': self.config.OUTPUT_DTYPE,
            'count': 1,
            'compress': self.config.OUTPUT_COMPRESSION,
            'nodata': self.config.NODATA_VALUE
        })
        
        os.makedirs(os.path.dirname(self.config.OUTPUT_RASTER), exist_ok=True)
        
        with rasterio.open(self.config.OUTPUT_RASTER, 'w', **output_profile) as dst:
            dst.write(
                np.full((height, width), self.config.NODATA_VALUE, dtype=np.float32),
                1
            )
            
            with tqdm(total=total_blocks, desc="Prediction Progress", unit="block") as pbar:
                successful_blocks = 0
                failed_blocks = 0
                
                for block_y in range(n_blocks_y):
                    for block_x in range(n_blocks_x):
                        y_start = block_y * block_size
                        y_end = min(y_start + block_size, height)
                        x_start = block_x * block_size
                        x_end = min(x_start + block_size, width)
                        
                        try:
                            block_data = self._read_block_features(
                                y_start, y_end, x_start, x_end
                            )
                            
                            if block_data is not None and len(block_data) > 0:
                                predictions = self._predict_block(block_data)
                                
                                window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
                                dst.write(
                                    predictions.reshape(y_end - y_start, x_end - x_start),
                                    1,
                                    window=window
                                )
                                successful_blocks += 1
                            else:
                                failed_blocks += 1
                        
                        except Exception:
                            failed_blocks += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': successful_blocks,
                            'Failed': failed_blocks
                        })
        
        print(f"\nPrediction complete! Results saved to: {self.config.OUTPUT_RASTER}")
        print(f"  Successful blocks: {successful_blocks}")
        print(f"  Failed blocks: {failed_blocks}")
    
    def _read_block_features(self, y_start, y_end, x_start, x_end):
        """Read feature data for a specific block"""
        try:
            block_features = []
            window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
            
            for feature in self.features:
                if feature in self.feature_paths:
                    with rasterio.open(self.feature_paths[feature]) as src:
                        data = src.read(1, window=window)
                        
                        data = data.astype(np.float32)
                        if src.nodata is not None:
                            data[data == src.nodata] = np.nan
                        
                        block_features.append(data.flatten())
                else:
                    shape = (y_end - y_start) * (x_end - x_start)
                    block_features.append(np.full(shape, np.nan))
            
            if not block_features:
                return None
            
            feature_array = np.column_stack(block_features)
            
            return feature_array
        
        except Exception:
            return None
    
    def _predict_block(self, block_data):
        """Predict canopy height for a block of data"""
        n_pixels = len(block_data)
        predictions = np.full(n_pixels, self.config.NODATA_VALUE, dtype=np.float32)
        
        valid_mask = ~np.isnan(block_data).any(axis=1)
        
        if valid_mask.any():
            try:
                valid_data = block_data[valid_mask]
                valid_predictions = self.model.predict(valid_data)
                
                predictions[valid_mask] = valid_predictions.astype(np.float32)
                
                predictions[predictions < 0] = 0
                predictions[predictions > 60] = 60
            
            except Exception:
                pass
        
        return predictions
    
    def generate_report(self):
        """Generate prediction statistics report"""
        print(f"\nGenerating prediction report...")
        
        try:
            with rasterio.open(self.config.OUTPUT_RASTER) as src:
                data = src.read(1)
                valid_data = data[data != self.config.NODATA_VALUE]
                
                if len(valid_data) > 0:
                    stats = {
                        'Valid pixels': f"{len(valid_data):,}",
                        'Total pixels': f"{data.size:,}",
                        'Valid ratio': f"{len(valid_data)/data.size*100:.2f}%",
                        'Min (m)': f"{valid_data.min():.2f}",
                        'Max (m)': f"{valid_data.max():.2f}",
                        'Mean (m)': f"{valid_data.mean():.2f}",
                        'Std (m)': f"{valid_data.std():.2f}",
                        'Median (m)': f"{np.median(valid_data):.2f}",
                        '25th percentile (m)': f"{np.percentile(valid_data, 25):.2f}",
                        '75th percentile (m)': f"{np.percentile(valid_data, 75):.2f}"
                    }
                    
                    print("\nCanopy Height Statistics:")
                    for key, value in stats.items():
                        print(f"  {key:<20s}: {value}")
                    
                    report_path = self.config.OUTPUT_RASTER.replace('.tif', '_report.txt')
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write("Canopy Height Prediction Report\n")
                        f.write("="*60 + "\n\n")
                        f.write(f"Model: {os.path.basename(self.config.MODEL_PATH)}\n")
                        f.write(f"Output: {os.path.basename(self.config.OUTPUT_RASTER)}\n\n")
                        f.write("Statistics:\n")
                        for key, value in stats.items():
                            f.write(f"  {key}: {value}\n")
                        f.write(f"\nFeatures used ({len(self.features)}):\n")
                        for i, feature in enumerate(self.features, 1):
                            f.write(f"  {i}. {feature}\n")
                    
                    print(f"Detailed report saved: {report_path}")
                else:
                    print("No valid prediction data")
        
        except Exception as e:
            print(f"Failed to generate report: {e}")
    
    def run_prediction(self, block_size=None):
        """Run complete prediction pipeline"""
        try:
            self.load_model_and_features()
            self.load_reference_info()
            available_count = self.find_feature_files()
            
            if available_count < len(self.features) * 0.5:
                print(f"\nWarning: Only {available_count}/{len(self.features)} features available")
                response = input("Continue with prediction? (y/n): ")
                if response.lower() != 'y':
                    print("Prediction cancelled")
                    return
            
            self.predict_by_blocks(block_size)
            self.generate_report()
            
            print("\n" + "="*60)
            print("Prediction pipeline complete!")
            print("="*60)
        
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Random Forest Canopy Height Prediction Pipeline'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['preprocess', 'train', 'predict', 'full'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable hyperparameter optimization'
    )
    
    parser.add_argument(
        '--optimization-method',
        type=str,
        default='random',
        choices=['random', 'grid'],
        help='Hyperparameter optimization method'
    )
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.optimize:
        config.OPTIMIZE_HYPERPARAMETERS = True
        config.OPTIMIZATION_METHOD = args.optimization_method
    
    config.create_output_dir()
    
    print("\n" + "="*80)
    print("RANDOM FOREST CANOPY HEIGHT PREDICTION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    
    if args.mode == 'preprocess' or args.mode == 'full':
        print("\n" + "="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80)
        
        preprocessor = DataPreprocessor(config)
        raw_data = preprocessor.load_data(config.RAW_DATA_PATH)
        cleaned_data, cleaning_report = preprocessor.clean_data(raw_data)
        split_results = preprocessor.split_data(cleaned_data, method='stratified')
        preprocessor.save_split_data(split_results, config.OUTPUT_DIR, method='stratified')
    
    if args.mode == 'train' or args.mode == 'full':
        print("\n" + "="*80)
        print("STEP 2: MODEL TRAINING")
        print("="*80)
        
        if args.mode == 'full':
            train_path = os.path.join(config.OUTPUT_DIR, "train_data_stratified.csv")
            test_path = os.path.join(config.OUTPUT_DIR, "test_data_stratified.csv")
        else:
            train_path = os.path.join(config.OUTPUT_DIR, "train_data_stratified.csv")
            test_path = os.path.join(config.OUTPUT_DIR, "test_data_stratified.csv")
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                print("Error: Preprocessed data not found. Please run preprocessing first.")
                return
        
        if config.OPTIMIZE_HYPERPARAMETERS:
            print(f"Hyperparameter optimization: ENABLED ({config.OPTIMIZATION_METHOD} search)")
        else:
            print("Hyperparameter optimization: DISABLED")
        
        trainer = ModelTrainer(config)
        results = trainer.train(train_path, test_path)
        
        print("\n" + "="*80)
        print("TRAINING RESULTS SUMMARY")
        print("="*80)
        print(f"Test R² Score: {results['metrics']['test_r2']:.4f}")
        print(f"Test RMSE: {results['metrics']['test_rmse']:.4f} m")
        print(f"Test MAE: {results['metrics']['test_mae']:.4f} m")
        print(f"Number of features: {len(results['selected_features'])}")
    
    if args.mode == 'predict' or args.mode == 'full':
        print("\n" + "="*80)
        print("STEP 3: RASTER PREDICTION")
        print("="*80)
        
        if not os.path.exists(config.MODEL_PATH):
            print("Error: Trained model not found. Please run training first.")
            return
        
        predictor = CanopyHeightPredictor(config)
        predictor.run_prediction()
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
