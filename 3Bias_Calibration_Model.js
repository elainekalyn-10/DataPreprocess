import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import rasterio
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
# DATA SPLITTING FUNCTIONS
# =====================================================================

def create_stratified_bins(y, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(y.dropna(), percentiles)
    bins[0] = y.min() - 0.001
    bins[-1] = y.max() + 0.001
    bin_labels = pd.cut(y, bins=bins, labels=False, include_lowest=True)
    return bin_labels


def random_sampling_split(df, target_col='CHM', train_size=0.7, random_state=42):
    df_clean = df.dropna(subset=[target_col]).copy()
    train, temp = train_test_split(df_clean, train_size=train_size, random_state=random_state)
    test, validation = train_test_split(temp, test_size=0.5, random_state=random_state)
    return train, test, validation


def stratified_sampling_split(df, target_col='CHM', train_size=0.7, random_state=42, n_bins=5):
    df_clean = df.dropna(subset=[target_col]).copy()
    stratify_labels = create_stratified_bins(df_clean[target_col], n_bins=n_bins)
    df_clean['stratify_bin'] = stratify_labels
    
    train, temp = train_test_split(
        df_clean,
        train_size=train_size,
        stratify=df_clean['stratify_bin'],
        random_state=random_state
    )
    
    test, validation = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp['stratify_bin'],
        random_state=random_state
    )
    
    train = train.drop('stratify_bin', axis=1)
    test = test.drop('stratify_bin', axis=1)
    validation = validation.drop('stratify_bin', axis=1)
    
    return train, test, validation


def save_datasets(train, test, validation, output_dir='./', prefix='dataset'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    train.to_csv(output_path / f'{prefix}_train.csv', index=False)
    test.to_csv(output_path / f'{prefix}_test.csv', index=False)
    validation.to_csv(output_path / f'{prefix}_validation.csv', index=False)


def split_data(file_path, target_col='CHM', method='stratified', output_dir='./', prefix='dataset'):
    df = pd.read_csv(file_path)
    
    if method == 'random':
        train, test, validation = random_sampling_split(df, target_col)
    elif method == 'stratified':
        train, test, validation = stratified_sampling_split(df, target_col)
    else:
        raise ValueError("method must be 'random' or 'stratified'")
    
    save_datasets(train, test, validation, output_dir, prefix)
    
    return train, test, validation


# =====================================================================
# MODEL TRAINING FUNCTIONS
# =====================================================================

def load_datasets(train_path, test_path, validation_path=None):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    validation_df = None
    if validation_path:
        try:
            validation_df = pd.read_csv(validation_path)
        except:
            pass
    return train_df, test_df, validation_df


def prepare_features_target(df, target_col='CHM', feature_cols=None):
    if feature_cols is None:
        feature_cols = ['height', 'dem', 'slope', 'aspect', 'hillshade']
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    if len(available_features) == 0:
        raise ValueError("No feature columns available")
    
    X = df[available_features]
    y = df[target_col]
    
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    return X, y, available_features


def build_random_forest_model(X_train, y_train, X_test, y_test):
    param_combinations = [
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1}
    ]
    
    best_model = None
    best_score = -np.inf
    best_params = None
    
    for params in param_combinations:
        rf = RandomForestRegressor(random_state=42, **params)
        rf.fit(X_train, y_train)
        
        y_test_pred = rf.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        
        if test_r2 > best_score:
            best_score = test_r2
            best_model = rf
            best_params = params
    
    return best_model, best_params


def evaluate_model(model, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    eval_results = {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'predictions': {
            'train': y_train_pred,
            'test': y_test_pred
        }
    }
    
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        eval_results.update({
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'predictions': {**eval_results['predictions'], 'validation': y_val_pred}
        })
    
    return eval_results


def save_model_and_metadata(model, feature_names, best_params, eval_results, output_dir='./models/'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = f'rf_model_{timestamp}.pkl'
    model_path = output_path / model_filename
    joblib.dump(model, model_path)
    
    performance_metrics = {
        'train_r2': float(eval_results['train_r2']),
        'train_rmse': float(eval_results['train_rmse']),
        'train_mae': float(eval_results['train_mae']),
        'test_r2': float(eval_results['test_r2']),
        'test_rmse': float(eval_results['test_rmse']),
        'test_mae': float(eval_results['test_mae'])
    }
    
    if 'val_r2' in eval_results:
        performance_metrics.update({
            'val_r2': float(eval_results['val_r2']),
            'val_rmse': float(eval_results['val_rmse']),
            'val_mae': float(eval_results['val_mae'])
        })
    
    metadata = {
        'model_info': {
            'model_type': 'RandomForestRegressor',
            'timestamp': timestamp,
            'feature_names': feature_names,
            'target_variable': 'CHM',
            'best_parameters': best_params
        },
        'performance': performance_metrics,
        'data_info': {
            'train_samples': len(eval_results['predictions']['train']),
            'test_samples': len(eval_results['predictions']['test']),
            'total_features': len(feature_names)
        }
    }
    
    if 'val_r2' in eval_results:
        metadata['data_info']['validation_samples'] = len(eval_results['predictions']['validation'])
    
    metadata_filename = f'model_metadata_{timestamp}.json'
    metadata_path = output_path / metadata_filename
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return {
        'model_path': str(model_path),
        'metadata_path': str(metadata_path)
    }


def train_model(train_path, test_path, validation_path=None, target_col='CHM', output_dir='./models/'):
    train_df, test_df, validation_df = load_datasets(train_path, test_path, validation_path)
    
    X_train, y_train, feature_names = prepare_features_target(train_df, target_col)
    X_test, y_test, _ = prepare_features_target(test_df, target_col, feature_cols=feature_names)
    
    X_val, y_val = None, None
    if validation_df is not None:
        X_val, y_val, _ = prepare_features_target(validation_df, target_col, feature_cols=feature_names)
    
    model, best_params = build_random_forest_model(X_train, y_train, X_test, y_test)
    
    eval_results = evaluate_model(model, X_train, y_train, X_test, y_test, X_val, y_val)
    
    saved_files = save_model_and_metadata(model, feature_names, best_params, eval_results, output_dir)
    
    return model, eval_results, saved_files


# =====================================================================
# PREDICTION AND VALIDATION FUNCTIONS
# =====================================================================

def load_model_and_metadata(model_path):
    model = joblib.load(model_path)
    
    model_dir = Path(model_path).parent
    timestamp = Path(model_path).stem.split('_')[-1]
    metadata_path = model_dir / f'model_metadata_{timestamp}.json'
    
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return model, metadata


def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    
    required_cols = ['height', 'lat', 'lon']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def extract_raster_values(df, raster_files):
    result_df = df.copy()
    
    for feature_name, raster_path in raster_files.items():
        with rasterio.open(raster_path) as src:
            coords = list(zip(df['lon'], df['lat']))
            values = list(src.sample(coords))
            feature_values = [val[0] if val.size > 0 else np.nan for val in values]
            result_df[feature_name] = feature_values
    
    return result_df


def prepare_prediction_data(df, feature_names):
    missing_features = [feat for feat in feature_names if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    X = df[feature_names].copy()
    valid_mask = ~X.isnull().any(axis=1)
    X_clean = X[valid_mask]
    valid_indices = df.index[valid_mask]
    
    if len(X_clean) == 0:
        raise ValueError("No valid samples for prediction")
    
    return X_clean, valid_indices


def make_predictions(model, X_features):
    predictions = model.predict(X_features)
    return predictions


def match_prediction_with_validation(pred_df, val_df, coordinate_tolerance=0.001):
    pred_valid = pred_df.dropna(subset=['CHM_predicted']).copy()
    matched_data = []
    
    for idx, pred_row in tqdm(pred_valid.iterrows(), total=len(pred_valid), desc="Matching"):
        pred_lat, pred_lon = pred_row['lat'], pred_row['lon']
        distances = np.sqrt((val_df['lat'] - pred_lat) ** 2 + (val_df['lon'] - pred_lon) ** 2)
        min_distance = distances.min()
        
        if min_distance <= coordinate_tolerance:
            closest_idx = distances.idxmin()
            val_row = val_df.loc[closest_idx]
            
            matched_data.append({
                'pred_idx': idx,
                'val_idx': closest_idx,
                'lat': pred_lat,
                'lon': pred_lon,
                'height': pred_row['height'],
                'CHM_predicted': pred_row['CHM_predicted'],
                'CHM_true': val_row['CHM'],
                'distance': min_distance,
                **{col: pred_row[col] for col in ['dem', 'slope', 'aspect', 'hillshade'] if col in pred_row}
            })
    
    if not matched_data:
        return None
    
    matched_df = pd.DataFrame(matched_data)
    return matched_df


def evaluate_predictions(matched_df):
    y_true = matched_df['CHM_true']
    y_pred = matched_df['CHM_predicted']
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    residuals = y_true - y_pred
    bias = residuals.mean()
    std_residuals = residuals.std()
    correlation = y_true.corr(y_pred)
    
    evaluation_results = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'std_residuals': std_residuals,
        'correlation': correlation,
        'n_samples': len(matched_df)
    }
    
    return evaluation_results, matched_df


def save_results(result_df, evaluation_results=None, matched_df=None, output_path=None):
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'predictions_{timestamp}.csv'
    
    result_df.to_csv(output_path, index=False)
    
    if evaluation_results:
        report_path = output_path.replace('.csv', '_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2)
    
    if matched_df is not None:
        validation_path = output_path.replace('.csv', '_validation_matched.csv')
        matched_df.to_csv(validation_path, index=False)
    
    return output_path


def predict_with_validation(csv_path, model_path, raster_files, validation_path=None, 
                           output_path=None, coordinate_tolerance=0.001):
    model, metadata = load_model_and_metadata(model_path)
    
    if metadata:
        feature_names = metadata['model_info']['feature_names']
    else:
        feature_names = ['height', 'dem', 'slope', 'aspect', 'hillshade']
    
    df = load_csv_data(csv_path)
    df_with_features = extract_raster_values(df, raster_files)
    
    X_features, valid_indices = prepare_prediction_data(df_with_features, feature_names)
    predictions = make_predictions(model, X_features)
    
    result_df = df_with_features.copy()
    result_df['CHM_predicted'] = np.nan
    result_df.loc[valid_indices, 'CHM_predicted'] = predictions
    
    evaluation_results = None
    matched_df = None
    
    if validation_path:
        val_df = pd.read_csv(validation_path)
        matched_df = match_prediction_with_validation(result_df, val_df, coordinate_tolerance)
        
        if matched_df is not None:
            evaluation_results, matched_df = evaluate_predictions(matched_df)
    
    output_file = save_results(result_df, evaluation_results, matched_df, output_path)
    
    return result_df, evaluation_results, output_file


# =====================================================================
# MAIN EXECUTION EXAMPLES
# =====================================================================

if __name__ == "__main__":
    
    # Example 1: Split dataset
    print("Example 1: Data Splitting")
    print("-" * 50)
    # train, test, validation = split_data(
    #     file_path='your_data.csv',
    #     target_col='CHM',
    #     method='stratified',
    #     output_dir='./',
    #     prefix='dataset'
    # )
    
    # Example 2: Train model
    print("\nExample 2: Model Training")
    print("-" * 50)
    # model, results, files = train_model(
    #     train_path='dataset_train.csv',
    #     test_path='dataset_test.csv',
    #     validation_path='dataset_validation.csv',
    #     target_col='CHM',
    #     output_dir='./models/'
    # )
    
    # Example 3: Predict with validation
    print("\nExample 3: Prediction with Validation")
    print("-" * 50)
    # raster_files = {
    #     'aspect': 'path/to/aspect.tif',
    #     'hillshade': 'path/to/hillshade.tif',
    #     'slope': 'path/to/slope.tif',
    #     'dem': 'path/to/dem.tif'
    # }
    # 
    # result_df, eval_results, output_file = predict_with_validation(
    #     csv_path='input_data.csv',
    #     model_path='models/rf_model.pkl',
    #     raster_files=raster_files,
    #     validation_path='validation_set.csv',
    #     output_path='predictions.csv',
    #     coordinate_tolerance=0.001
    # )
