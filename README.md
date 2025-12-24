# A 10m High-precision Canopy Height Product for Nanping City, Fujian Province, China
**The Pipeline of Generating Canopy Height Product**

This warehouse offers a modular processing flow for preprocessing GEDI L2A (Global Ecosystem Dynamics Survey) liDAR data and unmanned aerial vehicle (UAV) images, performing bias calibration on GEDI L2A, and generating high-precision canopy height products with a resolution of 10 meters. This workflow integrates Google Earth Engine (GEE), RStudio and Pycharm tools, aiming to prepare high-quality remote sensing data for ecological analysis, biomass estimation and vegetation structure modeling.

---

## 🔹 1. GEDI L2A Data Preprocessing (`1GEDI_Preprocess.js`)

This script processes GEDI L2A canopy height data from GEE's GEDI catalog. Key operations include:

* Filtering GEDI shots by quality flags and beam sensitivity
* Extracting canopy height metrics (RH98)
* Spatial filtering by region of interest (ROI)
* Temporal aggregation and outlier removal
* Exporting processed GEDI points to Google Drive

**Note:** GEDI data requires careful quality filtering due to variations in beam sensitivity, terrain complexity, and atmospheric conditions. This script implements recommended filtering protocols to ensure data reliability.

---

## 🔹 2. UAV Data Preprocessing (`2UAV_Preprocess.js`)

This step handles UAV-derived imagery and products, including:

* Digital Surface Model (DSM) and Digital Terrain Model (DTM) processing
* Canopy Height Model (CHM) generation from DSM-DTM differencing
---

## 🔹 3. Bias Calibration Model (`3Bias_Calibration_Model.js`)

A complete pipeline for calibrating GEDI data using Random Forest, including data splitting, model training, and prediction with validation.


### 1. Data Splitting
* Random sampling (70/15/15 split)
* Stratified sampling (70/15/15 split)
* Automatic dataset saving

### 2. Model Training
* Random Forest with parameter optimization
* Automatic feature selection
* Model evaluation and metadata storage

### 3. Prediction & Validation
* Terrain feature extraction from raster files
* Batch prediction
* Coordinate-based validation matching
* Performance evaluation

---
## 🔹 4. Canopy Height Model (`4Canopy_Height_Model.js`)

A complete pipeline for generating canopy height product.


### 1.  Data Preprocessing
- Removes missing values and infinite values
- Filters CHM values to specified range (0-50m default)
- Stratified sampling for balanced train/test split
- Removes zero-variance features

### 2. Feature Selection
- Correlation-based selection
- Selects top N features (default: 25)
- Ranks features by importance

### 3. Model Training
- Random Forest Regressor
- Hyperparameter optimization:
  - Random Search (faster, 20 iterations)
  - Grid Search (thorough, exhaustive)
- Cross-validation (5 folds)
- Comprehensive evaluation (R², RMSE, MAE)
### 4. Raster Prediction
- Block-wise processing (memory efficient)
- Automatic feature file discovery
- Handles NoData values
- Progress bar tracking

---
## 🔁 Extension

This preprocessing framework can be easily adapted and extended for:

**Temporal Extension:**
* Different time periods

**Spatial Extension:**
* Different geographic regions or study areas

**Data Source Extension:**
* Fusion with satellite imagery 

**Metric Customization:**
* Custom quality filters and threshold criteria

Simply modify the filtering criteria, spatial/temporal windows, and output parameters to suit your specific research objectives.

---

## 📁 Folder Structure
```
├── 1GEDI_Preprocess.js             # GEDI L2A data filtering and processing
├── 2UAV_Preprocess.js              # UAV imagery and CHM preprocessing
├── 3Bias_Calibration_Model.js      # GEDI L2A data filtering and processing
├── 4Canopy_Height_Model.js         # UAV imagery and CHM preprocessing
└── README.md                       # Project documentation
```

---

## 📌 Requirements

* **Google Earth Engine** account with access to Asset storage and Google Drive
* **R**  environment with raster/spatial packages (for CHM)
* **Python** Python environment (version 3.8 or above is recommended) and related scientific computing packages (such as numpy, pandas, scikit-learn, etc.)

---

## 📝 Citation
* If you use this code or pipeline in your research, please cite appropriately or reference this repository.


