# DataPreprocess
**GEDI L2A and UAV Data Preprocessing Pipeline**

This repository provides a modular pipeline for preprocessing GEDI L2A (Global Ecosystem Dynamics Investigation) LiDAR data and UAV (Unmanned Aerial Vehicle) imagery using Google Earth Engine (GEE) and RStudio. The workflow is designed to prepare high-quality remote sensing data for ecological analysis, biomass estimation, and vegetation structure modeling.

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

A complete pipeline for GEDI canopy height modeling using Random Forest, including data splitting, model training, and prediction with validation.


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
├── 1GEDI_Preprocess.js      # GEDI L2A data filtering and processing
├── 2UAV_Preprocess.js       # UAV imagery and CHM preprocessing
└── README.md                # Project documentation
```

---

## 📌 Requirements

* **Google Earth Engine** account with access to Asset storage and Google Drive
* **R**  environment with raster/spatial packages (for CHM)

---

## 📝 Citation
* If you use this code or pipeline in your research, please cite appropriately or reference this repository.


