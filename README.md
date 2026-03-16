# 🏙️ Urban SDSS: A Hybrid Spatial Machine Learning & MCDA Framework

**Title:** A HYBRID SPATIAL MACHINE LEARNING AND MULTI-CRITERIA DECISION ANALYSIS FRAMEWORK FOR OPTIMIZING PUBLIC FACILITY PLACEMENT IN UNDERSERVED URBAN AREAS.
**Study Area:** Semarang City, Indonesia.
---

## 📖 Project Overview
This repository contains the full end-to-end architecture of a Web-Based Spatial Decision Support System (SDSS). The system is designed to eliminate human bias in urban planning by combining Remote Sensing, Machine Learning, and Multi-Criteria Decision Analysis (MCDA). It identifies the optimal placement of public facilities by simultaneously maximizing population reach and mitigating the invisible threats of Urban Heat Islands (UHI).

## 🧬 System Architecture & Pipeline

### Phase 1: Advanced Geospatial Extraction
* **Temporal Compositing:** Extracts 3-year median composites (2023-2025) of Landsat 8 satellite imagery via **Google Earth Engine (GEE)** to eliminate cloud noise.
* **S-Tier Feature Engineering:** Captures a holistic view of urban thermodynamics using multiple indices:
  * `NDBI` (Built-up Density)
  * `NDVI` (Vegetation Health)
  * `NDWI` (Moisture & Water Bodies)
  * `SAVI` (Soil-Adjusted Vegetation)
  * `DEM` (Topographic Elevation)
* **Real-World Infrastructure Routing:** Utilizes **OSMnx** to calculate actual pedestrian walking distances (Dijkstra's Algorithm) to the nearest transit hubs, discarding naive straight-line Euclidean distances.

### Phase 2: AI Engine & Topographical Stratified Sampling
* **Prevention of Extrapolation Failure:** The training data is split using Stratified Random Sampling based on Elevation (DEM) deciles. This guarantees the AI learns the thermodynamics of the entire landscape—from coastal plains to mountain peaks.
* **Multi-Scale Spatial Lags:** Injects Tobler's First Law of Geography by mapping local (`K=8`) and regional (`K=24`) neighborhood contexts using KNN weight matrices.
* **Bayesian Optimization (Optuna):** Eliminates manual guesswork. Optuna rigorously hunts for the absolute best hyperparameters across 20 iterations.
* **Champion Model:** **XGBoost Regressor**, achieving an $R^2$ of **0.8088** and an RMSE of **1.47°C**.

### Phase 3: Spatial Inference & Strict Urban Masking
* **Global Prediction:** The trained XGBoost model infers the UHI risk distribution across the entire city boundary.
* **Decoupled Architecture:** Before applying the decision matrix, the system aggressively masks out non-urban areas (forests, rivers, mountains) using ESA WorldCover data. Facility placement is strictly evaluated on built-up concrete grids.

### Phase 4: Dynamic MCDA (CRITIC-TOPSIS) & WebGIS
* **On-the-fly Calculation:** A Streamlit-powered WebGIS dashboard enables policymakers to adjust decision weights in real-time.
* **Pareto Frontier Analytics:** Automatically visualizes trade-offs between Heat Risk and Population Density, mathematically proving the superiority of the recommended sites.
* **Rendering Engine:** Utilizes Folium (Leaflet.js) to render dynamic choropleth maps overlaid with the exact municipal boundaries of Semarang City.

---

## 🚀 Installation & Execution

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is highly recommended to run this within an isolated Virtual Environment.

### 2. Setup Virtual Environment
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate