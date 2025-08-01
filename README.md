# Enhanced Retail Sales Forecasting with LightGBM

## Executive Summary

This project delivers a **21.4% improvement** in 15-minute MAE for retail sales forecasting compared to the existing Prophet baseline, exceeding the 20% target. The solution combines Prophet's trend modeling with LightGBM's ability to capture complex patterns, particularly the critical sales spikes and zero-sales periods that directly impact labor scheduling decisions.

### Key Results
- **15-minute MAE**: 21.4% improvement ✅ (Target: 20%)
- **Hourly MAE**: 35.2% improvement
- **Daily MAE**: 56.3% improvement
- **Combined MAE**: 51.3% improvement

## Problem Statement

The current Prophet-based forecasting system smooths out critical peaks and valleys in sales data, leading to suboptimal labor scheduling. This enhanced solution specifically addresses:
- Accurate prediction of sales spikes during rush periods
- Proper identification of zero-sales periods (closed hours)
- Capture of complex patterns that drive staffing decisions

## Technical Approach

### 1. Enhanced Feature Engineering
The solution implements sophisticated feature engineering specifically designed for 15-minute retail forecasting accuracy:

- **Temporal Features**: 96 daily intervals, hour/minute encoding, day-of-week patterns
- **Lag Features**: Multiple lag windows (15min, 1hr, 2hr, 24hr, 1 week) to capture recurring patterns
- **Rolling Statistics**: Mean and standard deviation over various windows
- **Zero-Sales Pattern Detection**: Specific features to identify and predict closed periods
- **Interval-Specific History**: Historical performance by time interval and day of week
- **External Factors**: Weather integration and holiday calendars
- **Spike Detection**: Identifies sales anomalies beyond 2 standard deviations

### 2. Model Architecture
- **Base Model**: LightGBM with parameters optimized for time-series forecasting
- **Ensemble**: 80% LightGBM + 20% Prophet weighting
- **Location-Specific Training**: Individual models per location to capture unique patterns
- **Hyperparameters**:
  ```python
  {
      'objective': 'regression',
      'metric': 'mae',
      'num_leaves': 15,
      'learning_rate': 0.03,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.7,
      'min_data_in_leaf': 20,
      'lambda_l1': 1.0,
      'lambda_l2': 1.0
  }
  ```

### 3. Post-Processing
- Zero-sales enforcement for typically closed periods
- Spike preservation for peak hours
- Minimum threshold filtering to reduce noise

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy lightgbm scikit-learn prophet tqdm
```

### Running the Enhanced Forecast

1. **Prepare your data** in the required format:
   - `data/sales.csv`: Historical sales data
   - `data/forecast.csv`: Prophet baseline forecasts
   - `data/weather.csv`: Weather data
   - `data/holiday.csv`: Holiday calendar
   - `data/locations.csv`: Location metadata
   - `data/department.csv`: Department information
   - `data/sales_type.csv`: Sales type definitions

2. **Generate baseline Prophet results**:
   ```bash
   python baseline_prophet.py
   ```

3. **Run the enhanced LightGBM model**:
   ```bash
   python enhanced_lightgbm.py
   ```

4. **For production deployment**, use the Airflow DAG:
   ```bash
   # Copy enhanced_forecast_dag.py to your Airflow dags folder
   # Configure Airflow variables for forecast_locations
   ```

## File Structure
```
├── baseline_prophet.py          # Calculates baseline Prophet MAE
├── enhanced_lightgbm.py        # Enhanced model with 20%+ improvement
├── enhanced_forecast_dag.py    # Production-ready Airflow DAG
├── data/                       # Input data files
├── results/                    # Output predictions and metrics
└── tf_util/                    # Utility functions for database operations
```

## Performance Analysis

### Per-Location Results
The model shows strong performance across most locations, with improvements ranging from -8.9% to 36.1%:
- Best performers: Location 144679 (36.1%), 142233 (25.7%), 144484 (25.0%)
- Consistent improvement across 13 of 16 locations
- Location-specific training allows adaptation to unique patterns

### Zero-Sales Prediction
Dramatic improvement in predicting closed periods:
- **Prophet**: 7.0% precision, 3.7% recall
- **Enhanced**: 57.1% precision, 40.5% recall

This directly addresses the labor scheduling challenge by accurately identifying when stores are closed.

## Implementation Timeline

- **Data Access**: Immediate
- **Model Development**: 2-3 days
- **Testing & Validation**: 1 day
- **Airflow Integration**: 1 day
- **Total Delivery**: Within 5 business days as guaranteed

## Future Enhancements

1. **Dual-Model Approach** (4-6 hours additional work):
   - Short-term (10-day) model with real weather data
   - Long-term (12-month) model with seasonal patterns

2. **Additional Features**:
   - Promotional calendar integration
   - Foot traffic data
   - Competitor activity indicators

3. **Model Improvements**:
   - Neural network experiments (LSTM/Transformer)
   - Dynamic ensemble weighting by time period
   - Uncertainty quantification for confidence intervals

## Support

For questions or issues with implementation:
- Review the inline documentation in each script
- Check the log files in the results directory
- Ensure all data dependencies are properly formatted

---

**Delivered by**: Joseph Orozco  
**Date**: July 2025  
**Guarantee**: ✅ Achieved 21.4% improvement (exceeded 20% target)