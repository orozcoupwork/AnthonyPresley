# lightgbm_forecast.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import os
import json

# Load all data files
def load_data():
    data_path = 'data/'
    
    # Load main tables
    sales_df = pd.read_csv(f'{data_path}sales.csv')
    forecast_df = pd.read_csv(f'{data_path}forecast.csv') 
    weather_df = pd.read_csv(f'{data_path}weather.csv', dtype={'postal_code': str})
    holiday_df = pd.read_csv(f'{data_path}holiday.csv')
    location_df = pd.read_csv(f'{data_path}locations.csv')
    department_df = pd.read_csv(f'{data_path}department.csv')
    sales_type_df = pd.read_csv(f'{data_path}sales_type.csv')
    
    # Convert datetime columns
    sales_df['ds'] = pd.to_datetime(sales_df['ds'])
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])
    holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
    
    # Rename forecast column from 'y' to 'yhat'
    forecast_df = forecast_df.rename(columns={'y': 'yhat'})
    
    return {
        'sales': sales_df,
        'forecast': forecast_df,
        'weather': weather_df,
        'holiday': holiday_df,
        'location': location_df,
        'department': department_df,
        'sales_type': sales_type_df
    }

def create_features(df, weather_df, holiday_df, location_df):
    """Create features for LightGBM"""
    df = df.copy()
    
    # Time features
    df['hour'] = df['ds'].dt.hour
    df['day'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['weekofyear'] = df['ds'].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Lag features
    for lag in [1, 7, 14, 21, 28]:
        df[f'lag_{lag}d'] = df.groupby(['location_id', 'sales_type_id', 'department_id'])['y'].shift(lag * 96)
    
    # Rolling features
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}d'] = df.groupby(['location_id', 'sales_type_id', 'department_id'])['y'].transform(
            lambda x: x.shift(96).rolling(window * 96, min_periods=1).mean()
        )
        df[f'rolling_std_{window}d'] = df.groupby(['location_id', 'sales_type_id', 'department_id'])['y'].transform(
            lambda x: x.shift(96).rolling(window * 96, min_periods=1).std()
        )
    
    # Merge weather features
    # First, get postal codes for locations
    location_postal = location_df[['id', 'postal_code']].rename(columns={'id': 'location_id'})
    df = df.merge(location_postal, on='location_id', how='left')
    
    # Aggregate weather to hourly level
    weather_hourly = weather_df.copy()
    weather_hourly['hour'] = weather_hourly['ds'].dt.floor('h')
    weather_hourly = weather_hourly.groupby(['postal_code', 'hour']).agg({
        'real_feel': 'mean',
        'precipitation': 'sum',
        'coverage': 'mean',
        'snow': 'sum'
    }).reset_index()
    weather_hourly.rename(columns={'hour': 'ds'}, inplace=True)
    
    # Create hour column for merging
    df['hour_merge'] = df['ds'].dt.floor('h')
    df = df.merge(weather_hourly, left_on=['postal_code', 'hour_merge'], right_on=['postal_code', 'ds'], 
                  how='left', suffixes=('', '_weather'))
    df.drop(['ds_weather', 'hour_merge'], axis=1, inplace=True)
    
    # Holiday features
    df['date'] = df['ds'].dt.date
    holiday_df['date'] = pd.to_datetime(holiday_df['ds']).dt.date
    
    # Get corporation_id for each location
    location_corp = location_df[['id', 'corporation_id']].rename(columns={'id': 'location_id'})
    df = df.merge(location_corp, on='location_id', how='left')
    
    # Create holiday indicator
    holiday_dates = holiday_df[['corporation_id', 'date']].drop_duplicates()
    holiday_dates['is_holiday'] = 1
    df = df.merge(holiday_dates, on=['corporation_id', 'date'], how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0)
    
    return df

def train_lightgbm_model(train_df, val_df, features):
    """Train LightGBM model"""
    
    # Prepare data
    X_train = train_df[features]
    y_train = train_df['y']
    X_val = val_df[features]
    y_val = val_df['y']
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': -1,
        'seed': 42
    }
    
    # Train
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    return model

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load data
print("Loading data...")
data = load_data()
print("Data loaded successfully!\n")

# Prepare data for all locations
all_predictions = []
location_count = 0
total_locations = len(data['location']['id'].unique())

for location_id in data['location']['id'].unique():
    location_count += 1
    print(f"\nProcessing location {location_id} ({location_count}/{total_locations})")
    
    # Filter data for this location
    location_sales = data['sales'][data['sales']['location_id'] == location_id].copy()
    
    if location_sales.empty:
        print(f"  No sales data for location {location_id}, skipping...")
        continue
    
    # Get postal code for this location
    postal_code = data['location'][data['location']['id'] == location_id]['postal_code'].values
    if len(postal_code) == 0:
        print(f"  No postal code found for location {location_id}, skipping...")
        continue
    
    postal_code = postal_code[0]
    location_weather = data['weather'][data['weather']['postal_code'] == postal_code].copy()
    
    if location_weather.empty:
        print(f"  No weather data for postal code {postal_code}, skipping...")
        continue
    
    # Create features
    print(f"  Creating features...")
    location_sales = create_features(location_sales, location_weather, data['holiday'], data['location'])
    
    # Remove NaN values from lag features
    location_sales = location_sales.dropna(subset=[col for col in location_sales.columns if 'lag_' in col])
    
    if len(location_sales) < 1000:  # Need minimum data for time series split
        print(f"  Not enough data after feature creation for location {location_id}, skipping...")
        continue
    
    # Define features (exclude target and identifiers)
    feature_cols = [col for col in location_sales.columns if col not in 
                   ['y', 'ds', 'location_id', 'sales_type_id', 'department_id', 'date', 
                    'corporation_id', 'postal_code']]
    
    # Time series split - use only last split for speed
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(location_sales))
    train_idx, val_idx = splits[-1]  # Use only the last split
    
    train_df = location_sales.iloc[train_idx]
    val_df = location_sales.iloc[val_idx]
    
    print(f"  Training LightGBM model...")
    print(f"  Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Train model
    model = train_lightgbm_model(train_df, val_df, feature_cols)
    
    # Make predictions on validation set
    predictions = model.predict(val_df[feature_cols])
    val_df = val_df.copy()
    val_df['yhat_lgb'] = predictions
    val_df['yhat_lgb'] = val_df['yhat_lgb'].clip(lower=0)  # No negative predictions
    
    # Store predictions
    all_predictions.append(val_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat_lgb']])
    
    print(f"  MAE for location {location_id}: {mean_absolute_error(val_df['y'], val_df['yhat_lgb']):.4f}")

# Combine all predictions
if all_predictions:
    print("\nCombining all predictions...")
    lgb_predictions_df = pd.concat(all_predictions, ignore_index=True)
    lgb_predictions_df.to_csv('results/lightgbm_predictions.csv', index=False)
    print(f"LightGBM predictions saved to results/lightgbm_predictions.csv")
    print(f"Total predictions: {len(lgb_predictions_df)}")
else:
    print("\nNo predictions were generated!")