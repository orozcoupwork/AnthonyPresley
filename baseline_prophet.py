# baseline_prophet.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Load all data files
def load_data():
    data_path = 'data/'
    
    # Load main tables
    sales_df = pd.read_csv(f'{data_path}sales.csv')
    forecast_df = pd.read_csv(f'{data_path}forecast.csv') 
    weather_df = pd.read_csv(f'{data_path}weather.csv')
    holiday_df = pd.read_csv(f'{data_path}holiday.csv')
    location_df = pd.read_csv(f'{data_path}location.csv')
    department_df = pd.read_csv(f'{data_path}department.csv')
    sales_type_df = pd.read_csv(f'{data_path}sales_type.csv')
    
    # Convert datetime columns
    sales_df['ds'] = pd.to_datetime(sales_df['ds'])
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])
    holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
    
    return {
        'sales': sales_df,
        'forecast': forecast_df,
        'weather': weather_df,
        'holiday': holiday_df,
        'location': location_df,
        'department': department_df,
        'sales_type': sales_type_df
    }

# Calculate baseline MAE
def calculate_baseline_mae(data):
    sales_df = data['sales']
    forecast_df = data['forecast']
    
    # Merge actual and forecast
    merged_df = pd.merge(
        sales_df,
        forecast_df,
        on=['ds', 'location_id', 'sales_type_id', 'department_id'],
        how='inner'
    )
    
    # Calculate MAE at 15-minute level
    merged_df['ae_15min'] = abs(merged_df['y'] - merged_df['yhat'])
    
    # Group by store × sales_type × department
    mae_by_group = merged_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_15min'].mean()
    
    # Calculate hourly aggregates
    merged_df['hour'] = merged_df['ds'].dt.floor('H')
    hourly_df = merged_df.groupby(['location_id', 'sales_type_id', 'department_id', 'hour']).agg({
        'y': 'sum',
        'yhat': 'sum'
    }).reset_index()
    hourly_df['ae_hourly'] = abs(hourly_df['y'] - hourly_df['yhat'])
    mae_hourly = hourly_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_hourly'].mean()
    
    # Calculate daily aggregates
    merged_df['date'] = merged_df['ds'].dt.date
    daily_df = merged_df.groupby(['location_id', 'sales_type_id', 'department_id', 'date']).agg({
        'y': 'sum',
        'yhat': 'sum'
    }).reset_index()
    daily_df['ae_daily'] = abs(daily_df['y'] - daily_df['yhat'])
    mae_daily = daily_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_daily'].mean()
    
    # Calculate median MAE across all groups
    baseline_mae = {
        '15min': mae_by_group.median(),
        'hourly': mae_hourly.median(),
        'daily': mae_daily.median(),
        'combined': (mae_by_group.median() + mae_hourly.median() + mae_daily.median()) / 3
    }
    
    print(f"Baseline Prophet MAE:")
    print(f"  15-min: {baseline_mae['15min']:.4f}")
    print(f"  Hourly: {baseline_mae['hourly']:.4f}")
    print(f"  Daily: {baseline_mae['daily']:.4f}")
    print(f"  Combined: {baseline_mae['combined']:.4f}")
    
    return baseline_mae, merged_df

# Run baseline calculation
data = load_data()
baseline_mae, baseline_df = calculate_baseline_mae(data)

# Save baseline results
baseline_df.to_csv('results/baseline_prophet_results.csv', index=False)