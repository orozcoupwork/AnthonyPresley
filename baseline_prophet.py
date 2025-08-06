# baseline_prophet.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from prophet import Prophet
import argparse
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time
warnings.filterwarnings('ignore')

def load_data(location_id=None):
    """Load all data files"""
    data_path = 'data/'
    
    # Load main tables
    sales_df = pd.read_csv(f'{data_path}sales.csv')
    weather_df = pd.read_csv(f'{data_path}weather.csv', dtype={'postal_code': str})
    holiday_df = pd.read_csv(f'{data_path}holiday.csv')
    location_df = pd.read_csv(f'{data_path}locations.csv', dtype={'postal_code': str})
    department_df = pd.read_csv(f'{data_path}department.csv')
    sales_type_df = pd.read_csv(f'{data_path}sales_type.csv')
    
    # Convert datetime columns
    sales_df['ds'] = pd.to_datetime(sales_df['ds'])
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])
    holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
    
    # FIX: Prophet requires lower_window <= 0
    holiday_df['lower_window'] = holiday_df['lower_window'].apply(lambda x: -abs(x) if x > 0 else x)
    
    # IMPORTANT: Fill NaN department_ids with 0 for consistent grouping
    sales_df['department_id'] = sales_df['department_id'].fillna(0).astype(int)
    
    # Filter by location if specified
    if location_id:
        sales_df = sales_df[sales_df['location_id'] == location_id]
        location_df = location_df[location_df['id'] == location_id]
        print(f"Filtered to location {location_id}: {len(sales_df)} sales records")
    
    return {
        'sales': sales_df,
        'weather': weather_df,
        'holiday': holiday_df,
        'location': location_df,
        'department': department_df,
        'sales_type': sales_type_df
    }

def process_single_group(args):
    """Process a single group - designed for parallel execution"""
    group_key, group_df, holiday_df, min_records, verbose = args
    location_id, sales_type_id, department_id = group_key
    
    if verbose:
        print(f"Processing: location={location_id}, sales_type={sales_type_id}, dept={department_id}, records={len(group_df)}")
    
    if len(group_df) < min_records:
        return None
    
    # Prepare data for Prophet
    prophet_df = group_df[['ds', 'y']].copy()
    prophet_df = prophet_df.drop_duplicates(subset=['ds'])
    prophet_df = prophet_df.sort_values('ds')
    
    # Split into train/test (80/20)
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:train_size]
    test_df = prophet_df[train_size:]
    
    if len(train_df) < 50 or len(test_df) < 10:
        return None
    
    try:
        # Prepare holidays
        prophet_holidays = None
        if len(holiday_df) > 0:
            prophet_holidays = holiday_df[['ds', 'holiday', 'lower_window', 'upper_window']].copy()
            prophet_holidays['lower_window'] = prophet_holidays['lower_window'].astype(int)
            prophet_holidays['upper_window'] = prophet_holidays['upper_window'].astype(int)
        
        # Suppress Prophet output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Disable cmdstanpy logging
            import logging
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                holidays=prophet_holidays,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                interval_width=0.95,
                stan_backend='CMDSTANPY'  # Faster backend
            )
            
            model.fit(train_df)
            forecast = model.predict(test_df[['ds']])
        
        # Merge with actuals
        results_df = test_df.merge(forecast[['ds', 'yhat']], on='ds')
        results_df['location_id'] = location_id
        results_df['sales_type_id'] = sales_type_id
        results_df['department_id'] = department_id
        results_df['ae_15min'] = abs(results_df['y'] - results_df['yhat'])
        
        return results_df
        
    except Exception as e:
        if verbose:
            print(f"Error processing group {group_key}: {str(e)}")
        return None

def calculate_baseline_prophet_parallel(data, location_id=None, n_jobs=None, sample_frac=None):
    """Calculate baseline using Prophet with parallel processing"""
    sales_df = data['sales']
    holiday_df = data['holiday']
    location_df = data['location']
    
    # Set number of parallel jobs
    if n_jobs is None:
        n_jobs = min(cpu_count() - 1, 8)  # Leave one CPU free, max 8
    
    print(f"\nUsing {n_jobs} parallel workers")
    
    # Get unique combinations
    groups = sales_df.groupby(['location_id', 'sales_type_id', 'department_id'])
    group_keys = list(groups.groups.keys())
    
    # Sample if requested (for faster testing)
    if sample_frac is not None:
        n_sample = int(len(group_keys) * sample_frac)
        group_keys = np.random.choice(group_keys, n_sample, replace=False).tolist()
        print(f"Sampling {n_sample} out of {len(groups)} groups ({sample_frac*100:.0f}%)")
    
    print(f"\nProcessing {len(group_keys)} groups...")
    
    # Prepare arguments for parallel processing
    args_list = []
    for key in group_keys:
        loc_id, sales_type_id, dept_id = key
        
        if location_id and loc_id != location_id:
            continue
            
        group_df = groups.get_group(key)
        
        # Get holidays for this location's corporation
        loc_holidays = pd.DataFrame()
        if len(location_df) > 0:
            corp_id = location_df[location_df['id'] == loc_id]['corporation_id'].values
            if len(corp_id) > 0:
                loc_holidays = holiday_df[holiday_df['corporation_id'] == corp_id[0]].copy()
        
        args_list.append((
            key,
            group_df,
            loc_holidays,
            100,  # min_records
            False  # verbose (set to False for parallel)
        ))
    
    # Process in parallel
    start_time = time.time()
    
    with Pool(n_jobs) as pool:
        results = pool.map(process_single_group, args_list)
    
    # Filter out None results
    all_predictions = [r for r in results if r is not None]
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(args_list)} groups in {elapsed/60:.1f} minutes")
    print(f"Successfully processed {len(all_predictions)} groups")
    
    if not all_predictions:
        print("ERROR: No predictions generated!")
        return None, None
    
    # Combine all predictions
    final_df = pd.concat(all_predictions, ignore_index=True)
    print(f"Generated {len(final_df)} total predictions")
    
    # Calculate MAE at different levels
    mae_by_group = final_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_15min'].mean()
    
    # Hourly aggregates
    final_df['hour'] = final_df['ds'].dt.floor('h')
    hourly_df = final_df.groupby(['location_id', 'sales_type_id', 'department_id', 'hour']).agg({
        'y': 'sum',
        'yhat': 'sum'
    }).reset_index()
    hourly_df['ae_hourly'] = abs(hourly_df['y'] - hourly_df['yhat'])
    mae_hourly = hourly_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_hourly'].mean()
    
    # Daily aggregates
    final_df['date'] = final_df['ds'].dt.date
    daily_df = final_df.groupby(['location_id', 'sales_type_id', 'department_id', 'date']).agg({
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
    
    print(f"\nBaseline Prophet MAE:")
    print(f"  15-min: {baseline_mae['15min']:.4f}")
    print(f"  Hourly: {baseline_mae['hourly']:.4f}")
    print(f"  Daily: {baseline_mae['daily']:.4f}")
    print(f"  Combined: {baseline_mae['combined']:.4f}")
    
    return baseline_mae, final_df

def main(location_id=None, n_jobs=None, sample_frac=None):
    """Main execution"""
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created results directory")
    
    # Load data
    print("Loading data...")
    data = load_data(location_id)
    print("Data loaded successfully!")
    
    # Check if we have any data
    if location_id and len(data['sales']) == 0:
        print(f"\nERROR: No sales data found for location {location_id}")
        all_sales = pd.read_csv('data/sales.csv')
        locations = sorted(all_sales['location_id'].unique())
        print(f"\nAvailable locations: {locations}")
        return
    
    # Run baseline calculation with parallel processing
    print("\nCalculating baseline MAE using Prophet (parallel)...")
    baseline_mae, baseline_df = calculate_baseline_prophet_parallel(
        data, location_id, n_jobs, sample_frac
    )
    
    # Save baseline results if calculation was successful
    if baseline_df is not None:
        # Save to CSV
        baseline_df.to_csv('results/baseline_prophet_results.csv', index=False)
        print(f"\nResults saved to results/baseline_prophet_results.csv ({len(baseline_df)} rows)")
        
        # Save the baseline MAE values
        with open('results/baseline_mae.json', 'w') as f:
            json.dump(baseline_mae, f)
        print("✓ Baseline MAE values saved to results/baseline_mae.json")
        
        # Show sample predictions
        print("\nSample predictions (first 10 rows):")
        print(baseline_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat', 'ae_15min']].head(10))
    else:
        print("\nERROR: Could not calculate baseline MAE.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Prophet baseline forecast')
    parser.add_argument('--location', type=int, help='Location ID to process (optional)')
    parser.add_argument('--n_jobs', type=int, default=None, 
                        help='Number of parallel jobs (default: CPU count - 1, max 8)')
    parser.add_argument('--sample', type=float, default=None,
                        help='Fraction of groups to sample for faster testing (e.g., 0.1 for 10%)')
    args = parser.parse_args()
    
    main(args.location, args.n_jobs, args.sample)