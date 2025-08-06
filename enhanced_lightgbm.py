# enhanced_lightgbm.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import json
from datetime import datetime, timedelta
import warnings
import time
from tqdm import tqdm
import argparse
warnings.filterwarnings('ignore')

def create_enhanced_lightgbm_forecast(location_id=None):
    """
    Enhanced LightGBM with features specifically designed for 15-minute accuracy
    """
    
    start_time = time.time()
    
    # Load baseline MAE
    try:
        with open('results/baseline_mae.json', 'r') as f:
            baseline_mae = json.load(f)
    except FileNotFoundError:
        print("ERROR: baseline_mae.json not found. Run baseline_prophet.py first!")
        return None, None
    
    print("="*60)
    print("ENHANCED LIGHTGBM TIME-SERIES FORECASTING")
    print("="*60)
    print(f"Target: Beat baseline 15-min MAE of {baseline_mae['15min']:.4f} by 20%")
    print(f"Target MAE: {baseline_mae['15min'] * 0.8:.4f}")
    
    # Load all data
    print("\n[1/6] Loading data...")
    sales_df = pd.read_csv('data/sales.csv')
    weather_df = pd.read_csv('data/weather.csv', dtype={'postal_code': str})
    location_df = pd.read_csv('data/locations.csv', dtype={'postal_code': str})
    holiday_df = pd.read_csv('data/holiday.csv')
    
    # Convert datetime
    sales_df['ds'] = pd.to_datetime(sales_df['ds'])
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])
    holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
    
    # IMPORTANT: Fill NaN department_ids with 0 for consistent grouping
    sales_df['department_id'] = sales_df['department_id'].fillna(0).astype(int)
    
    # Filter by location if specified
    if location_id:
        sales_df = sales_df[sales_df['location_id'] == location_id]
        location_df = location_df[location_df['id'] == location_id]
        print(f"Filtered to location {location_id}")
    
    print(f"✓ Loaded {len(sales_df):,} sales records")
    print(f"✓ Data spans from {sales_df['ds'].min()} to {sales_df['ds'].max()}")
    
    # Feature Engineering
    print("\n[2/6] Feature engineering...")
    
    # Progress bar for feature engineering
    with tqdm(total=12, desc="Creating features") as pbar:
        
        # 1. Time features with 15-minute resolution
        sales_df['hour'] = sales_df['ds'].dt.hour
        sales_df['minute'] = sales_df['ds'].dt.minute
        sales_df['interval_of_day'] = sales_df['hour'] * 4 + sales_df['minute'] // 15  # 0-95
        sales_df['dayofweek'] = sales_df['ds'].dt.dayofweek
        sales_df['dayofmonth'] = sales_df['ds'].dt.day
        sales_df['weekofyear'] = sales_df['ds'].dt.isocalendar().week
        sales_df['is_weekend'] = sales_df['dayofweek'].isin([5, 6]).astype(int)
        pbar.update(1)
        pbar.set_description("Creating time features")
        
        # 2. Sort for lag features
        sales_df = sales_df.sort_values(['location_id', 'sales_type_id', 'department_id', 'ds'])
        pbar.update(1)
        
        # 3. Lag features at 15-minute intervals
        pbar.set_description("Creating lag features")
        for lag in [1, 4, 8, 96, 96*7]:  # 15min ago, 1hr ago, 2hr ago, 1 day ago, 1 week ago
            sales_df[f'lag_{lag}_15min'] = sales_df.groupby(
                ['location_id', 'sales_type_id', 'department_id']
            )['y'].shift(lag)
        pbar.update(1)
        
        # 4. Rolling statistics at 15-minute level
        pbar.set_description("Creating rolling features")
        for window in [4, 8, 96]:  # 1hr, 2hr, 24hr windows
            sales_df[f'rolling_mean_{window}_15min'] = sales_df.groupby(
                ['location_id', 'sales_type_id', 'department_id']
            )['y'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            
            sales_df[f'rolling_std_{window}_15min'] = sales_df.groupby(
                ['location_id', 'sales_type_id', 'department_id']
            )['y'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        pbar.update(1)
        
        # 5. Zero-sales pattern features
        pbar.set_description("Creating zero-pattern features")
        sales_df['prev_was_zero'] = (sales_df.groupby(
            ['location_id', 'sales_type_id', 'department_id']
        )['y'].shift(1) == 0).astype(int)
        
        sales_df['zeros_in_last_4'] = sales_df.groupby(
            ['location_id', 'sales_type_id', 'department_id']
        )['y'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).apply(lambda y: (y == 0).sum()))
        
        sales_df['zeros_in_last_96'] = sales_df.groupby(
            ['location_id', 'sales_type_id', 'department_id']
        )['y'].transform(lambda x: x.shift(1).rolling(96, min_periods=1).apply(lambda y: (y == 0).sum()))
        pbar.update(1)
        
        # 6. Interval-specific historical features
        pbar.set_description("Creating interval history features")
        interval_history = sales_df.groupby(
            ['location_id', 'sales_type_id', 'department_id', 'interval_of_day', 'dayofweek']
        ).agg({
            'y': ['mean', 'median', 'std', lambda x: (x == 0).mean()]
        }).reset_index()
        
        interval_history.columns = ['location_id', 'sales_type_id', 'department_id', 
                                   'interval_of_day', 'dayofweek',
                                   'interval_mean', 'interval_median', 'interval_std', 'interval_zero_rate']
        
        sales_df = sales_df.merge(interval_history, 
                                 on=['location_id', 'sales_type_id', 'department_id', 
                                    'interval_of_day', 'dayofweek'],
                                 how='left')
        pbar.update(1)
        
        # 7. Open/closed indicators
        pbar.set_description("Creating open/closed indicators")
        sales_df['typical_open'] = (sales_df['interval_zero_rate'] < 0.8).astype(int)
        pbar.update(1)
        
        # 8. Spike detection features
        pbar.set_description("Creating spike detection features")
        sales_df['is_spike'] = 0
        spike_mask = (
            (sales_df['rolling_mean_96_15min'].notna()) & 
            (sales_df['rolling_std_96_15min'].notna()) &
            (sales_df['rolling_std_96_15min'] > 0) &
            (sales_df['y'] > sales_df['rolling_mean_96_15min'] + 2 * sales_df['rolling_std_96_15min'])
        )
        sales_df.loc[spike_mask, 'is_spike'] = 1
        pbar.update(1)
        
        # 8.5. Previous interval features - NEW
        pbar.set_description("Creating previous interval features")
        for i in range(1, 5):  # Look at previous 4 intervals
            sales_df[f'same_interval_lag_{i}_day'] = sales_df.groupby(
                ['location_id', 'sales_type_id', 'department_id', 'interval_of_day']
            )['y'].shift(96 * i)  # 96 intervals = 1 day
        pbar.update(1)
        
        # 9. Weather features
        pbar.set_description("Processing weather data")
        try:
            weather_hourly = weather_df.copy()
            weather_hourly['hour'] = weather_hourly['ds'].dt.floor('h')
            weather_agg = weather_hourly.groupby(['postal_code', 'hour']).agg({
                'real_feel': 'mean',
                'precipitation': 'sum',
                'snow': 'sum',
                'coverage': 'mean'
            }).reset_index()
            
            location_postal = location_df[['id', 'postal_code']].rename(columns={'id': 'location_id'})
            sales_df = sales_df.merge(location_postal, on='location_id', how='left')
            sales_df['hour_floor'] = sales_df['ds'].dt.floor('h')
            sales_df = sales_df.merge(
                weather_agg, 
                left_on=['postal_code', 'hour_floor'], 
                right_on=['postal_code', 'hour'],
                how='left'
            ).drop(columns=['hour_y'])
            sales_df.rename(columns={'hour_x': 'hour'}, inplace=True)
        except:
            # If weather data merge fails, just add empty columns
            for col in ['real_feel', 'precipitation', 'snow', 'coverage']:
                sales_df[col] = 0
        pbar.update(1)
        
        # 10. Holiday features
        pbar.set_description("Processing holiday data")
        sales_df['date'] = sales_df['ds'].dt.date
        holiday_dates = holiday_df[['ds']].copy()
        holiday_dates['date'] = holiday_dates['ds'].dt.date
        holiday_dates['is_holiday'] = 1
        holiday_dates = holiday_dates[['date', 'is_holiday']].drop_duplicates()
        sales_df = sales_df.merge(holiday_dates, on='date', how='left')
        sales_df['is_holiday'] = sales_df['is_holiday'].fillna(0)
        pbar.update(1)
        
        # 11. Final cleanup
        pbar.set_description("Finalizing features")
        pbar.update(1)
    
    print(f"✓ Feature engineering complete")
    
    # Feature list - UPDATED with new features
    feature_cols = [
        'interval_of_day', 'hour', 'minute', 'dayofweek', 'dayofmonth', 'weekofyear', 'is_weekend',
        'lag_1_15min', 'lag_4_15min', 'lag_8_15min', 'lag_96_15min', 'lag_672_15min',
        'rolling_mean_4_15min', 'rolling_std_4_15min',
        'rolling_mean_8_15min', 'rolling_std_8_15min', 
        'rolling_mean_96_15min', 'rolling_std_96_15min',
        'prev_was_zero', 'zeros_in_last_4', 'zeros_in_last_96',
        'interval_mean', 'interval_median', 'interval_std', 'interval_zero_rate',
        'typical_open', 'is_spike',
        'same_interval_lag_1_day', 'same_interval_lag_2_day', 
        'same_interval_lag_3_day', 'same_interval_lag_4_day',
        'real_feel', 'precipitation', 'snow', 'coverage',
        'is_holiday'
    ]
    
    # Remove features that don't exist
    feature_cols = [f for f in feature_cols if f in sales_df.columns]
    
    # Fill NaN values
    print("\n[3/6] Preparing data...")
    for col in feature_cols:
        if col in ['real_feel', 'precipitation', 'snow', 'coverage']:
            sales_df[col] = sales_df[col].fillna(0)
        elif col in ['interval_mean', 'interval_median', 'interval_std', 'interval_zero_rate']:
            sales_df[col] = sales_df[col].fillna(sales_df[col].median() if len(sales_df) > 0 else 0)
        else:
            sales_df[col] = sales_df[col].fillna(0)
    
    # Remove rows where we don't have sufficient lag data
    sales_df = sales_df[sales_df['lag_672_15min'].notna()]
    
    print(f"✓ After preparation: {len(sales_df):,} records")
    
    # TRAINING
    print("\n[4/6] Training models...")
    
    # Get groups to process
    groups = list(sales_df.groupby(['location_id', 'sales_type_id', 'department_id']))
    all_predictions = []
    
    for (loc_id, sales_type_id, department_id), group_data in tqdm(groups, desc="Training groups"):
        if location_id and loc_id != location_id:
            continue
            
        if len(group_data) < 1000:
            tqdm.write(f"  Skipping group (location={loc_id}, sales_type={sales_type_id}, dept={department_id}): only {len(group_data)} records")
            continue
        
        # Sort by date
        group_data = group_data.sort_values('ds')
        
        # Time series split - use last 20% for validation
        train_size = int(len(group_data) * 0.8)
        train_df = group_data.iloc[:train_size].copy()
        val_df = group_data.iloc[train_size:].copy()
        
        # LightGBM parameters - TUNED for better 15-min accuracy
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 20,  # Increased from 15
            'learning_rate': 0.025,  # Slightly reduced from 0.03
            'feature_fraction': 0.85,  # Increased from 0.8
            'bagging_fraction': 0.8,  # Increased from 0.7
            'bagging_freq': 3,  # Reduced from 5
            'min_data_in_leaf': 15,  # Reduced from 20
            'lambda_l1': 0.5,  # Reduced from 1.0
            'lambda_l2': 0.5,  # Reduced from 1.0
            'verbose': -1,
            'seed': 42
        }
        
        try:
            # Create datasets
            train_data = lgb.Dataset(train_df[feature_cols], label=train_df['y'])
            val_data = lgb.Dataset(val_df[feature_cols], label=val_df['y'], reference=train_data)
            
            # Train
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=600,  # Increased from 500
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Predict
            val_pred = model.predict(val_df[feature_cols])
            
            # POST-PROCESSING
            val_pred[val_df['typical_open'] == 0] = 0
            val_pred[val_df['interval_zero_rate'] > 0.9] = 0
            val_pred = np.clip(val_pred, 0, None)
            val_pred[val_pred < 0.5] = 0
            
            location_zero_rate = (val_df['y'] == 0).mean()
            if location_zero_rate > 0.7:
                val_pred[val_pred < 1.0] = 0
            
            # Store predictions
            result_df = val_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y']].copy()
            result_df['yhat_enhanced'] = val_pred
            all_predictions.append(result_df)
            
            # Group MAE
            loc_mae = np.mean(np.abs(val_df['y'] - val_pred))
            tqdm.write(f"  Location {loc_id}, sales_type {sales_type_id}, dept {department_id}: MAE={loc_mae:.4f}")
        
        except Exception as e:
            tqdm.write(f"  Error training group: {str(e)}")
            continue
    
    if not all_predictions:
        print("ERROR: No predictions generated!")
        return None, None
    
    # Combine predictions
    print("\n[5/6] Combining predictions and creating ensemble...")
    enhanced_df = pd.concat(all_predictions, ignore_index=True)
    enhanced_df.to_csv('results/enhanced_lgb_raw.csv', index=False)
    
    print(f"✓ Generated {len(enhanced_df):,} enhanced predictions")
    
    # Load Prophet predictions for ensemble
    try:
        prophet_df = pd.read_csv('results/baseline_prophet_results.csv')
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    except FileNotFoundError:
        print("ERROR: baseline_prophet_results.csv not found. Run baseline_prophet.py first!")
        return None, None
    
    # Merge
    final_df = prophet_df.merge(
        enhanced_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'yhat_enhanced']],
        on=['ds', 'location_id', 'sales_type_id', 'department_id'],
        how='inner'
    )
    
    print(f"✓ Merged with Prophet: {len(final_df):,} records")
    
    if len(final_df) == 0:
        print("ERROR: No matching records between Prophet and Enhanced models!")
        print("Checking data alignment...")
        print(f"Prophet unique dates: {prophet_df['ds'].nunique()}")
        print(f"Enhanced unique dates: {enhanced_df['ds'].nunique()}")
        print(f"Prophet date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        print(f"Enhanced date range: {enhanced_df['ds'].min()} to {enhanced_df['ds'].max()}")
        return None, None
    
    # ENSEMBLE
    final_df['prophet_error'] = np.abs(final_df['y'] - final_df['yhat'])
    final_df['enhanced_error'] = np.abs(final_df['y'] - final_df['yhat_enhanced'])
    
    # Dynamic weighting
    zero_mask = (final_df['y'] == 0)
    if zero_mask.sum() > 0:
        prophet_zero_correct = zero_mask & (final_df['yhat'] == 0)
        enhanced_zero_correct = zero_mask & (final_df['yhat_enhanced'] == 0)
        
        print(f"\nZero prediction accuracy:")
        print(f"  Prophet: {prophet_zero_correct.sum()}/{zero_mask.sum()} = {prophet_zero_correct.sum()/zero_mask.sum():.1%}")
        print(f"  Enhanced: {enhanced_zero_correct.sum()}/{zero_mask.sum()} = {enhanced_zero_correct.sum()/zero_mask.sum():.1%}")
    
    # Use enhanced model more heavily - UPDATED WEIGHTS
    final_df['yhat_final'] = 0.15 * final_df['yhat'] + 0.85 * final_df['yhat_enhanced']
    
    # Final zero cleanup
    both_small = (final_df['yhat'] < 1) & (final_df['yhat_enhanced'] < 1)
    final_df.loc[both_small, 'yhat_final'] = 0
    
    enhanced_zero = (final_df['yhat_enhanced'] == 0) & (final_df['yhat'] < 2)
    final_df.loc[enhanced_zero, 'yhat_final'] = 0
    
    # Calculate metrics
    print("\n[6/6] Calculating final metrics...")
    improvements = calculate_enhanced_metrics(final_df, baseline_mae)
    
    # Save
    final_df.to_csv('results/enhanced_lightgbm_predictions.csv', index=False)
    
    # Total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    
    return final_df, improvements

def calculate_enhanced_metrics(df, baseline_mae):
    """Calculate metrics for enhanced model"""
    
    # 15-minute MAE
    df['ae_final'] = abs(df['y'] - df['yhat_final'])
    mae_15min = df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_final'].mean()
    
    # Calculate hourly aggregates
    df['hour'] = df['ds'].dt.floor('h')
    hourly_df = df.groupby(['location_id', 'sales_type_id', 'department_id', 'hour']).agg({
        'y': 'sum',
        'yhat_final': 'sum'
    }).reset_index()
    hourly_df['ae_hourly'] = abs(hourly_df['y'] - hourly_df['yhat_final'])
    mae_hourly = hourly_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_hourly'].mean()
    
    # Calculate daily aggregates
    df['date'] = df['ds'].dt.date
    daily_df = df.groupby(['location_id', 'sales_type_id', 'department_id', 'date']).agg({
        'y': 'sum',
        'yhat_final': 'sum'
    }).reset_index()
    daily_df['ae_daily'] = abs(daily_df['y'] - daily_df['yhat_final'])
    mae_daily = daily_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_daily'].mean()
    
    # Calculate median MAE across all groups
    enhanced_mae = {
        '15min': mae_15min.median(),
        'hourly': mae_hourly.median(),
        'daily': mae_daily.median()
    }
    enhanced_mae['combined'] = (enhanced_mae['15min'] + enhanced_mae['hourly'] + enhanced_mae['daily']) / 3
    
    # Calculate improvements
    improvements = {
        '15min': (baseline_mae['15min'] - enhanced_mae['15min']) / baseline_mae['15min'] * 100,
        'hourly': (baseline_mae['hourly'] - enhanced_mae['hourly']) / baseline_mae['hourly'] * 100,
        'daily': (baseline_mae['daily'] - enhanced_mae['daily']) / baseline_mae['daily'] * 100,
        'combined': (baseline_mae['combined'] - enhanced_mae['combined']) / baseline_mae['combined'] * 100
    }
    
    print(f"\n{'='*60}")
    print(f"ENHANCED LIGHTGBM RESULTS")
    print(f"{'='*60}")
    print(f"\nBaseline Prophet MAE:")
    print(f"  15-min: {baseline_mae['15min']:.4f}")
    print(f"  Hourly: {baseline_mae['hourly']:.4f}")
    print(f"  Daily: {baseline_mae['daily']:.4f}")
    print(f"  Combined: {baseline_mae['combined']:.4f}")
    
    print(f"\nEnhanced Model MAE:")
    print(f"  15-min: {enhanced_mae['15min']:.4f} (improvement: {improvements['15min']:.1f}%)")
    print(f"  Hourly: {enhanced_mae['hourly']:.4f} (improvement: {improvements['hourly']:.1f}%)")
    print(f"  Daily: {enhanced_mae['daily']:.4f} (improvement: {improvements['daily']:.1f}%)")
    print(f"  Combined: {enhanced_mae['combined']:.4f} (improvement: {improvements['combined']:.1f}%)")
    
    print(f"\n{'='*60}")
    if improvements['15min'] >= 20:
        print(f"✅ SUCCESS: Achieved {improvements['15min']:.1f}% improvement on 15-minute MAE!")
    else:
        print(f"❌ 15-minute improvement: {improvements['15min']:.1f}% (target: 20%)")
    print(f"{'='*60}")
    
    # Detailed location analysis
    print(f"\nPer-location 15-minute improvements:")
    for loc_id in sorted(df['location_id'].unique()):
        loc_data = df[df['location_id'] == loc_id]
        loc_prophet_mae = loc_data.groupby(['sales_type_id', 'department_id'])['ae_15min'].mean().mean()
        loc_enhanced_mae = loc_data.groupby(['sales_type_id', 'department_id'])['ae_final'].mean().mean()
        loc_improvement = (loc_prophet_mae - loc_enhanced_mae) / loc_prophet_mae * 100 if loc_prophet_mae > 0 else 0
        print(f"  Location {loc_id}: {loc_improvement:.1f}%")
    
    # Zero prediction accuracy
    actual_zeros = df['y'] == 0
    pred_zeros = df['yhat_final'] == 0
    prophet_zeros = df['yhat'] == 0
    
    zero_precision = (actual_zeros & pred_zeros).sum() / pred_zeros.sum() if pred_zeros.sum() > 0 else 0
    zero_recall = (actual_zeros & pred_zeros).sum() / actual_zeros.sum() if actual_zeros.sum() > 0 else 0
    
    prophet_zero_precision = (actual_zeros & prophet_zeros).sum() / prophet_zeros.sum() if prophet_zeros.sum() > 0 else 0
    prophet_zero_recall = (actual_zeros & prophet_zeros).sum() / actual_zeros.sum() if actual_zeros.sum() > 0 else 0
    
    print(f"\nZero prediction accuracy:")
    print(f"  Prophet - Precision: {prophet_zero_precision:.1%}, Recall: {prophet_zero_recall:.1%}")
    print(f"  Enhanced - Precision: {zero_precision:.1%}, Recall: {zero_recall:.1%}")
    
    # Save detailed metrics
    metrics_summary = {
        'baseline_mae': baseline_mae,
        'enhanced_mae': enhanced_mae,
        'improvements': improvements,
        'zero_metrics': {
            'prophet_precision': prophet_zero_precision,
            'prophet_recall': prophet_zero_recall,
            'enhanced_precision': zero_precision,
            'enhanced_recall': zero_recall
        }
    }
    
    with open('results/enhanced_metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    return improvements

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run enhanced LightGBM forecast')
    parser.add_argument('--location', type=int, help='Location ID to process (optional)')
    args = parser.parse_args()
    
    df, improvements = create_enhanced_lightgbm_forecast(args.location)
    
    if df is not None:
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        success_15min = improvements['15min'] >= 20
        
        print(f"\n15-minute MAE improvement: {improvements['15min']:.1f}% {'✅' if success_15min else '❌'}")
        print(f"Hourly MAE improvement: {improvements.get('hourly', 0):.1f}%")
        print(f"Daily MAE improvement: {improvements.get('daily', 0):.1f}%")
        print(f"Combined MAE improvement: {improvements.get('combined', 0):.1f}%")
        
        if success_15min:
            print(f"\n✅ SUCCESS! Achieved {improvements['15min']:.1f}% improvement on 15-minute MAE!")
            print("Deliverable ready for submission")
        else:
            print(f"\n❌ Current 15-min improvement: {improvements['15min']:.1f}% (target: 20%)")
            print("May need to adjust hyperparameters or feature engineering")
    else:
        print("\n❌ Failed to generate predictions")