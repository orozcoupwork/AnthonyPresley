# ensemble_forecast.py
def create_ensemble():
    # Load predictions
    prophet_df = pd.read_csv('results/baseline_prophet_results.csv')
    lgb_df = pd.read_csv('results/lightgbm_predictions.csv')
    
    # Merge predictions
    ensemble_df = pd.merge(
        prophet_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat']],
        lgb_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'yhat_lgb']],
        on=['ds', 'location_id', 'sales_type_id', 'department_id'],
        how='inner'
    )
    
    # Simple ensemble: weighted average
    ensemble_df['yhat_ensemble'] = 0.3 * ensemble_df['yhat'] + 0.7 * ensemble_df['yhat_lgb']
    
    # Calculate MAE for ensemble
    ensemble_df['ae_ensemble'] = abs(ensemble_df['y'] - ensemble_df['yhat_ensemble'])
    
    # Calculate metrics at different granularities
    mae_15min = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_ensemble'].mean()
    
    # Hourly
    ensemble_df['hour'] = pd.to_datetime(ensemble_df['ds']).dt.floor('H')
    hourly_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'hour']).agg({
        'y': 'sum',
        'yhat_ensemble': 'sum'
    }).reset_index()
    hourly_df['ae_hourly'] = abs(hourly_df['y'] - hourly_df['yhat_ensemble'])
    mae_hourly = hourly_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_hourly'].mean()
    
    # Daily
    ensemble_df['date'] = pd.to_datetime(ensemble_df['ds']).dt.date
    daily_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'date']).agg({
        'y': 'sum',
        'yhat_ensemble': 'sum'
    }).reset_index()
    daily_df['ae_daily'] = abs(daily_df['y'] - daily_df['yhat_ensemble'])
    mae_daily = daily_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_daily'].mean()
    
    # Calculate improvement
    ensemble_mae = {
        '15min': mae_15min.median(),
        'hourly': mae_hourly.median(),
        'daily': mae_daily.median(),
        'combined': (mae_15min.median() + mae_hourly.median() + mae_daily.median()) / 3
    }
    
    improvement = (baseline_mae['combined'] - ensemble_mae['combined']) / baseline_mae['combined'] * 100
    
    print(f"\nEnsemble MAE:")
    print(f"  15-min: {ensemble_mae['15min']:.4f}")
    print(f"  Hourly: {ensemble_mae['hourly']:.4f}")
    print(f"  Daily: {ensemble_mae['daily']:.4f}")
    print(f"  Combined: {ensemble_mae['combined']:.4f}")
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Save results
    ensemble_df.to_csv('results/ensemble_predictions.csv', index=False)
    
    return ensemble_df, improvement

ensemble_df, improvement = create_ensemble()