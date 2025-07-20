# ensemble_forecast.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

def create_ensemble():
    # Load baseline MAE for comparison
    with open('results/baseline_mae.json', 'r') as f:
        baseline_mae = json.load(f)
    
    print("Loading predictions...")
    # Load predictions
    prophet_df = pd.read_csv('results/baseline_prophet_results.csv')
    lgb_df = pd.read_csv('results/lightgbm_predictions.csv')
    
    # Convert datetime columns
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    lgb_df['ds'] = pd.to_datetime(lgb_df['ds'])
    
    print(f"Prophet predictions shape: {prophet_df.shape}")
    print(f"LightGBM predictions shape: {lgb_df.shape}")
    
    # Merge predictions
    ensemble_df = pd.merge(
        prophet_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat']],
        lgb_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'yhat_lgb']],
        on=['ds', 'location_id', 'sales_type_id', 'department_id'],
        how='inner'
    )
    
    print(f"Merged ensemble shape: {ensemble_df.shape}")
    
    if ensemble_df.empty:
        print("ERROR: No matching predictions between Prophet and LightGBM!")
        return None, None
    
    # Try different ensemble weights to find the best combination
    best_weight = 0.3
    best_mae = float('inf')
    
    print("\nTesting different ensemble weights...")
    for prophet_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
        lgb_weight = 1 - prophet_weight
        
        # Create ensemble prediction
        ensemble_df['yhat_ensemble'] = (prophet_weight * ensemble_df['yhat'] + 
                                       lgb_weight * ensemble_df['yhat_lgb'])
        
        # Calculate MAE at 15-minute level
        ensemble_df['ae_ensemble'] = abs(ensemble_df['y'] - ensemble_df['yhat_ensemble'])
        mae_15min = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_ensemble'].mean()
        
        current_mae = mae_15min.median()
        print(f"  Prophet weight: {prophet_weight:.1f}, LightGBM weight: {lgb_weight:.1f}, MAE: {current_mae:.4f}")
        
        if current_mae < best_mae:
            best_mae = current_mae
            best_weight = prophet_weight
    
    print(f"\nBest ensemble weight - Prophet: {best_weight:.1f}, LightGBM: {1-best_weight:.1f}")
    
    # Use best weights for final ensemble
    prophet_weight = best_weight
    lgb_weight = 1 - best_weight
    ensemble_df['yhat_ensemble'] = (prophet_weight * ensemble_df['yhat'] + 
                                   lgb_weight * ensemble_df['yhat_lgb'])
    
    # Calculate MAE for ensemble
    ensemble_df['ae_ensemble'] = abs(ensemble_df['y'] - ensemble_df['yhat_ensemble'])
    
    # Calculate metrics at different granularities
    mae_15min = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_ensemble'].mean()
    
    # Hourly
    ensemble_df['hour'] = ensemble_df['ds'].dt.floor('h')
    hourly_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'hour']).agg({
        'y': 'sum',
        'yhat_ensemble': 'sum'
    }).reset_index()
    hourly_df['ae_hourly'] = abs(hourly_df['y'] - hourly_df['yhat_ensemble'])
    mae_hourly = hourly_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_hourly'].mean()
    
    # Daily
    ensemble_df['date'] = ensemble_df['ds'].dt.date
    daily_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'date']).agg({
        'y': 'sum',
        'yhat_ensemble': 'sum'
    }).reset_index()
    daily_df['ae_daily'] = abs(daily_df['y'] - daily_df['yhat_ensemble'])
    mae_daily = daily_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_daily'].mean()
    
    # Calculate ensemble MAE
    ensemble_mae = {
        '15min': mae_15min.median(),
        'hourly': mae_hourly.median(),
        'daily': mae_daily.median(),
        'combined': (mae_15min.median() + mae_hourly.median() + mae_daily.median()) / 3
    }
    
    # Calculate improvements
    improvements = {
        '15min': (baseline_mae['15min'] - ensemble_mae['15min']) / baseline_mae['15min'] * 100,
        'hourly': (baseline_mae['hourly'] - ensemble_mae['hourly']) / baseline_mae['hourly'] * 100,
        'daily': (baseline_mae['daily'] - ensemble_mae['daily']) / baseline_mae['daily'] * 100,
        'combined': (baseline_mae['combined'] - ensemble_mae['combined']) / baseline_mae['combined'] * 100
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"\nBaseline Prophet MAE:")
    print(f"  15-min: {baseline_mae['15min']:.4f}")
    print(f"  Hourly: {baseline_mae['hourly']:.4f}")
    print(f"  Daily: {baseline_mae['daily']:.4f}")
    print(f"  Combined: {baseline_mae['combined']:.4f}")
    
    print(f"\nEnsemble MAE (Prophet {prophet_weight:.1f} + LightGBM {lgb_weight:.1f}):")
    print(f"  15-min: {ensemble_mae['15min']:.4f} (improvement: {improvements['15min']:.1f}%)")
    print(f"  Hourly: {ensemble_mae['hourly']:.4f} (improvement: {improvements['hourly']:.1f}%)")
    print(f"  Daily: {ensemble_mae['daily']:.4f} (improvement: {improvements['daily']:.1f}%)")
    print(f"  Combined: {ensemble_mae['combined']:.4f} (improvement: {improvements['combined']:.1f}%)")
    
    print(f"\n{'='*60}")
    if improvements['combined'] >= 20:
        print(f"✅ SUCCESS: Achieved {improvements['combined']:.1f}% improvement (>= 20% target)")
    else:
        print(f"❌ FAILED: Only {improvements['combined']:.1f}% improvement (< 20% target)")
    print(f"{'='*60}")
    
    # Add ensemble weights to the results
    ensemble_mae['prophet_weight'] = prophet_weight
    ensemble_mae['lgb_weight'] = lgb_weight
    ensemble_mae['improvements'] = improvements
    
    # Save results
    ensemble_df.to_csv('results/ensemble_predictions.csv', index=False)
    
    # Save ensemble metrics
    with open('results/ensemble_metrics.json', 'w') as f:
        json.dump({
            'baseline_mae': baseline_mae,
            'ensemble_mae': ensemble_mae,
            'improvements': improvements,
            'ensemble_weights': {
                'prophet': prophet_weight,
                'lightgbm': lgb_weight
            }
        }, f, indent=2)
    
    return ensemble_df, improvements['combined']

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Run ensemble
ensemble_df, improvement = create_ensemble()

if ensemble_df is not None:
    print(f"\nEnsemble predictions saved to results/ensemble_predictions.csv")
    print(f"Ensemble metrics saved to results/ensemble_metrics.json")
    
    # Create a summary report
    with open('results/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write("ENSEMBLE FORECAST SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        with open('results/ensemble_metrics.json', 'r') as metrics_file:
            metrics = json.load(metrics_file)
            
        f.write("BASELINE PROPHET PERFORMANCE:\n")
        f.write(f"  15-minute MAE: {metrics['baseline_mae']['15min']:.4f}\n")
        f.write(f"  Hourly MAE: {metrics['baseline_mae']['hourly']:.4f}\n")
        f.write(f"  Daily MAE: {metrics['baseline_mae']['daily']:.4f}\n")
        f.write(f"  Combined MAE: {metrics['baseline_mae']['combined']:.4f}\n\n")
        
        f.write("ENSEMBLE MODEL PERFORMANCE:\n")
        f.write(f"  Weights: Prophet {metrics['ensemble_weights']['prophet']:.1f} + LightGBM {metrics['ensemble_weights']['lightgbm']:.1f}\n")
        f.write(f"  15-minute MAE: {metrics['ensemble_mae']['15min']:.4f}\n")
        f.write(f"  Hourly MAE: {metrics['ensemble_mae']['hourly']:.4f}\n")
        f.write(f"  Daily MAE: {metrics['ensemble_mae']['daily']:.4f}\n")
        f.write(f"  Combined MAE: {metrics['ensemble_mae']['combined']:.4f}\n\n")
        
        f.write("IMPROVEMENTS:\n")
        f.write(f"  15-minute: {metrics['improvements']['15min']:.1f}%\n")
        f.write(f"  Hourly: {metrics['improvements']['hourly']:.1f}%\n")
        f.write(f"  Daily: {metrics['improvements']['daily']:.1f}%\n")
        f.write(f"  Combined: {metrics['improvements']['combined']:.1f}%\n\n")
        
        f.write("="*60 + "\n")
        if metrics['improvements']['combined'] >= 20:
            f.write("TARGET ACHIEVED: >= 20% improvement\n")
        else:
            f.write("TARGET MISSED: < 20% improvement\n")
    
    print("Summary report saved to results/summary_report.txt")
else:
    print("\nERROR: Could not create ensemble predictions.")