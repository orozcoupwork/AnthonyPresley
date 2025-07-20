# timeforge_ensemble_forecast_dag.py
"""
TimeForge Ensemble Forecast DAG
Author: Joseph Orozco
Description: Ensemble forecast combining Prophet and LightGBM with 20% MAE improvement guarantee
"""

"""
DEPLOYMENT NOTES:
1. Place this file in your Airflow DAGs folder
2. Ensure the Python scripts (baseline_prophet.py, lightgbm_forecast.py, ensemble_forecast.py) 
   are in the same directory as the DAG or in PYTHONPATH
3. Data files should be in a 'data/' subdirectory
4. Results will be written to 'results/' subdirectory
5. The DAG will fail if the 20% improvement target is not met
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import os
import json
import pandas as pd
import numpy as np

# Default arguments for the DAG
default_args = {
    'owner': 'joseph_orozco',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['anthony@timeforge.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'timeforge_ensemble_forecast',
    default_args=default_args,
    description='Ensemble forecast combining Prophet and LightGBM with 20% MAE improvement guarantee',
    schedule_interval=None,  # Run on demand
    catchup=False,
    tags=['forecasting', 'ensemble', 'timeforge'],
)

def validate_data(**context):
    """Validate input data exists"""
    required_files = [
        'data/sales.csv',
        'data/forecast.csv',
        'data/weather.csv',
        'data/holiday.csv',
        'data/locations.csv',
        'data/department.csv',
        'data/sales_type.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        raise Exception(f"Missing required data files: {missing_files}")
    
    print("All required data files found")
    return True

def calculate_baseline_mae(**context):
    """Calculate baseline Prophet MAE"""
    try:
        # Import and run baseline calculation
        exec(open('baseline_prophet.py').read())
        
        # Load results
        with open('results/baseline_mae.json', 'r') as f:
            baseline_mae = json.load(f)
        
        print(f"Baseline MAE calculated successfully")
        print(f"Combined MAE: {baseline_mae['combined']:.4f}")
        
        # Push to XCom for downstream tasks
        context['task_instance'].xcom_push(key='baseline_mae', value=baseline_mae)
        
        return baseline_mae
        
    except Exception as e:
        raise Exception(f"Failed to calculate baseline MAE: {str(e)}")

def train_lightgbm(**context):
    """Train LightGBM models"""
    try:
        # Import and run LightGBM training
        exec(open('lightgbm_forecast.py').read())
        
        # Verify output exists
        if not os.path.exists('results/lightgbm_predictions.csv'):
            raise Exception("LightGBM predictions file not created")
        
        # Get row count
        lgb_df = pd.read_csv('results/lightgbm_predictions.csv')
        print(f"LightGBM training completed")
        print(f"Generated {len(lgb_df)} predictions")
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to train LightGBM: {str(e)}")

def create_ensemble(**context):
    """Create ensemble and verify improvement"""
    try:
        # Get baseline MAE from upstream task
        baseline_mae = context['task_instance'].xcom_pull(
            task_ids='calculate_baseline',
            key='baseline_mae'
        )
        
        # Import and run ensemble
        exec(open('ensemble_forecast.py').read())
        
        # Load ensemble metrics
        with open('results/ensemble_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        improvement = metrics['improvements']['combined']
        
        print(f"Ensemble created successfully")
        print(f"Improvement: {improvement:.1f}%")
        
        # Check if 20% improvement was achieved
        if improvement < 20:
            raise Exception(
                f"FAILED TO MEET GUARANTEE: Only {improvement:.1f}% improvement (< 20% target). "
                f"Baseline MAE: {baseline_mae['combined']:.4f}, "
                f"Ensemble MAE: {metrics['ensemble_mae']['combined']:.4f}"
            )
        
        # Push metrics to XCom
        context['task_instance'].xcom_push(key='ensemble_metrics', value=metrics)
        context['task_instance'].xcom_push(key='improvement', value=improvement)
        
        return metrics
        
    except Exception as e:
        raise Exception(f"Failed to create ensemble: {str(e)}")

def generate_final_report(**context):
    """Generate delivery report"""
    try:
        # Get metrics from upstream tasks
        baseline_mae = context['task_instance'].xcom_pull(
            task_ids='calculate_baseline',
            key='baseline_mae'
        )
        ensemble_metrics = context['task_instance'].xcom_pull(
            task_ids='create_ensemble',
            key='ensemble_metrics'
        )
        improvement = context['task_instance'].xcom_pull(
            task_ids='create_ensemble',
            key='improvement'
        )
        
        # Generate report
        report_content = f"""
MILESTONE 1 DELIVERY REPORT
{'='*60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Contractor: Joseph Orozco
Project: TimeForge Retail Forecast Enhancement

GUARANTEE: At least 20% drop in median MAE vs Prophet baseline
RESULT: {'✅ ACHIEVED' if improvement >= 20 else '❌ FAILED'}

BASELINE PROPHET PERFORMANCE:
  15-minute MAE: {baseline_mae['15min']:.4f}
  Hourly MAE: {baseline_mae['hourly']:.4f}
  Daily MAE: {baseline_mae['daily']:.4f}
  Combined MAE: {baseline_mae['combined']:.4f}

ENSEMBLE MODEL PERFORMANCE:
  Model: Prophet ({ensemble_metrics['ensemble_weights']['prophet']:.1f}) + LightGBM ({ensemble_metrics['ensemble_weights']['lightgbm']:.1f})
  15-minute MAE: {ensemble_metrics['ensemble_mae']['15min']:.4f} ({ensemble_metrics['improvements']['15min']:.1f}% improvement)
  Hourly MAE: {ensemble_metrics['ensemble_mae']['hourly']:.4f} ({ensemble_metrics['improvements']['hourly']:.1f}% improvement)
  Daily MAE: {ensemble_metrics['ensemble_mae']['daily']:.4f} ({ensemble_metrics['improvements']['daily']:.1f}% improvement)
  Combined MAE: {ensemble_metrics['ensemble_mae']['combined']:.4f} ({ensemble_metrics['improvements']['combined']:.1f}% improvement)

DELIVERABLES COMPLETED:
  ✓ Baseline audit and accuracy gap report
  ✓ LightGBM model with weather/holiday features
  ✓ Ensemble model combining Prophet and LightGBM
  ✓ Airflow DAG for production deployment
  ✓ Comprehensive comparison report
  ✓ All code and documentation

STATUS: Milestone 1 Complete - Ready for payment
"""
        
        # Save report
        with open('results/airflow_delivery_report.txt', 'w') as f:
            f.write(report_content)
        
        print(report_content)
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to generate report: {str(e)}")

# Define task instances
start = DummyOperator(
    task_id='start',
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    provide_context=True,
    dag=dag,
)

calculate_baseline_task = PythonOperator(
    task_id='calculate_baseline',
    python_callable=calculate_baseline_mae,
    provide_context=True,
    dag=dag,
)

train_lightgbm_task = PythonOperator(
    task_id='train_lightgbm',
    python_callable=train_lightgbm,
    provide_context=True,
    dag=dag,
)

create_ensemble_task = PythonOperator(
    task_id='create_ensemble',
    python_callable=create_ensemble,
    provide_context=True,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_final_report,
    provide_context=True,
    dag=dag,
)

end = DummyOperator(
    task_id='end',
    dag=dag,
)

# Set task dependencies
start >> validate_data_task >> calculate_baseline_task >> train_lightgbm_task >> create_ensemble_task >> generate_report_task >> end