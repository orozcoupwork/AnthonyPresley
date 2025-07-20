# lightgbm_forecast.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def create_features(df, weather_df, holiday_df):
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
    
    # Lag features
    for lag in [1, 7, 14, 21, 28]:
        df[f'lag_{lag}d'] = df.groupby(['location_id', 'sales_type_id', 'department_id'])['y'].shift(lag * 96)  # 96 = 24*4 (15-min intervals)
    
    # Rolling features
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}d'] = df.groupby(['location_id', 'sales_type_id', 'department_id'])['y'].transform(
            lambda x: x.rolling(window * 96, min_periods=1).mean()
        )
        df[f'rolling_std_{window}d'] = df.groupby(['location_id', 'sales_type_id', 'department_id'])['y'].transform(
            lambda x: x.rolling(window * 96, min_periods=1).std()
        )
    
    # Merge weather features
    df['date'] = df['ds'].dt.date
    weather_daily = weather_df.groupby(weather_df['ds'].dt.date).agg({
        'real_feel': 'mean',
        'precipitation': 'sum',
        'coverage': 'mean',
        'snow': 'sum'
    }).reset_index()
    weather_daily.columns = ['date', 'temp_mean', 'precip_sum', 'coverage_mean', 'snow_sum']
    df = df.merge(weather_daily, on='date', how='left')
    
    # Holiday features
    holiday_df['date'] = holiday_df['ds'].dt.date
    holiday_binary = holiday_df[['corporation_id', 'date', 'holiday']].copy()
    holiday_binary['is_holiday'] = 1
    
    # Merge holidays through location -> corporation mapping
    location_corp = data['location'][['id', 'corporation_id']].rename(columns={'id': 'location_id'})
    df = df.merge(location_corp, on='location_id', how='left')
    df = df.merge(holiday_binary, left_on=['corporation_id', 'date'], right_on=['corporation_id', 'date'], how='left')
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
        'num_threads': -1
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

# Prepare data for all locations
all_predictions = []

for location_id in data['location']['id'].unique():
    print(f"\nProcessing location {location_id}")
    
    # Filter data for this location
    location_sales = data['sales'][data['sales']['location_id'] == location_id].copy()
    location_weather = data['weather'][data['weather']['postal_code'].isin(
        data['location'][data['location']['id'] == location_id]['postal_code'].values
    )].copy()
    
    # Create features
    location_sales = create_features(location_sales, location_weather, data['holiday'])
    
    # Remove NaN values from lag features
    location_sales = location_sales.dropna(subset=[col for col in location_sales.columns if 'lag_' in col])
    
    # Define features (exclude target and identifiers)
    feature_cols = [col for col in location_sales.columns if col not in ['y', 'ds', 'location_id', 'sales_type_id', 'department_id', 'date', 'corporation_id']]
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    
    for train_idx, val_idx in tscv.split(location_sales):
        train_df = location_sales.iloc[train_idx]
        val_df = location_sales.iloc[val_idx]
        
        # Train model
        model = train_lightgbm_model(train_df, val_df, feature_cols)
        
        # Make predictions
        predictions = model.predict(val_df[feature_cols])
        val_df['yhat_lgb'] = predictions
        val_df['yhat_lgb'] = val_df['yhat_lgb'].clip(lower=0)  # No negative predictions
        
        all_predictions.append(val_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat_lgb']])

# Combine all predictions
lgb_predictions_df = pd.concat(all_predictions, ignore_index=True)
lgb_predictions_df.to_csv('results/lightgbm_predictions.csv', index=False)