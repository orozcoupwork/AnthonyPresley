from datetime import date, time, datetime

from tf.db import get_weather_for_regressors
import numpy as np
import pandas as pd


def get_weather_regressors(postal_code: str, start_date: datetime.date, end_date: datetime.date,
                           start_hour: datetime.time, end_hour: datetime.time) -> pd.DataFrame:

    # TODO: introduce lag period
    weather_df = get_weather_for_regressors(postal_code, start_date, end_date, start_hour, end_hour)

    conditions = [
        # Level 1: Very comfortable: 18–24°C
        ((weather_df['real_feel'] >= 18) & (weather_df['real_feel'] <= 24)),
        # Level 2: Slightly uncomfortable: Cool: 10–17°C or Warm: 25–30°C
        (((weather_df['real_feel'] >= 10) & (weather_df['real_feel'] < 18)) |
         ((weather_df['real_feel'] > 24) & (weather_df['real_feel'] <= 30))),
        # Level 3: Moderately uncomfortable: Cold: 0–9°C or Hot: 31–35°C
        (((weather_df['real_feel'] >= 0) & (weather_df['real_feel'] < 10)) |
         ((weather_df['real_feel'] > 30) & (weather_df['real_feel'] <= 35))),
        # Level 4: Very uncomfortable: Cold: -10 to -1°C or Hot: 36–40°C
        (((weather_df['real_feel'] >= -10) & (weather_df['real_feel'] <= 0)) |
         ((weather_df['real_feel'] > 35) & (weather_df['real_feel'] <= 40))),
        # Level 5: Extremely uncomfortable: ≤ -11°C or ≥ 41°C
        ((weather_df['real_feel'] < -10) | (weather_df['real_feel'] > 41))
    ]
    choices = [1, 2, 3, 4, 5]
    weather_df['comfort_level'] = np.select(conditions, choices, default=np.nan)

    mapping = {
        'VL': 1,  # very light
        'L': 2,   # light
        'H': 4,   # heavy
        'VH': 5   # very heavy
    }

    # Apply mapping: any code not found will default to 3 for moderate
    weather_df['intensity'] = weather_df['intensity_code'].apply(lambda x: mapping.get(x, 1))

    # Override intensity based on weather_code:
    # For weather_code in ['CL', 'FW', 'SC', 'BK'], set intensity to 0
    weather_df.loc[weather_df['weather_code'].isin(['CL', 'FW', 'SC', 'BK']), 'intensity'] = 0
    # For weather_code 'OV', set intensity to 1
    weather_df.loc[weather_df['weather_code'] == 'OV', 'intensity'] = 1

    weather_df.drop(columns=['real_feel', 'coverage_code', 'intensity_code', 'weather_code'], inplace=True)
    weather_df.set_index('ds', inplace=True)
    daily_weather_df = (
        weather_df
        .resample('D')
        .mean()
        .reset_index()
    )
    daily_weather_df['ds'] = daily_weather_df['ds'].dt.date
    daily_weather_df['intensity'] = daily_weather_df['intensity']/5
    daily_weather_df['comfort_level'] = daily_weather_df['comfort_level']/5
    return daily_weather_df

