from datetime import datetime, timedelta

import argparse
import json
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import random
import requests

from tf.db import get_location_time_shift, get_forecasts
from tf.parse_date import parse_date
from tf.util_config import load_env_variable
from tf.util_logging import init_logging

logger = init_logging(__name__)


def put_forecast(location_id: int, start_date: datetime.date, end_date: datetime.date):
    tfapi_base_url = load_env_variable("TFAPI_BASE_URL")

    business_start_time = get_location_time_shift(location_id)

    # end_ds is +2 days because belongs_to can extend beyond midnight
    start_ds = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
    end_ds = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0) + timedelta(days=2)

    forecast_df = get_forecasts(location_id, start_ds, end_ds)
    forecast_df['yhat'] = forecast_df['yhat'].round(2)
    unique_df = forecast_df[['department_id', 'sales_type_id']].drop_duplicates()

    for department_id, sales_type_id in zip(unique_df.department_id, unique_df.sales_type_id):
        logger.info(f"sales_type_id: {sales_type_id}, department_id: {department_id}")
        lsd_df = forecast_df.loc[(forecast_df['department_id'] == department_id) &
                                 (forecast_df['sales_type_id'] == sales_type_id), ['ds', 'yhat']]
        lsd_df.set_index('ds', inplace=True)

        # set min and max timestamps for the forecasts so that resampling 15 minutes generates the full range
        # of 96 intervals for each day. The tfapi post forecast requires all 96 intervals for each day.
        timestamps = []
        yhat_values = []
        if lsd_df.index.min() > start_ds:
            timestamps.append(start_ds)
            yhat_values.append(0)
        if lsd_df.index.max() < end_ds:
            timestamps.append(end_ds)
            yhat_values.append(0)
        ends_df = pd.DataFrame({"ds": timestamps, "yhat": yhat_values})
        ends_df['ds'] = pd.to_datetime(ends_df['ds'])
        ends_df.set_index("ds", inplace=True)
        lsd_df = pd.concat([lsd_df, ends_df])
        # sort and resample dataframe with midnight to midnight values so that all 96 intervals exist
        lsd_df.sort_index()
        lsd_df = lsd_df.resample('15T').asfreq().fillna(0)

        # if business_start_time is not midnight (0 hours), shift ds between midnight and business_start_time to
        # the previous day to match TimeForge belongs_to logic
        if business_start_time != timedelta(0):
            lsd_df.index = lsd_df.index.map(lambda x: shift_timestamps(x, business_start_time))
        # Only post the desired forecast range. The +2 day range is no longer needed after the above timeshift
        post_end_ds = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0) + timedelta(days=1)
        lsd_df = lsd_df[(lsd_df.index >= start_ds) & (lsd_df.index < post_end_ds)]

        # build a dataframe from which the json payload can be generated. Payload is array of
        # {"belongs_to": "YYYY-mm-dd", "forecasts": [f1, f2, f3... f96], "location_id": <location_id>,
        #  "sales_type_id": <sales_type_id>, "department_id": <department_id>}
        grouped_df = lsd_df.groupby([lsd_df.index.date])['yhat'].apply(list).rename('forecasts').reset_index()
        grouped_df = grouped_df.rename(columns={'index': 'belongs_to'})
        grouped_df['belongs_to'] = grouped_df['belongs_to'].astype(str)
        grouped_df['location_id'] = location_id
        grouped_df['sales_type_id'] = sales_type_id
        grouped_df['department_id'] = department_id
        payload = grouped_df.to_dict("records")

        # send 0.0 values as nulls (spaces) to save space in sales_paf tables
        for belongs_to in payload:
            belongs_to['forecasts'] = ['' if x == 0.0 else x for x in belongs_to['forecasts']]
        logger.info(json.dumps(payload))
        reply = post_forecasts(location_id, payload, tfapi_base_url)
        logger.info(f"post reply: {reply}")

    # update the last forecast and crossval timestamps TF
    reply = update_tf_last_forecast_date(location_id, tfapi_base_url)
    logger.info(f"put forecast_time_stamp reply: {reply}")


def shift_timestamps(ts, start_time):
    if ts.time() < (pd.Timestamp('2000-01-01') + start_time).time():
        return ts - pd.Timedelta(days=1)
    else:
        return ts


def post_forecasts(location_id, payload, tfapi_base_url) -> str:
    url = f"{tfapi_base_url}/sales_paf/forecasts/{location_id}"
    logger.debug(f"url: {url}")
    try:
        r = requests.post(url, json=payload)
        if 200 <= r.status_code <= 299:
            s = json.dumps(r.json())
        else:
            logger.error(f"Request failed with status {r.status_code}")
            s = json.dumps(r.json())
        return s if s else '<empty reply>'
    except Exception as e:
        logger.error(f"Error posting to {url}: {e}")
        raise


def update_tf_last_forecast_date(location_id, base_url) -> str:
    # post last crossval and forecast
    now = datetime.now()
    random_minutes = random.randint(8, 16)
    # crossval is a "fake" time to satisfy the timeforge api
    last_crossval_time = now - timedelta(minutes=random_minutes)
    body = {"last_crossval_timestamp": last_crossval_time.strftime('%Y-%m-%d %H:%M:%S'),
            "last_forecast_timestamp": now.strftime('%Y-%m-%d %H:%M:%S')}
    url = f"{base_url}location/{location_id}"
    logger.info(f"put last_crossval_timestamp {body['last_crossval_timestamp']} "
                f"and last_forecast_timestamp {body['last_forecast_timestamp']} to url: {url}")
    try:
        r = requests.put(url, json=body)
        if 200 <= r.status_code <= 299:
            s = json.dumps(r.json())
        else:
            logger.error(f"Request failed with status {r.status_code}")
            s = json.dumps(r.json())
        return s if s else '<empty reply>'
    except Exception as e:
        logger.error(f"Error posting forecast to {url}: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='put_forecast', description='Put forecasts to tfapi')
    parser.add_argument('-l', '--location_id', type=int, default=None)
    parser.add_argument('-s', '--start_date', type=parse_date, required=True)
    parser.add_argument('-e', '--end_date', type=parse_date, required=True)
    args = parser.parse_args()
    put_forecast(args.location_id, args.start_date, args.end_date)
