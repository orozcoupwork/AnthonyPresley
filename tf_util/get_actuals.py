import argparse
from datetime import datetime, time, timedelta
from itertools import product

from tf.db import (get_location_ids, get_location_corporation_id, get_location_time_shift, get_sales_type_ids,
                   get_department_ids, insert_actual)
from tf.parse_date import parse_date
from tf.tf_api import get_sales_paf
from tf.util_config import load_env_variable
from tf.util_logging import init_logging

from dotenv import load_dotenv

logger = init_logging(__name__)


def get_corporation_actuals(corporation_id: int, start_date: datetime.date, end_date: datetime.date):

    location_ids = get_location_ids(corporation_id)

    for location_id in location_ids:
        get_location_actuals(location_id, start_date, end_date)


def get_location_actuals(location_id: int, start_date: datetime.date, end_date: datetime.date):

    load_dotenv() 
    logger.info(f'getting sales for location_id: {location_id}')
    time_shift_interval = get_location_time_shift(location_id)
    logger.info(f"{time_shift_interval=}")

    start_ds = datetime.combine(start_date, time()) + time_shift_interval
    end_ds = datetime.combine(end_date, time()) + time_shift_interval + timedelta(minutes=1425)

    corporation_id = get_location_corporation_id(location_id)
    sales_type_list = get_sales_type_ids(corporation_id)
    location_sales_type_list = [sale[0] for sale in sales_type_list if sale[1] == 'L']
    department_sales_type_list = [sale[0] for sale in sales_type_list if sale[1] == 'D']

    department_list = get_department_ids(corporation_id)

    # retrieve location-level actuals
    for sales_type_id in location_sales_type_list:
        logger.info(f"getting actuals for location_id: {location_id} sales_type_id: {sales_type_id}, "
                    f"department_id: None")
        paf_df = get_sales_paf(location_id, sales_type_id, None, start_ds, end_ds, time_shift_interval)
        print(paf_df)

    # retrieve department-level actuals
    for sales_type_id, department_id in product(department_sales_type_list, department_list):
        logger.info(f"getting actuals for location_id: {location_id} sales_type_id: {sales_type_id}, "
                    f"department_id: {department_id}")
        paf_df = get_sales_paf(location_id, sales_type_id, department_id, start_date, end_date, time_shift_interval)
        rows = insert_actual(paf_df, location_id, sales_type_id, department_id, start_ds, end_ds)
        logger.info(f'retrieved {rows} rows')

    # retrieve location-level actuals
    for sales_type_id in location_sales_type_list:
        logger.info(f"getting actuals for location_id: {location_id} sales_type_id: {sales_type_id}, "
                    f"department_id: ''")
        paf_df = get_sales_paf(location_id, sales_type_id, None, start_date, end_date, time_shift_interval)
        rows = insert_actual(paf_df, location_id, sales_type_id, None, start_ds, end_ds)
        logger.info(f'retrieved {rows} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='get_actuals', description='Retrieve actuals from TimeForge')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', '--corporation_id', type=int)
    group.add_argument('-l', '--location_id', type=int)
    parser.add_argument('-s', '--start_date', type=parse_date, required=True)
    parser.add_argument('-e', '--end_date', type=parse_date, required=True)
    args = parser.parse_args()

    if args.corporation_id:
        get_corporation_actuals(args.corporation_id, args.start_date, args.end_date)
    else:
        get_location_actuals(args.location_id, args.start_date, args.end_date)
