from contextlib import contextmanager
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from icecream import ic
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base

from tf.util_config import load_variable
from tf.util_logging import init_logging

logger = init_logging(__name__)

load_dotenv()

engine_url = load_variable("FORECAST_DATABASE_URL")

try:
    engine = create_engine(engine_url)
    logger.info(f"Connecting to {engine_url}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except SQLAlchemyError as session_error:
    logger.error(f"error creating engine to {engine_url}: {session_error}")
    raise


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope():
    db_gen = get_db()
    session = next(db_gen)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        # Clean up the session
        next(db_gen, None)


def get_location_corporation_id(location_id: int) -> int:
    sql_query = text("select corporation_id from location where id = :location_id")
    with session_scope() as session:
        try:
            session.begin()
            corporation_id = session.execute(sql_query, {"location_id": location_id}).scalar()
            session.commit()
            return corporation_id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)
            raise
        except Exception as e:
            session.rollback()
            logger.error(e)
            raise


def get_location_default_sales_type_id(location_id: int) -> int:
    sql_query = text("select default_sales_type_forecast from location where id = :location_id")
    with session_scope() as session:
        try:
            session.begin()
            corporation_id = session.execute(sql_query, {"location_id": location_id}).scalar()
            session.commit()
            return corporation_id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)
            raise
        except Exception as e:
            session.rollback()
            logger.error(e)
            raise

def get_sales_type_ids(corporation_id: int) -> list[tuple[int, str]]:
    sql_query = text("select id, level from sales_type where corporation_id=:corporation_id and forecast_enabled='t'")
    with session_scope() as session:
        try:
            session.begin()
            results = session.execute(sql_query, {'corporation_id': corporation_id})
            session.commit()
            sales_type_list = [row for row in results]
            return sales_type_list
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)
            raise
        except Exception as e:
            session.rollback()
            logger.error(e)
            raise


def get_department_ids(corporation_id: int) -> list[int]:
    sql_query = text('select id from department where corporation_id=:corporation_id')
    with session_scope() as session:
        try:
            session.begin()
            results = session.execute(sql_query, {'corporation_id': corporation_id})
            session.commit()
            department_id_list = [row[0] for row in results]
            return department_id_list
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)
            raise
        except Exception as e:
            session.rollback()
            logger.error(e)
            raise


def get_location_ids(corporation_id: int) -> list[int]:
    sql_query = text('select id from location where corporation_id=:corporation_id')
    with session_scope() as session:
        try:
            session.begin()
            results = session.execute(sql_query, {'corporation_id': corporation_id})
            session.commit()
            location_id_list = [row[0] for row in results]
            return location_id_list
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)
            raise
        except Exception as e:
            session.rollback()
            logger.error(e)
            raise


def get_location_time_shift(location_id: int) -> timedelta:
    sql_query = text('select start_time_interval from corporation c join location l on c.id=l.corporation_id '
                     'where l.id=:location_id')
    with session_scope() as session:
        try:
            session.begin()
            result = session.execute(sql_query, {'location_id': location_id})
            time_interval = result.scalar()
            session.commit()
            return time_interval
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error reading time interval: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error reading time interval: {e}")
            raise


def insert_actual(df_actual: pd.DataFrame, location_id: int, sales_type_id: int, department_id: int,
                  start_ds: datetime, end_ds: datetime) -> int:

    delete_query = text("delete from actual where location_id = :location_id and "
                        "sales_type_id = :sales_type_id and department_id = :department_id and "
                        "ds between :start_date and :delete_end_date")
    delete_query_params = {"location_id": location_id,
                           "sales_type_id": sales_type_id,
                           "department_id": department_id,
                           "start_date": start_ds,
                           "delete_end_date": end_ds}
    logger.info(f"{delete_query.text=}, {delete_query_params=}")
    insert_actual_query = ("INSERT INTO actual (ds, location_id, sales_type_id, department_id, y) "
                "VALUES (:ds, :location_id, :sales_type_id, :department_id, :y)" )
    with session_scope() as session:
        try:
            session.begin()
            session.execute(delete_query, delete_query_params)
            logger.info(f"df_actual len: {len(df_actual)}")
            actual_data = df_actual.to_dict(orient="records")
            if actual_data :
                session.execute(insert_actual_query, actual_data)
            session.commit()
            inserted_count = len(df_actual)
            return inserted_count
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)
        except Exception as e:
            session.rollback()
            logger.error(e)


def get_location_actual(location_id: int, start_date: datetime.date, end_date: datetime.date,
                        time_shift: timedelta) -> pd.DataFrame:

    start_ds = datetime.combine(start_date, time()) - time_shift
    end_ds = datetime.combine(end_date, time()) - time_shift + timedelta(minutes=1425)
    params = {"location_id": location_id,
              "start_ds": start_ds,
              "end_ds": end_ds}

    logger.info(f"params: {params}")
    sql_query = text("SELECT a.ds, a.sales_type_id, coalesce(a.department_id,0) as department_id, a.y "
                     "FROM actual a join sales_type st on a.sales_type_id=st.id and st.forecast_enabled='t' "
                     "LEFT JOIN department d on a.department_id=d.id and d.forecast_enabled='t' "
                     "WHERE a.location_id = :location_id and a.ds between :start_ds and :end_ds")

    with session_scope() as session:
        try:
            session.begin()
            result = session.execute(sql_query, params)
            df_location = pd.DataFrame(result.fetchall(), columns=result.keys())
            session.commit()
            first_timestamp = df_location["ds"].min()
            last_timestamp = df_location["ds"].max()
            logger.info(f"{len(df_location)} rows read for {start_ds} to {end_ds}, first_timestamp: {first_timestamp}, "
                        f"latest_date: {last_timestamp}")
            df_location["ds"] = pd.to_datetime(df_location["ds"]) - time_shift
            return df_location
        # TODO: differentiate SQLAlachemy Exceptions from general exceptions
        except Exception as e:
            session.rollback()
            logger.error(f"Error reading df_location or df_holiday: {e}")
            raise


def get_location_holidays(location_id: int) -> pd.DataFrame:
    sql_query = text("SELECT corporation_id, ds, holiday, lower_window, upper_window "
                     "FROM ml_get_location_holidays(:location_id)")

    with session_scope() as session:
        try:
            session.begin()
            result = session.execute(sql_query, {"location_id": location_id})
            df_holidays = pd.DataFrame(result.fetchall(), columns=result.keys())
            session.commit()
            return df_holidays
        except Exception as e:
            session.rollback()
            logger.error(f"Error reading df_location or df_holiday: {e}")
            raise

def get_location_hours(location_id: int) -> dict:
    """
    Fetches the operational hours for a location from the database.

    Args:
        location_id (int): The ID of the location.

    Returns:
        dict: A dictionary containing operational hours or None if not found.
    """
    sql_query = text("SELECT hours FROM location WHERE id = :location_id")
    with session_scope() as session:
        try:
            session.begin()
            result = session.execute(sql_query, {"location_id": location_id})
            hours = result.scalar()  # Fetch the `hours` JSONB data
            session.commit()
            return hours
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error fetching hours for location_id {location_id}: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error: {e}")
            raise

def insert_forecasts(df_forecast: pd.DataFrame, location_id: int, sales_type_id: int,
                     start_date: datetime.date, end_date: datetime.date, time_shift: timedelta):

    # timeshift back to the real chronological time
    df_forecast['ds'] = df_forecast['ds'] + time_shift
    df_forecast = df_forecast[df_forecast['yhat'] != 0]
    # department 0 should be stored as null.
    df_forecast.loc[df_forecast['yhat'] == 0, 'yhat'] = np.nan

    start_ds = datetime.combine(start_date, time()) + time_shift
    #calculate the next date
    start_ds_next_day = start_ds + timedelta(days=1)
    end_ds = datetime.combine(end_date, time()) + time_shift + timedelta(minutes=1425)
    delete_query = text("DELETE FROM forecast WHERE location_id=:location_id AND "
                        "sales_type_id=:sales_type_id and "
                        "ds>= :forecast_start_date and ds<=:forecast_end_date")
    delete_params = {"location_id": location_id,
                     "sales_type_id": sales_type_id,
                     "forecast_start_date": start_ds_next_day,
                     "forecast_end_date": end_ds}
    logger.debug(f"delete_query: {delete_query}, delete_params: {delete_params}")

    # do not save yhat values. forecast table should be sparse
    df_forecast = df_forecast[(df_forecast["yhat"] > 0) & (~pd.isna(df_forecast["yhat"]))]
    insert_query = ("INSERT INTO forecast (ds, location_id, sales_type_id, department_id, yhat) "
                "VALUES (:ds, :location_id, :sales_type_id, :department_id, :yhat)" )
    with session_scope() as session:
        try:
            session.begin()
            delete_result = session.execute(delete_query, delete_params)
            forecast_data = df_forecast.to_dict(orient="records")
            logger.info(f"forecast_data len: {len(forecast_data)}")
            if forecast_data:
                session.execute(insert_query, forecast_data)
            session.commit()
            logger.info(f"deleted {delete_result.rowcount} rows from forecast table for location:{location_id} and "
                        f"sales_type_id:{sales_type_id} between {start_ds} to {start_ds}.")
            logger.info(f"wrote {len(df_forecast)} forecast rows")
        except Exception as e:
            session.rollback()
            logger.error(f"Exception {e} during delete/add of forecasts")
            raise


def get_forecasts(location_id: int, start_ds: datetime, end_ds: datetime) -> pd.DataFrame:

    logger.info(f"location_id: {location_id} for ds between {start_ds} through {end_ds}")
    forecast_query = text("select ds, sales_type_id, department_id, yhat "
                          "from forecast where location_id=:location_id and ds between "
                          ":start_ds and :end_ds")
    forecast_params = {"location_id": location_id,
                       "start_ds": start_ds,
                       "end_ds": end_ds}

    with session_scope() as session:
        try:
            session.begin()
            logger.info(f"forecast_query: {forecast_query}, forecast_params: {forecast_params}")
            result = session.execute(forecast_query, forecast_params)
            forecast_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            session.commit()
            return forecast_df
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Transaction failed with {e}")
            raise

def get_location_postal_code(location_id: int) -> str|None:
    """
    Fetch the postal code for a given location ID.
    """
    sql_query = text("SELECT postal_code FROM location WHERE id = :location_id")
    with session_scope() as session:
        try:
            session.begin()
            postal_code = session.execute(sql_query, {"location_id": location_id}).scalar()
            session.commit()
            if not postal_code:
                logger.warning(f"No postal code found for location_id {location_id}")
            return postal_code
        except Exception as e:
            session.rollback()
            logger.error(f"Error fetching postal code for location_id {location_id}: {e}")
            return None


def query_missing_weather_dates(postal_code: str, start_date: datetime.date, end_date: datetime.date) -> list[datetime.date]:

    query = ("SELECT dates.generated_date "
             "FROM generate_series(:start_date, :end_date, '1 day'::interval) AS dates(generated_date)) "
             "LEFT JOIN weather w ON dates.generated_date = w.weather_date and  postal_code=:postal_code "
             "where w.postal_code is NULL")
    params = {"postal_code": postal_code,
              "start_date": start_date,
              "end_date": end_date}
    with session_scope() as session:
        result = session.execute(query, params)
        return [row.generated_date for row in result]


def get_weather_data(postal_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    logger.info(f"Fetching weather data for postal_code {postal_code} between {start_date} and {end_date}")
    with session_scope() as session:
        query = ("SELECT weather_date AS ds, real_feel, precipitation, coverage, snow "
                "FROM weather WHERE postal_code = :postal_code AND weather_date BETWEEN :start_date AND :end_date "
                "ORDER BY weather_date")
        params = {"postal_code": postal_code,
                  "start_date": start_date,
                  "end_date": end_date}
        result = session.execute(query, params)
        df_weather = pd.DataFrame(result.fetchall(), columns=result.keys())
    if df_weather.empty:
        logger.warning(f"No weather data found for postal_code {postal_code} between {start_date} and {end_date}")
    return df_weather

def get_weather_sliced_data(postal_code: str, start_date: datetime.date, end_date: datetime.date,
                            start_time: datetime.time, end_time: datetime.time) -> pd.DataFrame:
    logger.info(f"Fetching weather data for postal_code {postal_code} between {start_date} and {end_date}"
                "sliced by times: {start_time} and {end_time}")
    with session_scope() as session:
        query = ("SELECT ds::date, avg(real_feel) as real_feel, sum(precipitation) as precipitation, "
                 "avg(coverage) as coverage, sum(snow) as snow "
                 "FROM weather_hourly WHERE postal_code = :postal_code AND ds BETWEEN :start_date AND :end_date "
                 "AND ds::time BETWEEN :start_time AND :end_time "
                 "GROUP BY ds::date "
                 "ORDER BY ds::date")
        params = {"postal_code": postal_code,
                  "start_date": start_date,
                  "end_date": end_date,
                  "start_time": start_time,
                  "end_time": end_time}
        result = session.execute(query, params)
        df_weather = pd.DataFrame(result.fetchall(), columns=result.keys())
    if df_weather.empty:
        logger.warning(f"No weather data found for postal_code {postal_code} between {start_date} and {end_date}")
    return df_weather

def get_weather_for_regressors(postal_code: str, start_date: datetime.date, end_date: datetime.date,
                            start_time: datetime.time, end_time: datetime.time) -> pd.DataFrame:
    logger.info(f"Fetching weather data for postal_code {postal_code} between {start_date} and {end_date}"
                "sliced by times: {start_time} and {end_time}")
    with session_scope() as session:
        query = ("SELECT ds, real_feel, split_part(weather_code, ':', 1) as coverage_code, "
                 "split_part(weather_code, ':', 2) as intensity_code, split_part(weather_code, ':', 3) as weather_code "
                 "FROM weather_hourly WHERE postal_code = :postal_code AND ds BETWEEN :start_date AND :end_date "
                 "AND ds::time BETWEEN :start_time AND :end_time "
                 "ORDER BY ds::date")
        params = {"postal_code": postal_code,
                  "start_date": start_date,
                  "end_date": end_date,
                  "start_time": start_time,
                  "end_time": end_time}
        result = session.execute(query, params)
        df_weather = pd.DataFrame(result.fetchall(), columns=result.keys())
    if df_weather.empty:
        logger.warning(f"No weather data found for postal_code {postal_code} between {start_date} and {end_date}")
    return df_weather



def get_future_weather_data(postal_code: str, start_date: datetime.date, days: int) -> pd.DataFrame:
    end_date = start_date + timedelta(days=days)
    query = ("SELECT weather_date AS ds, real_feel, precipitation, coverage, snow "
             "FROM weather "
             "WHERE postal_code = :postal_code AND weather_date BETWEEN :start_date AND :end_date "
             "ORDER BY weather_date")
    params = {"postal_code": postal_code,
              "start_date": start_date,
              "end_date": end_date}

    with session_scope() as session:
        result = session.execute(query, params)
        df_future_weather = pd.DataFrame(result.fetchall(), columns=result.keys())
    if df_future_weather.empty:
        logger.warning(f"No future weather data found for postal_code {postal_code} from {start_date} to {end_date}")
    elif len(df_future_weather) < days:
        logger.info(f"Partial future weather data found for postal_code {postal_code} (from {start_date} to {end_date})")
    return df_future_weather

def get_usda_zone(postal_code: str) -> str|None:
    query = ("SELECT zone FROM postal_code_weather_zone "
             "WHERE postal_code = :postal_code")
    params = {"postal_code": postal_code}

    with session_scope() as session:
        result = session.execute(query, params).fetchone()
        if not result:
            logger.warning(f"No USDA zone found for postal_code {postal_code}")
        return result['zone'] if result else None

def get_prior_scales_for_zone(zone: str) -> dict:
    with session_scope() as session:
        query = "SELECT prior_scales FROM weather_zone WHERE zone = :zone"
        results = session.execute(query, {'zone': zone}).fetchone()
    if not results:
        logger.warning(f"No prior scales found for zone {zone}. Using default scales.")
    ic(results['prior_scales'])
    return results['prior_scales']
