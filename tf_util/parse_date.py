from datetime import datetime
def parse_date(date_string) -> datetime.date:
    try:
        return datetime.strptime(date_string, "%Y-%m-%d").date()
    except ValueError as e:
        msg = f"Not a valid date: '{date_string}'. Expected format: YYYY-mm-dd."
        raise e