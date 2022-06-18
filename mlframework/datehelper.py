"""Provides utility functions working with dates.
"""
__author__ = 'khanhtruong'
__date__ = '2022-06-18'


import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


def date_range(
    start: str = None,
    end: str = None,
    periods: int = None,
    freq: str = 'D',
    output_format: str = '%Y-%m-%d',
) -> list[str]:
    """Generate list of dates. Of the four parameters start, end, periods, and
        freq, exactly three must be specified.

    Args:
        start (str, optional): Starting date. Defaults to None.
        end (str, optional): Ending date. Defaults to None.
        periods (int, optional): Number of periods to generate. Defaults to None.
        freq (str, optional): Frequency of dates. Defaults to 'D'.
            'D': day
            'W': week end (Sunday)
            'M': month end
            'MS': month begin
            More detail at:
            https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        output_format (str, optional): [description]. Defaults to '%Y-%m-%d'. More detail at:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

    Returns:
        list[str]: Dates in the range.
    """
    dates = (
        pd.date_range(start=start, end=end, periods=periods, freq=freq)
        .strftime(output_format)
        .to_list()
    )
    return dates


def get_last_date_from_month(
    input: str,
    input_format: str = '%Y-%m',
    output_format: str = '%Y-%m-%d'
) -> str:
    """Get last date of a month.

    Args:
        input (str): Input date.
        input_format (str, optional): Input format. Defaults to '%Y-%m'.
        output_format (str, optional): Output format. Defaults to '%Y-%m-%d'.

    Returns:
        str: Last date of the month.
    """
    dinput = datetime.datetime.strptime(input, input_format)
    dlast = dinput + relativedelta(day=31)
    dlast = datetime.datetime.strftime(dlast, output_format)
    return dlast


def get_hours_minutes_seconds(timedelta: datetime.timedelta) -> tuple[int]:
    """Convert time delta to hours, minutes, seconds

    Args:
        timedelta (datetime.timedelta): time delta between two time points.
            Ex: datetime.timedelta(0, 9, 494935)

    Returns:
        tuple[int]: three integers corresponding to number of hours, minutes and seconds
    """
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds
