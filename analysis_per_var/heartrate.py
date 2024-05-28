import pandas as pd
import pandas as pd
import datetime
import pytz
def convert_unix_to_datetime(unix_timestamp):
    return datetime.datetime.fromtimestamp(unix_timestamp)


def add_date_to_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Create a dictionary that maps 'summaryId' to 'calendarDate' in df2
    date_dict = df2.set_index('summaryId')['calendarDate'].to_dict()

    # Add a new column 'date' to df1 by mapping 'dailiessummaryId' to 'calendarDate' using date_dict
    df1['date'] = df1['dailiessummaryId'].map(date_dict)
    # Ensure 'date' column is of datetime type
    df1['date'] = pd.to_datetime(df1['date'])

    # Convert 'timeOffsetHeartRateSample' from seconds to timedelta
    df1['time'] = pd.to_timedelta(df1['timeOffsetHeartRateSamples'], unit='s')

    # Add 'date' and 'timeOffsetHeartRateSample' to create a datetime representing the time of day
    df1['datetime'] = df1['date'] + df1['time']

    return df1.drop(columns='time')

def analyze_heart_rate(df, dailies):
    heart_rate_df = add_date_to_df(df, dailies)
    x = 5