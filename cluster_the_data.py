import pickle
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict

import analysis_per_var.heartrate
from analysis_per_var.epoch_walking import calc_epoch_running_walking


def generate_stats(df):
    # Assuming your DataFrame is named df and the column is 'timestamp_column'
    # Convert the 'timestamp_column' to pandas datetime format
    df['startTimeDate'] = pd.to_datetime(df['startTimeDate'])
    date_counts = df['startTimeDate'].dt.date.value_counts()
    unique_days = len(date_counts)
    data_points_per_day = df.groupby(df['startTimeDate'].dt.date)[
        'startTimeDate'].nunique()
    # Calculate the average number of timings per day
    average_data_points_per_day = data_points_per_day.mean()
    return unique_days, average_data_points_per_day


def hist_visualize_stats_per_all(data_dict):
    x_values = [inner_dict['deep_sleep_avg_datapoints_per_day']
                for inner_dict in data_dict.values()]
    # Create a histogram to visualize the distribution
    plt.hist(x_values, bins=50, edgecolor='black')
    plt.xlabel(' deep_sleep_avg_datapoints_per_day')
    plt.ylabel('soldiers count')
    plt.title('Distribution of deep_sleep_avg_datapoints_per_day per day')
    plt.savefig(
        'Distribution of deep_sleep_avg_datapoints_per_day per day.png', dpi=300)
    plt.show()


def calc_epoch_statistics(data, unique_ids):
    df = data['epoch']
    import pandas as pd
    import numpy as np

    # Assuming df is your DataFrame and it has columns ['userId', 'steps', 'distanceInMeters', 'startTimeDate']

    # Convert distanceInMeters to km
    df['distanceInKm'] = df['distanceInMeters'] / 1000

    # Parse startTimeDate to datetime
    df['startTimeDate'] = pd.to_datetime(df['startTimeDate'])
    df['day'] = df['startTimeDate'].dt.date
    df['week'] = df['startTimeDate'].dt.week

    # Define the hour bins with labels
    bins = np.arange(0, 25, 1)
    labels = [
        f'{str(i).zfill(2)}:00-{str(i + 1).zfill(2)}:00' for i in bins[:-1]]
    df['hour_bin'] = pd.cut(df['startTimeDate'].dt.hour, bins=bins,
                            labels=labels, include_lowest=True, right=False)

    # Calculate daily total
    daily_total = df.groupby(['userId', 'day']).agg(
        {'steps': ['sum'], 'distanceInKm': ['sum']}).reset_index()
    daily_total.columns = ['userId', 'day', 'daily_steps', 'daily_distance']

    # Calculate the number of days with zero distance for each user
    zero_distance_days = df[df['distanceInKm'] == 0].groupby('userId')['day'].nunique(
    ).reset_index().rename(columns={'day': 'zero_distance_days'})

    # Calculate the number of days with non-zero steps and distance for each user
    non_zero_days = df[df['distanceInKm'] != 0].groupby(
        'userId')['day'].nunique().reset_index().rename(columns={'day': 'non_zero_days'})
    non_zero_steps_days = df[df['steps'] != 0].groupby('userId')['day'].nunique(
    ).reset_index().rename(columns={'day': 'non_zero_steps_days'})

    # Calculate weekly total
    weekly_total = df.groupby(['userId', 'week']).agg(
        {'steps': ['sum'], 'distanceInKm': ['sum']}).reset_index()
    weekly_total.columns = ['userId', 'week',
                            'weekly_steps', 'weekly_distance']

    # Calculate the number of unique days and weeks for each user
    num_days = df.groupby('userId')['day'].nunique(
    ).reset_index().rename(columns={'day': 'num_days'})
    num_weeks = df.groupby('userId')['week'].nunique(
    ).reset_index().rename(columns={'week': 'num_weeks'})

    # Join the total steps and distance with the number of days and weeks and zero distance days
    daily = daily_total.groupby('userId').agg(
        {'daily_steps': ['sum'], 'daily_distance': ['sum']}).reset_index()
    daily.columns = ['userId', 'total_steps', 'total_distance']
    daily = daily.merge(num_days, on='userId', how='left')
    daily = daily.merge(non_zero_days, on='userId', how='left')
    daily = daily.merge(non_zero_steps_days, on='userId', how='left')
    daily = daily.merge(zero_distance_days, on='userId', how='left')

    weekly = weekly_total.groupby('userId').agg(
        {'weekly_steps': ['sum'], 'weekly_distance': ['sum']}).reset_index()
    weekly.columns = ['userId', 'total_steps_week', 'total_distance_week']
    weekly = weekly.merge(num_weeks, on='userId', how='left')

    # Calculate the averages
    daily['avg_steps_per_day'] = daily['total_steps'] / daily['num_days']
    daily['avg_steps_per_day_non_zero'] = daily['total_steps'] / \
        daily['non_zero_steps_days']
    daily['avg_distance_per_day'] = daily['total_distance'] / daily['num_days']
    daily['avg_distance_per_day_non_zero'] = daily['total_distance'] / \
        daily['non_zero_days']

    weekly['avg_steps_per_week'] = weekly['total_steps_week'] / \
        weekly['num_weeks']
    weekly['avg_distance_per_week'] = weekly['total_distance_week'] / \
        weekly['num_weeks']

    # Calculate time bins
    hourly = df.groupby(['userId', 'hour_bin']).agg(
        {'steps': ['mean'], 'distanceInKm': ['mean']}).reset_index()
    hourly.columns = ['userId', 'hour_bin',
                      'avg_steps_hourly', 'avg_distance_hourly']

    # Convert 'hour_bin' to string
    hourly['hour_bin'] = hourly['hour_bin'].astype(str)

    # Pivot the hourly dataframe
    hourly_pivot = hourly.pivot_table(index='userId', columns='hour_bin', values=[
                                      'avg_steps_hourly', 'avg_distance_hourly']).reset_index()
    hourly_pivot.columns = [f'{y}_{x}' if y !=
                            '' else x for x, y in hourly_pivot.columns]

    # Merge all dataframes
    result = pd.merge(daily, weekly, on='userId', how='outer')
    result = pd.merge(result, hourly_pivot, on='userId', how='outer')
    result.to_csv("epoch-stats.csv", index=False)
    x = 5

    # Load the result DataFrame from the CSV file
    df = pd.read_csv("epoch-stats.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # List of users
    users = df['userId'].unique()

    # Hour bins
    hour_bins = [
        f'{str(i).zfill(2)}:00-{str(i + 1).zfill(2)}:00' for i in range(24)]

    # Loop through each user
    for user in users:
        user_data = df[df['userId'] == user]

        # Plot for average steps per hour
        user_data.filter(regex='avg_steps_hourly').mean(
            axis=0).plot(ax=axes[0], marker='o', label=user)

    axes[0].set_title('Average Steps per Hour for Each User')
    axes[0].set_xticks(range(len(hour_bins)))
    axes[0].set_xticklabels(hour_bins, rotation=45)
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Average Steps')
    # axes[0].legend()

    for user in users:
        user_data = df[df['userId'] == user]

        # Plot for average distance in km per hour
        user_data.filter(regex='avg_distance_hourly').mean(
            axis=0).plot(ax=axes[1], marker='o', label=user)

    axes[1].set_title('Average Distance in Km per Hour for Each User')
    axes[1].set_xticks(range(len(hour_bins)))
    axes[1].set_xticklabels(hour_bins, rotation=45)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Average Distance in Km')
    # axes[1].legend()

    plt.tight_layout()  # To prevent overlap of the plots
    plt.savefig('Average Distance in Km per Hour for Each User.png', dpi=300)
    plt.show()
    x = 5

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot for average steps per day
    result.groupby('userId')['avg_steps_per_day'].mean().plot(
        kind='bar', ax=axes[0], color='g')
    axes[0].set_title('Average Steps per Day for Each User')
    axes[0].set_ylabel('Average Steps')
    axes[0].set_xticklabels([])  # Remove x-axis labels

    # Plot for average steps per week
    result.groupby('userId')['avg_steps_per_week'].mean().plot(
        kind='bar', ax=axes[1], color='b')
    axes[1].set_title('Average Steps per Week for Each User')
    axes[1].set_ylabel('Average Steps')
    axes[1].set_xticklabels([])  # Remove x-axis labels

    plt.tight_layout()  # To prevent overlap of the plots

    # Save the figure
    plt.savefig('average_steps_per_day_and_week.png', dpi=300)

    plt.show()

    return result


def create_sleep_state_dataframe(unique_dict_ids):
    new_dataframes = {}

    for id, sleep_data in unique_dict_ids.items():
        # Create a time index for the entire day, one entry per hour
        time_index = pd.date_range('2023-01-01', periods=24, freq='H')

        # Initialize the new DataFrame with the time index
        new_df = pd.DataFrame(index=time_index, columns=['sleep_state'])
        new_df.index.name = 'time'

        # Map sleep states for each hour
        for state in ['light_sleep', 'awake_sleep', 'deep_sleep']:
            df = sleep_data[state]
            for _, row in df.iterrows():
                start_time = pd.Timestamp(
                    '2023-01-01') + pd.Timedelta(seconds=row['startTimeInSeconds'])
                end_time = pd.Timestamp(
                    '2023-01-01') + pd.Timedelta(seconds=row['endTimeInSeconds'])
                new_df.loc[start_time:end_time, 'sleep_state'] = state

        # Fill any missing values with 'awake'
        new_df['sleep_state'].fillna('awake', inplace=True)

        # Add the new DataFrame to the dictionary
        new_dataframes[id] = new_df

    return new_dataframes


def unify_sleep_data(light_sleep, awake_sleep, deep_sleep):
    # Function to process each sleep DataFrame
    def process_sleep_df(df, sleep_state, unified_df):
        for index, row in df.iterrows():
            start_minute = row['startTimeInSeconds'] // 60
            end_minute = row['endTimeInSeconds'] // 60
            date = row['startTimeDate']
            for minute in range(start_minute, end_minute):
                minute_of_hour = minute % 60
                unified_df = unified_df.append({'Date': str(date).split(" ")[0],
                                                'Hour': date.hour,
                                                'Minute': minute_of_hour,
                                                'SleepState': sleep_state}, ignore_index=True)

        return unified_df

    # Create a new DataFrame for the unified data
    unified_sleep = pd.DataFrame(
        columns=['Date', 'Hour', 'Minute', 'SleepState'])

    # Process each sleep DataFrame
    unified_sleep = process_sleep_df(light_sleep, 'Light Sleep', unified_sleep)
    unified_sleep = process_sleep_df(awake_sleep, 'Awake', unified_sleep)
    unified_sleep = process_sleep_df(deep_sleep, 'Deep Sleep', unified_sleep)

    # Fill in missing minutes
    unique_dates = unified_sleep['Date'].unique()
    for date in unique_dates:
        for hour in range(24):
            for minute in range(60):
                if not ((unified_sleep['Date'] == date) &
                        (unified_sleep['Hour'] == hour) &
                        (unified_sleep['Minute'] == minute)).any():
                    unified_sleep = unified_sleep.append(
                        {'Date': date, 'Hour': hour, 'Minute': minute, 'SleepState': '?'}, ignore_index=True)

    # Sort the DataFrame
    unified_sleep = unified_sleep.sort_values(
        by=['Date', 'Hour', 'Minute']).reset_index(drop=True)

    return unified_sleep


def sleep_by_weeks(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # Define a custom function to calculate the week number with Sunday as the first day
    def get_week_number(date):
        # Adjust the date so that Sunday is the first day of the week
        adjusted_date = date - pd.Timedelta(days=(date.weekday() + 1) % 7)
        # Return the ISO week number of the adjusted date
        return adjusted_date.isocalendar()[1]

    # Apply the custom function to calculate the week number for each row
    df['WeekNumber'] = df['Date'].apply(get_week_number)
    # Initialize an empty dictionary to hold the DataFrames for each week
    weekly_dataframes = {}
    # Group by 'WeekNumber' and create a DataFrame for each week
    for week_number, group in df.groupby('WeekNumber'):
        weekly_dataframes[week_number] = group
    return weekly_dataframes


def preprocess_data(data):
    data['epoch'] = data['epoch'][['userId', 'userAccessToken', 'summaryId', 'activeKilocalories',
                                   'steps', 'distanceInMeters', 'activeTimeInSeconds', 'startTimeInSeconds']]
    data['deep_sleep']['totalSeconds'] = data['deep_sleep']['endTimeInSeconds'] - \
        data['deep_sleep']['startTimeInSeconds']
    data['light_sleep']['totalSeconds'] = data['light_sleep']['endTimeInSeconds'] - \
        data['light_sleep']['startTimeInSeconds']
    data['awake_sleep']['totalSeconds'] = data['awake_sleep']['endTimeInSeconds'] - \
        data['awake_sleep']['startTimeInSeconds']
    data['epoch']['startTimeDate'] = pd.to_datetime(
        data['epoch']['startTimeInSeconds'], unit='s')
    data['deep_sleep']['startTimeDate'] = pd.to_datetime(
        data['deep_sleep']['startTimeInSeconds'], unit='s')
    data['light_sleep']['startTimeDate'] = pd.to_datetime(
        data['light_sleep']['startTimeInSeconds'], unit='s')
    data['awake_sleep']['startTimeDate'] = pd.to_datetime(
        data['awake_sleep']['startTimeInSeconds'], unit='s')
    unique_ids = data['heart_rate_daily']['userId'].unique()
    unique_dict_ids = {}

    # analysis_per_var.heartrate.analyze_heart_rate(data['heart_rate_daily'],data['dailies_summary'])

    # calc_epoch_statistics(data, unique_ids)
    # calc_epoch_running_walking(data,unique_ids)
    for id in unique_ids:
        filtered_light_sleep = data['light_sleep'][data['light_sleep']['userId'] == id] \
            .reset_index().drop(columns=['index', 'summaryId'])
        filtered_awake_sleep = data['awake_sleep'][data['awake_sleep']['userId'] == id] \
            .reset_index().drop(columns=['index', 'summaryId'])
        filtered_deep_sleep = data['deep_sleep'][data['deep_sleep']['userId'] == id] \
            .reset_index().drop(columns=['index', 'summaryId'])
        filtered_rem_sleep = data['rem_sleep'][data['rem_sleep']['userId'] == id] \
            .reset_index().drop(columns=['index', 'summaryId'])
        filtered_epoch = data['epoch'][data['epoch']['userId'] == id] \
            .reset_index().drop(columns=['index', 'userAccessToken', 'summaryId'])
        filtered_hr = data['heart_rate_daily'][data['heart_rate_daily']['userId'] == id] \
            .reset_index().drop(columns=['index', 'dailiessummaryId'])

        epoch_unique_days, epoch_avg_datapoints__per_day = generate_stats(filtered_epoch)
        deep_sleep_unique_days, deep_sleep_avg_datapoints__per_day = generate_stats(filtered_deep_sleep)
        awake_sleep_unique_days, awake_sleep_avg_datapoints_per_day = generate_stats(filtered_awake_sleep)
        light_sleep_unique_days, light_sleep_avg_datapoints_per_day = generate_stats(filtered_light_sleep)
        rem_sleep_unique_days, rem_sleep_avg_datapoints_per_day = generate_stats(filtered_rem_sleep)

        unique_dict_ids[id] = {
            'light_sleep': filtered_light_sleep,
            'awake_sleep': filtered_awake_sleep,
            'deep_sleep': filtered_deep_sleep,
            'rem_sleep': filtered_rem_sleep,
            'epoch': filtered_epoch,
            'heart_rate': filtered_hr,
            'epoch_unique_days': epoch_unique_days,
            'epoch_avg_datapoints__per_day': epoch_avg_datapoints__per_day,
            'deep_sleep_unique_days': deep_sleep_unique_days,
            'deep_sleep_avg_datapoints_per_day': deep_sleep_avg_datapoints__per_day,
            'awake_sleep_unique_days': awake_sleep_unique_days,
            'awake_sleep_avg_datapoints_per_day': awake_sleep_avg_datapoints_per_day,
            'light_sleep_unique_days': light_sleep_unique_days,
            'light_sleep_avg_datapoints_per_day': light_sleep_avg_datapoints_per_day,
            'rem_sleep_unique_days': rem_sleep_unique_days,
            'rem_sleep_avg_datapoints_per_day': rem_sleep_avg_datapoints_per_day
        }

        unified_sleep = unify_sleep_data(unique_dict_ids[id]['light_sleep'],
                                         unique_dict_ids[id]['awake_sleep'],
                                         unique_dict_ids[id]['deep_sleep'])
        # Convert 'Date' column to datetime
        unique_dict_ids[id]['sleep_by_minutes'] = unified_sleep
        weekly_dataframes = sleep_by_weeks(unified_sleep)
        unique_dict_ids[id]['sleep_by_weeks'] = weekly_dataframes
        x = 6

    with open('sleeping.pkl', 'wb') as file:
        pickle.dump(unique_dict_ids, file)
    x = 6
    x = 5

    # hist_visualize_stats_per_all(unique_dict_ids)
