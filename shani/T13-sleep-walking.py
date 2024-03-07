import pickle
from datetime import timedelta
import itertools
import pandas as pd


def sleep_analysis_13(df):
    import pandas as pd

    data_days = df

    # Step 1: Convert durationInSeconds to minutes
    data_days['durationInMinutes'] = data_days['durationInSeconds'] / 60

    # Calculate the average duration per day
    average_duration_per_day = data_days.groupby('ID')['durationInMinutes'].mean().reset_index()

    # Calculate the average duration per week
    data_days['calendarDate'] = pd.to_datetime(data_days['calendarDate'])
    data_days['week'] = data_days['calendarDate'].dt.isocalendar().week
    average_duration_per_week = data_days.groupby(['ID', 'week'])['durationInMinutes'].mean().reset_index()

    # Calculate the number of full weeks per ID
    full_weeks_count = data_days.groupby('ID')['week'].nunique().reset_index()
    full_weeks_count.rename(columns={'week': 'FullWeeks'}, inplace=True)

    # Calculate the number of partial weeks and their date ranges per ID
    partial_weeks = data_days[data_days['calendarDate'].dt.dayofweek != 0]
    partial_weeks_count = partial_weeks.groupby('ID')['week'].nunique().reset_index()
    partial_weeks_count.rename(columns={'week': 'PartialWeeks'}, inplace=True)

    partial_weeks_dates = partial_weeks.groupby(['ID', 'week'])['calendarDate'].agg(['min', 'max']).reset_index()
    partial_weeks_dates['PartialWeeksDates'] = partial_weeks_dates.apply(
        lambda row: f"{row['min'].strftime('%d-%m-%Y')}:{row['max'].strftime('%d-%m-%Y')}"
        if not pd.isna(row['min']) and not pd.isna(row['max']) else '', axis=1)
    partial_weeks_dates = partial_weeks_dates.groupby('ID')['PartialWeeksDates'].agg(list).reset_index()

    # Calculate the number of missing weeks and their date ranges per ID
    all_weeks = pd.DataFrame(data_days['week'].drop_duplicates())
    all_combinations = all_weeks.merge(data_days['ID'].drop_duplicates(), how='cross')
    missing_weeks = pd.merge(all_combinations, data_days, on=['ID', 'week'], how='left', indicator=True)
    missing_weeks = missing_weeks[missing_weeks['_merge'] == 'left_only'].drop(columns=['_merge'])
    missing_weeks_dates = missing_weeks.groupby(['ID', 'week'])['calendarDate'].agg(['min', 'max']).reset_index()
    missing_weeks_dates['MissingWeeks'] = missing_weeks_dates.apply(
        lambda row: f"{row['min'].strftime('%d-%m-%Y')}:{row['max'].strftime('%d-%m-%Y')}"
        if not pd.isna(row['min']) and not pd.isna(row['max']) else '', axis=1)
    missing_weeks_dates = missing_weeks_dates.groupby('ID')['MissingWeeks'].agg(list).reset_index()

    # Merge all the calculated statistics into the final output DataFrame
    output_df = average_duration_per_day.merge(average_duration_per_week.pivot(index='ID', columns='week', values='durationInMinutes'),
                                               on='ID')
    output_df = output_df.merge(full_weeks_count, on='ID', how='left').fillna(0)
    output_df = output_df.merge(partial_weeks_count, on='ID', how='left').fillna(0)
    output_df = output_df.merge(partial_weeks_dates, on='ID', how='left').fillna('')
    output_df = output_df.merge(missing_weeks_dates, on='ID', how='left').fillna('')

    # Convert the week numbers to date ranges in column names
    output_df.rename(columns={col: f"Week {pd.to_datetime('2023-1-1') + pd.to_timedelta(col * 7 - 1, 'D')}:"
                                   f"{pd.to_datetime('2023-1-1') + pd.to_timedelta(col * 7 + 5, 'D')}"
                              for col in output_df.columns if isinstance(col, int)}, inplace=True)

    # Save the final output DataFrame to a CSV file
    output_df.to_csv('output_statistics.csv', index=False)


def calc_daily(df):
    # Step 1: Calculate the overall average distanceInMeters for each ID
    overall_average_distance = df.groupby('ID')['distanceInMeters'].mean().reset_index()
    overall_average_distance.rename(columns={'distanceInMeters': 'average_daily_total'}, inplace=True)
    overall_average_distance['average_daily_total'] = overall_average_distance['average_daily_total'].round(2)

    # Step 2: Convert 'calendarDate' to a datetime object
    df['calendarDate'] = pd.to_datetime(df['calendarDate'])

    # Step 3: Calculate the week number for each 'calendarDate'
    df['week'] = df['calendarDate'].dt.strftime('%U')

    # Step 4: Calculate the average distanceInMeters per week for each unique week present in the data
    weekly_average_distance = df.groupby(['ID', 'week'])['distanceInMeters'].mean().reset_index()
    weekly_average_distance['distanceInMeters'] = weekly_average_distance['distanceInMeters'].round(2)

    # Step 5: Calculate the total distance covered per week for each ID
    weekly_total_distance = df.groupby(['ID', 'week'])['distanceInMeters'].sum().reset_index()
    weekly_total_distance['distanceInMeters'] = weekly_total_distance['distanceInMeters'].round(2)

    # Step 6: Pivot the weekly_average_distance and weekly_total_distance DataFrames with explicit column names
    weekly_pivot = weekly_average_distance.pivot(index='ID', columns='week', values='distanceInMeters')
    weekly_pivot.columns = [f'Week_{col}_avg' for col in weekly_pivot.columns]

    weekly_total_pivot = weekly_total_distance.pivot(index='ID', columns='week', values='distanceInMeters')
    weekly_total_pivot.columns = [f'Week_{col}_total' for col in weekly_total_pivot.columns]

    # Step 7: Merge the overall average distanceInMeters, weekly averages, and weekly totals based on 'ID'
    final_df = overall_average_distance.merge(weekly_pivot, on='ID').merge(weekly_total_pivot, on='ID')

    # Step 8: Sort columns to have average_daily_total followed by weekly averages and totals for each week
    sorted_columns = ['ID', 'average_daily_total'] + [col for col in sorted(final_df.columns) if col not in ['ID', 'average_daily_total']]
    final_df = final_df[sorted_columns]
    final_df.to_csv('daily_walking_statistics.csv', index=False)
    print(final_df)


sleep_summary = pd.read_csv('/Users/barakgahtan/PycharmProjects/injuries-wearables/sleep_summary_fix (1).csv')
daily_summart = pd.read_csv('/Users/barakgahtan/PycharmProjects/injuries-wearables/daily_summary.csv')
# sleep_analysis_13(sleep_summary)
calc_daily(daily_summart)
x = 5
