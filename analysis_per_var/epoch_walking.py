import numpy as np
import pandas as pd
def calc_epoch_running_walking(data, unique_ids):
    import pandas as pd
    df = data['epoch'][['userId', 'distanceInMeters','startTimeDate']]
    # Assuming df is your DataFrame

    # Convert 'startTimeDate' to datetime if it's not
    df['startTimeDate'] = pd.to_datetime(df['startTimeDate'])

    # Extract date, hour and 30 minute interval from 'startTimeDate'
    df['date'] = df['startTimeDate'].dt.date
    df['half_hour'] = df['startTimeDate'].dt.hour + df['startTimeDate'].dt.minute // 30 / 2

    # Calculate speed in meters per 15 minutes
    df['speed'] = df['distanceInMeters'] / ((df['startTimeDate'].dt.minute % 15) + 1)

    # Split into rows with speed above and below threshold
    above_threshold = df[df['speed'] > 1875]
    below_threshold = df[df['speed'] <= 1875]

    # Group by userId, date and half_hour, then sum distance
    above_threshold_summed = above_threshold.groupby(['userId', 'date', 'half_hour'])['distanceInMeters'].sum().reset_index()
    below_threshold_summed = below_threshold.groupby(['userId', 'date', 'half_hour'])['distanceInMeters'].sum().reset_index()

    # Group by userId and half_hour, then calculate average distance
    above_threshold_averaged = above_threshold_summed.groupby(['userId', 'half_hour'])['distanceInMeters'].mean()
    below_threshold_averaged = below_threshold_summed.groupby(['userId', 'half_hour'])['distanceInMeters'].mean()

    # Initialize empty dictionary to hold results
    result_dict = {}

    # Iterate over each user
    for user in df['userId'].unique():
        # Initialize inner dictionary for this user
        user_dict = {}

        # Iterate over each half_hour
        for half_hour in np.arange(0, 24, 0.5):
            # Get average distance above and below threshold for this half_hour
            avg_distance_above = above_threshold_averaged.get((user, half_hour), 0)
            avg_distance_below = below_threshold_averaged.get((user, half_hour), 0)

            # Add to inner dictionary
            # Add to inner dictionary
            user_dict[half_hour] = (avg_distance_above if not np.isnan(avg_distance_above) else 0,
                                    avg_distance_below if not np.isnan(avg_distance_below) else 0)

        # Add inner dictionary to result_dict
        result_dict[user] = user_dict

    # Initialize list for storing rows
    data_rows = []

    # Loop through each user
    for user in result_dict:
        # Loop through each half_hour interval
        for half_hour in result_dict[user]:
            # Extract the distances above and below threshold
            dist_above, dist_below = result_dict[user][half_hour]

            # Add this information to our data_rows list
            data_rows.append([user, half_hour, dist_above, dist_below])

    # Convert the data_rows list into a DataFrame
    df_result = pd.DataFrame(data_rows, columns=['userId', 'half_hour', 'dist_above', 'dist_below'])

    # Save DataFrame to CSV
    df_result.to_csv('result.csv', index=False)
    # Initialize two dictionaries to hold daily totals for each user
    daily_totals_above = {user: 0 for user in result_dict.keys()}
    daily_totals_below = {user: 0 for user in result_dict.keys()}

    # For each user
    for user in result_dict.keys():
        # For each half hour for this user
        for half_hour in result_dict[user].keys():
            # Get the distances above and below threshold
            dist_above, dist_below = result_dict[user][half_hour]

            # Convert speed from m/15min to km/h
            speed_above = dist_above * 4 / 1000
            speed_below = dist_below * 4 / 1000

            # If speed is above 7.5 km/h, add to daily total above threshold
            if speed_above > 7.5:
                daily_totals_above[user] += dist_above

            # If speed is below or at 7.5 km/h, add to daily total below threshold
            if speed_below <= 7.5:
                daily_totals_below[user] += dist_below

    # Count number of unique dates for each user to calculate the average
    unique_dates = df['date'].nunique()

    # Divide by number of unique dates to get average daily distances
    average_daily_above = {user: total / unique_dates / 1000 for user, total in daily_totals_above.items()}
    average_daily_below = {user: total / unique_dates / 1000 for user, total in daily_totals_below.items()}

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, time

    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Create an ordered list of users for the x-axis
    users = list(result_dict.keys())

    # For the above threshold plot
    # Get the distances for the users in the same order as `users`
    distances_above = [average_daily_above[user] for user in users]
    axs[0].bar(users, distances_above, color='b', alpha=0.7)

    # Set the labels and title for the first plot
    axs[0].set_xlabel('Users')
    axs[0].set_ylabel('Average daily distance above threshold (km)')
    axs[0].set_title('Distances above threshold')

    # For the below threshold plot
    # Get the distances for the users in the same order as `users`
    distances_below = [average_daily_below[user] for user in users]
    axs[1].bar(users, distances_below, color='r', alpha=0.7)

    # Set the labels and title for the second plot
    axs[1].set_xlabel('Users')
    axs[1].set_ylabel('Average daily distance below threshold (km)')
    axs[1].set_title('Distances below threshold')
    # Remove x-axis labels for both plots
    axs[0].set_xticklabels(['' for _ in users])
    axs[1].set_xticklabels(['' for _ in users])
    # Show the plots
    plt.tight_layout()
    plt.savefig("averagerunnnigndistance.png", dpi=300)
    plt.show()

    # import matplotlib.pyplot as plt
    # import matplotlib.dates as mdates
    # from datetime import datetime, time
    #
    # # Define a function to convert half hours to datetime.datetime format
    # def half_hours_to_datetime(half_hours_list):
    #     times_list = []
    #     for half_hour in half_hours_list:
    #         hour = int(half_hour)
    #         minute = int((half_hour % 1) * 60)
    #         times_list.append(datetime.combine(datetime.today(), time(hour, minute)))
    #     return times_list
    #
    # # Set up the figure and subplots
    # fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    #
    # # For each user, plot their speeds above threshold
    # for user in result_dict.keys():
    #     # Get the half_hours and speeds for this user
    #     half_hours = half_hours_to_datetime([half_hour for half_hour in result_dict[user].keys()])
    #     # Convert speed from m/15min to km/h
    #     speeds_above = [dist[0] * 4 / 1000 for dist in result_dict[user].values()]
    #
    #     axs[0].plot(half_hours, speeds_above)
    #
    # # Set the labels and title for the first plot
    # axs[0].set_xlabel('Time of the day')
    # axs[0].set_ylabel('Average speed above threshold (km/h)')
    # axs[0].set_title('Speeds above threshold')
    # axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #
    # # For each user, plot their speeds below or at threshold
    # for user in result_dict.keys():
    #     # Get the half_hours and speeds for this user
    #     half_hours = half_hours_to_datetime([half_hour for half_hour in result_dict[user].keys()])
    #     # Convert speed from m/15min to km/h
    #     speeds_below = [dist[1] * 4 / 1000 for dist in result_dict[user].values()]
    #
    #     axs[1].plot(half_hours, speeds_below)
    #
    # # Set the labels and title for the second plot
    # axs[1].set_xlabel('Time of the day')
    # axs[1].set_ylabel('Average speed below or at threshold (km/h)')
    # axs[1].set_title('Speeds below or at threshold')
    # axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #
    # # For each user, plot their speeds under or equal to 17 km/h
    # for user in result_dict.keys():
    #     # Get the half_hours and speeds for this user
    #     half_hours = half_hours_to_datetime([half_hour for half_hour in result_dict[user].keys()])
    #     # Convert speed from m/15min to km/h and filter out speeds above 17 km/h
    #     speeds = [(dist[0] * 4 / 1000 if dist[0] * 4 / 1000 <= 17 else None,
    #                dist[1] * 4 / 1000 if dist[1] * 4 / 1000 <= 17 else None) for dist in result_dict[user].values()]
    #
    #     speeds_under_17 = [(speed if speed is not None else None) for speed in speeds]
    #
    #     half_hours, speeds_under_17 = zip(*[(half_hour, speed) for half_hour, speed in zip(half_hours, speeds_under_17) if speed is not None])
    #
    #     axs[2].plot(half_hours, speeds_under_17)
    #
    # # Set the labels and title for the third plot
    # axs[2].set_xlabel('Time of the day')
    # axs[2].set_ylabel('Average speed under or at 17 km/h (km/h)')
    # axs[2].set_title('Everything under 17 km/h')
    # axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #
    # # Show the plots
    # plt.tight_layout()
    # plt.savefig("speed.png", dpi=300)
    # plt.show()
def preprocess_data(data):
    data['epoch'] = data['epoch'][['userId','userAccessToken','summaryId','activeKilocalories','steps','distanceInMeters','activeTimeInSeconds', 'startTimeInSeconds']]
    data['deep_sleep']['totalSeconds'] = data['deep_sleep']['endTimeInSeconds'] - data['deep_sleep']['startTimeInSeconds']
    data['light_sleep']['totalSeconds'] = data['light_sleep']['endTimeInSeconds'] - data['light_sleep']['startTimeInSeconds']
    data['awake_sleep']['totalSeconds'] = data['awake_sleep']['endTimeInSeconds'] - data['awake_sleep']['startTimeInSeconds']
    data['epoch']['startTimeDate'] = pd.to_datetime(data['epoch']['startTimeInSeconds'], unit='s')
    data['deep_sleep']['startTimeDate'] = pd.to_datetime(data['deep_sleep']['startTimeInSeconds'], unit='s')
    data['light_sleep']['startTimeDate'] = pd.to_datetime(data['light_sleep']['startTimeInSeconds'], unit='s')
    data['awake_sleep']['startTimeDate'] = pd.to_datetime(data['awake_sleep']['startTimeInSeconds'], unit='s')
    unique_ids = data['heart_rate_daily']['userId'].unique()
    unique_dict_ids = {}
    # calc_epoch_statistics(data, unique_ids)
    calc_epoch_running_walking(data,unique_ids)