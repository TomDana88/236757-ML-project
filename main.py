import pickle
from datetime import timedelta
import itertools
import pandas as pd
from cluster_the_data import preprocess_data as ps



input_file_sleep = 'sleeping.pkl'  # File path of the pickle file
with open(input_file_sleep, 'rb') as f:
    loaded_dict_sleep = pickle.load(f)


input_file = 'data_dict.pickle'  # File path of the pickle file
with open(input_file, 'rb') as f:
    loaded_dict = pickle.load(f)
sleep_keys = ['rem_sleep', 'heart_rate_daily', 'deep_sleep', 'dailies_summary', 'leep_summary', 'awake_sleep', 'light_sleep','epoch']

keys_to_keep = ['heart_rate_daily','light_sleep','awake_sleep','deep_sleep','epoch',"dailies_summary"]
filtered_data = {key: loaded_dict[key] for key in sleep_keys if key in loaded_dict}
ps(filtered_data)


