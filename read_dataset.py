import os
import pandas as pd
import pickle
data_folder = 'data'  # Folder path where the CSV files are located
data_dict = {}  # Dictionary to store the DataFrames

# Get a list of CSV files in the data folder
csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

# Loop through each CSV file and create a DataFrame
for file in csv_files:
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)
    data_dict[file.strip(".csv")] = df


output_file = 'data_dict.pickle'  # File path to save the pickle file
# Dump the dictionary into a pickle file
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)