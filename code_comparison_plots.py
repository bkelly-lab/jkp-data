#portfolio code comparison
import pandas as pd
import polars as pl
import os
import matplotlib.pyplot as plt

# Specify folder paths
new_folder = 'path'   # Replace with your path for parquet files
old_folder = 'path'   # Replace with your path for csv files
output_folder = 'path'  # Replace with the path where you want to save images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# # List all files in both folders
# new_files = {os.path.splitext(f)[0]: os.path.join(new_folder, f) for f in os.listdir(new_folder) if f.endswith('.csv')}
# old_files = {os.path.splitext(f)[0]: os.path.join(old_folder, f) for f in os.listdir(old_folder) if f.endswith('.csv')}

new_files={}
new_files['USA'] = 'file.csv'
new_files['WORLD_EX_US'] = 'file.csv'

old_files={}
old_files['USA'] = 'old_file.csv'
old_files['WORLD_EX_US'] = 'old_file.csv'

# Iterate over all countries (files present in both new and old folders)
for country in new_files.keys() & old_files.keys():
    # Read the new parquet file and old CSV file
    new_data = pd.read_csv(new_files[country])
    old_data = pd.read_csv(old_files[country])
    
    # Ensure the required columns exist
    if {'characteristic', 'eom', 'ret_vw'}.issubset(new_data.columns) and {'characteristic', 'eom', 'ret_vw'}.issubset(old_data.columns):
        # Convert 'eom' columns to datetime in both datasets
        new_data['eom'] = pd.to_datetime(new_data['eom'])
        old_data['eom'] = pd.to_datetime(old_data['eom'])
        
        # Perform the join on 'characteristic' and 'eom'
        merged_data = new_data.merge(old_data, on=['characteristic', 'eom'], suffixes=('_new', '_old'))

        # Calculate correlations grouped by 'characteristic'
        correlations = merged_data.groupby('characteristic').apply(
            lambda x: x['ret_vw_new'].corr(x['ret_vw_old'])
        ).dropna()

        # Sort the characteristics alphabetically
        correlations = correlations.sort_index()

        # Calculate average 'ret_vw' for each characteristic
        avg_new = merged_data.groupby('characteristic')['ret_vw_new'].mean()
        avg_old = merged_data.groupby('characteristic')['ret_vw_old'].mean()

        # Create a combined plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 40))

        # Plot the histogram of correlations for this country on the first subplot
        correlations.plot(kind='bar', ax=ax1)
        ax1.set_xlabel('Characteristics (sorted alphabetically)', fontsize=10)
        ax1.set_ylabel('Correlation', fontsize=10)
        ax1.set_title(f'Correlations for {country}', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)

        # Create a scatter plot for average values on the second subplot
        ax2.scatter(avg_old, avg_new, alpha=0.7)
        # Add characteristic names as labels to the points
        for char in avg_new.index:
            ax2.text(avg_old[char], avg_new[char], char, fontsize=8, alpha=0.7)

        # Add a y = x line for reference
        min_val = min(min(avg_old), min(avg_new))
        max_val = max(max(avg_old), max(avg_new))
        ax2.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', linewidth=1)
        ax2.set_xlabel('Average Ret_VW (Old)', fontsize=10)
        ax2.set_ylabel('Average Ret_VW (New)', fontsize=10)
        ax2.set_title(f'Average Comparison for {country}', fontsize=12)
        ax2.grid(True)

        # Adjust layout and save the figure to the output folder
        plt.tight_layout()
        output_path = os.path.join(output_folder, f'{country}_comparison.png')
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory

        print(f'Saved combined plot for {country} at {output_path}')
