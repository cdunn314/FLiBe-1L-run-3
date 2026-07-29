"""
Script to process and combine temperature CSV files.

This script:
1. Reads all CSV files from data/temperature
2. Combines them into a single dataframe
3. Sorts by date and time
4. Reformats DATE and TIME columns into a single DATETIME column (MM/DD/YYYY HH:MM in 24h format)
5. Saves the processed data to data/temperature/processed/
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime


def load_general_json(json_path):
    """Load the general.json file and extract run_nb."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['general_data']['run_nb']


def process_temperature_csvs(data_dir, output_dir, run_nb):
    """
    Process temperature CSV files.
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing the input CSV files
    output_dir : Path
        Directory to save the processed CSV
    run_nb : int
        Run number from general.json
    """
    # Get all CSV files from the temperature data directory
    csv_files = list(data_dir.glob('*.CSV')) + list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # Read and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(
            csv_file,
            header=None,
            names=['DATE', 'TIME', 'TC1', 'TC2', 'TC3', 'TC4']
        )
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal rows before processing: {len(combined_df)}")
    
    # Create a datetime column by combining DATE and TIME
    # Convert to datetime object first to handle the conversion properly
    combined_df['datetime_obj'] = pd.to_datetime(
        combined_df['DATE'] + ' ' + combined_df['TIME'],
        format='%m/%d/%Y %I:%M:%S %p'
    )
    
    # Sort by datetime
    combined_df = combined_df.sort_values('datetime_obj')
    
    # Format as MM/DD/YYYY HH:MM (24-hour format)
    combined_df['DATETIME'] = combined_df['datetime_obj'].dt.strftime('%m/%d/%Y %H:%M')
    
    # Create final dataframe with desired columns
    final_df = combined_df[['DATETIME', 'TC1', 'TC2', 'TC3', 'TC4']]
    
    # Remove duplicate DATETIME entries, keeping only the first occurrence
    rows_before = len(final_df)
    final_df = final_df.drop_duplicates(subset=['DATETIME'], keep='first')
    rows_after = len(final_df)
    
    if rows_before > rows_after:
        print(f"Removed {rows_before - rows_after} duplicate timestamp(s)")
    
    print(f"Total rows after processing: {len(final_df)}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_dir / f"FLiBe_1L_run_{run_nb}_temp.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\nProcessed data saved to: {output_file}")
    print(f"Date range: {final_df['DATETIME'].iloc[0]} to {final_df['DATETIME'].iloc[-1]}")


def main():
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'temperature'
    output_dir = project_root / 'data' / 'temperature' / 'processed'
    general_json_path = project_root / 'data' / 'general.json'
    
    # Load run number from general.json
    run_nb = load_general_json(general_json_path)
    print(f"Processing temperature data for Run #{run_nb}\n")
    
    # Process the CSV files
    process_temperature_csvs(data_dir, output_dir, run_nb)


if __name__ == '__main__':
    main()
