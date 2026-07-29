"""
Temperature data handling module for FLiBe experiments.

This module provides classes to load and access temperature data from thermocouples
and dip heater setpoints, synchronized with the experiment timeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path


class TemperatureStream:
    """
    Handles temperature data from a single thermocouple.
    
    Provides interface similar to GasStream for consistency with tritium data.
    """
    
    def __init__(self, timestamps, temperatures, tc_label, start_time):
        """
        Initialize temperature stream.
        
        Args:
            timestamps: List of datetime objects
            temperatures: Array of temperature values in Celsius
            tc_label: Label for this thermocouple
            start_time: Experiment start time (datetime)
        """
        self.timestamps = np.array(timestamps)
        self.temperatures = np.array(temperatures)
        self.tc_label = tc_label
        self.start_time = start_time
        
        # Calculate relative times in days from start
        self.relative_times_days = np.array([
            (ts - start_time).total_seconds() / 86400.0 
            for ts in self.timestamps
        ])
    
    def get_temperature(self):
        """
        Get temperature array.
        
        Returns:
            numpy array of temperatures in Celsius
        """
        return self.temperatures
    
    def get_times_days(self):
        """
        Get relative times in days from experiment start.
        
        Returns:
            numpy array of times in days
        """
        return self.relative_times_days
    
    def get_times_absolute(self):
        """
        Get absolute timestamps.
        
        Returns:
            numpy array of datetime objects
        """
        return self.timestamps
    
    def __repr__(self):
        return f"TemperatureStream('{self.tc_label}', {len(self.temperatures)} points)"


class DipHeaterSetpoints:
    """
    Handles dip heater setpoint data with step changes.
    
    Interprets setpoints as instantaneous changes at specified times.
    """
    
    def __init__(self, setpoints, start_time):
        """
        Initialize dip heater setpoints.
        
        Args:
            setpoints: List of dicts with 'value', 'unit', and 'set_time'
            start_time: Experiment start time (datetime)
        """
        self.start_time = start_time
        self.setpoints = []
        
        for sp in setpoints:
            set_time = datetime.strptime(sp['set_time'], "%m/%d/%Y %H:%M")
            relative_time_days = (set_time - start_time).total_seconds() / 86400.0
            
            # Convert temperature to Celsius if needed
            temp = sp['value']
            if sp.get('unit', 'C') == 'F':
                temp = (temp - 32) * 5/9
            
            self.setpoints.append({
                'time_days': relative_time_days,
                'time_absolute': set_time,
                'temperature': temp
            })
        
        # Sort by time
        self.setpoints.sort(key=lambda x: x['time_days'])
    
    def get_temperature_at_time(self, time_days):
        """
        Get setpoint temperature at a specific time.
        
        Args:
            time_days: Time in days from experiment start (scalar or array)
            
        Returns:
            Temperature in Celsius (scalar or array)
        """
        if np.isscalar(time_days):
            return self._get_single_temp(time_days)
        else:
            return np.array([self._get_single_temp(t) for t in time_days])
    
    def _get_single_temp(self, time_days):
        """Get temperature for a single time point."""
        # Find the most recent setpoint before or at this time
        temp = self.setpoints[0]['temperature']  # Default to first setpoint
        
        for sp in self.setpoints:
            if sp['time_days'] <= time_days:
                temp = sp['temperature']
            else:
                break
        
        return temp
    
    def get_step_function(self, time_array_days):
        """
        Get step function values for plotting.
        
        Args:
            time_array_days: Array of times in days
            
        Returns:
            Array of temperatures corresponding to each time
        """
        return self.get_temperature_at_time(time_array_days)
    
    def get_setpoint_times(self):
        """
        Get times when setpoints change.
        
        Returns:
            Array of times in days
        """
        return np.array([sp['time_days'] for sp in self.setpoints])
    
    def get_setpoint_values(self):
        """
        Get setpoint temperature values.
        
        Returns:
            Array of temperatures in Celsius
        """
        return np.array([sp['temperature'] for sp in self.setpoints])
    
    def __repr__(self):
        return f"DipHeaterSetpoints({len(self.setpoints)} setpoints)"


def load_temperature_data(temp_csv_path=None, general_json_path=None):
    """
    Load temperature data from CSV and general.json.
    
    Args:
        temp_csv_path: Path to processed temperature CSV (optional)
        general_json_path: Path to general.json (optional)
        
    Returns:
        dict with keys:
            - TC streams as 'TC1', 'TC2', 'TC3', 'TC4' (labeled)
            - 'dip_heater': DipHeaterSetpoints object
            - 'start_time': experiment start time
    """
    # Default paths
    if general_json_path is None:
        general_json_path = "../../data/general.json"
    
    # Load general.json for metadata
    with open(general_json_path, 'r') as f:
        general_data = json.load(f)

    if temp_csv_path is None:
        run_nb = general_data['general_data']['run_nb']
        temp_csv_path = (
            Path(general_json_path).parent
            / "temperature"
            / "processed"
            / f"FLiBe_1L_run_{run_nb}_temp.csv"
        )
    
    # Get start time
    start_time_str = general_data['timestamps']['run_start']
    start_time = datetime.strptime(start_time_str, "%m/%d/%Y %H:%M")
    
    # Get TC labels
    temp_config = general_data['general_data']['temperature']['logs']
    tc_labels = {
        'TC1': temp_config['TC1_label'],
        'TC2': temp_config['TC2_label'],
        'TC3': temp_config['TC3_label'],
        'TC4': temp_config['TC4_label']
    }
    
    # Load temperature CSV
    df = pd.read_csv(temp_csv_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%m/%d/%Y %H:%M')
    
    # Create TemperatureStream objects for each TC
    temp_streams = {}
    for tc_col in ['TC1', 'TC2', 'TC3', 'TC4']:
        temp_streams[tc_col] = TemperatureStream(
            timestamps=df['DATETIME'].tolist(),
            temperatures=df[tc_col].values,
            tc_label=tc_labels[tc_col],
            start_time=start_time
        )
    
    # Create DipHeaterSetpoints object
    dip_setpoints = general_data['general_data']['temperature']['dip_setpoints']
    dip_heater = DipHeaterSetpoints(dip_setpoints, start_time)

    # Create Filter heater setpoints object (same structure as dip setpoints)
    filter_setpoints = general_data['general_data']['temperature'].get('filter_heaters', [])
    filter_heater = DipHeaterSetpoints(filter_setpoints, start_time)
    
    return {
        'TC1': temp_streams['TC1'],
        'TC2': temp_streams['TC2'],
        'TC3': temp_streams['TC3'],
        'TC4': temp_streams['TC4'],
        'dip_heater': dip_heater,
        'filter_heater': filter_heater,
        'start_time': start_time,
        'tc_labels': tc_labels
    }


# Convenience function to create a time array for plotting
def create_time_array(start_days=0, end_days=20, num_points=1000):
    """
    Create a time array for plotting.
    
    Args:
        start_days: Start time in days
        end_days: End time in days
        num_points: Number of points
        
    Returns:
        numpy array of times in days
    """
    return np.linspace(start_days, end_days, num_points)
