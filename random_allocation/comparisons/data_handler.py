"""
Data handling utilities for experiments.

This module provides functions for saving and loading experiment data
with full preservation of all information.
"""

import os
import json
from typing import Dict, List, Union, Any, Optional, Collection, cast

import numpy as np
import pandas as pd

# Define type aliases locally to avoid circular imports
MethodList = List[str]
DataDict = Dict[str, Any]


def save_experiment_data(data: DataDict, methods: MethodList, experiment_name: str) -> None:
    """
    Save experiment data with complete information.
    
    Args:
        data: The experiment data dictionary
        methods: List of methods used in the experiment
        experiment_name: Name of the experiment for the output file (full path)
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(experiment_name), exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'x data': data['x data'].tolist() if isinstance(data['x data'], np.ndarray) else data['x data'],
        'x name': data.get('x name', ''),
        'y name': data.get('y name', ''),
        'title': data.get('title', ''),
        'y data': {}
    }
    
    # Process y data for each method
    for method in methods:
        if method in data['y data']:
            save_data['y data'][method] = data['y data'][method].tolist() if isinstance(data['y data'][method], np.ndarray) else data['y data'][method]
        
        # Save standard deviation data if available
        std_key = method + '- std'
        if std_key in data['y data']:
            save_data['y data'][std_key] = data['y data'][std_key].tolist() if isinstance(data['y data'][std_key], np.ndarray) else data['y data'][std_key]
    
    # Save any additional parameters stored in the data dict
    for key, value in data.items():
        if key not in ['x data', 'y data', 'x name', 'y name', 'title']:
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(value, np.ndarray):
                save_data[key] = value.tolist()
            else:
                save_data[key] = value
    
    # Save as JSON for complete data preservation
    with open(f"{experiment_name}.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Also save as CSV for compatibility
    df_data: Dict[str, Union[Collection[float], str]] = {'x': data['x data']}
    
    # Add y data for each method to the DataFrame
    for method in methods:
        if method in data['y data']:
            df_data[method] = data['y data'][method]
        
        std_key = method + '- std'
        if std_key in data['y data']:
            df_data[f"{method}_std"] = data['y data'][std_key]
    
    # Include additional metadata
    df_data['title'] = data.get('title', '')
    df_data['x name'] = data.get('x name', '')
    df_data['y name'] = data.get('y name', '')
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(df_data)
    df.to_csv(experiment_name, index=False)


def load_experiment_data(experiment_name: str, methods: MethodList) -> Optional[DataDict]:
    """
    Load experiment data with complete information.
    
    Args:
        experiment_name: Name of the experiment file (full path, without extension)
        methods: List of methods used in the experiment
    
    Returns:
        The loaded experiment data dictionary or None if file doesn't exist
    """
    # First try to load from JSON which has complete information
    json_file = f"{experiment_name}.json"
    if os.path.exists(json_file):
        print(f"Reading data from {json_file}")
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Convert loaded data to correct format
        data: DataDict = {
            'x data': np.array(loaded_data['x data']) if 'x data' in loaded_data else np.array([]),
            'y data': {},
            'x name': loaded_data.get('x name', ''),
            'y name': loaded_data.get('y name', ''),
            'title': loaded_data.get('title', '')
        }
        
        # Process y data
        if 'y data' in loaded_data:
            for method, values in loaded_data['y data'].items():
                data['y data'][method] = np.array(values)
        
        # Add any additional parameters
        for key, value in loaded_data.items():
            if key not in ['x data', 'y data', 'x name', 'y name', 'title']:
                # Convert list back to numpy array if needed
                if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    data[key] = np.array(value)
                else:
                    data[key] = value
        
        return data
    
    # Fallback to CSV if JSON doesn't exist
    csv_file = experiment_name
    if os.path.exists(csv_file):
        print(f"Reading data from {csv_file}")
        # Read the CSV and convert to expected DataDict format
        df = pd.read_csv(csv_file)
        
        csv_data: DataDict = {'y data': {}, 'x data': np.array(df['x'].tolist())}
        
        for method in methods:
            if method in df.columns:
                csv_data['y data'][method] = df[method].values
            if f"{method}_std" in df.columns:
                csv_data['y data'][method + '- std'] = df[f"{method}_std"].values
        
        # Extract metadata
        if 'title' in df.columns and not df.empty:
            csv_data['title'] = df['title'].iloc[0] 
        if 'x name' in df.columns and not df.empty:
            csv_data['x name'] = df['x name'].iloc[0]
        if 'y name' in df.columns and not df.empty:
            csv_data['y name'] = df['y name'].iloc[0]
        
        return csv_data
    
    # If neither file exists, return None
    return None 