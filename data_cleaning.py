"""Data Cleaning Module for Traffic Accident Analysis

This module handles data loading, cleaning, and preprocessing for traffic accident data.
It includes functions for handling missing values, data type conversions, and basic
data quality checks.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Load traffic accident data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def check_data_quality(df):
    """
    Perform initial data quality checks.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing quality metrics
    """
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    print("\n=== Data Quality Report ===")
    print(f"Total Rows: {quality_metrics['total_rows']:,}")
    print(f"Total Columns: {quality_metrics['total_columns']}")
    print(f"Missing Values: {quality_metrics['missing_values']:,}")
    print(f"Duplicate Rows: {quality_metrics['duplicate_rows']:,}")
    print(f"Memory Usage: {quality_metrics['memory_usage_mb']:.2f} MB")
    
    return quality_metrics

def handle_missing_values(df, strategy='drop', threshold=0.5):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('drop', 'fill', 'smart')
        threshold (float): Threshold for dropping columns (% of missing values)
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print(f"\nHandling missing values (strategy: {strategy})...")
    initial_shape = df.shape
    
    # Remove columns with too many missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    print(f"Shape after handling missing values: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)")
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"\nRemoved {removed:,} duplicate rows")
    else:
        print("\nNo duplicate rows found")
    
    return df

def standardize_columns(df):
    """
    Standardize column names and data types.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with standardized columns
    """
    print("\nStandardizing column names...")
    
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    
    # Attempt to convert date columns
    date_keywords = ['date', 'time', 'timestamp']
    for col in df.columns:
        if any(keyword in col for keyword in date_keywords):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"Converted {col} to datetime")
            except:
                pass
    
    return df

def clean_traffic_data(filepath, missing_strategy='fill', save_output=True):
    """
    Main function to clean traffic accident data.
    
    Args:
        filepath (str): Path to the raw data file
        missing_strategy (str): Strategy for handling missing values
        save_output (bool): Whether to save cleaned data
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("="*60)
    print("Starting Data Cleaning Pipeline")
    print("="*60)
    
    # Load data
    df = load_data(filepath)
    
    # Check initial quality
    quality_before = check_data_quality(df)
    
    # Standardize columns
    df = standardize_columns(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Final quality check
    print("\n" + "="*60)
    print("Final Data Quality Report")
    print("="*60)
    quality_after = check_data_quality(df)
    
    # Save cleaned data
    if save_output:
        output_path = filepath.replace('.csv', '_cleaned.csv')
        df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Data Cleaning Complete!")
    print("="*60)
    
    return df

if __name__ == "__main__":
    # Example usage
    print("Data Cleaning Module")
    print("To use this module, import it and call clean_traffic_data()")
    print("\nExample:")
    print("  from data_cleaning import clean_traffic_data")
    print("  df_clean = clean_traffic_data('traffic_accidents.csv')")
