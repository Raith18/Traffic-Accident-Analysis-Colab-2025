"""Exploratory Data Analysis (EDA) Module for Traffic Accident Analysis

This module performs comprehensive exploratory data analysis on traffic accident data,
including statistical summaries, distribution analysis, and pattern identification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def generate_data_summary(df):
    """
    Generate comprehensive data summary statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing summary statistics
    """
    print("="*60)
    print("DATA SUMMARY REPORT")
    print("="*60)
    
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    print(f"\nDataset Shape: {summary['shape'][0]:,} rows × {summary['shape'][1]} columns")
    print(f"Memory Usage: {summary['memory_usage_mb']:.2f} MB")
    
    print("\n" + "-"*60)
    print("Column Information:")
    print("-"*60)
    
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        unique = df[col].nunique()
        
        print(f"{col:30} | {str(dtype):15} | Non-null: {non_null:8,} | Unique: {unique:6,} | Missing: {null_pct:5.1f}%")
    
    return summary

def analyze_numeric_features(df):
    """
    Analyze numeric features in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns found in the dataset.")
        return
    
    print("\n" + "="*60)
    print("NUMERIC FEATURES ANALYSIS")
    print("="*60)
    
    print(f"\nFound {len(numeric_cols)} numeric features: {', '.join(numeric_cols)}")
    
    print("\n" + "-"*60)
    print("Descriptive Statistics:")
    print("-"*60)
    
    desc = df[numeric_cols].describe()
    print(desc.to_string())
    
    # Additional statistics
    print("\n" + "-"*60)
    print("Additional Statistics:")
    print("-"*60)
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            cv = (data.std() / data.mean() * 100) if data.mean() != 0 else 0
            
            print(f"\n{col}:")
            print(f"  Skewness: {skewness:.3f} {'(right-skewed)' if skewness > 0 else '(left-skewed)' if skewness < 0 else '(symmetric)'}")
            print(f"  Kurtosis: {kurtosis:.3f} {'(heavy-tailed)' if kurtosis > 0 else '(light-tailed)'}")
            print(f"  Coefficient of Variation: {cv:.2f}%")

def analyze_categorical_features(df, max_categories=20):
    """
    Analyze categorical features in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        max_categories (int): Maximum number of categories to display
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found in the dataset.")
        return
    
    print("\n" + "="*60)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*60)
    
    print(f"\nFound {len(categorical_cols)} categorical features: {', '.join(categorical_cols)}")
    
    for col in categorical_cols:
        print("\n" + "-"*60)
        print(f"Feature: {col}")
        print("-"*60)
        
        value_counts = df[col].value_counts()
        unique_count = len(value_counts)
        
        print(f"Unique values: {unique_count:,}")
        print(f"Most common: {value_counts.index[0]} ({value_counts.iloc[0]:,} occurrences, {value_counts.iloc[0]/len(df)*100:.1f}%)")
        
        if unique_count <= max_categories:
            print(f"\nValue distribution:")
            for idx, (value, count) in enumerate(value_counts.items(), 1):
                pct = count / len(df) * 100
                print(f"  {idx:2}. {str(value):30} : {count:8,} ({pct:5.1f}%)")
        else:
            print(f"\nTop {max_categories} values:")
            for idx, (value, count) in enumerate(value_counts.head(max_categories).items(), 1):
                pct = count / len(df) * 100
                print(f"  {idx:2}. {str(value):30} : {count:8,} ({pct:5.1f}%)")
            print(f"  ... and {unique_count - max_categories} more values")

def analyze_temporal_patterns(df, datetime_column):
    """
    Analyze temporal patterns in the accident data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        datetime_column (str): Name of datetime column
    """
    if datetime_column not in df.columns:
        print(f"Column '{datetime_column}' not found in dataset.")
        return
    
    print("\n" + "="*60)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("="*60)
    
    # Ensure datetime format
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Extract time components
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['day_of_week'] = df[datetime_column].dt.dayofweek
    df['hour'] = df[datetime_column].dt.hour
    
    print("\n" + "-"*60)
    print("Yearly Distribution:")
    print("-"*60)
    yearly = df['year'].value_counts().sort_index()
    for year, count in yearly.items():
        print(f"  {year}: {count:,} accidents")
    
    print("\n" + "-"*60)
    print("Monthly Distribution:")
    print("-"*60)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = df['month'].value_counts().sort_index()
    for month, count in monthly.items():
        print(f"  {month_names[month-1]:3}: {count:,} accidents ({count/len(df)*100:.1f}%)")
    
    print("\n" + "-"*60)
    print("Day of Week Distribution:")
    print("-"*60)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                 'Friday', 'Saturday', 'Sunday']
    dow = df['day_of_week'].value_counts().sort_index()
    for day, count in dow.items():
        print(f"  {day_names[day]:9}: {count:,} accidents ({count/len(df)*100:.1f}%)")
    
    print("\n" + "-"*60)
    print("Hourly Distribution:")
    print("-"*60)
    hourly = df['hour'].value_counts().sort_index()
    
    # Group by time periods
    early_morning = hourly[hourly.index.isin(range(0, 6))].sum()
    morning = hourly[hourly.index.isin(range(6, 12))].sum()
    afternoon = hourly[hourly.index.isin(range(12, 18))].sum()
    evening = hourly[hourly.index.isin(range(18, 24))].sum()
    
    print(f"  Early Morning (00:00-05:59): {early_morning:,} ({early_morning/len(df)*100:.1f}%)")
    print(f"  Morning (06:00-11:59):       {morning:,} ({morning/len(df)*100:.1f}%)")
    print(f"  Afternoon (12:00-17:59):     {afternoon:,} ({afternoon/len(df)*100:.1f}%)")
    print(f"  Evening (18:00-23:59):       {evening:,} ({evening/len(df)*100:.1f}%)")
    
    peak_hour = hourly.idxmax()
    print(f"\n  Peak hour: {peak_hour:02d}:00 with {hourly[peak_hour]:,} accidents")

def identify_outliers(df, numeric_columns=None, method='iqr'):
    """
    Identify outliers in numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (list): List of numeric columns to check
        method (str): Method for outlier detection ('iqr' or 'zscore')
    """
    print("\n" + "="*60)
    print(f"OUTLIER ANALYSIS (Method: {method.upper()})")
    print("="*60)
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        else:  # zscore
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
        
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(data)) * 100
        
        if outlier_count > 0:
            print(f"\n{col}:")
            print(f"  Outliers found: {outlier_count:,} ({outlier_pct:.2f}%)")
            if method == 'iqr':
                print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Min outlier: {outliers.min():.2f}")
            print(f"  Max outlier: {outliers.max():.2f}")

def perform_full_eda(df, datetime_column=None):
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        datetime_column (str): Name of datetime column for temporal analysis
    """
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "EXPLORATORY DATA ANALYSIS" + " "*18 + "#")
    print("#" + " "*58 + "#")
    print("#"*60 + "\n")
    
    # 1. Data Summary
    generate_data_summary(df)
    
    # 2. Numeric Features
    analyze_numeric_features(df)
    
    # 3. Categorical Features
    analyze_categorical_features(df)
    
    # 4. Temporal Patterns
    if datetime_column:
        analyze_temporal_patterns(df, datetime_column)
    
    # 5. Outlier Detection
    identify_outliers(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)

if __name__ == "__main__":
    print("EDA Module for Traffic Accident Analysis")
    print("\nAvailable functions:")
    print("  - generate_data_summary()")
    print("  - analyze_numeric_features()")
    print("  - analyze_categorical_features()")
    print("  - analyze_temporal_patterns()")
    print("  - identify_outliers()")
    print("  - perform_full_eda()")
    print("\nExample usage:")
    print("  from eda_analysis import perform_full_eda")
    print("  perform_full_eda(df, datetime_column='accident_date')")
