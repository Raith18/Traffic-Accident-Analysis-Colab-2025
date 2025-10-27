"""Visualization Module for Traffic Accident Analysis

This module provides functions for creating various visualizations of traffic accident data,
including distribution plots, time series analysis, geographic visualizations, and more.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def plot_accident_distribution(df, column, title=None, top_n=15, figsize=(12, 6)):
    """
    Plot distribution of accidents by a categorical variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to plot
        title (str): Plot title
        top_n (int): Number of top categories to display
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get value counts and select top N
    value_counts = df[column].value_counts().head(top_n)
    
    # Create bar plot
    ax = value_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    
    # Formatting
    if title is None:
        title = f'Distribution of Accidents by {column.replace("_", " ").title()}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(column.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(value_counts.values):
        ax.text(i, v + max(value_counts.values) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop {top_n} {column} by accident count:")
    print(value_counts)

def plot_time_series(df, date_column, freq='M', figsize=(14, 6)):
    """
    Plot time series of accident frequency.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of date column
        freq (str): Frequency for grouping ('D'=daily, 'M'=monthly, 'Y'=yearly)
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by time period
    time_series = df.groupby(pd.Grouper(key=date_column, freq=freq)).size()
    
    # Plot
    plt.plot(time_series.index, time_series.values, marker='o', linewidth=2, 
             markersize=6, color='darkblue')
    
    # Formatting
    freq_labels = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Y': 'Yearly'}
    freq_label = freq_labels.get(freq, freq)
    
    plt.title(f'{freq_label} Accident Frequency Over Time', fontsize=14, 
              fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTime series statistics:")
    print(f"Total accidents: {time_series.sum():,}")
    print(f"Average {freq_label.lower()} accidents: {time_series.mean():.2f}")
    print(f"Peak period: {time_series.idxmax()} with {time_series.max():,} accidents")

def plot_correlation_heatmap(df, numeric_columns=None, figsize=(12, 10)):
    """
    Plot correlation heatmap for numeric variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (list): List of numeric columns to include
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Select numeric columns
    if numeric_columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[numeric_columns]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap of Numeric Variables', fontsize=14, 
              fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\nStrongest correlations (absolute value > 0.5):")
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                strong_corrs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if strong_corrs:
        for var1, var2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {var1} <-> {var2}: {corr:.3f}")
    else:
        print("  No strong correlations found")

def plot_accident_severity(df, severity_column, figsize=(10, 6)):
    """
    Plot accident severity distribution.
    
    Args:
        df (pd.DataFrame): Input dataframe
        severity_column (str): Name of severity column
        figsize (tuple): Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    severity_counts = df[severity_column].value_counts().sort_index()
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(severity_counts)))
    
    ax1.bar(range(len(severity_counts)), severity_counts.values, color=colors, 
            edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(severity_counts)))
    ax1.set_xticklabels(severity_counts.index)
    ax1.set_xlabel('Severity Level', fontsize=12)
    ax1.set_ylabel('Number of Accidents', fontsize=12)
    ax1.set_title('Accident Count by Severity', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax2.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Severity Distribution (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nSeverity statistics:")
    print(severity_counts)
    print(f"\nTotal accidents: {len(df):,}")
    for level, count in severity_counts.items():
        print(f"Severity {level}: {count:,} ({count/len(df)*100:.1f}%)")

def plot_hourly_pattern(df, datetime_column, figsize=(14, 6)):
    """
    Plot accident patterns by hour of day.
    
    Args:
        df (pd.DataFrame): Input dataframe
        datetime_column (str): Name of datetime column
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Extract hour from datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    hourly_counts = df[datetime_column].dt.hour.value_counts().sort_index()
    
    # Create bar plot
    colors = ['darkblue' if 6 <= h <= 18 else 'darkred' for h in hourly_counts.index]
    plt.bar(hourly_counts.index, hourly_counts.values, color=colors, 
            edgecolor='black', alpha=0.7)
    
    plt.title('Accident Distribution by Hour of Day', fontsize=14, 
              fontweight='bold', pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.xticks(range(24))
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkblue', alpha=0.7, label='Daytime (6am-6pm)'),
        Patch(facecolor='darkred', alpha=0.7, label='Nighttime (6pm-6am)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    print("\nHourly statistics:")
    print(f"Peak hour: {hourly_counts.idxmax()}:00 with {hourly_counts.max():,} accidents")
    print(f"Lowest hour: {hourly_counts.idxmin()}:00 with {hourly_counts.min():,} accidents")

def create_summary_dashboard(df, date_column, severity_column=None):
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of date column
        severity_column (str): Name of severity column (optional)
    """
    print("="*60)
    print("Creating Comprehensive Visualization Dashboard")
    print("="*60)
    
    # Time series plot
    print("\n1. Time Series Analysis")
    plot_time_series(df, date_column, freq='M')
    
    # Hourly pattern
    print("\n2. Hourly Pattern Analysis")
    plot_hourly_pattern(df, date_column)
    
    # Severity analysis (if available)
    if severity_column and severity_column in df.columns:
        print("\n3. Severity Analysis")
        plot_accident_severity(df, severity_column)
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        print("\n4. Correlation Analysis")
        plot_correlation_heatmap(df)
    
    print("\n" + "="*60)
    print("Dashboard Complete!")
    print("="*60)

if __name__ == "__main__":
    print("Visualization Module for Traffic Accident Analysis")
    print("\nAvailable functions:")
    print("  - plot_accident_distribution()")
    print("  - plot_time_series()")
    print("  - plot_correlation_heatmap()")
    print("  - plot_accident_severity()")
    print("  - plot_hourly_pattern()")
    print("  - create_summary_dashboard()")
    print("\nExample usage:")
    print("  from visualization import plot_time_series")
    print("  plot_time_series(df, 'accident_date')")
