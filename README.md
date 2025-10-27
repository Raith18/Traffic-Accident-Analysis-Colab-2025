# Traffic Accident Analysis - Colab 2025

## Project Overview
This project provides a reproducible pipeline to analyze traffic accident data. Core modules are split into standalone Python files for data cleaning, EDA, visualization, clustering, and association rule mining. Each module can be run as a script or imported as a library.

## Repository Structure
- README.md — Project overview, outputs, and how to run
- data_cleaning.py — Clean and standardize raw CSVs, handle missing data, deduplicate
- eda_analysis.py — Statistical summaries, numeric/categorical profiling, temporal patterns, outliers
- visualization.py — Time series, severity, hourly pattern, correlation heatmaps, dashboard
- clustering_model.py — KMeans/DBSCAN clustering with summaries and JSON output
- association_rules.py — Apriori/FP-Growth association rule mining for categorical patterns

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pandas numpy matplotlib seaborn scipy scikit-learn mlxtend
```

## How to Run as Scripts
Assume your data file is data/traffic_accidents.csv and has a datetime column accident_date and a severity column severity.

1) Data Cleaning
```bash
python data_cleaning.py
# Import usage
python -c "import pandas as pd; from data_cleaning import clean_traffic_data; clean_traffic_data('data/traffic_accidents.csv')"
```

2) EDA
```bash
python - << 'PY'
import pandas as pd
from eda_analysis import perform_full_eda

df = pd.read_csv('data/traffic_accidents_cleaned.csv')
perform_full_eda(df, datetime_column='accident_date')
PY
```

3) Visualization
```bash
python - << 'PY'
import pandas as pd
from visualization import plot_time_series, create_summary_dashboard

df = pd.read_csv('data/traffic_accidents_cleaned.csv')
plot_time_series(df, 'accident_date', freq='M')
create_summary_dashboard(df, date_column='accident_date', severity_column='severity')
PY
```

4) Clustering
```bash
python clustering_model.py \
  --data data/traffic_accidents_cleaned.csv \
  --features speed_limit vehicles_involved latitude longitude \
  --method kmeans --k 4 --groupby severity
```

5) Association Rules
```bash
python association_rules.py \
  --data data/traffic_accidents_cleaned.csv \
  --categorical weather road_type light_condition severity \
  --numeric speed_limit \
  --bins 5 --method apriori --min_support 0.05 --metric lift --min_threshold 1.2 --top_n 20
```

## Module Overviews
- data_cleaning.py
  - Functions: load_data, check_data_quality, standardize_columns, remove_duplicates, handle_missing_values, clean_traffic_data
  - Saves cleaned CSV as <input>_cleaned.csv
- eda_analysis.py
  - Functions: generate_data_summary, analyze_numeric_features, analyze_categorical_features, analyze_temporal_patterns, identify_outliers, perform_full_eda
- visualization.py
  - Functions: plot_accident_distribution, plot_time_series, plot_correlation_heatmap, plot_accident_severity, plot_hourly_pattern, create_summary_dashboard
- clustering_model.py
  - Functions: prepare_features, run_kmeans, run_dbscan, optimal_k_elbow, summarize_clusters, run_clustering_pipeline
  - CLI prints JSON summary with model, params, n_clusters, silhouette
- association_rules.py
  - Functions: bin_numeric, to_transaction_df, mine_frequent_itemsets, mine_association_rules, run_association_pipeline

## Outputs (Sample)
- Data Cleaning
  - Example: Removed 1,245 duplicate rows; Missing values filled for 3 numeric and 2 categorical columns; Memory usage reduced from 42.1 MB to 31.7 MB
  - Output file: data/traffic_accidents_cleaned.csv

- EDA
  - Numeric stats: mean speed_limit ≈ 45.6, vehicles_involved median = 2
  - Outliers: ~2.3% in vehicles_involved by IQR method
  - Temporal: Peak accidents on Fridays; Peak hour 18:00

- Visualization
  - Time series: Monthly accidents trend showing seasonal peaks in winter months
  - Severity: Distribution shows ~72% minor, 22% serious, 6% fatal (example)
  - Correlation: vehicles_involved vs damage_cost strong positive correlation (example)

- Clustering
  - KMeans (k=4): silhouette ~0.41; clusters separate by urban/night vs suburban/day patterns
  - DBSCAN: 3 clusters, 8% noise; silhouette ~0.35 (after removing noise)
  - JSON example
    ```json
    {"model": "KMeans", "params": {"k": 4}, "n_clusters": 4, "silhouette": 0.41}
    ```

- Association Rules
  - Frequent itemset example: {weather=Rain, light=Night} support 0.12
  - Top rule example: {road_type=Highway, weather=Rain} -> {severity=Serious}
    - support=0.09, confidence=0.62, lift=1.8

## Notes
- Replace column names to match your dataset (e.g., accident_date, severity, road_type, weather, light_condition, latitude, longitude, speed_limit, vehicles_involved, damage_cost).
- If running in Colab, you can upload your CSV and then call these modules similarly.
- The notebook (if used) can now import these modules to keep the analysis organized.
