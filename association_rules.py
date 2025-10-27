"""Association Rules Mining Module for Traffic Accident Analysis

This module applies market-basket-like association rule mining (Apriori/FP-Growth) on categorical
features to find frequent co-occurring patterns in accident data.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


def bin_numeric(df, columns, bins=5):
    """
    Discretize numeric columns into bins for association mining.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col + '_bin'] = pd.qcut(df[col], q=bins, duplicates='drop')
    return df


def to_transaction_df(df, categorical_columns):
    """
    Convert dataframe to one-hot encoded transaction format for mlxtend.
    """
    trans_df = pd.get_dummies(df[categorical_columns].astype('category'))
    return trans_df


def mine_frequent_itemsets(df, categorical_columns, method='apriori', min_support=0.05, use_colnames=True):
    """
    Mine frequent itemsets using Apriori or FP-Growth.
    """
    trans_df = to_transaction_df(df, categorical_columns)
    if method == 'apriori':
        itemsets = apriori(trans_df, min_support=min_support, use_colnames=use_colnames)
    else:
        itemsets = fpgrowth(trans_df, min_support=min_support, use_colnames=use_colnames)
    itemsets = itemsets.sort_values('support', ascending=False).reset_index(drop=True)
    print(f"Found {len(itemsets)} itemsets with support >= {min_support}")
    return itemsets


def mine_association_rules(itemsets, metric='lift', min_threshold=1.2, top_n=20):
    """
    Generate association rules from frequent itemsets.
    """
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    rules = rules.sort_values(metric, ascending=False).head(top_n)
    print(f"Top {top_n} rules by {metric}:")
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    return rules


def run_association_pipeline(df, categorical_columns, numeric_to_bin=None, bins=5, method='apriori', min_support=0.05, metric='lift', min_threshold=1.2, top_n=20):
    """
    Full pipeline to mine association rules from the accident dataset.
    """
    if numeric_to_bin:
        df = bin_numeric(df, numeric_to_bin, bins=bins)
        categorical_columns = categorical_columns + [col + '_bin' for col in numeric_to_bin]
    
    itemsets = mine_frequent_itemsets(df, categorical_columns, method=method, min_support=min_support)
    rules = mine_association_rules(itemsets, metric=metric, min_threshold=min_threshold, top_n=top_n)
    return itemsets, rules

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mine association rules from traffic accident data')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--categorical', type=str, nargs='+', required=True, help='Categorical columns')
    parser.add_argument('--numeric', type=str, nargs='*', default=None, help='Numeric columns to discretize')
    parser.add_argument('--bins', type=int, default=5, help='Number of quantile bins for numeric discretization')
    parser.add_argument('--method', type=str, default='apriori', choices=['apriori', 'fpgrowth'])
    parser.add_argument('--min_support', type=float, default=0.05)
    parser.add_argument('--metric', type=str, default='lift', choices=['confidence', 'lift', 'leverage', 'conviction'])
    parser.add_argument('--min_threshold', type=float, default=1.2)
    parser.add_argument('--top_n', type=int, default=20)
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    itemsets, rules = run_association_pipeline(
        df,
        categorical_columns=args.categorical,
        numeric_to_bin=args.numeric,
        bins=args.bins,
        method=args.method,
        min_support=args.min_support,
        metric=args.metric,
        min_threshold=args.min_threshold,
        top_n=args.top_n
    )
