"""Utility functions for I/O operations."""

import csv
import os


def load__influence_scores(path: str) -> dict:
    """Load influence scores"""
    if not os.path.exists(path):
        return {}

    scores = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["airport"]] = float(row["score"])

    return scores


def save_output(path: str, text: str):
    """Write output to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(text)


def save_dataframe_csv(df, path):
    """Write a Spark DataFrame to a single local CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(df.columns)
        writer.writerows(df.collect())
