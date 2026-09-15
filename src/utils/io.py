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
