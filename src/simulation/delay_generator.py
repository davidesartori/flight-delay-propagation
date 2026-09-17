"""
This module generates random delay files for a list of airports.
The generated files are CSV files containing airport codes and their corresponding extra delays."""

import time
import random
import csv
import os
import shutil

def generate_random_delay_files(airports, output_dir="delay_streaming",
                                  interval_sec=10, probability=0.001,
                                  delay_range=(5, 30), stop_event=None):
    """Generates random delay files for a list of airports."""
    clean_streaming_dir(output_dir)
    file_counter = 0
    while stop_event is None or not stop_event.is_set():
        time.sleep(interval_sec)
        rows = [
            (airport, random.randint(*delay_range))
            for airport in airports
            if random.random() < probability
        ]
        if rows:
            file_path = os.path.join(output_dir, f"delays_{file_counter}.csv")
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["airport", "extra_delay"])
                writer.writerows(rows)
            file_counter += 1

def clean_streaming_dir(output_dir="delay_streaming"):
    """Cleans the streaming input directory by removing all files and recreating the directory."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
