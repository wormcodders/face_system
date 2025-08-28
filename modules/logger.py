import csv
from datetime import datetime
import os

class FaceLogger:
    def __init__(self, log_path="logs.csv"):
        self.log_path = log_path
        # create log file with header if not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "name"])

    def log(self, name: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name])
        print(f"[LOG] {timestamp} - {name}")
