# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides test dataset
"""

import argparse
from pathlib import Path
import pandas as pd
import mlflow

TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = []

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    parser.add_argument("--table_name", type=str, default="mlmonitoring", help="Table name in ADX for logging")

    args = parser.parse_args()

    return args

def log_test_data(df, table_name):
    from obs.collector import Online_Collector
    collector = Online_Collector(table_name)
    collector.batch_collect(df)

def main(args):
    '''Read and save test dataset'''

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    data = pd.read_csv(Path(args.raw_data))
    data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]

    # ------------- Save Test Data ------------- #
    # ------------------------------------------ #

    mlflow.log_metric('test size', data.shape[0])

    data.to_parquet((Path(args.test_data) / "test.parquet"))

    if args.enable_monitoring.lower() == 'true' or args.enable_monitoring == '1' or args.enable_monitoring.lower() == 'yes':
        log_test_data(data, args.table_name)

if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Test dataset output path: {args.test_data}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()