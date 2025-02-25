import argparse
import glob
import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shuffle_parquet.py",
        description="Shuffle parquet files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--num_processes", action="store", type=int, required=True)
    parser.add_argument("--input_prefix", action="store", type=str, required=True)
    parser.add_argument("--output_prefix", action="store", type=str, required=True)
    parser.add_argument("--shards", action="store", type=int, required=False, default=1)
    args = parser.parse_args()
    parquet_files = glob.glob(os.path.join(args.input_prefix, "*.parquet"))

    logging.info(f"Found {len(parquet_files)} parquet files.")
    spark = (
        SparkSession.builder.appName("shuffle")
        .master("local[*]")
        .config("spark.executor.instances", str(args.num_processes))
        .config("spark.sql.shuffle.partitions", str(4 * args.num_processes))
        .config("spark.executor.memory", "10g")
        .config("spark.driver.memory", "24g")
        .getOrCreate()
    )
    df = spark.read.parquet(*parquet_files)
    shuffled_df = df.orderBy(rand())
    shuffled_df = shuffled_df.repartition(args.shards)
    shuffled_df.write.parquet(args.output_prefix, mode="overwrite")
    spark.stop()


if __name__ == "__main__":
    main()
