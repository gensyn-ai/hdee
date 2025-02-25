import argparse
import logging
import os
import sys
from multiprocessing import Manager, Process
from typing import List

import pyarrow as pa  # type: ignore
from datasets import load_dataset, VerificationMode
from pyarrow.parquet import ParquetWriter  # type: ignore
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


class OutputWriter:
    rank: int
    output_prefix: str
    number_of_samplers_per_file: int
    samples: int
    _schema: pa.lib.Schema
    _writer: ParquetWriter
    file_count: int

    def __init__(self, rank: int, output_prefix: str, number_of_samplers_per_file: int):
        self.rank = rank
        self.output_prefix = output_prefix
        self.number_of_samplers_per_file = number_of_samplers_per_file
        self.samples = 0
        self._schema = pa.schema(
            [
                ("input_ids", pa.list_(pa.int32())),
                ("label", pa.list_(pa.int32())),
            ]
        )
        self.file_count = 0
        self._writer: ParquetWriter = ParquetWriter(
            self._file_name(),
            schema=self._schema,
        )
        self.file_count += 1

    def files(self) -> List[str]:
        return [
            os.path.join(self.output_prefix, f"data_{self.rank}_{file_index}.parquet")
            for file_index in range(self.file_count)
        ]

    def close(self) -> None:
        self._writer.close()

    def _file_name(self) -> str:
        return os.path.join(
            self.output_prefix, f"data_{self.rank}_{self.file_count}.parquet"
        )

    def write(self, input_ids: List[int], label: List[int]) -> None:
        self.samples += 1
        table = pa.Table.from_arrays(
            [
                pa.array([input_ids], type=pa.list_(pa.int32())),
                pa.array([label], type=pa.list_(pa.int32())),
            ],
            schema=self._schema,
        )
        self._writer.write(table)

        if self.samples == self.number_of_samplers_per_file:
            self.samples = 0
            self.file_count += 1
            self._writer.close()
            self._writer = ParquetWriter(
                self._file_name(),
                schema=self._schema,
            )


def process_data(
    rank: int,
    world_size: int,
    dataset_name: str,
    domain_name: str,
    partition: str,
    textfield_name: str,
    output_prefix: str,
    tokenizer_name: str,
    access_token: str,
    sequence_length: int,
    number_of_samplers_per_file: int,
    output_files: List[str],
) -> None:
    if number_of_samplers_per_file <= 0:
        number_of_samplers_per_file = sys.maxsize
    buffer: List[int] = []
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        token=access_token,
    )
    dataset = load_dataset(
        dataset_name,domain_name, split=partition, verification_mode=VerificationMode.NO_CHECKS
    )
    writer = OutputWriter(
        rank=rank,
        output_prefix=output_prefix,
        number_of_samplers_per_file=number_of_samplers_per_file,
    )

    for index, sample in enumerate(dataset):
        if rank == 0 and (index + 1) % 100_000 == 0:
            logging.info(f"Processed {index + 1} samples.")

        if index % world_size != rank:
            continue
        line = sample[textfield_name]
        if not line:
            continue

        tokens = tokenizer.encode(line)
        buffer += tokens

        while len(buffer) > sequence_length:
            input_ids = buffer[:sequence_length]
            label = buffer[1 : (sequence_length + 1)]
            writer.write(input_ids, label)
            buffer = buffer[sequence_length:]
    output_files.extend(writer.files())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pretokenize_data.py",
        description="Pretokenize text data contained in HF datasets.",
    )
    parser.add_argument("--num_processes", action="store", type=int, required=True)
    parser.add_argument("--dataset_name", action="store", type=str, required=True)
    parser.add_argument("--domain_name", action="store", type=str, required=True)
    parser.add_argument("--partition", action="store", type=str, required=True)
    parser.add_argument("--textfield_name", action="store", type=str, required=True)
    parser.add_argument("--output_prefix", action="store", type=str, required=True)
    parser.add_argument("--tokenizer_name", action="store", type=str, required=True)
    parser.add_argument(
        "--access_token", action="store", type=str, required=False, default=None
    )
    parser.add_argument("--sequence_length", action="store", type=int)
    parser.add_argument("--number_of_samplers_per_file", action="store", type=int)
    args = parser.parse_args()
    manager = Manager()
    output_files = manager.list()
    processes = []

    # Preload the dataset to cache files.
    _ = load_dataset(
        args.dataset_name,
        args.domain_name,
        split=args.partition,
        num_proc=args.num_processes,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    logging.info("Dataset files loaded.")

    for rank in range(args.num_processes):
        processes.append(
            Process(
                target=process_data,
                args=(
                    rank,
                    args.num_processes,
                    args.dataset_name,
                    args.domain_name,
                    args.partition,
                    args.textfield_name,
                    args.output_prefix,
                    args.tokenizer_name,
                    args.access_token,
                    args.sequence_length,
                    args.number_of_samplers_per_file,
                    output_files,
                ),
            )
        )
        processes[-1].start()

    for process in processes:
        process.join()

    with open(os.path.join(args.output_prefix, "manifest.txt"), "w") as fout:
        fout.write("\n".join(output_files))


if __name__ == "__main__":
    main()
