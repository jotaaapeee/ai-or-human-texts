import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def converter_csv_para_parquet(input_path, output_path, compression, chunksize):
    print(f"Lendo CSV em chunks de {chunksize} linhas...")

    writer = None

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
        print(f"Processando chunk {i}...")

        table = pa.Table.from_pandas(chunk)

        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression=compression
            )

        writer.write_table(table)

    if writer:
        writer.close()

    print("Conversão finalizada!")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("-c", "--compression", default="snappy")

    args = parser.parse_args()

    converter_csv_para_parquet(
        args.input,
        args.output,
        args.compression,
        args.chunksize
    )

if __name__ == "__main__":
    main()