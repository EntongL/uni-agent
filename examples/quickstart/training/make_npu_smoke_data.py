"""Create tiny parquet files for the single-node NPU RL smoke test.

The script keeps the original VERL row schema and only truncates the number
of rows, so the generated files can be passed directly to ``data.train_files``
and ``data.val_files``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def make_smoke_file(source: Path, target: Path, rows: int) -> None:
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    if not source.is_file():
        raise FileNotFoundError(f"source parquet does not exist: {source}")

    import pyarrow.parquet as pq

    table = pq.read_table(source)
    if table.num_rows == 0:
        raise ValueError(f"source parquet is empty: {source}")

    import pyarrow as pa

    if rows > table.num_rows:
        repeats = (rows + table.num_rows - 1) // table.num_rows
        table = pa.concat_tables([table] * repeats)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = table.slice(0, rows)
    pq.write_table(written, target)
    print(f"Wrote {written.num_rows} rows: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    make_smoke_file(args.source.expanduser(), args.target.expanduser(), args.rows)


if __name__ == "__main__":
    main()
