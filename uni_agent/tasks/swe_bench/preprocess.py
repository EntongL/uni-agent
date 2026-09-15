# ruff: noqa: E501
"""Preprocess SWE-bench Verified into the new-framework SWE task format.

Provider-agnostic on purpose: the row only carries the canonical *open-source*
image ref (the published ``swebench/sweb.eval.x86_64.<id>``); mapping that to a
service-provider image is the sandbox provider's job (e.g. ``_to_vefaas_image``),
decided at run time, not baked into the dataset.

Example::

    python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir ~/data/swe_agent

    python -m uni_agent.tasks.swe_bench.preprocess \\
        --dataset-dir /path/to/SWE-bench_Verified \\
        --local-save-dir ~/data/swe_agent
"""

import argparse
import glob
import os

from datasets import load_dataset


def get_image_name(instance_id: str) -> str:
    """Canonical open-source image ref (mirrors swebench's ``instance_image_key``).

    Provider-agnostic; a provider maps this to its own registry at run time.
    """
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"


DATA_SOURCE = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"


def local_parquet_files(data_dir: str, split: str = DEFAULT_SPLIT) -> list[str]:
    """Return parquet shards for ``split`` under a local dataset snapshot."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    patterns = (
        os.path.join(data_dir, "data", f"{split}-*"),
        os.path.join(data_dir, f"{split}-*"),
        os.path.join(data_dir, "data", f"{split}.parquet"),
        os.path.join(data_dir, f"{split}.parquet"),
    )
    for pattern in patterns:
        files = [path for path in sorted(glob.glob(pattern)) if os.path.isfile(path)]
        if files:
            return files
    return []


def dataset_from_parquet_files(files: list[str]):
    """Load parquet shards without parsing Hugging Face feature metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset

    tables = [pq.read_table(path).replace_schema_metadata(None) for path in files]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    if hasattr(Dataset, "from_arrow"):
        return Dataset.from_arrow(table)
    return Dataset(table)


def load_raw_dataset(dataset_dir: str | None = None, split: str = DEFAULT_SPLIT):
    """Load SWE-bench Verified from a local snapshot, or from Hugging Face."""
    if dataset_dir:
        dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Local SWE-bench directory not found: {dataset_dir}")

        parquet_files = local_parquet_files(dataset_dir, split=split)
        if parquet_files:
            print(
                f"Loading the {DATA_SOURCE} {split} split from {len(parquet_files)} "
                f"local parquet file(s) under {dataset_dir}...",
                flush=True,
            )
            return dataset_from_parquet_files(parquet_files)

        print(f"Loading the {DATA_SOURCE} dataset from local snapshot {dataset_dir}...", flush=True)
        return load_dataset(dataset_dir, split=split)

    print(f"Loading the {DATA_SOURCE} dataset from huggingface...", flush=True)
    return load_dataset(DATA_SOURCE, split=split)


def build_swe_bench_verified(max_instances: int | None = None, dataset_dir: str | None = None):
    def process(example):
        instance_id = example["instance_id"]

        metadata = {
            "instance_id": instance_id,
            "repo": example["repo"],
            "version": str(example["version"]),
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": example["problem_statement"],
            "FAIL_TO_PASS": example["FAIL_TO_PASS"],
            "PASS_TO_PASS": example["PASS_TO_PASS"],
        }
        task_config = {
            "name": "swe_bench",
            "sandbox": {"image": get_image_name(instance_id)},
            "metadata": metadata,
        }

        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": example["problem_statement"]}],
            "extra_info": {
                "tools_kwargs": {"task": task_config},
            },
        }

    dataset = load_raw_dataset(dataset_dir=dataset_dir)
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Local SWE-bench Verified directory from a ModelScope/Hugging Face download. "
        "If omitted, the dataset is loaded from Hugging Face.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances kept (smoke testing).",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_swe_bench_verified(max_instances=args.max_instances, dataset_dir=args.dataset_dir)
    out_path = f"{save_dir}/swe_bench_verified.parquet"
    dataset.to_parquet(out_path)
    print(f"Wrote {len(dataset)} instances to {out_path}", flush=True)
