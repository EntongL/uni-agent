from __future__ import annotations

from copy import deepcopy

import pytest

from uni_agent.tasks import TaskConfig, TaskConfigResolver
from uni_agent.tasks.swe_bench import preprocess as swe_bench_preprocess
from uni_agent.tasks.swe_bench_multilingual import preprocess as multilingual_preprocess
from uni_agent.tasks.swe_rebench import preprocess as swe_rebench_preprocess


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return _FakeDataset([self.rows[index] for index in indices])

    def map(self, function, remove_columns):
        return _FakeDataset([function(deepcopy(row)) for row in self.rows])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("module", "build_name", "row"),
    [
        (
            swe_bench_preprocess,
            "build_swe_bench_verified",
            {
                "instance_id": "org__repo-1",
                "repo": "org/repo",
                "version": "1",
                "base_commit": "base",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
                "problem_statement": "Canonical source problem",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            },
        ),
        (
            swe_rebench_preprocess,
            "build_swe_rebench",
            {
                "instance_id": "org__repo-2",
                "repo": "org/repo",
                "base_commit": "base",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
                "problem_statement": "Canonical source problem",
                "FAIL_TO_PASS": "[]",
                "FAIL_TO_FAIL": "[]",
                "PASS_TO_PASS": "[]",
                "PASS_TO_FAIL": "[]",
                "install_config": {"install": "install", "log_parser": "parser", "test_cmd": "test"},
            },
        ),
        (
            multilingual_preprocess,
            "build_swe_bench_multilingual",
            {
                "instance_id": "redis__redis-3",
                "repo": "redis/redis",
                "version": "1",
                "base_commit": "base",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
                "problem_statement": "Canonical source problem",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            },
        ),
    ],
)
def test_swe_preprocess_emits_source_prompt_without_nested_rendered_prompt(monkeypatch, module, build_name, row):
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: _FakeDataset([row]))

    output = getattr(module, build_name)()[0]

    expected_prompt = [{"role": "user", "content": "Canonical source problem"}]
    assert output["prompt"] == expected_prompt
    task_config = output["extra_info"]["tools_kwargs"]["task"]
    assert "prompt" not in task_config
    assert task_config["metadata"]["problem_statement"] == "Canonical source problem"
    if module is multilingual_preprocess:
        assert task_config["metadata"]["language"] == "C"


def _write_swe_rebench_parquet(path):
    import json

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    install_config = pa.struct(
        [
            pa.field("install", pa.string()),
            pa.field("log_parser", pa.string()),
            pa.field("test_cmd", pa.string()),
        ]
    )
    table = pa.table(
        {
            "instance_id": ["org__repo-2"],
            "repo": ["org/repo"],
            "base_commit": ["base"],
            "patch": ["SECRET GOLD PATCH"],
            "test_patch": ["SECRET TEST PATCH"],
            "problem_statement": ["Canonical source problem"],
            "FAIL_TO_PASS": [["fail"]],
            "FAIL_TO_FAIL": [[]],
            "PASS_TO_PASS": [["pass"]],
            "PASS_TO_FAIL": [[]],
            "install_config": [
                {"install": "install", "log_parser": "parser", "test_cmd": "test"},
            ],
        },
        schema=pa.schema(
            [
                pa.field("instance_id", pa.string()),
                pa.field("repo", pa.string()),
                pa.field("base_commit", pa.string()),
                pa.field("patch", pa.string()),
                pa.field("test_patch", pa.string()),
                pa.field("problem_statement", pa.string()),
                pa.field("FAIL_TO_PASS", pa.list_(pa.string())),
                pa.field("FAIL_TO_FAIL", pa.list_(pa.string())),
                pa.field("PASS_TO_PASS", pa.list_(pa.string())),
                pa.field("PASS_TO_FAIL", pa.list_(pa.string())),
                pa.field("install_config", install_config),
            ]
        ),
    )
    # Mimic Hub parquet metadata that older/newer `datasets` cannot deserialize.
    table = table.replace_schema_metadata(
        {
            b"huggingface": json.dumps(
                {
                    "info": {
                        "features": {
                            "install_config": {"_type": "ThisTypeDoesNotExist"},
                        }
                    }
                }
            ).encode("utf-8")
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_swe_bench_parquet(path):
    import json

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pa.table(
        {
            "instance_id": ["org__repo-1"],
            "repo": ["org/repo"],
            "version": ["1"],
            "base_commit": ["base"],
            "patch": ["SECRET GOLD PATCH"],
            "test_patch": ["SECRET TEST PATCH"],
            "problem_statement": ["Canonical source problem"],
            "FAIL_TO_PASS": [["fail"]],
            "PASS_TO_PASS": [["pass"]],
        },
        schema=pa.schema(
            [
                pa.field("instance_id", pa.string()),
                pa.field("repo", pa.string()),
                pa.field("version", pa.string()),
                pa.field("base_commit", pa.string()),
                pa.field("patch", pa.string()),
                pa.field("test_patch", pa.string()),
                pa.field("problem_statement", pa.string()),
                pa.field("FAIL_TO_PASS", pa.list_(pa.string())),
                pa.field("PASS_TO_PASS", pa.list_(pa.string())),
            ]
        ),
    )
    # Mimic Hub parquet metadata that can fail feature deserialization.
    table = table.replace_schema_metadata(
        {
            b"huggingface": json.dumps(
                {
                    "info": {
                        "features": {
                            "FAIL_TO_PASS": {"_type": "ThisTypeDoesNotExist"},
                        }
                    }
                }
            ).encode("utf-8")
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.mark.cpu
@pytest.mark.level0
def test_swe_rebench_local_parquet_files(tmp_path):
    data_dir = tmp_path / "swe_rebench"
    shard_dir = data_dir / "data"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "filtered-00000-of-00001.parquet"
    shard.write_bytes(b"parquet")

    files = swe_rebench_preprocess.local_parquet_files(str(data_dir))
    assert files == [str(shard)]


@pytest.mark.cpu
@pytest.mark.level0
def test_swe_rebench_loads_local_parquet_without_hub_metadata(tmp_path, monkeypatch):
    data_dir = tmp_path / "swe_rebench"
    shard = data_dir / "data" / "filtered-00000-of-00001.parquet"
    _write_swe_rebench_parquet(shard)

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError(f"load_dataset should not be called for local parquet: {args} {kwargs}")

    monkeypatch.setattr(swe_rebench_preprocess, "load_dataset", fail_load_dataset)

    output = swe_rebench_preprocess.build_swe_rebench(dataset_dir=str(data_dir))[0]
    task_config = output["extra_info"]["tools_kwargs"]["task"]
    assert output["prompt"] == [{"role": "user", "content": "Canonical source problem"}]
    assert task_config["metadata"]["install"] == "install"
    assert task_config["metadata"]["log_parser"] == "parser"
    assert task_config["metadata"]["test_cmd"] == "test"


@pytest.mark.cpu
@pytest.mark.level0
def test_swe_bench_local_parquet_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "swe_bench_verified"
    shard = data_dir / "data" / "test-00000-of-00001.parquet"
    _write_swe_bench_parquet(shard)

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError(f"load_dataset should not be called for local parquet: {args} {kwargs}")

    monkeypatch.setattr(swe_bench_preprocess, "load_dataset", fail_load_dataset)

    output = swe_bench_preprocess.build_swe_bench_verified(dataset_dir=str(data_dir))[0]
    task_config = output["extra_info"]["tools_kwargs"]["task"]
    assert output["prompt"] == [{"role": "user", "content": "Canonical source problem"}]
    assert task_config["metadata"]["version"] == "1"
    assert task_config["metadata"]["FAIL_TO_PASS"] == ["fail"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("recipe_path", "task_name", "expects_submit", "expects_language"),
    [
        ("examples/quickstart/inference/task_config_react.yaml", "swe_bench", True, False),
        ("examples/quickstart/inference/task_config_react.yaml", "swe_bench_multilingual", True, True),
        ("examples/quickstart/inference/task_config_claude_code.yaml", "swe_bench", False, False),
        ("examples/quickstart/inference/task_config_claude_code.yaml", "swe_bench_multilingual", False, True),
        ("examples/quickstart/training/task_config_react.yaml", "swe_bench", True, False),
        ("examples/quickstart/training/task_config_react.yaml", "swe_rebench", True, False),
        ("examples/quickstart/training/task_config_react.yaml", "swe_bench_multilingual", True, True),
        ("examples/quickstart/training/task_config_claude_code.yaml", "swe_bench", False, False),
        ("examples/quickstart/training/task_config_claude_code.yaml", "swe_rebench", False, False),
        ("examples/quickstart/training/task_config_claude_code.yaml", "swe_bench_multilingual", False, True),
    ],
)
def test_swe_recipe_renders_complete_metadata_prompt(recipe_path, task_name, expects_submit, expects_language):
    source_problem = "Dataset source problem"
    metadata_problem = "Metadata problem"
    language = "TestLanguageSentinel"
    metadata = {
        "problem_statement": metadata_problem,
        "patch": "SECRET GOLD PATCH",
        "test_patch": "SECRET TEST PATCH",
    }
    if expects_language:
        metadata["language"] = language
    image_prefix = "swerebench" if task_name == "swe_rebench" else "swebench"

    resolved = TaskConfigResolver.from_file(recipe_path).resolve(
        {
            "name": task_name,
            "sandbox": {"image": f"{image_prefix}/test-repo:latest"},
            "prompt": [{"role": "user", "content": source_problem}],
            "metadata": metadata,
        }
    )

    rendered_messages = TaskConfig(**resolved).prompt
    rendered_text = "\n".join(str(message["content"]) for message in rendered_messages)

    assert metadata_problem in rendered_text
    assert source_problem not in rendered_text
    assert "SECRET GOLD PATCH" not in rendered_text
    assert "SECRET TEST PATCH" not in rendered_text
    assert ("submit" in rendered_text.lower()) is expects_submit
    assert (language in rendered_text) is expects_language
