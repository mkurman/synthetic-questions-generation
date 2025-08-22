import os


def is_local_file(dataset_name: str) -> bool:
    """Check if the dataset name is a local file path"""
    return (
        dataset_name.endswith(".jsonl") or dataset_name.endswith(".json")
    ) and os.path.exists(dataset_name)


def is_local_parquet(dataset_name: str) -> bool:
    """Check if the dataset name is a local parquet file path"""
    return dataset_name.endswith(".parquet") and os.path.exists(dataset_name)
