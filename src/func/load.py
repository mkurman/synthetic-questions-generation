from typing import List, Dict, Any
import json
import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON on line {line_num}: {e}")
                        continue
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading JSONL file {file_path}: {e}")
