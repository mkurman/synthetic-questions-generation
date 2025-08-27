import argparse
from typing import Optional, List
from .config_loader import ConfigLoader


def get_parser():
    parser = argparse.ArgumentParser(
        description="Question Generation System based on dataset text column"
    )
    parser.add_argument(
        "dataset", nargs="?", help="HuggingFace dataset name or local JSONL file path"
    )
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--custom-prompts", help="Path to custom prompts directory")
    parser.add_argument(
        "--provider",
        choices=[
            "featherless",
            "openai",
            "anthropic",
            "qwen",
            "qwen-deepinfra",
            "kimi",
            "z.ai",
            "openrouter",
            "cerebras",
            "together",
            "groq",
            "gemini",
            "ollama",
            "chutes",
            "huggingface",
            "other",
        ],
        help="API provider to use",
    )
    parser.add_argument("--model", help="Model name")
    parser.add_argument(
        "--provider-url",
        help="Base URL for 'other' provider (required when using --provider other)",
    )
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name containing text to generate questions from",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=3,
        help="Number of questions to generate per text",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Maximum tokens per response"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of concurrent workers"
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")
    parser.add_argument(
        "--max-items", type=int, help="Maximum number of items to process"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for dataset items (0-based)",
    )
    parser.add_argument(
        "--end-index", type=int, help="End index for dataset items (exclusive, 0-based)"
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use (ignored for local files)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Sleep time in seconds between API requests (rate limiting)",
    )
    parser.add_argument(
        "--sleep-between-items",
        type=float,
        default=0.0,
        help="Sleep time in seconds between processing dataset items",
    )
    parser.add_argument(
        "--rate-limit-wait",
        type=float,
        default=15.0,
        help=(
            "Seconds to wait before retrying when the provider returns a rate limit or service unavailable error (429/503). "
            "If the provider returns a Retry-After header, that value will be used instead."
        ),
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=1,
        help="Number of retry attempts on 429/503 responses before failing",
    )
    parser.add_argument(
        "--style",
        help=(
            "Optional style or comma-separated styles to sample randomly. "
            "If omitted, a style is randomly chosen from defaults."
        ),
    )
    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Generate questions without any style instructions",
    )
    parser.add_argument(
        "--styles-file", help="Path to a file containing styles (one per line)"
    )
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="Generate answers for each question using the model",
    )
    parser.add_argument(
        "--with-options",
        action="store_true",
        help="Generate multiple-choice questions with options (A, B, C, D, E)",
    )
    parser.add_argument(
        "--answer-provider",
        choices=[
            "featherless",
            "openai",
            "anthropic",
            "qwen",
            "qwen-deepinfra",
            "kimi",
            "z.ai",
            "openrouter",
            "cerebras",
            "together",
            "groq",
            "gemini",
            "ollama",
            "chutes",
            "huggingface",
            "other",
        ],
        help="API provider to use for answering questions (if not set, uses the same provider as --provider)",
    )
    parser.add_argument(
        "--answer-model",
        help="Model to use for answering questions (if not set, uses the same model as --model)",
    )
    parser.add_argument(
        "--answer-single-request",
        action="store_true",
        help="Answer all questions in a single request instead of one question per request",
    )

    return parser


def parse_args_with_config(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments with optional YAML configuration file support.

    Args:
        args: Optional list of arguments to parse (defaults to sys.argv)

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If required arguments are missing or invalid
    """
    parser = get_parser()

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # If config file is provided, load and merge with CLI args
    if parsed_args.config:
        config_loader = ConfigLoader()
        try:
            config = config_loader.load_config(parsed_args.config)
            config_loader.validate_config(config)

            # Merge config with CLI args (CLI args take precedence)
            merged_args = config_loader.merge_config_with_args(
                config, parsed_args, args
            )

            # Handle dataset from config if not provided via CLI
            if not parsed_args.dataset and "dataset" in config:
                merged_args.dataset = config["dataset"]
            elif parsed_args.dataset:
                merged_args.dataset = parsed_args.dataset

            return merged_args

        except Exception as e:
            parser.error(f"Error loading configuration: {e}")

    # Validate required arguments when not using config
    if not parsed_args.config:
        required_args = ["dataset", "provider", "model", "output_dir"]
        missing_args = []

        for arg in required_args:
            if not getattr(parsed_args, arg.replace("-", "_")):
                missing_args.append(f"--{arg}")

        if missing_args:
            parser.error(
                f"The following arguments are required: {', '.join(missing_args)}"
            )

    return parsed_args
