import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Question Generation System based on dataset text column")
    parser.add_argument("dataset", help="HuggingFace dataset name or local JSONL file path")
    parser.add_argument("--provider", required=True,
                        choices=["featherless", "openai", "anthropic", "qwen", "qwen-deepinfra", "kimi",
                                "z.ai", "openrouter", "cerebras", "together", "groq", "gemini", "ollama", "chutes", "huggingface", "other"],
                        help="API provider to use")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--provider-url", help="Base URL for 'other' provider (required when using --provider other)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--text-column", default="text", help="Column name containing text to generate questions from")
    parser.add_argument("--num-questions", type=int, default=3, help="Number of questions to generate per text")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per response")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")
    parser.add_argument("--max-items", type=int, help="Maximum number of items to process")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for dataset items (0-based)")
    parser.add_argument("--end-index", type=int, help="End index for dataset items (exclusive, 0-based)")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use (ignored for local files)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--sleep-between-requests", type=float, default=0.0, 
                        help="Sleep time in seconds between API requests (rate limiting)")
    parser.add_argument("--sleep-between-items", type=float, default=0.0, 
                        help="Sleep time in seconds between processing dataset items")
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
        help="Generate questions without any style instructions"
    )
    parser.add_argument(
        "--styles-file",
        help="Path to a file containing styles (one per line)"
    )
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="Generate answers for each question using the model"
    )
    parser.add_argument(
        "--answer-provider",
        choices=["featherless", "openai", "anthropic", "qwen", "qwen-deepinfra", "kimi",
                "z.ai", "openrouter", "cerebras", "together", "groq", "gemini", "ollama", "chutes", "huggingface", "other"],
        help="API provider to use for answering questions (if not set, uses the same provider as --provider)"
    )
    parser.add_argument(
        "--answer-model",
        help="Model to use for answering questions (if not set, uses the same model as --model)"
    )
    parser.add_argument(
        "--answer-single-request",
        action="store_true",
        help="Answer all questions in a single request instead of one question per request"
    )
    
    return parser
