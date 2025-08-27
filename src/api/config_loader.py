"""
Configuration loader for synthetic questions generation.
Supports loading YAML configuration files and converting them to command-line arguments.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Handles loading and parsing of YAML configuration files."""

    def __init__(self):
        """Initialize the configuration loader."""
        pass

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the configuration

        Raises:
            FileNotFoundError: If the config file is not found
            yaml.YAMLError: If the YAML file is invalid
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            logger.info(f"Loaded configuration from: {config_path}")
            return config

        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid YAML in configuration file {config_path}: {e}"
            )
        except Exception as e:
            raise Exception(f"Error loading configuration file {config_path}: {e}")

    def config_to_args(self, config: Dict[str, Any]) -> List[str]:
        """
        Convert configuration dictionary to command-line arguments.

        Args:
            config: Configuration dictionary

        Returns:
            List of command-line arguments
        """
        args = []

        # Handle positional argument (dataset)
        if "dataset" in config:
            args.append(str(config["dataset"]))

        # Handle optional arguments
        for key, value in config.items():
            if key == "dataset":
                continue  # Already handled as positional

            # Convert key to command-line format
            arg_name = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(arg_name)
            elif isinstance(value, list):
                # Handle comma-separated values (like styles)
                if value:
                    args.extend([arg_name, ",".join(map(str, value))])
            elif value is not None:
                args.extend([arg_name, str(value)])

        return args

    def merge_config_with_args(
        self,
        config: Dict[str, Any],
        cli_args: argparse.Namespace,
        provided_args: Optional[List[str]] = None,
    ) -> argparse.Namespace:
        """
        Merge configuration with command-line arguments.
        Command-line arguments take precedence over configuration file values.

        Args:
            config: Configuration dictionary from YAML file
            cli_args: Parsed command-line arguments

        Returns:
            Merged arguments namespace
        """
        # Convert config keys to match argument names
        config_normalized = {}
        for key, value in config.items():
            # Convert underscores to hyphens and normalize
            normalized_key = key.replace("-", "_")
            config_normalized[normalized_key] = value

        # Create a new namespace with config defaults
        merged_args = argparse.Namespace()

        # First, set all values from config
        for key, value in config_normalized.items():
            if key != "dataset":  # dataset is positional, handled separately
                setattr(merged_args, key, value)

        # Then override with any CLI arguments that were explicitly provided
        cli_dict = vars(cli_args)

        # Determine which boolean flags were explicitly provided
        explicitly_provided_flags = set()
        if provided_args:
            for arg in provided_args:
                if arg.startswith("--"):
                    flag_name = arg[2:].replace("-", "_")
                    explicitly_provided_flags.add(flag_name)

        for key, value in cli_dict.items():
            # For boolean flags, only override if explicitly provided
            if isinstance(value, bool):
                if key in explicitly_provided_flags:
                    setattr(merged_args, key, value)
            # For other values, only override if not None
            elif value is not None:
                setattr(merged_args, key, value)

        return merged_args

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If the configuration is invalid
        """
        required_fields = ["dataset", "provider", "model", "output_dir"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required configuration field missing: {field}")

        # Validate provider choices
        valid_providers = [
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
        ]

        if config["provider"] not in valid_providers:
            raise ValueError(
                f"Invalid provider: {config['provider']}. Must be one of: {', '.join(valid_providers)}"
            )

        # Validate provider-url requirement for 'other' provider
        if config["provider"] == "other" and "provider_url" not in config:
            raise ValueError("provider_url is required when provider is 'other'")

        # Validate numeric fields
        numeric_fields = {
            "num_questions": int,
            "max_tokens": int,
            "num_workers": int,
            "start_index": int,
            "end_index": int,
            "max_items": int,
            "sleep_between_requests": float,
            "sleep_between_items": float,
            "rate_limit_wait": float,
            "rate_limit_retries": int,
        }

        for field, expected_type in numeric_fields.items():
            if field in config:
                try:
                    if expected_type == int:
                        int(config[field])
                    elif expected_type == float:
                        float(config[field])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid value for {field}: must be a {expected_type.__name__}"
                    )

        # Validate boolean fields
        boolean_fields = [
            "shuffle",
            "verbose",
            "debug",
            "no_style",
            "with_answer",
            "answer_single_request",
            "with_options",
        ]

        for field in boolean_fields:
            if field in config and not isinstance(config[field], bool):
                raise ValueError(
                    f"Invalid value for {field}: must be a boolean (true/false)"
                )

        logger.info("Configuration validation passed")

    def get_example_config(self) -> Dict[str, Any]:
        """
        Get an example configuration dictionary.

        Returns:
            Example configuration dictionary
        """
        return {
            "dataset": "mkurman/hindawi-journals-2007-2023",
            "provider": "openrouter",
            "model": "qwen/qwen3-235b-a22b-2507",
            "output_dir": "./data/questions_openrouter",
            "start_index": 0,
            "end_index": 10,
            "num_questions": 5,
            "text_column": "text",
            "verbose": True,
            "max_tokens": 4096,
            "num_workers": 1,
            "sleep_between_requests": 0.0,
            "rate_limit_wait": 15.0,
            "rate_limit_retries": 1,
        }
