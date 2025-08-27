"""
Prompt loading system for synthetic questions generation.
Supports loading default prompts and custom prompts with tag replacement.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PromptLoader:
    """Handles loading and formatting of system prompts with replaceable tags."""

    def __init__(self, custom_prompt_dir: Optional[str] = None):
        """
        Initialize the prompt loader.

        Args:
            custom_prompt_dir: Optional path to custom prompt directory.
                              If None, only default prompts will be available.
        """
        self.repo_root = Path(__file__).parent.parent.parent
        self.default_prompt_dir = self.repo_root / "prompt" / "default"
        self.custom_prompt_dir = Path(custom_prompt_dir) if custom_prompt_dir else None

        # Cache for loaded prompts
        self._prompt_cache: Dict[str, str] = {}

    def load_prompt(self, prompt_name: str, use_custom: bool = True) -> str:
        """
        Load a prompt by name, with optional custom prompt override.

        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            use_custom: Whether to check for custom prompts first

        Returns:
            The prompt content as a string

        Raises:
            FileNotFoundError: If the prompt file is not found
        """
        cache_key = f"{prompt_name}_{use_custom}_{self.custom_prompt_dir}"

        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        prompt_content = None

        # Try custom prompt first if enabled and custom dir exists
        if use_custom and self.custom_prompt_dir and self.custom_prompt_dir.exists():
            custom_path = self.custom_prompt_dir / f"{prompt_name}.txt"
            if custom_path.exists():
                try:
                    with open(custom_path, "r", encoding="utf-8") as f:
                        prompt_content = f.read().strip()
                    logger.info(f"Loaded custom prompt: {custom_path}")
                except Exception as e:
                    logger.warning(f"Failed to load custom prompt {custom_path}: {e}")

        # Fall back to default prompt
        if prompt_content is None:
            default_path = self.default_prompt_dir / f"{prompt_name}.txt"
            if not default_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_name}.txt")

            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    prompt_content = f.read().strip()
                logger.debug(f"Loaded default prompt: {default_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load default prompt {default_path}: {e}"
                )

        self._prompt_cache[cache_key] = prompt_content
        return prompt_content

    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load and format a prompt with the given parameters.

        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            **kwargs: Parameters to substitute in the prompt

        Returns:
            The formatted prompt string
        """
        prompt_template = self.load_prompt(prompt_name)

        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing parameter for prompt {prompt_name}: {e}")
            raise ValueError(
                f"Missing required parameter for prompt {prompt_name}: {e}"
            )
        except Exception as e:
            logger.error(f"Error formatting prompt {prompt_name}: {e}")
            raise ValueError(f"Error formatting prompt {prompt_name}: {e}")

    def get_question_generation_prompts(
        self,
        num_questions: int,
        selected_style: Optional[str] = None,
        text: str = "",
        with_options: bool = False,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Get formatted system and user prompts for question generation.

        Args:
            num_questions: Number of questions to generate
            selected_style: Optional style for questions
            text: The text to generate questions from
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Prepare style-related parameters
        if selected_style:
            style_instruction = f"- Use a {selected_style} style"
            style_note = f"Style: {selected_style}"
            style_suffix = f" in a {selected_style} style"
            casual_greeting_instruction = (
                f"- Try to use greetings where applicable - like human would do when the style is informal"
                if "casual" in selected_style.lower()
                else ""
            )
        else:
            style_instruction = "- Generate questions in a straightforward manner"
            style_note = ""
            style_suffix = ""
            casual_greeting_instruction = ""

        # Choose prompt templates based on whether options are requested
        if with_options:
            system_prompt_name = "question_generation_with_options"
            user_prompt_name = "question_generation_with_options_user"
        else:
            system_prompt_name = "question_generation"
            user_prompt_name = "question_generation_user"

        # Format system prompt
        system_prompt = self.format_prompt(
            system_prompt_name,
            num_questions=num_questions,
            style_instruction=style_instruction,
            style_note=style_note,
            casual_greeting_instruction=casual_greeting_instruction,
            **kwargs,
        )

        # Format user prompt
        user_prompt = self.format_prompt(
            user_prompt_name,
            num_questions=num_questions,
            text=text,
            style_suffix=style_suffix,
            **kwargs,
        )

        return system_prompt, user_prompt

    def get_answer_generation_prompts(
        self,
        question: str,
        source_text: str,
        options: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Get formatted system and user prompts for single answer generation.

        Args:
            question: The question to answer
            source_text: The source text to base the answer on
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Choose prompt templates based on whether options are provided
        if options:
            system_prompt_name = "answer_generation_with_options"
            user_prompt_name = "answer_generation_with_options_user"

            # Format options for display
            options_text = "\n".join(
                [f"{letter}) {text}" for letter, text in options.items()]
            )

            system_prompt = self.format_prompt(system_prompt_name, **kwargs)
            user_prompt = self.format_prompt(
                user_prompt_name,
                question=question,
                options_text=options_text,
                source_text=source_text,
                **kwargs,
            )
        else:
            system_prompt = self.format_prompt("answer_generation", **kwargs)
            user_prompt = self.format_prompt(
                "answer_generation_user",
                question=question,
                source_text=source_text,
                **kwargs,
            )

        return system_prompt, user_prompt

    def get_batch_answer_generation_prompts(
        self,
        questions: list[str],
        source_text: str,
        questions_with_options: Optional[list[Dict[str, Any]]] = None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Get formatted system and user prompts for batch answer generation.

        Args:
            questions: List of questions to answer
            source_text: The source text to base the answers on
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Choose prompt templates based on whether options are provided
        if questions_with_options:
            system_prompt_name = "batch_answer_generation_with_options"
            user_prompt_name = "batch_answer_generation_with_options_user"

            # Format questions with options for display
            questions_with_options_text = []
            for i, q_data in enumerate(questions_with_options):
                question_text = q_data["question"]
                options = q_data["options"]
                options_text = "\n".join(
                    [f"  {letter}) {text}" for letter, text in options.items()]
                )
                questions_with_options_text.append(
                    f"{i+1}. {question_text}\n{options_text}"
                )

            questions_formatted = "\n\n".join(questions_with_options_text)

            system_prompt = self.format_prompt(system_prompt_name, **kwargs)
            user_prompt = self.format_prompt(
                user_prompt_name,
                questions_with_options_text=questions_formatted,
                source_text=source_text,
                **kwargs,
            )
        else:
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

            system_prompt = self.format_prompt("batch_answer_generation", **kwargs)
            user_prompt = self.format_prompt(
                "batch_answer_generation_user",
                questions_text=questions_text,
                source_text=source_text,
                **kwargs,
            )

        return system_prompt, user_prompt

    def list_available_prompts(
        self, include_custom: bool = True
    ) -> Dict[str, list[str]]:
        """
        List all available prompts.

        Args:
            include_custom: Whether to include custom prompts in the listing

        Returns:
            Dictionary with 'default' and optionally 'custom' keys containing lists of prompt names
        """
        result = {"default": []}

        # List default prompts
        if self.default_prompt_dir.exists():
            for prompt_file in self.default_prompt_dir.glob("*.txt"):
                result["default"].append(prompt_file.stem)

        # List custom prompts if requested and directory exists
        if (
            include_custom
            and self.custom_prompt_dir
            and self.custom_prompt_dir.exists()
        ):
            result["custom"] = []
            for prompt_file in self.custom_prompt_dir.glob("*.txt"):
                result["custom"].append(prompt_file.stem)

        return result
