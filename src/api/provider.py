import os
import asyncio
import aiohttp
import random
from typing import Any, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from default.styles import load_default_styles
from api.prompt_loader import PromptLoader

QUESTION_STYLES: List[str] = load_default_styles()


class APIProvider:
    """API provider for question generation"""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        max_tokens: int = 8192,
        custom_url: Optional[str] = None,
        custom_prompt_dir: Optional[str] = None,
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.custom_url = custom_url
        self.rate_limit_wait = 15.0
        self.rate_limit_retries = 1
        self.base_url = self._get_base_url()
        self.headers = self._get_headers()
        self.prompt_loader = PromptLoader(custom_prompt_dir)

    def _get_base_url(self) -> str:
        """Get the base URL for the provider"""
        if self.provider_name == "other":
            if not self.custom_url:
                raise ValueError(
                    "--provider-url is required when using --provider other"
                )
            return self.custom_url

        provider_urls = {
            "featherless": "https://api.featherless.ai/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "qwen": "https://api.qwen.com/v1",
            "qwen-deepinfra": "https://api.deepinfra.com/v1/openai",
            "kimi": "https://api.moonshot.ai/v1",
            "z.ai": "https://api.z.ai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "cerebras": "https://api.cerebras.ai/v1",
            "together": "https://api.together.xyz/v1",
            "groq": "https://api.groq.com/openai/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
            "ollama": "http://localhost:11434/v1",
            "chutes": "https://llm.chutes.ai/v1",
            "huggingface": "https://api-inference.huggingface.co/v1",
        }
        return provider_urls.get(self.provider_name, "https://api.featherless.ai/v1")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for the provider"""
        # Handle environment variable naming edge cases
        provider_env_mapping = {
            "qwen-deepinfra": "QWEN_DEEPINFRA_API_KEY",
            "z.ai": "Z_AI_API_KEY",
            "other": "OTHER_API_KEY",
        }

        env_var_name = provider_env_mapping.get(
            self.provider_name,
            f"{self.provider_name.upper().replace('.', '_').replace('-', '_')}_API_KEY",
        )

        api_key = os.getenv(env_var_name)

        # Ollama doesn't require an API key
        if self.provider_name.lower() == "ollama":
            if not api_key:
                api_key = "ollama-local"
        else:
            if not api_key:
                raise ValueError(
                    f"API key for {self.provider_name} not found in environment variables (expected: {env_var_name})"
                )

        # Provider-specific authentication
        if self.provider_name.lower() == "anthropic":
            return {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        elif self.provider_name.lower() == "openrouter":
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        elif self.provider_name.lower() == "gemini":
            return {"Content-Type": "application/json"}
        elif self.provider_name.lower() == "ollama":
            return {"Content-Type": "application/json"}
        else:
            # Default to OpenAI-compatible format (includes "other" provider)
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

    async def generate_questions(
        self,
        text: str,
        num_questions: int,
        session: aiohttp.ClientSession,
        selected_style: Optional[str] = None,
        with_options: bool = False,
    ) -> tuple[str, str]:
        """Generate questions based on the provided text and return response with selected style"""
        try:
            messages, selected_style = self._prepare_messages(
                text, num_questions, selected_style, with_options
            )

            if self.provider_name.lower() == "gemini":
                response = await self._generate_gemini_response(messages, session)
            elif self.provider_name.lower() == "anthropic":
                response = await self._generate_anthropic_response(messages, session)
            else:
                response = await self._generate_openai_compatible_response(
                    messages, session
                )

            return response, selected_style

        except Exception as e:
            logger.error(f"Error generating questions with {self.provider_name}: {e}")
            raise

    def _prepare_messages(
        self,
        text: str,
        num_questions: int,
        selected_style: Optional[str],
        with_options: bool = False,
    ) -> tuple[List[Dict[str, str]], str]:
        """Prepare messages for question generation and return the selected style.

        If selected_style is None, a style is chosen randomly from QUESTION_STYLES.
        If selected_style is an empty string, no style is used (completely neutral).
        """
        # Choose style (respect override if provided)
        if selected_style is None:
            selected_style = random.choice(QUESTION_STYLES)
        elif selected_style == "":
            # No style requested - keep it as empty string for neutral handling
            pass

        # Use prompt loader to get formatted prompts
        system_prompt, user_prompt = self.prompt_loader.get_question_generation_prompts(
            num_questions=num_questions,
            selected_style=selected_style,
            text=text,
            with_options=with_options,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], selected_style or "no-style"

    async def _generate_openai_compatible_response(
        self, messages: List[Dict[str, str]], session: aiohttp.ClientSession
    ) -> str:
        """Generate response using OpenAI-compatible API"""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.8,  # Higher temperature for more creative questions
            "top_p": 0.9,
        }

        result = await self._post_json_with_rate_limit(session, url, payload)
        return result["choices"][0]["message"]["content"].strip()

    async def _generate_anthropic_response(
        self, messages: List[Dict[str, str]], session: aiohttp.ClientSession
    ) -> str:
        """Generate response using Anthropic API"""
        url = f"{self.base_url}/messages"

        # Convert messages format for Anthropic
        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), ""
        )
        user_messages = [msg for msg in messages if msg["role"] != "system"]

        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": user_messages,
            "temperature": 0.8,
        }

        if system_message:
            payload["system"] = system_message

        result = await self._post_json_with_rate_limit(
            session, url, payload, provider_name="Anthropic"
        )
        return result["content"][0]["text"].strip()

    async def _generate_gemini_response(
        self, messages: List[Dict[str, str]], session: aiohttp.ClientSession
    ) -> str:
        """Generate response using Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={api_key}"

        # Convert messages to Gemini format
        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), ""
        )
        user_content = next(
            (msg["content"] for msg in messages if msg["role"] == "user"), ""
        )

        full_prompt = (
            f"{system_message}\n\n{user_content}" if system_message else user_content
        )

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": self.max_tokens,
            },
        }

        result = await self._post_json_with_rate_limit(
            session, url, payload, provider_name="Gemini"
        )
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def _post_json_with_rate_limit(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: Dict[str, Any],
        provider_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST JSON with rate limit handling and retries.

        - Retries on HTTP 429/503 with Retry-After header (seconds) or fallback to self.rate_limit_wait.
        - Raises Exception on non-OK responses after retries.
        """
        attempts = self.rate_limit_retries + 1
        last_error_text = None
        last_status = None
        prov = provider_name or self.provider_name

        for attempt in range(1, attempts + 1):
            async with session.post(
                url, headers=self.headers, json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()

                last_status = response.status
                last_error_text = await response.text()

                # Handle rate limit or service unavailable
                if response.status in (429, 503):
                    retry_after = response.headers.get("Retry-After")
                    wait_s: float
                    if retry_after:
                        try:
                            wait_s = float(retry_after)
                        except ValueError:
                            wait_s = self.rate_limit_wait
                    else:
                        wait_s = self.rate_limit_wait

                    logger.warning(
                        f"Rate limit detected from {prov} (status {response.status}). "
                        f"Waiting {wait_s}s before retry {attempt}/{attempts-1 if attempts>1 else 0}..."
                    )
                    await asyncio.sleep(wait_s)
                    continue

                # Non-retryable error
                break

        raise Exception(
            f"{prov} API request failed with status {last_status}: {last_error_text}"
        )

    async def generate_answer(
        self,
        question: str,
        source_text: str,
        session: aiohttp.ClientSession,
        options: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate an answer for a given question based on the source text"""
        try:
            messages = self._prepare_answer_messages(question, source_text, options)

            if self.provider_name.lower() == "gemini":
                response = await self._generate_gemini_response(messages, session)
            elif self.provider_name.lower() == "anthropic":
                response = await self._generate_anthropic_response(messages, session)
            else:
                response = await self._generate_openai_compatible_response(
                    messages, session
                )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating answer with {self.provider_name}: {e}")
            raise

    async def generate_answers_batch(
        self,
        questions: List[str],
        source_text: str,
        session: aiohttp.ClientSession,
        questions_with_options: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate answers for multiple questions in a single request"""
        try:
            messages = self._prepare_batch_answer_messages(
                questions, source_text, questions_with_options
            )

            if self.provider_name.lower() == "gemini":
                response = await self._generate_gemini_response(messages, session)
            elif self.provider_name.lower() == "anthropic":
                response = await self._generate_anthropic_response(messages, session)
            else:
                response = await self._generate_openai_compatible_response(
                    messages, session
                )

            return response.strip()

        except Exception as e:
            logger.error(
                f"Error generating batch answers with {self.provider_name}: {e}"
            )
            raise

    def _prepare_answer_messages(
        self, question: str, source_text: str, options: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for answering a single question"""
        system_prompt, user_prompt = self.prompt_loader.get_answer_generation_prompts(
            question=question, source_text=source_text, options=options
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _prepare_batch_answer_messages(
        self,
        questions: List[str],
        source_text: str,
        questions_with_options: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare messages for answering multiple questions in a single request"""
        system_prompt, user_prompt = (
            self.prompt_loader.get_batch_answer_generation_prompts(
                questions=questions,
                source_text=source_text,
                questions_with_options=questions_with_options,
            )
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
