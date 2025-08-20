"""
Question Generation System using multi-provider APIs.
This system generates questions based on the 'text' column from datasets,
saving each question as a separate JSON record with metadata.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_default_styles() -> List[str]:
    """Load default question styles from file"""
    default_styles_file = Path(__file__).parent.parent / "default_styles.txt"
    
    try:
        with open(default_styles_file, 'r', encoding='utf-8') as f:
            styles = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return styles
    except Exception as e:
        logger.warning(f"Could not load default styles file: {e}. Using fallback styles.")
        # Fallback styles if file can't be loaded
        return [
            "formal and academic",
            "casual and conversational",
            "funny and humorous",
            "thought-provoking and philosophical",
            "practical and application-focused",
            "analytical and critical thinking",
            "creative and imaginative",
            "simple and straightforward",
            "detailed and comprehensive",
            "concise and direct",
        ]


# Load default question styles
QUESTION_STYLES: List[str] = load_default_styles()


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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


def is_local_file(dataset_name: str) -> bool:
    """Check if the dataset name is a local file path"""
    return (dataset_name.endswith('.jsonl') or dataset_name.endswith('.json')) and os.path.exists(dataset_name)

def is_local_parquet(dataset_name: str) -> bool:
    """Check if the dataset name is a local parquet file path"""
    return dataset_name.endswith('.parquet') and os.path.exists(dataset_name)

class APIProvider:
    """API provider for question generation"""
    
    def __init__(self, provider_name: str, model_name: str, max_tokens: int = 8192):
        self.provider_name = provider_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.base_url = self._get_base_url()
        self.headers = self._get_headers()
    
    def _get_base_url(self) -> str:
        """Get the base URL for the provider"""
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
        }
        
        env_var_name = provider_env_mapping.get(
            self.provider_name, 
            f"{self.provider_name.upper().replace('.', '_').replace('-', '_')}_API_KEY"
        )
        
        api_key = os.getenv(env_var_name)
        
        # Ollama doesn't require an API key
        if self.provider_name.lower() == "ollama":
            if not api_key:
                api_key = "ollama-local"
        else:
            if not api_key:
                raise ValueError(f"API key for {self.provider_name} not found in environment variables (expected: {env_var_name})")
        
        # Provider-specific authentication
        if self.provider_name.lower() == "anthropic":
            return {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
        elif self.provider_name.lower() == "openrouter":
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        elif self.provider_name.lower() == "gemini":
            return {
                "Content-Type": "application/json"
            }
        elif self.provider_name.lower() == "ollama":
            return {
                "Content-Type": "application/json"
            }
        else:
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
    
    async def generate_questions(self, text: str, num_questions: int, session: aiohttp.ClientSession, selected_style: Optional[str] = None) -> tuple[str, str]:
        """Generate questions based on the provided text and return response with selected style"""
        try:
            messages, selected_style = self._prepare_messages(text, num_questions, selected_style)
            
            if self.provider_name.lower() == "gemini":
                response = await self._generate_gemini_response(messages, session)
            elif self.provider_name.lower() == "anthropic":
                response = await self._generate_anthropic_response(messages, session)
            else:
                response = await self._generate_openai_compatible_response(messages, session)
            
            return response, selected_style
        
        except Exception as e:
            logger.error(f"Error generating questions with {self.provider_name}: {e}")
            raise
    
    def _prepare_messages(self, text: str, num_questions: int, selected_style: Optional[str]) -> tuple[List[Dict[str, str]], str]:
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
        
        system_prompt = f"""You are an expert question generator. Your task is to create {num_questions} diverse and engaging questions based on the provided text.

Guidelines:
- Generate exactly {num_questions} questions
{f"- Use a {selected_style} style" if selected_style else "- Generate questions in a straightforward manner"}
- Questions should be relevant to the content, but don't directly include any phrases like "in this text", "in this article", "the text", "regarding to the text", etc. Use references instead if possible, and when not, be more general. We want answers to the questions to be discoverable through web search
- Vary question types (what, how, why, when, where, is, are, etc.)
{f"- Try to use greetings where applicable - like human would do when the style is informal" if selected_style and "casual" in selected_style.lower() else ""}
- Make questions engaging and thought-provoking
- Each question should be on a separate line
- Number each question (1., 2., 3., etc.)
- Questions should be self-contained and understandable without the original text

{f"Style: {selected_style}" if selected_style else ""}"""
        
        user_prompt = f"""Based on the following text, generate {num_questions} questions{f" in a {selected_style} style" if selected_style else ""}:

Text:
{text}

Please generate exactly {num_questions} numbered questions:"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], selected_style or "no-style"
    
    async def _generate_openai_compatible_response(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        """Generate response using OpenAI-compatible API"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.8,  # Higher temperature for more creative questions
            "top_p": 0.9
        }
        
        async with session.post(url, headers=self.headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"].strip()
    
    async def _generate_anthropic_response(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        """Generate response using Anthropic API"""
        url = f"{self.base_url}/messages"
        
        # Convert messages format for Anthropic
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_messages = [msg for msg in messages if msg["role"] != "system"]
        
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": user_messages,
            "temperature": 0.8
        }
        
        if system_message:
            payload["system"] = system_message
        
        async with session.post(url, headers=self.headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Anthropic API request failed with status {response.status}: {error_text}")
            
            result = await response.json()
            return result["content"][0]["text"].strip()
    
    async def _generate_gemini_response(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        """Generate response using Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={api_key}"
        
        # Convert messages to Gemini format
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        full_prompt = f"{system_message}\n\n{user_content}" if system_message else user_content
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": self.max_tokens
            }
        }
        
        async with session.post(url, headers=self.headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Gemini API request failed with status {response.status}: {error_text}")
            
            result = await response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def generate_answer(self, question: str, source_text: str, session: aiohttp.ClientSession) -> str:
        """Generate an answer for a given question based on the source text"""
        try:
            messages = self._prepare_answer_messages(question, source_text)

            if self.provider_name.lower() == "gemini":
                response = await self._generate_gemini_response(messages, session)
            elif self.provider_name.lower() == "anthropic":
                response = await self._generate_anthropic_response(messages, session)
            else:
                response = await self._generate_openai_compatible_response(messages, session)

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating answer with {self.provider_name}: {e}")
            raise

    async def generate_answers_batch(self, questions: List[str], source_text: str, session: aiohttp.ClientSession) -> str:
        """Generate answers for multiple questions in a single request"""
        try:
            messages = self._prepare_batch_answer_messages(questions, source_text)

            if self.provider_name.lower() == "gemini":
                response = await self._generate_gemini_response(messages, session)
            elif self.provider_name.lower() == "anthropic":
                response = await self._generate_anthropic_response(messages, session)
            else:
                response = await self._generate_openai_compatible_response(messages, session)

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating batch answers with {self.provider_name}: {e}")
            raise

    def _prepare_answer_messages(self, question: str, source_text: str) -> List[Dict[str, str]]:
        """Prepare messages for answering a single question"""
        system_prompt = """You are an expert assistant that provides accurate, comprehensive answers to questions based on provided text.

Guidelines:
- Answer the question directly and thoroughly based on the provided text
- Use information from the text to support your answer
- Answer should be relevant to the text, but don't directly include any phrases like "in this text", "in this article", "the text", "regarding to the text", etc. We want answers to the questions to be based on the context without explicitly mentioning it
- If the text doesn't contain enough information to fully answer the question, acknowledge this limitation
- Be clear, concise, and well-structured in your response
- Do not make up information that isn't in the provided text
- If the question cannot be answered from the text, explain why"""

        user_prompt = f"""Based on the following text, please answer this question:

Question: {question}

Text:
{source_text}

Please provide a comprehensive answer based on the information in the text:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _prepare_batch_answer_messages(self, questions: List[str], source_text: str) -> List[Dict[str, str]]:
        """Prepare messages for answering multiple questions in a single request"""
        system_prompt = """You are an expert assistant that provides accurate, comprehensive answers to multiple questions based on provided text.

Guidelines:
- Answer each question directly and thoroughly based on the provided text
- Use information from the text to support your answers
- Answer should be relevant to the text, but don't directly include any phrases like "in this text", "in this article", "the text", "regarding to the text", etc. We want answers to the questions to be based on the context without explicitly mentioning it
- If the text doesn't contain enough information to fully answer a question, acknowledge this limitation
- Be clear, concise, and well-structured in your responses
- Do not make up information that isn't in the provided text
- Number each answer to correspond with the question number
- Format each answer on a separate line starting with the number (1., 2., 3., etc.)"""

        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        user_prompt = f"""Based on the following text, please answer these questions:

Questions:
{questions_text}

Text:
{source_text}

Please provide comprehensive answers based on the information in the text, numbering each answer:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class QuestionGenerator:
    """Manages question generation process"""

    def __init__(self, provider: APIProvider, num_questions: int = 3, verbose: bool = False, sleep_between_requests: float = 0.0, styles: Optional[List[str]] = None,
                 answer_provider: Optional[APIProvider] = None, answer_single_request: bool = False):
        self.provider = provider
        self.num_questions = num_questions
        self.verbose = verbose
        self.sleep_between_requests = sleep_between_requests
        self.styles = styles
        self.answer_provider = answer_provider
        self.answer_single_request = answer_single_request
    
    async def generate_questions_for_text(self, text: str, metadata: Optional[Dict[str, Any]], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Generate questions for a single text and return structured result"""
        try:
            if self.verbose:
                print(f"\n🤖 Generating {self.num_questions} questions using {self.provider.model_name}...")
                print(f"📝 Text preview: {text[:100]}...")
            
            # Choose style for this item (random if multiple provided)
            chosen_style = None
            if self.styles is not None:
                if len(self.styles) > 0:
                    chosen_style = random.choice(self.styles)
                else:
                    # Empty list means --no-style was used
                    chosen_style = ""
            # If self.styles is None, use default behavior (random from QUESTION_STYLES)

            # Generate questions
            questions_response, selected_style = await self.provider.generate_questions(text, self.num_questions, session, chosen_style)
            
            if self.verbose:
                print(f"🎨 Using style: {selected_style}")
            
            # Rate limiting
            if self.sleep_between_requests > 0:
                if self.verbose:
                    print(f"⏱️  Sleeping {self.sleep_between_requests}s for rate limiting...")
                await asyncio.sleep(self.sleep_between_requests)
            
            # Parse questions from response
            questions = self._parse_questions(questions_response)

            if self.verbose:
                print(f"✅ Generated {len(questions)} questions successfully!")
                for i, q in enumerate(questions, 1):
                    print(f"  {i}. {q}")

            # Generate answers if requested
            answers = []
            answer_errors = []
            if self.answer_provider and questions:
                if self.verbose:
                    print(f"🤖 Generating answers using {self.answer_provider.model_name}...")

                try:
                    if self.answer_single_request:
                        # Generate all answers in a single request
                        if self.verbose:
                            print(f"📝 Generating all {len(questions)} answers in single request...")

                        batch_response = await self.answer_provider.generate_answers_batch(questions, text, session)
                        answers = self._parse_batch_answers(batch_response, len(questions))

                        # Ensure we have the right number of answers
                        while len(answers) < len(questions):
                            answers.append("Error: Could not generate answer")
                            answer_errors.append(f"Missing answer for question {len(answers)}")

                        if self.verbose:
                            print(f"✅ Generated {len(answers)} answers in batch!")
                    else:
                        # Generate answers one by one
                        for i, question in enumerate(questions):
                            if self.verbose:
                                print(f"📝 Generating answer {i+1}/{len(questions)}...")

                            try:
                                answer = await self.answer_provider.generate_answer(question, text, session)
                                answers.append(answer)

                                if self.verbose:
                                    print(f"✅ Answer {i+1}: {answer[:100]}...")

                                # Rate limiting between answer requests
                                if self.sleep_between_requests > 0 and i < len(questions) - 1:
                                    if self.verbose:
                                        print(f"⏱️  Sleeping {self.sleep_between_requests}s for rate limiting...")
                                    await asyncio.sleep(self.sleep_between_requests)

                            except Exception as e:
                                error_msg = f"Error generating answer for question {i+1}: {str(e)}"
                                answers.append("Error: Could not generate answer")
                                answer_errors.append(error_msg)
                                if self.verbose:
                                    print(f"❌ {error_msg}")

                except Exception as e:
                    error_msg = f"Error in batch answer generation: {str(e)}"
                    answer_errors.append(error_msg)
                    # Fill with error messages
                    answers = ["Error: Could not generate answer"] * len(questions)
                    if self.verbose:
                        print(f"❌ {error_msg}")

            result = {
                "source_text": text,
                "questions": questions,
                "raw_response": questions_response,
                "metadata": metadata or {},
                "generation_settings": {
                    "provider": self.provider.provider_name,
                    "model": self.provider.model_name,
                    "style": selected_style,
                    "num_questions_requested": self.num_questions,
                    "num_questions_generated": len(questions),
                    "max_tokens": self.provider.max_tokens
                },
                "timestamp": datetime.now().isoformat()
            }

            # Add answer-related fields if answers were generated
            if self.answer_provider:
                result["answers"] = answers
                result["generation_settings"]["answer_provider"] = self.answer_provider.provider_name
                result["generation_settings"]["answer_model"] = self.answer_provider.model_name
                result["generation_settings"]["answer_single_request"] = self.answer_single_request
                if answer_errors:
                    result["answer_errors"] = answer_errors

            return result
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return {
                "source_text": text,
                "questions": [],
                "error": str(e),
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_questions(self, response: str) -> List[str]:
        """Parse individual questions from the model response"""
        lines = response.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.) and clean up
            # Handle various numbering formats
            import re
            cleaned = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. ", "2. ", etc.
            cleaned = re.sub(r'^\d+\)\s*', '', cleaned)  # Remove "1) ", "2) ", etc.
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)  # Remove "- " or "* "
            cleaned = cleaned.strip()
            
            if cleaned and cleaned.endswith('?'):  # Only add actual questions
                questions.append(cleaned)
        
        return questions

    def _parse_batch_answers(self, response: str, expected_count: int) -> List[str]:
        """Parse individual answers from a batch response"""
        lines = response.strip().split('\n')
        answers = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering (1., 2., etc.) and clean up
            import re
            cleaned = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. ", "2. ", etc.
            cleaned = re.sub(r'^\d+\)\s*', '', cleaned)  # Remove "1) ", "2) ", etc.
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)  # Remove "- " or "* "
            cleaned = cleaned.strip()

            if cleaned:  # Add any non-empty cleaned line as an answer
                answers.append(cleaned)

        # If we didn't get enough answers, try to split by common patterns
        if len(answers) < expected_count and len(answers) == 1:
            # Try to split a single long response
            single_response = answers[0]
            # Look for numbered patterns within the response
            import re
            numbered_parts = re.split(r'\n\s*\d+\.\s*', single_response)
            if len(numbered_parts) > 1:
                # Remove empty first part if it exists
                if not numbered_parts[0].strip():
                    numbered_parts = numbered_parts[1:]
                answers = [part.strip() for part in numbered_parts if part.strip()]

        return answers


async def process_dataset_item(
    item: Dict[str, Any],
    question_generator: QuestionGenerator,
    session: aiohttp.ClientSession,
    item_index: int,
    text_column: str = "text"
) -> List[Dict[str, Any]]:
    """Process a single dataset item and generate multiple question records"""
    
    # Extract text from the specified column
    text = item.get(text_column, "")
    if not text:
        logger.warning(f"No text found in column '{text_column}' for item {item_index}")
        return []
    
    # Create metadata
    metadata = {
        "original_item_index": item_index,
        "text_column": text_column,
    }
    
    # Generate questions
    result = await question_generator.generate_questions_for_text(text, metadata, session)
    
    # Create separate records for each question
    question_records = []
    if "error" not in result:
        answers = result.get("answers", [])
        answer_errors = result.get("answer_errors", [])

        for i, question in enumerate(result["questions"]):
            question_record = {
                "input": question,  # The generated question becomes the input
                "source_text": result["source_text"],
                "question_index": i + 1,
                "total_questions": len(result["questions"]),
                "metadata": result["metadata"],
                "generation_settings": result["generation_settings"],
                "timestamp": result["timestamp"]
            }

            # Add answer if available
            if i < len(answers):
                answer = answers[i]
                if answer == "Error: Could not generate answer":
                    question_record["output"] = "error"
                    question_record["answer_error"] = "Unable to generate answer for this question"
                else:
                    question_record["output"] = answer
            elif answers:  # If we have some answers but not for this question
                question_record["output"] = "error"
                question_record["answer_error"] = "Unable to generate answer for this question"

            # Add answer errors if any
            if answer_errors:
                question_record["answer_errors"] = answer_errors

            question_records.append(question_record)
    else:
        # Create error record
        error_record = {
            "input": "",
            "source_text": text,
            "error": result["error"],
            "metadata": metadata,
            "timestamp": result["timestamp"]
        }
        question_records.append(error_record)
    
    return question_records


async def main():
    parser = argparse.ArgumentParser(description="Question Generation System based on dataset text column")
    parser.add_argument("dataset", help="HuggingFace dataset name or local JSONL file path")
    parser.add_argument("--provider", required=True, 
                        choices=["featherless", "openai", "anthropic", "qwen", "qwen-deepinfra", "kimi", 
                                "z.ai", "openrouter", "cerebras", "together", "groq", "gemini", "ollama", "chutes", "huggingface"],
                        help="API provider to use")
    parser.add_argument("--model", required=True, help="Model name")
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
                "z.ai", "openrouter", "cerebras", "together", "groq", "gemini", "ollama", "chutes", "huggingface"],
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

    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if args.verbose:
        if is_local_file(args.dataset):
            print(f"📂 Loading local JSONL file: {args.dataset}")
        elif is_local_parquet(args.dataset):
            print(f"📂 Loading local Parquet file: {args.dataset}")
        else:
            print(f"📂 Loading HuggingFace dataset: {args.dataset}")
    
    try:
        if is_local_file(args.dataset):
            # Load local JSONL file
            dataset_list = load_jsonl_file(args.dataset)
            if args.verbose:
                print(f"✅ Loaded {len(dataset_list)} items from local JSONL file")
        elif is_local_parquet(args.dataset):
            # Load local Parquet file
            dataset = load_from_disk("parquet", args.dataset)
            dataset_list = list(dataset)
            if args.verbose:
                print(f"✅ Loaded {len(dataset_list)} items from local Parquet file")
        else:
            # Load HuggingFace dataset
            dataset = load_dataset(args.dataset, split=args.dataset_split)
            dataset_list = list(dataset)
            if args.verbose:
                print(f"✅ Loaded {len(dataset_list)} items from HuggingFace dataset")
    except Exception as e:
        if is_local_file(args.dataset):
            logger.error(f"Failed to load JSONL file {args.dataset}: {e}")
        elif is_local_parquet(args.dataset):
            logger.error(f"Failed to load Parquet file {args.dataset}: {e}")
        else:
            logger.error(f"Failed to load HuggingFace dataset {args.dataset}: {e}")
        return
    
    # Shuffle if requested
    if args.shuffle:
        random.shuffle(dataset_list)
        if args.verbose:
            print("🔀 Dataset shuffled")
    
    # Apply start and end index slicing
    original_size = len(dataset_list)
    start_idx = max(0, args.start_index)
    end_idx = args.end_index if args.end_index is not None else len(dataset_list)
    end_idx = min(end_idx, len(dataset_list))
    
    if start_idx >= len(dataset_list):
        logger.error(f"Start index {start_idx} is beyond dataset size {len(dataset_list)}")
        return
    
    if start_idx >= end_idx:
        logger.error(f"Start index {start_idx} must be less than end index {end_idx}")
        return
    
    dataset_list = dataset_list[start_idx:end_idx]
    
    if args.verbose and (start_idx > 0 or end_idx < original_size):
        print(f"📊 Selected range: items {start_idx} to {end_idx-1} (total: {len(dataset_list)} items)")
    
    # Limit items if specified (this applies after range selection)
    if args.max_items:
        dataset_list = dataset_list[:args.max_items]
        if args.verbose:
            print(f"📊 Limited to {len(dataset_list)} items")
    
    dataset_size = len(dataset_list)
    if args.verbose:
        print(f"📋 Dataset loaded with {dataset_size} items")
        print(f"📝 Text column: '{args.text_column}'")
        print(f"❓ Questions per text: {args.num_questions}")
    
    # Check if text column exists in dataset
    if dataset_size > 0 and args.text_column not in dataset_list[0]:
        sample = dataset_list[0]
        available_cols = list(sample.keys()) if isinstance(sample, dict) else []
        logger.error(
            f"Column '{args.text_column}' not found in dataset. Available columns: {available_cols}"
        )
        return
    
    # Initialize provider
    try:
        provider = APIProvider(args.provider, args.model, args.max_tokens)
        if args.verbose:
            print(f"🤖 Initialized provider: {args.provider}/{args.model}")
    except Exception as e:
        logger.error(f"Failed to initialize provider: {e}")
        return

    # Initialize answer provider if requested
    answer_provider = None
    if args.with_answer:
        try:
            answer_provider_name = args.answer_provider if args.answer_provider else args.provider
            answer_model = args.answer_model if args.answer_model else args.model
            answer_provider = APIProvider(answer_provider_name, answer_model, args.max_tokens)
            if args.verbose:
                print(f"🤖 Initialized answer provider: {answer_provider_name}/{answer_model}")
                if args.answer_single_request:
                    print("📝 Answer mode: Single request for all questions")
                else:
                    print("📝 Answer mode: One request per question")
        except Exception as e:
            logger.error(f"Failed to initialize answer provider: {e}")
            return
    
    # Initialize question generator
    # Prepare styles list (if provided)
    styles_list: Optional[List[str]] = None
    
    # Handle mutually exclusive style options
    style_options_count = sum([
        bool(args.style), 
        bool(args.no_style), 
        bool(args.styles_file)
    ])
    
    if style_options_count > 1:
        logger.error("Only one of --style, --no-style, or --styles-file can be specified")
        return
    
    if args.no_style:
        styles_list = []  # Empty list means no style
    elif args.styles_file:
        try:
            with open(args.styles_file, 'r', encoding='utf-8') as f:
                styles_list = [line.strip() for line in f if line.strip()]
            if args.verbose:
                print(f"📋 Loaded {len(styles_list)} styles from {args.styles_file}")
        except Exception as e:
            logger.error(f"Failed to load styles file {args.styles_file}: {e}")
            return
    elif args.style:
        styles_list = [s.strip() for s in str(args.style).split(',') if s.strip()]

    question_generator = QuestionGenerator(
        provider=provider,
        num_questions=args.num_questions,
        verbose=args.verbose,
        sleep_between_requests=args.sleep_between_requests,
        styles=styles_list,
        answer_provider=answer_provider,
        answer_single_request=args.answer_single_request if args.with_answer else False,
    )
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if is_local_file(args.dataset):
        # Use the base filename for local files
        dataset_name = Path(args.dataset).stem
    elif is_local_parquet(args.dataset):
        dataset_name = Path(args.dataset).stem
    else:
        # Use the dataset name for HuggingFace datasets
        dataset_name = args.dataset.replace('/', '_')
    
    # Add range information to filename if a custom range is used
    range_suffix = ""
    if args.start_index > 0 or args.end_index is not None:
        end_str = str(args.end_index) if args.end_index is not None else "end"
        range_suffix = f"_range{args.start_index}-{end_str}"
    
    filename = f"questions_{timestamp}_{dataset_name}_{args.provider}_{args.model.replace('/', '_')}{range_suffix}.jsonl"
    output_file = output_dir / filename
    
    if args.verbose:
        print(f"\n🚀 Starting question generation...")
        print(f"📊 Processing {dataset_size} items")
        if args.start_index > 0 or args.end_index is not None:
            end_display = args.end_index if args.end_index is not None else "end"
            print(f"📍 Range: items {args.start_index} to {end_display} from original dataset")
        if is_local_file(args.dataset):
            print(f"📁 Source: Local JSONL file ({Path(args.dataset).name})")
        elif is_local_parquet(args.dataset):
            print(f"📂 Source: Local Parquet file ({Path(args.dataset).name})")
        else:
            print(f"🤗 Source: HuggingFace dataset ({args.dataset})")
        print(f"🤖 Model: {args.provider}/{args.model}")
        if args.with_answer:
            answer_provider_display = args.answer_provider if args.answer_provider else args.provider
            answer_model_display = args.answer_model if args.answer_model else args.model
            print(f"💬 Answer model: {answer_provider_display}/{answer_model_display}")
            print(f"📝 Answer mode: {'Single request' if args.answer_single_request else 'One per question'}")
        print(f"❓ Expected total questions: {dataset_size * args.num_questions}")
        print(f"💾 Output file: {output_file}")
        print("=" * 80)
    
    # Process dataset
    connector = aiohttp.TCPConnector(limit=args.num_workers * 2)
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(args.num_workers)
        
        async def process_with_semaphore(item, index):
            async with semaphore:
                return await process_dataset_item(item, question_generator, session, index, args.text_column)
        
        tasks = [
            process_with_semaphore(item, args.start_index + idx) 
            for idx, item in enumerate(dataset_list)
        ]
        
        successful_questions = 0
        successful_answers = 0
        failed_items = 0

        if args.with_answer:
            progress_desc = "Generating Q&A" if not args.verbose else "Q&A generation (verbose)"
        else:
            progress_desc = "Generating questions" if not args.verbose else "Question generation (verbose)"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=progress_desc, disable=False):
                try:
                    question_records = await task
                    
                    if question_records:
                        for record in question_records:
                            if "error" not in record:
                                successful_questions += 1
                                if "output" in record and record["output"] != "error":
                                    successful_answers += 1
                                if args.verbose:
                                    question_preview = record['input'][:100]
                                    if args.with_answer and "output" in record and record["output"] != "error":
                                        answer_preview = record['output'][:50] if record['output'] else "No answer"
                                        print(f"✅ Q&A {successful_questions}: {question_preview}... → {answer_preview}...")
                                    else:
                                        print(f"✅ Question {successful_questions}: {question_preview}...")
                            else:
                                failed_items += 1
                                if args.verbose:
                                    print(f"❌ Failed to generate questions: {record['error']}")

                            # Write each question as a separate JSON line
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            f.flush()
                    else:
                        failed_items += 1
                        if args.verbose:
                            print(f"❌ No questions generated for item")
                    
                    # Rate limiting between items
                    if args.sleep_between_items > 0:
                        if args.verbose:
                            print(f"⏱️  Sleeping {args.sleep_between_items}s between items...")
                        await asyncio.sleep(args.sleep_between_items)
                
                except Exception as e:
                    failed_items += 1
                    logger.error(f"Task failed: {e}")
                    if args.verbose:
                        print(f"❌ Task failed: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    if args.with_answer:
        print(f"🎉 Question & Answer Generation Complete!")
    else:
        print(f"🎉 Question Generation Complete!")
    print(f"{'='*60}")
    print(f"📊 Dataset items processed: {dataset_size}")
    print(f"✅ Questions generated: {successful_questions}")
    if args.with_answer:
        print(f"💬 Answers generated: {successful_answers}")
        print(f"📈 Answer success rate: {(successful_answers/successful_questions*100):.1f}%" if successful_questions > 0 else "📈 Answer success rate: 0.0%")
    print(f"❌ Failed items: {failed_items}")
    print(f"📈 Question success rate: {(successful_questions/(dataset_size*args.num_questions)*100):.1f}%")
    print(f"💾 Output saved to: {output_file}")
    print(f"🔧 Provider used: {args.provider}/{args.model}")
    if args.with_answer:
        answer_provider_display = args.answer_provider if args.answer_provider else args.provider
        answer_model_display = args.answer_model if args.answer_model else args.model
        print(f"💬 Answer provider used: {answer_provider_display}/{answer_model_display}")


if __name__ == "__main__":
    asyncio.run(main())
