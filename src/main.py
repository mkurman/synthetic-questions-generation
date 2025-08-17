"""
Question Generation System using multi-provider APIs.
This system generates questions based on the 'text' column from datasets,
saving each question as a separate JSON record with metadata.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from datasets import load_dataset
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
    async def generate_questions(self, text: str, num_questions: int, session: aiohttp.ClientSession) -> tuple[str, str]:
        """Generate questions based on the provided text and return response with selected style"""
        try:
            messages, selected_style = self._prepare_messages(text, num_questions)
            
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
    
    def _prepare_messages(self, text: str, num_questions: int) -> tuple[List[Dict[str, str]], str]:
        """Prepare messages for question generation and return the selected style"""
        # Random question styles for variety
        styles = [
            "formal and academic",
            "casual and conversational", 
            "funny and humorous",
            "thought-provoking and philosophical",
            "practical and application-focused",
            "analytical and critical thinking",
            "creative and imaginative",
            "simple and straightforward",
            "detailed and comprehensive",
            "concise and direct"
        ]
        
        selected_style = random.choice(styles)
        
        system_prompt = f"""You are an expert question generator. Your task is to create {num_questions} diverse and engaging questions based on the provided text.

Guidelines:
- Generate exactly {num_questions} questions
- Use a {selected_style} style
- Questions should be relevant to the content, but don't directly include any phrases like "in this text", "in this article", "regarding to the text", etc. Use references instead if possible, and when not, be more general. We want answers to the questions to be discoverable through web search
- Vary question types (what, how, why, when, where, is, are, etc.)
- Try to use greetings where applicable - like human would do when the style is informal
- Make questions engaging and thought-provoking
- Each question should be on a separate line
- Number each question (1., 2., 3., etc.)
- Questions should be self-contained and understandable without the original text

Style: {selected_style}"""
        
        user_prompt = f"""Based on the following text, generate {num_questions} questions in a {selected_style} style:

Text:
{text}

Please generate exactly {num_questions} numbered questions:"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], selected_style
    
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


class QuestionGenerator:
    """Manages question generation process"""
    
    def __init__(self, provider: APIProvider, num_questions: int = 3, verbose: bool = False, sleep_between_requests: float = 0.0):
        self.provider = provider
        self.num_questions = num_questions
        self.verbose = verbose
        self.sleep_between_requests = sleep_between_requests
    
    async def generate_questions_for_text(self, text: str, metadata: Optional[Dict[str, Any]], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Generate questions for a single text and return structured result"""
        try:
            if self.verbose:
                print(f"\n🤖 Generating {self.num_questions} questions using {self.provider.model_name}...")
                print(f"📝 Text preview: {text[:100]}...")
            
            # Generate questions
            questions_response, selected_style = await self.provider.generate_questions(text, self.num_questions, session)
            
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
            
            return {
                "source_text": text,
                "questions": questions,
                "raw_response": questions_response,
                "metadata": metadata or {},
                "generation_settings": {
                    "provider": self.provider.provider_name,
                    "model": self.provider.model_name,
                    "num_questions_requested": self.num_questions,
                    "num_questions_generated": len(questions),
                    "max_tokens": self.provider.max_tokens
                },
                "timestamp": datetime.now().isoformat()
            }
            
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
        "source_dataset_item": item
    }
    
    # Generate questions
    result = await question_generator.generate_questions_for_text(text, metadata, session)
    
    # Create separate records for each question
    question_records = []
    if "error" not in result:
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
        else:
            print(f"📂 Loading HuggingFace dataset: {args.dataset}")
    
    try:
        if is_local_file(args.dataset):
            # Load local JSONL file
            dataset_list = load_jsonl_file(args.dataset)
            if args.verbose:
                print(f"✅ Loaded {len(dataset_list)} items from local JSONL file")
        else:
            # Load HuggingFace dataset
            dataset = load_dataset(args.dataset, split=args.dataset_split)
            dataset_list = list(dataset)
            if args.verbose:
                print(f"✅ Loaded {len(dataset_list)} items from HuggingFace dataset")
    except Exception as e:
        if is_local_file(args.dataset):
            logger.error(f"Failed to load JSONL file {args.dataset}: {e}")
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
        logger.error(f"Column '{args.text_column}' not found in dataset. Available columns: {list(dataset_list[0].keys())}")
        return
    
    # Initialize provider
    try:
        provider = APIProvider(args.provider, args.model, args.max_tokens)
        if args.verbose:
            print(f"🤖 Initialized provider: {args.provider}/{args.model}")
    except Exception as e:
        logger.error(f"Failed to initialize provider: {e}")
        return
    
    # Initialize question generator
    question_generator = QuestionGenerator(
        provider=provider,
        num_questions=args.num_questions,
        verbose=args.verbose,
        sleep_between_requests=args.sleep_between_requests
    )
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if is_local_file(args.dataset):
        # Use the base filename for local files
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
        else:
            print(f"🤗 Source: HuggingFace dataset ({args.dataset})")
        print(f"🤖 Model: {args.provider}/{args.model}")
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
        failed_items = 0
        
        progress_desc = "Generating questions" if not args.verbose else "Question generation (verbose)"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=progress_desc, disable=False):
                try:
                    question_records = await task
                    
                    if question_records:
                        for record in question_records:
                            if "error" not in record:
                                successful_questions += 1
                                if args.verbose:
                                    print(f"✅ Question {successful_questions}: {record['input'][:100]}...")
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
    print(f"🎉 Question Generation Complete!")
    print(f"{'='*60}")
    print(f"📊 Dataset items processed: {dataset_size}")
    print(f"✅ Questions generated: {successful_questions}")
    print(f"❌ Failed items: {failed_items}")
    print(f"📈 Success rate: {(successful_questions/(dataset_size*args.num_questions)*100):.1f}%")
    print(f"💾 Output saved to: {output_file}")
    print(f"🔧 Provider used: {args.provider}/{args.model}")


if __name__ == "__main__":
    asyncio.run(main())
