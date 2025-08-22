"""
Question Generation System using multi-provider APIs.
This system generates questions based on the 'text' column from datasets,
saving each question as a separate JSON record with metadata.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import asyncio
import aiohttp
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

import sys
from pathlib import Path

src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from api.provider import APIProvider
from api.question import QuestionGenerator
from api.parser import get_parser
from func.load import load_jsonl_file
from func.process import process_dataset_item
from func.validate import is_local_file, is_local_parquet

import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    parser = get_parser()
    args = parser.parse_args()

    # Validate rate limit args
    if args.rate_limit_wait < 0:
        parser.error("--rate-limit-wait must be >= 0")
    if args.rate_limit_retries < 0:
        parser.error("--rate-limit-retries must be >= 0")

    # Validate "other" provider requirements
    if args.provider == "other" and not args.provider_url:
        parser.error("--provider-url is required when using --provider other")

    if args.answer_provider == "other" and not args.provider_url:
        parser.error("--provider-url is required when using --answer-provider other")

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
        provider = APIProvider(args.provider, args.model, args.max_tokens, args.provider_url)
        # Apply rate-limit configuration
        provider.rate_limit_wait = args.rate_limit_wait
        provider.rate_limit_retries = args.rate_limit_retries
        if args.verbose:
            print(f"🤖 Initialized provider: {args.provider}/{args.model}")
            if args.provider == "other" and args.provider_url:
                print(f"🔗 Custom URL: {args.provider_url}")
            print(f"⏳ Rate limit: wait {args.rate_limit_wait}s, retries {args.rate_limit_retries}")
    except Exception as e:
        logger.error(f"Failed to initialize provider: {e}")
        return

    # Initialize answer provider if requested
    answer_provider = None
    if args.with_answer:
        try:
            answer_provider_name = args.answer_provider if args.answer_provider else args.provider
            answer_model = args.answer_model if args.answer_model else args.model
            # Use custom URL if answer provider is "other", otherwise use the same URL as main provider if it's also "other"
            answer_custom_url = args.provider_url if (answer_provider_name == "other" or args.provider == "other") else None
            answer_provider = APIProvider(answer_provider_name, answer_model, args.max_tokens, answer_custom_url)
            # Apply rate-limit configuration to answer provider as well
            answer_provider.rate_limit_wait = args.rate_limit_wait
            answer_provider.rate_limit_retries = args.rate_limit_retries
            if args.verbose:
                print(f"🤖 Initialized answer provider: {answer_provider_name}/{answer_model}")
                if answer_provider_name == "other" and answer_custom_url:
                    print(f"🔗 Answer provider custom URL: {answer_custom_url}")
                if args.answer_single_request:
                    print("📝 Answer mode: Single request for all questions")
                else:
                    print("📝 Answer mode: One request per question")
                print(f"⏳ Answer rate limit: wait {args.rate_limit_wait}s, retries {args.rate_limit_retries}")
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
                styles_list = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
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
