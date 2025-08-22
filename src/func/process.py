from typing import Any, Dict, List
import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from api.question import QuestionGenerator
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

