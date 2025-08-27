from typing import Any, Dict, List, Optional
import sys
import re
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from api.question import QuestionGenerator
import aiohttp
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _parse_multiple_choice_answer_output(answer: str) -> Optional[Dict[str, str]]:
    """Parse a multiple-choice answer output to extract letter and explanation"""
    if not answer or not isinstance(answer, str):
        return None

    # Try to find "Answer: X" pattern
    answer_match = re.search(r"Answer:\s*([A-E])", answer, re.IGNORECASE)
    answer_letter = answer_match.group(1).upper() if answer_match else None

    # Try to find "Explanation:" pattern
    explanation_match = re.search(
        r"Explanation:\s*(.+)", answer, re.IGNORECASE | re.DOTALL
    )
    explanation = explanation_match.group(1).strip() if explanation_match else None

    # If we found both components, return them
    if answer_letter and explanation:
        return {"answer": answer_letter, "explanation": explanation}

    return None


async def process_dataset_item(
    item: Dict[str, Any],
    question_generator: QuestionGenerator,
    session: aiohttp.ClientSession,
    item_index: int,
    text_column: str = "text",
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
    result = await question_generator.generate_questions_for_text(
        text, metadata, session
    )

    # Create separate records for each question
    question_records = []
    if "error" not in result:
        answers = result.get("answers", [])
        answer_errors = result.get("answer_errors", [])

        for i, question in enumerate(result["questions"]):
            # Handle both string questions and dictionary questions with options
            if isinstance(question, dict):
                # Multiple-choice question with options
                question_text = question["question"]
                question_record = {
                    "input": question_text,  # The question text becomes the input
                    "options": question["options"],  # Add the options
                    "source_text": result["source_text"],
                    "question_index": i + 1,
                    "total_questions": len(result["questions"]),
                    "metadata": result["metadata"],
                    "generation_settings": result["generation_settings"],
                    "timestamp": result["timestamp"],
                }
            else:
                # Regular string question
                question_record = {
                    "input": question,  # The generated question becomes the input
                    "source_text": result["source_text"],
                    "question_index": i + 1,
                    "total_questions": len(result["questions"]),
                    "metadata": result["metadata"],
                    "generation_settings": result["generation_settings"],
                    "timestamp": result["timestamp"],
                }

            # Add answer if available
            if i < len(answers):
                answer = answers[i]
                if answer == "Error: Could not generate answer":
                    question_record["output"] = "error"
                    question_record["answer_error"] = (
                        "Unable to generate answer for this question"
                    )
                else:
                    question_record["output"] = answer

                    # Parse multiple-choice answer components if using options
                    if isinstance(question, dict) and "options" in question_record:
                        parsed_answer = _parse_multiple_choice_answer_output(answer)
                        if parsed_answer:
                            question_record["correct_answer"] = parsed_answer["answer"]
                            question_record["explanation"] = parsed_answer[
                                "explanation"
                            ]
            elif answers:  # If we have some answers but not for this question
                question_record["output"] = "error"
                question_record["answer_error"] = (
                    "Unable to generate answer for this question"
                )

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
            "timestamp": result["timestamp"],
        }
        question_records.append(error_record)

    return question_records
