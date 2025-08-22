import asyncio
import aiohttp
import random
from datetime import datetime
import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from api.provider import APIProvider
from typing import List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Manages question generation process"""

    def __init__(
        self,
        provider: APIProvider,
        num_questions: int = 3,
        verbose: bool = False,
        sleep_between_requests: float = 0.0,
        styles: Optional[List[str]] = None,
        answer_provider: Optional[APIProvider] = None,
        answer_single_request: bool = False,
    ):
        self.provider = provider
        self.num_questions = num_questions
        self.verbose = verbose
        self.sleep_between_requests = sleep_between_requests
        self.styles = styles
        self.answer_provider = answer_provider
        self.answer_single_request = answer_single_request

    async def generate_questions_for_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Generate questions for a single text and return structured result"""
        try:
            if self.verbose:
                print(
                    f"\n🤖 Generating {self.num_questions} questions using {self.provider.model_name}..."
                )
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
            questions_response, selected_style = await self.provider.generate_questions(
                text, self.num_questions, session, chosen_style
            )

            if self.verbose:
                print(f"🎨 Using style: {selected_style}")

            # Rate limiting
            if self.sleep_between_requests > 0:
                if self.verbose:
                    print(
                        f"⏱️  Sleeping {self.sleep_between_requests}s for rate limiting..."
                    )
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
                    print(
                        f"🤖 Generating answers using {self.answer_provider.model_name}..."
                    )

                try:
                    if self.answer_single_request:
                        # Generate all answers in a single request
                        if self.verbose:
                            print(
                                f"📝 Generating all {len(questions)} answers in single request..."
                            )

                        batch_response = (
                            await self.answer_provider.generate_answers_batch(
                                questions, text, session
                            )
                        )
                        answers = self._parse_batch_answers(
                            batch_response, len(questions)
                        )

                        # Ensure we have the right number of answers
                        while len(answers) < len(questions):
                            answers.append("Error: Could not generate answer")
                            answer_errors.append(
                                f"Missing answer for question {len(answers)}"
                            )

                        if self.verbose:
                            print(f"✅ Generated {len(answers)} answers in batch!")
                    else:
                        # Generate answers one by one
                        for i, question in enumerate(questions):
                            if self.verbose:
                                print(f"📝 Generating answer {i+1}/{len(questions)}...")

                            try:
                                answer = await self.answer_provider.generate_answer(
                                    question, text, session
                                )
                                answers.append(answer)

                                if self.verbose:
                                    print(f"✅ Answer {i+1}: {answer[:100]}...")

                                # Rate limiting between answer requests
                                if (
                                    self.sleep_between_requests > 0
                                    and i < len(questions) - 1
                                ):
                                    if self.verbose:
                                        print(
                                            f"⏱️  Sleeping {self.sleep_between_requests}s for rate limiting..."
                                        )
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
                    "max_tokens": self.provider.max_tokens,
                },
                "timestamp": datetime.now().isoformat(),
            }

            # Add answer-related fields if answers were generated
            if self.answer_provider:
                result["answers"] = answers
                result["generation_settings"][
                    "answer_provider"
                ] = self.answer_provider.provider_name
                result["generation_settings"][
                    "answer_model"
                ] = self.answer_provider.model_name
                result["generation_settings"][
                    "answer_single_request"
                ] = self.answer_single_request
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
                "timestamp": datetime.now().isoformat(),
            }

    def _parse_questions(self, response: str) -> List[str]:
        """Parse individual questions from the model response"""
        lines = response.strip().split("\n")
        questions = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering (1., 2., etc.) and clean up
            # Handle various numbering formats
            import re

            cleaned = re.sub(r"^\d+\.\s*", "", line)  # Remove "1. ", "2. ", etc.
            cleaned = re.sub(r"^\d+\)\s*", "", cleaned)  # Remove "1) ", "2) ", etc.
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)  # Remove "- " or "* "
            cleaned = cleaned.strip()

            if cleaned and cleaned.endswith("?"):  # Only add actual questions
                questions.append(cleaned)

        return questions

    def _parse_batch_answers(self, response: str, expected_count: int) -> List[str]:
        """Parse individual answers from a batch response"""
        lines = response.strip().split("\n")
        answers = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering (1., 2., etc.) and clean up
            import re

            cleaned = re.sub(r"^\d+\.\s*", "", line)  # Remove "1. ", "2. ", etc.
            cleaned = re.sub(r"^\d+\)\s*", "", cleaned)  # Remove "1) ", "2) ", etc.
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)  # Remove "- " or "* "
            cleaned = cleaned.strip()

            if cleaned:  # Add any non-empty cleaned line as an answer
                answers.append(cleaned)

        # If we didn't get enough answers, try to split by common patterns
        if len(answers) < expected_count and len(answers) == 1:
            # Try to split a single long response
            single_response = answers[0]
            # Look for numbered patterns within the response
            import re

            numbered_parts = re.split(r"\n\s*\d+\.\s*", single_response)
            if len(numbered_parts) > 1:
                # Remove empty first part if it exists
                if not numbered_parts[0].strip():
                    numbered_parts = numbered_parts[1:]
                answers = [part.strip() for part in numbered_parts if part.strip()]

        return answers
