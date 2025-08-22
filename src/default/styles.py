from pathlib import Path
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_default_styles() -> List[str]:
    """Load default question styles from file"""
    default_styles_file = Path(__file__).parent.parent.parent / "default_styles.txt"
    
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