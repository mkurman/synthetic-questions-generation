# Synthetic Questions Generation

Generate diverse, engaging questions from text using multiple LLM providers (OpenAI-compatible, Anthropic, Gemini, OpenRouter, Groq, Together, Cerebras, Qwen/DeepInfra, Kimi, Z.ai, Ollama, Chutes, Hugging Face).

To diversify outputs, the generator randomly selects a writing style for each item (e.g., formal and academic; casual and conversational; funny and humorous; thought‑provoking and philosophical; practical and application‑focused; analytical and critical; creative and imaginative; simple and straightforward; detailed and comprehensive; or concise and direct).

## Quick start

```bash
# 1) Create a venv (optional)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set an API key for your chosen provider (example: OpenRouter)
export OPENROUTER_API_KEY=your_api_key_here

# 4) Run
python3 src/main.py mkurman/hindawi-journals-2007-2023 \
	--provider openrouter \
	--model qwen/qwen3-235b-a22b-2507 \
	--output-dir ./data/questions_openrouter \
	--start-index 0 \
	--end-index 10 \
	--num-questions 5 \
	--text-column text \
	--verbose
```

See `example.sh` for a ready-to-run snippet.

## New Features

### YAML Configuration Support

You can now use YAML configuration files for easier management:

```bash
# Using YAML configuration
python3 src/main.py --config configs/example.yaml

# Override specific settings
python3 src/main.py --config configs/example.yaml --provider anthropic --model claude-3-sonnet
```

### Custom System Prompts

Customize the system prompts used for question and answer generation:

```bash
# Using custom prompts
python3 src/main.py --config configs/example.yaml --custom-prompts ./my_prompts
```

### Multiple-Choice Questions

Generate multiple-choice questions with options A, B, C, D, E:

```bash
# Generate multiple-choice questions
python3 src/main.py --config configs/example.yaml --with-options
```

See [CONFIGURATION.md](CONFIGURATION.md) for detailed documentation on these features.

## Requirements

- Python 3.9+ (uses modern typing like `list[str]`)
- Install Python packages: `pip install -r requirements.txt`

Contents of `requirements.txt`:

- aiohttp
- datasets
- tqdm
- PyYAML

## Usage

The tool accepts either a Hugging Face dataset name (e.g., `mkurman/hindawi-journals-2007-2023`) or a path to a local `.jsonl`/`.json` file. It reads a text field (default `text`), asks an LLM to generate N questions, and writes each question as a JSONL record.

Basic CLI:

```bash
python3 src/main.py <dataset_or_jsonl_path> \
	--provider <provider> \
	--model <model_name> \
	--output-dir <dir>
```

Key options:

- --text-column TEXT          Column containing text to prompt from (default: text)
- --num-questions INT         Questions per text (default: 3)
- --max-tokens INT            Max tokens per response (default: 4096)
- --provider-url URL          Base URL for 'other' provider (required when using --provider other)
- --num-workers INT           Concurrency (default: 1)
- --shuffle                   Shuffle dataset items
- --max-items INT             Limit number of items
- --start-index INT           Start index (0-based)
- --end-index INT             End index (exclusive, 0-based)
- --dataset-split SPLIT       HF split for remote datasets (default: train)
- --sleep-between-requests S  Rate-limit between API calls
- --sleep-between-items S     Rate-limit between items
- --style STYLE               Optional. Single style or comma-separated list; one is chosen randomly per item
- --no-style                  Generate questions without any style instructions (neutral tone)
- --styles-file FILE          Load styles from a file (one per line, # for comments)
- --with-answer               Generate answers for each question using the model
- --answer-provider PROVIDER  API provider to use for answering questions (if not set, uses the same provider as --provider)
- --answer-model MODEL        Model to use for answering questions (if not set, uses the same model as --model)
- --answer-single-request     Answer all questions in a single request instead of one question per request
- --verbose                   Verbose logging
- --debug                     Debug logging

Supported providers for `--provider`:

featherless, openai, anthropic, qwen, qwen-deepinfra, kimi, z.ai, openrouter, cerebras, together, groq, gemini, ollama, chutes, huggingface, other

The `other` provider allows you to use any OpenAI-compatible API endpoint by specifying `--provider-url`.

### Styles

You can control question styles in several ways:

1. **Default behavior** (no style flags): Randomly selects from 35+ built-in styles per item (see `default_styles.txt`)
2. **Custom styles** (`--style`): Single style or comma-separated list; one chosen randomly per item  
3. **No styling** (`--no-style`): Generates neutral, straightforward questions without style instructions
4. **Styles from file** (`--styles-file`): Load styles from a text file (one per line, `#` for comments)

**Note**: Only one style option can be used at a time.

Built-in default styles include academic, creative, informal, analytical, practical, philosophical, and more. The complete list is in `default_styles.txt` and includes styles like:

- formal and academic, professional and business-focused
- creative and imaginative, artistic and expressive, humorous and entertaining
- casual and conversational, friendly and approachable, informal and relaxed
- analytical and critical thinking, investigative and probing
- practical and application-focused, hands-on and actionable
- thought-provoking and philosophical, reflective and contemplative
- simple and straightforward, clear and concise
- detailed and comprehensive, thorough and exhaustive

Examples:

```bash
# Single custom style
python3 src/main.py <dataset> \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  --output-dir ./data/out \
  --num-questions 5 \
  --style "formal and academic"

# Multiple custom styles (random per item)
python3 src/main.py <dataset> \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  --output-dir ./data/out \
  --num-questions 5 \
  --style "casual and conversational,funny and humorous,concise and direct"

# No styling (neutral questions)
python3 src/main.py <dataset> \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  --output-dir ./data/out \
  --num-questions 5 \
  --no-style

# Load styles from file
python3 src/main.py <dataset> \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  --output-dir ./data/out \
  --num-questions 5 \
  --styles-file ./styles_sample.txt
```

See `styles_sample.txt` for an example styles file format and `default_styles.txt` for the complete list of built-in styles.

### Answer Generation

The system can optionally generate answers for each question using the `--with-answer` flag. This creates question-answer pairs where each question is answered based on the original source text.

Key features:

- **Answer generation**: Use `--with-answer` to enable answer generation for each question
- **Custom answer provider**: Use `--answer-provider` to specify a different API provider for answering (defaults to the same provider used for questions)
- **Custom answer model**: Use `--answer-model` to specify a different model for answering (defaults to the same model used for questions)
- **Batch vs individual**: Use `--answer-single-request` to generate all answers in one request, or process one question at a time (default)
- **Error handling**: If answer generation fails, the output field is set to "error" with an appropriate error message

Examples:

```bash
# Generate questions with answers using the same model
python3 src/main.py <dataset> \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  --output-dir ./data/qa_output \
  --num-questions 3 \
  --with-answer

# Use a different model for answers
python3 src/main.py <dataset> \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  --answer-model qwen/qwen3-4b-instruct \
  --output-dir ./data/qa_output \
  --num-questions 3 \
  --with-answer

# Use a different provider and model for answers
python3 src/main.py <dataset> \
  --provider openrouter \
  --model openai/gpt-oss-120b \
  --answer-provider anthropic \
  --answer-model moonshotai/kimi-k2 \
  --output-dir ./data/qa_output \
  --num-questions 3 \
  --with-answer

# Generate all answers in a single request (more efficient but less granular error handling)
python3 src/main.py <dataset> \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  --output-dir ./data/qa_output \
  --num-questions 3 \
  --with-answer \
  --answer-single-request

# Use custom provider for questions and standard provider for answers
export OTHER_API_KEY=your_custom_api_key
export ANTHROPIC_API_KEY=your_anthropic_key
python3 src/main.py <dataset> \
  --provider other \
  --provider-url https://your-custom-api.com/v1 \
  --model your-custom-model \
  --answer-provider anthropic \
  --answer-model claude-3-haiku-20240307 \
  --output-dir ./data/qa_output \
  --num-questions 3 \
  --with-answer
```

When `--with-answer` is used, the output format includes an `output` field containing the generated answer, or "error" if answer generation failed.

## Authentication (API keys)

Provide API keys via environment variables. General rule: `<PROVIDER>_API_KEY` using uppercase and replacing `.` or `-` with `_`. Special cases are handled automatically.

- openai → `OPENAI_API_KEY`
- anthropic → `ANTHROPIC_API_KEY`
- openrouter → `OPENROUTER_API_KEY`
- groq → `GROQ_API_KEY`
- together → `TOGETHER_API_KEY`
- cerebras → `CEREBRAS_API_KEY`
- qwen → `QWEN_API_KEY`
- qwen-deepinfra → `QWEN_DEEPINFRA_API_KEY`
- kimi (Moonshot) → `KIMI_API_KEY`
- z.ai → `Z_AI_API_KEY`
- featherless → `FEATHERLESS_API_KEY`
- chutes → `CHUTES_API_KEY`
- hugging face → `HUGGINGFACE_API_KEY`
- gemini → `GEMINI_API_KEY` (note: Gemini uses a query param; still export as shown)
- ollama → no API key required (assumes local Ollama at http://localhost:11434)
- other → `OTHER_API_KEY` (for custom OpenAI-compatible endpoints)

Example:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

### Using Custom Providers ("other")

The `other` provider allows you to use any OpenAI-compatible API endpoint. This is useful for:

- Custom or self-hosted models
- New providers not yet directly supported
- Local inference servers that implement OpenAI-compatible APIs

Requirements:
- Set `--provider other`
- Provide `--provider-url` with the base URL of your API endpoint
- Set `OTHER_API_KEY` environment variable with your API key

Example:

```bash
export OTHER_API_KEY=your_custom_api_key
python3 src/main.py dataset.jsonl \
  --provider other \
  --provider-url https://your-custom-api.com/v1 \
  --model your-custom-model \
  --output-dir ./output \
  --num-questions 3
```

The system will use OpenAI-compatible request format with your custom endpoint.

## Datasets

You can pass either:

- Hugging Face dataset name: `org/dataset` (uses `datasets.load_dataset(..., split=...)`)
- Local JSONL/JSON file: path ending with `.jsonl` or `.json`
- Local Parquet file: path ending with `.parquet`

Default text column is `text`. Change with `--text-column` if your data uses another key.

Local JSONL example (one JSON per line):

```json
{"text": "Large Language Models excel at generating diverse questions from text."}
{"text": "Neural networks can learn complex patterns from large datasets."}
```

Local Parquet example:

```bash
python3 src/main.py /path/to/data.parquet \
	--provider openrouter \
	--model qwen/qwen3-235b-a22b-2507 \
	--output-dir ./data/questions_parquet \
	--text-column text \
	--num-questions 3
```

## Output

Writes to `<output-dir>/questions_{YYYY-MM-DD_HH-MM-SS}_{dataset}_{provider}_{model}[optional_range].jsonl`

Each line is a JSON record. For successful question generations:

```json
{
	"input": "What practical applications benefit most from question generation using LLMs?",
	"source_text": "...original text...",
	"question_index": 1,
	"total_questions": 5,
	"metadata": { "original_item_index": 0, "text_column": "text" },
	"generation_settings": {
		"provider": "openrouter",
		"model": "qwen/qwen3-235b-a22b-2507",
        "style": "formal and academic",
		"num_questions_requested": 5,
		"num_questions_generated": 5,
		"max_tokens": 4096
	},
	"timestamp": "2025-08-17T12:34:56.789012"
}
```

When using `--with-answer`, each record also includes an `output` field with the generated answer:

```json
{
	"input": "What practical applications benefit most from question generation using LLMs?",
	"output": "Question generation using LLMs has several practical applications including educational content creation, chatbot training data, assessment generation for online courses, and synthetic dataset augmentation for machine learning models...",
	"source_text": "...original text...",
	"question_index": 1,
	"total_questions": 5,
	"metadata": { "original_item_index": 0, "text_column": "text" },
	"generation_settings": {
		"provider": "openrouter",
		"model": "qwen/qwen3-235b-a22b-2507",
        "style": "formal and academic",
		"answer_provider": "anthropic",
		"answer_model": "claude-3-haiku-20240307",
		"answer_single_request": false,
		"num_questions_requested": 5,
		"num_questions_generated": 5,
		"max_tokens": 4096
	},
	"timestamp": "2025-08-17T12:34:56.789012"
}
```

When using `--with-options`, each record includes an `options` field with multiple-choice options:

```json
{
	"input": "What is the primary purpose of machine learning?",
	"options": {
		"A": "To replace human intelligence completely",
		"B": "To enable computers to learn and make decisions from data",
		"C": "To create robots that look like humans",
		"D": "To store large amounts of data efficiently",
		"E": "To generate synthetic questions from text"
	},
	"source_text": "...original text...",
	"question_index": 1,
	"total_questions": 3,
	"metadata": { "original_item_index": 0, "text_column": "text" },
	"generation_settings": {
		"provider": "openrouter",
		"model": "qwen/qwen3-235b-a22b-2507",
		"style": "formal and academic",
		"with_options": true,
		"num_questions_requested": 3,
		"num_questions_generated": 3,
		"max_tokens": 4096
	},
	"timestamp": "2025-08-17T12:34:56.789012"
}
```

When using both `--with-options` and `--with-answer`, the answer includes the correct letter and explanation in separate fields:

```json
{
	"input": "What is the primary purpose of machine learning?",
	"options": {
		"A": "To replace human intelligence completely",
		"B": "To enable computers to learn and make decisions from data",
		"C": "To create robots that look like humans",
		"D": "To store large amounts of data efficiently",
		"E": "To generate synthetic questions from text"
	},
	"output": "Answer: B | Explanation: This is the correct answer because it enables computers to learn from data and make intelligent decisions, which is the fundamental purpose of machine learning.",
	"correct_answer": "B",
	"explanation": "This is the correct answer because it enables computers to learn from data and make intelligent decisions, which is the fundamental purpose of machine learning.",
	"source_text": "...original text...",
	"generation_settings": {
		"with_options": true,
		"with_answer": true,
		...
	}
}
```

The system automatically extracts:
- **`correct_answer`**: The letter (A, B, C, D, or E) for programmatic use
- **`explanation`**: The detailed explanation text
- **`output`**: The full formatted answer (for backward compatibility)

If answer generation fails for a question, the `output` field is set to "error" and an `answer_error` field provides details.

If generation fails for an item, an error record is emitted with `error` instead of `questions` fields.

## Examples

OpenRouter (Qwen):

```bash
export OPENROUTER_API_KEY=your_api_key_here
python3 src/main.py mkurman/hindawi-journals-2007-2023 \
	--provider openrouter \
	--model qwen/qwen3-235b-a22b-2507 \
	--output-dir ./data/questions_openrouter \
	--start-index 0 \
	--end-index 10 \
	--num-questions 5 \
	--text-column text \
	--verbose
```

OpenRouter with Answer Generation:

```bash
export OPENROUTER_API_KEY=your_api_key_here
python3 src/main.py mkurman/hindawi-journals-2007-2023 \
	--provider openrouter \
	--model qwen/qwen3-235b-a22b-2507 \
	--output-dir ./data/qa_openrouter \
	--start-index 0 \
	--end-index 10 \
	--num-questions 3 \
	--with-answer \
	--answer-single-request \
	--verbose
```

Multi-Provider Q&A Generation (Questions from OpenRouter, Answers from Anthropic):

```bash
export OPENROUTER_API_KEY=your_openrouter_key
export ANTHROPIC_API_KEY=your_anthropic_key
python3 src/main.py mkurman/hindawi-journals-2007-2023 \
	--provider openrouter \
	--model qwen/qwen3-235b-a22b-2507 \
	--answer-provider anthropic \
	--answer-model claude-3-haiku-20240307 \
	--output-dir ./data/qa_multi_provider \
	--start-index 0 \
	--end-index 5 \
	--num-questions 2 \
	--with-answer \
	--verbose
```

Ollama (local):

```bash
# Ensure Ollama is running and the model is pulled locally
python3 src/main.py ./data/articles.jsonl \
	--provider ollama \
	--model hf.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
	--output-dir ./data/questions_ollama \
	--num-questions 3
```

## Tips

- Increase `--num-workers` for concurrency, and use `--sleep-between-requests` for rate limits.
- Use `--shuffle` to randomize items, and `--start-index/--end-index` to slice large datasets.
- Ensure `*_API_KEY` is set (where * is the provider name)

## License

Apache 2.0. See [LICENSE](LICENSE).

