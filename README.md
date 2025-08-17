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

## Requirements

- Python 3.9+ (uses modern typing like `list[str]`)
- Install Python packages: `pip install -r requirements.txt`

Contents of `requirements.txt`:

- aiohttp
- datasets
- tqdm

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
- --num-workers INT           Concurrency (default: 1)
- --shuffle                   Shuffle dataset items
- --max-items INT             Limit number of items
- --start-index INT           Start index (0-based)
- --end-index INT             End index (exclusive, 0-based)
- --dataset-split SPLIT       HF split for remote datasets (default: train)
- --sleep-between-requests S  Rate-limit between API calls
- --sleep-between-items S     Rate-limit between items
- --verbose                   Verbose logging
- --debug                     Debug logging

Supported providers for `--provider`:

featherless, openai, anthropic, qwen, qwen-deepinfra, kimi, z.ai, openrouter, cerebras, together, groq, gemini, ollama, chutes

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

Example:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

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

Each line is a JSON record. For successful generations:

```json
{
	"input": "What practical applications benefit most from question generation using LLMs?",
	"source_text": "...original text...",
	"question_index": 1,
	"total_questions": 5,
	"metadata": { "original_item_index": 0, "text_column": "text", "source_dataset_item": { /* original item */ } },
	"generation_settings": {
		"provider": "openrouter",
		"model": "qwen/qwen3-235b-a22b-2507",
		"num_questions_requested": 5,
		"num_questions_generated": 5,
		"max_tokens": 4096
	},
	"timestamp": "2025-08-17T12:34:56.789012"
}
```

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

Apache 2.0. See `LICENSE`.

