# Configuration Guide

This document describes the new configuration features added to the synthetic questions generation system.

## YAML Configuration

You can now use YAML configuration files instead of command-line arguments for easier management and repeatability.

### Basic Usage

```bash
python3 src/main.py --config configs/example.yaml
```

### Configuration File Format

```yaml
# Dataset configuration
dataset: "mkurman/hindawi-journals-2007-2023"

# Provider configuration
provider: "openrouter"
model: "qwen/qwen3-235b-a22b-2507"

# Output configuration
output_dir: "./data/questions_openrouter"

# Processing configuration
start_index: 0
end_index: 10
num_questions: 5
text_column: "text"
verbose: true

# Optional configurations
max_tokens: 4096
num_workers: 1
shuffle: false
sleep_between_requests: 0.0
rate_limit_wait: 15.0
rate_limit_retries: 1

# Style configuration
style: "formal and academic"
# styles_file: "./styles_sample.txt"
# no_style: false

# Answer generation
with_answer: true
answer_provider: "openrouter"  # Optional, defaults to main provider
answer_model: "qwen/qwen3-235b-a22b-2507"  # Optional, defaults to main model
answer_single_request: false

# Multiple-choice questions
with_options: true  # Generate questions with A, B, C, D, E options

# Custom prompts
custom_prompts: "./my_custom_prompts"  # Optional

# Custom provider URL (required when provider is "other")
# provider_url: "https://api.example.com/v1"
```

### Command-Line Override

You can override any configuration file setting with command-line arguments:

```bash
python3 src/main.py --config configs/example.yaml --provider anthropic --model claude-3-sonnet
```

### Backward Compatibility

The system maintains full backward compatibility. You can still use command-line arguments exclusively:

```bash
python3 src/main.py dataset_name --provider openai --model gpt-4 --output-dir ./output
```

## Custom Prompts

You can now customize the system prompts used for question and answer generation.

### Directory Structure

Create a directory with custom prompt files:

```
my_custom_prompts/
├── question_generation.txt
├── question_generation_user.txt
├── answer_generation.txt
├── answer_generation_user.txt
├── batch_answer_generation.txt
└── batch_answer_generation_user.txt
```

### Prompt Templates

Prompts support replaceable tags using Python's string formatting:

#### question_generation.txt
```
You are an expert question generator. Create {num_questions} questions.

Guidelines:
- Generate exactly {num_questions} questions
{style_instruction}
- Make questions engaging
{casual_greeting_instruction}

{style_note}
```

#### Available Tags

**Question Generation:**
- `{num_questions}` - Number of questions to generate
- `{style_instruction}` - Style-specific instruction
- `{style_note}` - Style note for context
- `{casual_greeting_instruction}` - Casual greeting instruction (conditional)

**User Prompts:**
- `{text}` - The source text
- `{style_suffix}` - Style suffix for user prompt

**Answer Generation:**
- `{question}` - The question to answer
- `{source_text}` - The source text for answering
- `{questions_text}` - Formatted list of questions (batch mode)

### Using Custom Prompts

#### Via Configuration File
```yaml
custom_prompts: "./my_custom_prompts"
```

#### Via Command Line
```bash
python3 src/main.py --custom-prompts ./my_custom_prompts --config configs/example.yaml
```

### Fallback Behavior

If a custom prompt file is not found, the system automatically falls back to the default prompt. This allows you to customize only specific prompts while keeping others as default.

## Environment Variables

The system still requires appropriate API keys as environment variables:

```bash
export OPENROUTER_API_KEY=your_api_key_here
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
# ... other provider keys
```

## Migration Guide

### From Command Line to YAML

1. Create a YAML file with your current settings
2. Test with `--config your_config.yaml`
3. Remove command-line arguments once verified

### Adding Custom Prompts

1. Create a custom prompts directory
2. Copy default prompts from `prompt/default/` as starting points
3. Modify the prompts with your custom instructions
4. Add `custom_prompts: "./your_directory"` to your config
5. Test to ensure prompts work as expected

## Multiple-Choice Questions

The system can generate multiple-choice questions with options A, B, C, D, and E using the `--with-options` flag.

### Usage

#### Via Command Line
```bash
python3 src/main.py dataset --provider openai --model gpt-4 --output-dir ./output --with-options
```

#### Via Configuration File
```yaml
with_options: true
```

### Output Format

When using `--with-options`, the output format includes the options:

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
  "metadata": {...},
  "generation_settings": {
    "with_options": true,
    ...
  }
}
```

### Answer Generation for Multiple-Choice Questions

When using `--with-options` together with `--with-answer`, the system automatically:

1. **Includes options in answer prompts**: The model sees all A, B, C, D, E options when generating answers
2. **Uses specialized prompts**: Automatically switches to multiple-choice answer generation prompts
3. **Parses structured answers**: Extracts both the correct letter (A, B, C, D, E) and explanation

#### Answer Format

The answer output follows this format:
```
Answer: B | Explanation: This is the correct answer because it enables computers to learn from data and make intelligent decisions, which is the fundamental purpose of machine learning.
```

#### Example with Answers

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

#### Separate Fields for Better Usability

When using multiple-choice questions with answers, the system automatically extracts and stores:

- **`correct_answer`**: The letter of the correct answer (A, B, C, D, or E)
- **`explanation`**: The detailed explanation for why this answer is correct
- **`output`**: The full formatted answer (for backward compatibility)

This makes it easy to:
- Quickly identify the correct answer programmatically
- Use the explanation for educational purposes
- Maintain compatibility with existing processing pipelines

### Custom Prompts for Multiple-Choice

You can customize the multiple-choice question generation by creating custom prompt files:

- `question_generation_with_options.txt` - System prompt for multiple-choice questions
- `question_generation_with_options_user.txt` - User prompt template

The system automatically uses these prompts when `with_options: true` is set.

## Examples

See `configs/example.yaml` for a complete example configuration file.
