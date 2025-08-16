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