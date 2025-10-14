# AutoEval

## Files
- `configs/auto_eval.yaml`: Prompt for the auto eval.
- `script_eval.py`: Runs auto-judge to generate results.
- `script_winrate.py`: Get the win rate results using the scores.

## Run VLM-as-a-Judge
```
export ROOT_PATH="example_outputs"
python script_eval.py mode="gpt" root_path=$ROOT_PATH
python script_eval.py mode="qwen" root_path=$ROOT_PATH
python script_eval.py mode="gemini" root_path=$ROOT_PATH
```

The folder `$ROOT_PATH` must contain model outputs in the format:
```bash
# Template
$ROOT_PATH/<split_name>/<model_name>/<id>.jpg
# Example
$ROOT_PATH/text_to_image/bagel/1904601298038906936.jpg
```

Running `script_eval.py` produces compiled csvs under `runs/<split_name>/<mode>.csv` (composed of the columns `question_id,model,score`) as well as raw outputs as JSON files.

## Compute Win Rate
Run script_winrate.py on the csv file(s) to get win rate results:
```bash
python script_winrate.py mode="text_to_image" "judge_names=['gpt']"
```