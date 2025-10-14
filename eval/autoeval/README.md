# AutoEval

## Files
- `configs/auto_eval.yaml`: prompt for the auto eval.
- `script_eval.py`: runs auto-judge to generate results.
- `script_winrate.py`: get the winrate results on the model's scores.

## Run
```bash
python script_eval.py --mode "gpt" --root-path "../results"
python script_eval.py --mode "qwen" --root-path "../results"
python script_eval.py --mode "gemini" --root-path "../results"
```

--root-path must contain model generations in the format:
```bash
../results/<config_name>/<model_name>/<id>.jpg
```
Example:
```bash
../results/image_to_image_synthetic/bagel/1904601298038906936.jpg
```

## Output
Running script_eval.py produces JSON results under runs/, e.g.:
```bash
runs/image_to_image_synthetic/bagel.json
```

## Convert to CSV
Organize the JSON results into a CSV with columns:
question_id,model,score

## Win-rate
Run winrate.py on the CSV file(s) to get win-rate results:
```bash
python winrate.py
```