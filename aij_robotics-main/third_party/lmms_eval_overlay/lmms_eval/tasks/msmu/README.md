# MSMU-Bench (`msmu`)

| Resource | Link |
|----------|------|
| Dataset | [cpystan/MSMU](https://huggingface.co/datasets/cpystan/MSMU) |
| Official code | [cpystan/SD-VLM](https://github.com/cpystan/SD-VLM) |
| Paper | SD-VLM: Spatial Measuring and Understanding with Depth-Encoded VLMs |

Spatial measuring and understanding VQA: **image + text question** → **concise text / numeric answer**.

## Dataset (verified on HF)

| Split | Rows | `type` field |
|-------|------|--------------|
| `train` | 25378 | always `None` (SFT only) |
| **`test`** | **987** | MSMU-Bench categories |

Fields:

- `image` — RGB image
- `conversations` — `value[0]` human (`<image>\n{question}`), `value[1]` reference answer
- `type` — raw category (test split only)

**lmms-eval uses split `test`** (same as `evaluation/inference.py`).

## Category mapping (dataset `type` → metric name)

| HF leaderboard column | Dataset `type` | lmms-eval metric |
|----------------------|----------------|------------------|
| Existence | `existence` | `msmu_existence` |
| Object Counting | `count` | `msmu_object_counting` |
| Scale Estimation | `scale_estimation` | `msmu_scale_estimation` |
| Grounding | `grounding` | `msmu_grounding` |
| Relative Position | `relative_position` | `msmu_relative_position` |
| Absolute Distance | `absolute_distance` | `msmu_absolute_distance` |
| Scale Comparison | `scale_compare` | `msmu_scale_comparison` |
| Reference Object Estimation | `refer_obj_estimation` | `msmu_reference_object_estimation` |
| **Average** | (macro over 8) | `msmu_average` |

Additional: `msmu_overall_accuracy` — **micro** fraction correct over all samples (not the same as Average).

## Official vs lmms-eval evaluation

Official MSMU-Bench ([SD-VLM README](https://github.com/cpystan/SD-VLM)):

```bash
python evaluation/inference.py --model_path 'SD-VLM-7B' --data 'MSMU' --vision_tower 'clip-vit-large-patch14-336'
python evaluation/llm_generate.py --path './preds.json' --key '<OPENAI_API_KEY>'
python evaluation/evaluate_final.py --path './preds_gpt.json'
```

`evaluate_final.py` computes **Average** as the mean of per-category scores over **8 categories** (macro).

**This task** uses **deterministic** local scoring (no OpenAI API):

- Quantitative types: ratio test aligned with official δ1 (`max(pred/gt, gt/pred) < 1.25` on extracted numbers)
- Qualitative types: overlap / yes-no / comparative heuristics
- Mixed fallback: numeric match when possible, else normalized exact match

Deterministic scores **will not match** official GPT-judged leaderboard numbers.

## Commands

Smoke (`--limit 5`):

```bash
python -m lmms_eval \
  --model qwen3_5 \
  --model_args pretrained=Qwen/Qwen3.5-4B,device_map=auto \
  --tasks msmu \
  --batch_size 1 \
  --limit 5 \
  --output_path ./logs/msmu_smoke
```

Full benchmark (987 samples):

```bash
python -m lmms_eval \
  --model qwen3_5 \
  --model_args pretrained=Qwen/Qwen3.5-4B,device_map=auto \
  --tasks msmu \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/msmu_full
```

## Export for official GPT judging (optional)

```bash
python scripts/export_msmu_predictions.py \
  --samples ./logs/msmu_full/<model_dir>/*_samples_msmu.jsonl \
  --output ./preds.json
```

Then run SD-VLM `llm_generate.py` and `evaluate_final.py` (OpenAI key required).

## Known limitations

- No unit conversion to meters (official GPT judge handles units)
- Qualitative scoring is heuristic, not GPT-based
- No depth input (SD-VLM uses depth-encoded pipeline)
- `--limit` runs: `msmu_average` averages only categories present in the subset (full eval divides by 8)
- Coordinate / geometry answers may need a richer official-compatible scorer later
