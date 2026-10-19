# ERQA (Embodied Reasoning QA)

[ERQA](https://github.com/embodiedreasoning/ERQA) is a multimodal embodied-reasoning multiple-choice benchmark. The official release ships as `data/erqa.tfrecord`. **lmms-eval does not read TFRecord at evaluation time** — convert the dataset offline once, then point evaluation at the converted JSONL and images.

## 1. Download the official dataset

Clone the ERQA repository and obtain `data/erqa.tfrecord`:

```bash
git clone https://github.com/embodiedreasoning/ERQA.git
cd ERQA
# erqa.tfrecord should be at ./data/erqa.tfrecord
```

## 2. Convert TFRecord → JSONL + images (offline)

TensorFlow is only needed for conversion:

```bash
pip install tensorflow-cpu  # or: pip install tensorflow
```

```bash
python lmms_eval/tasks/erqa/convert_erqa.py \
  --input /path/to/ERQA/data/erqa.tfrecord \
  --output_dir /path/to/erqa_converted
```

This writes:

- `/path/to/erqa_converted/erqa.jsonl`
- `/path/to/erqa_converted/images/`

Then set:

```bash
export ERQA_DATA_DIR=/path/to/erqa_converted
```

If `ERQA_DATA_DIR` points at a `.tfrecord` file or a folder that still only contains TFRecord, evaluation fails with a message to run the converter first.

## 3. Run evaluation with a local VLM (no API keys)

### Qwen3-VL (Hugging Face checkpoint)

```bash
export ERQA_DATA_DIR=/path/to/erqa_converted

python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=/path/to/local/Qwen3-VL-8B-Instruct,device_map=auto \
  --tasks erqa \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/erqa_qwen3vl
```

### Qwen3.5-VL

```bash
python -m lmms_eval \
  --model qwen3_5 \
  --model_args pretrained=Qwen/Qwen3.5-4B,device_map=auto \
  --tasks erqa \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/erqa_qwen35
```

Other local models supported by lmms-eval (LLaVA, InternVL, etc.) work the same way; adjust `--model` and `--model_args` per the model registry.

## 4. Metrics

- **erqa_acc**: exact-match accuracy on answer letters **A/B/C/D**.
- Per-`question_type` breakdown is logged during aggregation (not separate leaderboard columns).

Answer parsing accepts variants such as `B`, `(B)`, `Answer: B`, and `The answer is B.`

## 5. Custom data path (optional)

Default config loads `${ERQA_DATA_DIR}/erqa.jsonl`. To override paths, copy `erqa.yaml` and set `dataset_kwargs.data_files` to your JSONL, or use `--include_path` with a local YAML override (see `docs/guides/task_guide.md`).

## 6. Sanity check

```bash
python -m lmms_eval --tasks list | grep erqa
```
