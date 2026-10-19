# lmms-eval overlay for AIJ Score self-check

Participants **do not** need the private cv-team `lmms-eval` fork.

## What this is

Small patch set applied on top of the **public** repo:

<https://github.com/EvolvingLMMs-Lab/lmms-eval>

| Overlay piece | Why |
|---------------|-----|
| `scripts/run_many_tasks.py` | GreenVLA multi-task launcher (not in upstream) |
| `lmms_eval/models/simple/smolvlm2.py` | SmolVLM2 local eval model |
| `lmms_eval/tasks/erqa/` | Local `ERQA_DATA_DIR` JSONL layout (matches judge) |
| `lmms_eval/tasks/msmu/` | MSMU bench (not in upstream at pin time) |
| registry line `smolvlm2` in `models/__init__.py` | model discovery |

Pinned base commit: see `BASE_REF.txt`.

## Apply

```bash
./scripts/setup_score_selfcheck.sh
# or only overlay onto an existing checkout:
./scripts/apply_lmms_eval_overlay.sh third_party/lmms-eval
```

ERQA images still come from Hugging Face via `scripts/prepare_erqa.py`
([FlagEval/ERQA](https://huggingface.co/datasets/FlagEval/ERQA)).
