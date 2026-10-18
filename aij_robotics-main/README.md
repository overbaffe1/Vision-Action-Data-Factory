# AIJ Robotics — пайплайн участника

Обучение **SmolVLM2 → SmolVLA** и сборка `submission/` для хакатона.

```text
ShareGPT JSONL  →  SFT SmolVLM2  →  SmolVLA (LIBERO)  →  runs/<run>/submission/
                                                              ├── smolvlm2/
                                                              └── action_expert/
```

| | |
|--|--|
| Репозиторий | [aij_robotics](https://gitlab.frontierai.ru/frontierai/teams/embodied-ai/cv-team/aij_robotics) · ветка **`main`** |
| Базовая VLM | [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) |
| Датасет VLA | [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) @ **v3.0** |
| Запуск | Docker (рекомендуется) или локальный env |

---

## Оглавление

1. [Быстрый старт](#1-быстрый-старт)
2. [Конфиги](#2-конфиги)
3. [Данные](#3-данные)
4. [Полное обучение](#4-полное-обучение)
5. [Проверка submission](#5-проверка-submission)
6. [Score self-check (локальная метрика)](#6-score-self-check-локальная-метрика)
7. [Требования и GPU](#7-требования-и-gpu)
8. [Скрипты и структура](#8-скрипты-и-структура)
9. [Частые проблемы](#9-частые-проблемы)

---

## 1. Быстрый старт

### 1.1. Клон и образ

```bash
git clone -b main https://gitlab.frontierai.ru/frontierai/teams/embodied-ai/cv-team/aij_robotics.git
cd aij_robotics
docker build -t aij-robotics:latest .
```

Сборка долгая (CUDA 12.8 + torch + `hf-libero`).

Перед стартом:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 1.2. Smoke на мини-данных из репо

```bash
mkdir -p runs

docker run --gpus all --rm --shm-size=8g \
  -e CUDA_HOME=/usr/local/cuda -e MUJOCO_GL=egl \
  -v "$PWD/configs:/workspace/configs:ro" \
  -v "$PWD/examples/smoke:/data/vlm:ro" \
  -v "$PWD/examples/smoke/libero_mini:/data/vla:ro" \
  -v "$PWD/runs:/workspace/runs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  aij-robotics:latest \
  ./scripts/run_pipeline.sh configs/smoke.yaml --smoke
```

`--smoke` ≈ короткий VLM + 5 LIBERO eps + 10 VLA steps.

Проверка весов:

```bash
docker run --gpus all --rm --shm-size=4g \
  -e MUJOCO_GL=egl \
  -v "$PWD/runs:/workspace/runs" \
  aij-robotics:latest \
  ./scripts/validate_submission.sh runs/smoke_libero/submission
```

Полезные флаги пайплайна: `--skip-vla`, `--skip-vlm`, `--dry-run`, `--vlm-protocol full|lora`.

### 1.3. Score self-check (после `submission/`)

Приватный lmms-eval **не нужен**. Один раз:

```bash
./scripts/setup_score_selfcheck.sh   # GitHub lmms-eval + overlay + FlagEval/ERQA
```

Дальше — [§6](#6-score-self-check-локальная-метрика).

---

## 2. Конфиги

| Файл | Назначение |
|------|------------|
| [`configs/smoke.yaml`](configs/smoke.yaml) | Smoke: пути к мини-данным |
| [`configs/participant.yaml`](configs/participant.yaml) | Полный прогон: **ваши** пути |
| `configs/vlm_sft*.yaml` | Шаблон SFT VLM (full / LoRA / smoke) — можно менять |
| `configs/vla_libero*.yaml` | Шаблон VLA — можно менять |
| [`configs/selfcheck_score.example.yaml`](configs/selfcheck_score.example.yaml) | Пути для Score self-check |
| [`configs/greenvla_vlm_tasks.yaml`](configs/greenvla_vlm_tasks.yaml) | 8 VLM-бенчей (как у судьи) |

Пути в YAML — **внутри контейнера**; на хосте монтируйте через `-v`.  
Чтобы правки шаблонов попадали в контейнер, монтируйте весь `configs/`.

---

## 3. Данные

### 3.1. VLM — ShareGPT JSONL

```text
my_sharegpt/
├── train.jsonl
├── val.jsonl          # опционально
└── images/...
```

В JSONL предпочтительны относительные пути: `"images": ["images/0001.jpg"]`  
(резолвятся от `vlm.media_dir`, по умолчанию — каталог `data_path`).

### 3.2. VLA — обучение vs валидация

| | Что | Где |
|--|-----|-----|
| **Train** | Демо LeRobot v3.0 | `HuggingFaceVLA/libero` @ `v3.0` → `vla.dataset_dir` |
| **Self-check sim** | Suite `libero_spatial` (10 задач) | `run_libero_selfcheck` / Score self-check |
| **Официальный LB** | отдельный закрытый протокол | только судья (`aij-robotics-judge`) |

Фичи датасета: `observation.images.image` + `image2` (256×256), `state` **(8,)**, `action` **(7,)**.  
Полный шаблон VLA ожидает **1693** эпизода.

Хелперы: `scripts/make_mini_libero_dataset.py`, `scripts/make_libero_hf_slice.py`.

### 3.3. Пример `participant.yaml`

```yaml
run_name: team_demo
vlm:
  data_path: /data/vlm/train.jsonl
  eval_data_path: /data/vlm/val.jsonl   # или null
  media_dir: /data/vlm
vla:
  suite: libero
  dataset_dir: /data/vla
runtime:
  gpu_indices: [0]
  output_root: /workspace/runs
```

---

## 4. Полное обучение

Сначала всегда прогоните smoke ([§1.2](#12-smoke-на-мини-данных-из-репо)). Полный прогон — долгий.

```bash
docker run --gpus all --rm --shm-size=8g \
  -e CUDA_HOME=/usr/local/cuda -e MUJOCO_GL=egl \
  -v "$PWD/configs:/workspace/configs:ro" \
  -v /path/to/my_sharegpt:/data/vlm:ro \
  -v /path/to/HuggingFaceVLA-libero-v3:/data/vla:ro \
  -v "$PWD/runs:/workspace/runs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  aij-robotics:latest \
  ./scripts/run_pipeline.sh configs/participant.yaml
```

На WSL держите репозиторий в `~/...` (ext4), не на `/mnt/c/...`.

---

## 5. Проверка submission

### 5.1. Что сдавать

```text
runs/<run>/submission/
├── smolvlm2/          # HF-веса VLM
├── action_expert/     # LeRobot SmolVLA
├── metadata.json
└── participant.yaml
```

### 5.2. Валидация весов

```bash
./scripts/validate_submission.sh runs/<run>/submission
```

Проверяет layout, грузит чекпоинты, делает tiny forward (ожидает LIBERO `image`/`image2`).

### 5.3. Только sim (`libero_spatial`)

```bash
export MUJOCO_GL=egl
# по умолчанию: 1 задача × 1 ep; для полного spatial:
SUITE=libero_spatial TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' N_EPISODES=1 \
  ./scripts/run_libero_selfcheck.sh runs/<run>/submission
```

Метрики: `…/libero_selfcheck/eval_info.json` → `overall.pc_success`.

На коротком smoke SR часто **0** — нормально, если нет ошибок.

---

## 6. Score self-check (локальная метрика)

Локальный прокси официального Score **той же формулой**, что у судьи:

```text
Score = 100 × (0.4 × VLM + 0.6 × VLA)   # без округления
```

| Компонент | Как считается |
|-----------|----------------|
| **VLM** ∈ [0,1] | Среднее primary-accuracy по **8** бенчам: blink, crosspoint, embspatial, erqa, mmmu_val, mmmu_pro_standard, mmmu_pro_vision, msmu |
| **VLA** ∈ [0,1] | По `eval_info.json`: для каждой задачи `SR_task = successes/episodes`, затем **mean по задачам** (равный вес задачи) |

### Чем отличается от лидерборда

| | Self-check (этот репо) | Официальный судья |
|--|------------------------|-------------------|
| VLM | те же 8 бенчей lmms-eval | те же 8 бенчей |
| VLA | vanilla **`libero_spatial`** | закрытый протокол (не публикуется здесь) |
| Score | та же формула | та же формула |

Числа **сопоставимы по формуле**, но **не равны** официальному Score.

### 6.1. Один раз: зависимости Score

Всё ставится **из этого репозитория** (приватный GitLab **не нужен**):

```bash
./scripts/setup_score_selfcheck.sh
```

Что делает скрипт:

| Шаг | Куда | Источник |
|-----|------|----------|
| clone публичного **lmms-eval** @ pin | `third_party/lmms-eval` | [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) |
| overlay (SmolVLM2, launcher, ERQA, MSMU) | поверх checkout | `third_party/lmms_eval_overlay/` **в этом репо** |
| скачать + сконвертировать **ERQA** | `data/erqa_converted` | [FlagEval/ERQA](https://huggingface.co/datasets/FlagEval/ERQA) |

Только ERQA:

```bash
python3 scripts/prepare_erqa.py
```

Только overlay на уже скачанный lmms-eval:

```bash
./scripts/apply_lmms_eval_overlay.sh third_party/lmms-eval
```

Нужен доступ к **GitHub** и **Hugging Face**. `libero_spatial` — через `hf-libero` (Docker / `setup_env.sh`).

`setup_score_selfcheck.sh` ставит torch **cu128** (под драйвер CUDA 12.x) и `num2words`.  
Скрипты self-check выставляют `PYTHONNOUSERSITE=1`, чтобы `~/.local` не ломал numpy/numba.

Конфиг: [`configs/selfcheck_score.example.yaml`](configs/selfcheck_score.example.yaml).

### 6.2. Запуск

```bash
export API_TYPE=dummy MUJOCO_GL=egl

# полный self-check: 8 VLM + 10 spatial задач × 1 ep
./scripts/run_score_selfcheck.sh \
  --submission runs/<run>/submission \
  --config configs/selfcheck_score.example.yaml

# быстрый smoke: VLM limit=5, 1 ep × 10 spatial
./scripts/run_score_selfcheck.sh \
  --submission runs/<run>/submission \
  --config configs/selfcheck_score.example.yaml \
  --smoke
```

Число эпизодов VLA задаётся в конфиге (`vla.episodes_per_task`, по умолчанию **1** → **10 ep** на все spatial-задачи).

### Артефакты

```text
…/score_selfcheck/          # или --output-dir
├── vlm_benches/summary/results_merged.csv
├── libero_spatial/eval_info.json
└── summary/
    ├── leaderboard_score.json   # полный payload
    ├── leaderboard_score.txt    # одна строка — Score (float)
    └── selfcheck_report.json
```

Пересчёт из готовых артефактов:

```bash
python3 scripts/compute_leaderboard_score.py \
  --vlm-csv …/vlm_benches/summary/results_merged.csv \
  --libero-eval-info …/libero_spatial/eval_info.json \
  --output …/summary/leaderboard_score.json
```

---

## 7. Требования и GPU

| | Windows 11 | Linux |
|--|------------|--------|
| GPU NVIDIA | драйвер на Windows | драйвер на хосте |
| Docker | Desktop + WSL2 | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Репозиторий | только `~/…` в WSL (не `/mnt/c`) | любой быстрый диск |

Драйвер на **хосте**; CUDA/torch — в образе.

| GPU | Образ |
|-----|--------|
| 10xx–40xx | текущий **cu128** обычно ок |
| 50xx (Blackwell) | нужен драйвер Blackwell + **cu128** (иначе `no kernel image`) |

### VRAM (ориентиры, RTX 5080 16 GB)

| Режим | Примерный peak |
|-------|----------------|
| Smoke LoRA (rank 8, bs=1) | ~15 GB |
| Full SFT bs=1…4 | ~5–14 GB |
| Full SFT bs=8 | ~28 GB — не влезет на 16 GB |

При OOM: LoRA, меньше batch, несколько GPU (`runtime.gpu_indices`).

### Без Docker (Linux)

```bash
./scripts/setup_env.sh
conda activate vla-hackathon   # или ваш env из setup
./scripts/run_pipeline.sh configs/participant.yaml
```

Проверка sim-стека:

```bash
python -c "import libero; from libero.libero import benchmark; assert 'libero_spatial' in benchmark.get_benchmark_dict()"
command -v lerobot-eval
```

---

## 8. Скрипты и структура

| Путь | Назначение |
|------|------------|
| `scripts/run_pipeline.sh` | Обучение VLM + VLA → `submission/` |
| `scripts/run_smoke.sh` | Smoke + validate |
| `scripts/validate_submission.sh` | Проверка весов |
| `scripts/run_libero_selfcheck.sh` | Sim `libero_spatial` |
| `scripts/setup_score_selfcheck.sh` | публичный lmms-eval + overlay + ERQA |
| `scripts/apply_lmms_eval_overlay.sh` | накатить `third_party/lmms_eval_overlay` |
| `scripts/prepare_erqa.py` | FlagEval/ERQA → `data/erqa_converted` |
| `third_party/lmms_eval_overlay/` | SmolVLM2 / launcher / ERQA / MSMU (в git) |
| `scripts/run_score_selfcheck.sh` | **Score:** 8 VLM + spatial → `leaderboard_score.txt` |
| `scripts/run_vlm_selfcheck.py` | Оркестрация lmms-eval |
| `scripts/compute_leaderboard_score.py` | Формула Score |
| `third_party/LLaMA-Factory` | SFT SmolVLM2 |
| `third_party/scaling-lawyer` | SmolVLA |
| `examples/smoke/` | Мини-данные |
| `Dockerfile` | Образ участника |

---

## 9. Частые проблемы

| Симптом | Что сделать |
|---------|-------------|
| `could not read Username` при clone | GitLab token / SSH |
| Нет GPU в Docker | WSL integration / NVIDIA Container Toolkit |
| `no kernel image` / RTX 50xx | Драйвер Blackwell + образ cu128 |
| `unable to allocate shared memory` | `--shm-size=8g` |
| CUDA OOM | LoRA / меньше batch / больше GPU |
| `Expected LeRobot meta/info.json` | Датасет **v3.0**, корень с `meta/` |
| VLM: нет картинок | `media_dir` + относительные пути |
| `No module named 'libero'` / нет `lerobot-eval` | Пересобрать образ или `./scripts/setup_env.sh` |
| MuJoCo / blank GL | `MUJOCO_GL=egl` |
| Score self-check: нет lmms / ERQA | `./scripts/setup_score_selfcheck.sh`; `API_TYPE=dummy` |
| lmms-eval pin устарел | обновить `third_party/lmms_eval_overlay/BASE_REF.txt` + overlay |
| Self-check SR=0 на smoke | Часто нормально на коротком прогоне |
| Репо на `/mnt/c` тормозит | Клон в `~/` внутри WSL |
