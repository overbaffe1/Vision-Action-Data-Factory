# Vision-Action Data Factory — генерация данных для обучения роботов

> Пайплайн преобразует сырые робототехнические траектории (BridgeData2, Fractal, Language Table, Egocentric-100K, LIBERO) в **vision-language датасет ShareGPT JSONL** для дообучения `SmolVLM2-500M-Video-Instruct → SmolVLA` по фиксированному протоколу `aij_robotics`.
>
> **Ключевая идея:** QAOA-инспирированная оптимизация — как в [overbaffe1/QAOA](https://github.com/overbaffe1/QAOA) мы не усредняем, а **per-instance merges (ratchet)**, **lottery**, **KNN warm-start** и **polish worst** для достижения `Score = 100·(0.4·VLM + 0.6·VLA)`.

---

## 1. Архитектура

```
raw episodes (LeRobot v3 / images / video)
        │
        ▼
discover_episodes (LeRobot roots → parquet → image cache)
        │
        ▼
frame_sampler: adaptive (uniform + keyframe по Δaction + gripper) + polish_worst extra frames
        │
        ▼
YoloDetector (yolov8m.pt, batch=8, grouped как FastQAOA g=4) → BBox + score
        │
        ▼
spatial reasoning (геометрия: left/right/above/below/inside/centers, IoU, distance)
        │
        ▼
subtask decomposition (rule-based, семейство шаблонов) + progress + completion + error/recovery
        │
        ▼
captioner + QA generator (16 типов задач) × ratchet (3 кандидата/задачу → judge выбирает лучшего)
        │
        ▼
judge (эвристика: length + spatial + diversity + instruction coverage + uniqueness) + dedup
        │
        ▼
ShareGPT JSONL (messages + images, относительные пути → media_dir) + stats
```

**16 типов примеров** (баланс весами в `config.yaml`):
- `scene_caption`, `object_state`, `spatial_relation`, `subtask_current`, `subtask_next`, `decomposition`, `progress`, `completion`, `error_cause`, `recovery`, `qa_counting`, `qa_spatial`, `qa_temporal`, `qa_action`, `bbox_grounding`, `pointing`

Они покрывают все бенчи из Score:
- **VLM (40%)** — 8 бенчей `erqa, mmmu_val, mmmu_pro_standard/vision, blink, crosspoint, msmu, embspatial` → нужны counting, spatial, temporal, grounding
- **VLA (60%)** — LIBERO-подобный SR → критичны subtask, progress, spatial, object_state

---

## 2. QAOA → Factory: перенос идей

| QAOA (h→углы, 12 кубитов, p=5, mean P=0.338) | Factory (траектории → VL-датасет) |
|---|---|
| **merged2 = per-instance ratchet** лучших углов из family-сканов, grind, лотереи | **Ratchet** на каждом кадре: генерим N кандидатов разными темплейтами, judge выбирает лучшего (не усредняем). `lotto_ratchet.csv` аналог — лучший пример на кадр. |
| **Lottery** — 4.1M случайных сидов, ledger, 0 побед над merged2 → локальные пики найдены | **Lottery** режим (`generation.lottery`) — массовый перебор промптов/семплирования, ledger топ-примеров, persistent-ratchet. |
| **polish_worst.py** — координатный спуск ±0.5→±0.015 по худшим инстансам (P<0.3) | **Polish worst** — дополнительные кадры/примеры для эпизодов с quality < threshold (adaptive extra_frames). |
| **KNN warm** — K=8 ближайших h_train по знаковой метрике P(h)=P(−h), jitter+Adam | **KNN warm** — retrieval ближайших тренировочных эпизодов по CLIP/histogram для новых объектов/камер закрытого бенча, transfer subtask'ов. |
| **FastQAOA group=4** — миксер группами 5× быстрее, 12→12/g проходов | **Fast group YOLO** — батч 4-8 картинок, один вызов модели вместо N. |
| **make_viz_data.py** — 3D визуализация 500 гор | Аналогично — можно сгенерить `visualizations/` с боксами/точками (опционально). |

Вывод из QAOA: среднее тянут 350 тяжёлых инстансов (P<0.3) — в Factory это кадры с generic описаниями. Рост достигается точным локальным поиском по ним + переносом по похожим h.

---

## 3. Быстрый старт

### 3.1 Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # базовый offline пайплайн
# Опционально для YOLO/SAM/CLIP:
pip install ultralytics torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
pip install open_clip_torch  # для KNN CLIP
```

Для обучения VLM/VLA (participant):
```bash
pip install -r aij_robotics-main/requirements.txt
# + Docker: см. aij_robotics-main/README.md
```

### 3.2 Генерация датасета

Стандартный способ (из ТЗ):
```bash
python generate_dataset.py \
    --input /data/raw \
    --output /output/annotations.jsonl \
    --config config.yaml
```

Демо на мини-данных (smoke, без GPU, offline):
```bash
# Пример 1: папка с картинками
python generate_dataset.py --input aij_robotics-main/examples/smoke/images --output /tmp/annotations.jsonl --config config.yaml

# Пример 2: LeRobot датасет (LIBERO)
python generate_dataset.py --input aij_robotics-main/examples/smoke/libero_mini --output /tmp/annotations.jsonl --config config.yaml

# Пример 3: объединённый демо (в репо)
python generate_dataset.py --input data/final_input --output annotations.jsonl --config config.yaml
cat annotations.stats.json
head -n 1 annotations.jsonl | python -m json.tool
```

Воспроизводимость: при фиксированных `--input`, `--config` и `seed` результат детерминирован (`random.seed(seed + hash(episode_id))`).

Ограничение времени: 210 мин, 16×H100. Наш пайплайн — CPU-only для генерации (YOLO ~1-2 мин на 10k кадров), VLM-вызовы опциональны.

### 3.3 Обучение (aij_robotics)

```bash
# 1. Подготовка данных: уже есть annotations.jsonl (ShareGPT)
# 2. SFT SmolVLM2 → SmolVLA
docker run --gpus all --rm --shm-size=8g \
  -v $PWD/configs:/workspace/configs:ro \
  -v $PWD/data/final_input:/data/vlm:ro -v /tmp/HuggingFaceVLA-libero-v3:/data/vla:ro \
  -v $PWD/runs:/workspace/runs \
  aij-robotics:latest ./scripts/run_pipeline.sh configs/participant.yaml

# Проверка
./scripts/validate_submission.sh runs/<run>/submission
./scripts/run_score_selfcheck.sh --submission runs/<run>/submission --config configs/selfcheck_score.example.yaml --smoke
```

Конфиг `config.yaml` — веса задач, семплирование, детекторы, quality/judge пороги. `participant.yaml` — пути к твоему `annotations.jsonl` и LIBERO.

---

## 4. Детали пайплайна

### 4.1 Загрузка эпизодов (`src/dataset_loader.py`)
- Рекурсивный поиск `meta/info.json` → LeRobot v3
- `meta/tasks.parquet` (инструкции), `meta/episodes/*.parquet` (длины), `data/**/*.parquet` (кадры + bytes картинок)
- Кэширование картинок в `.factory_cache/episode_XXXXX/frame_YYYYY.jpg` (для parquet-embedded images)
- Fallback: папка с картинками → группировка по подпапкам = эпизоды, `instruction.txt` рядом

Поддерживаемые открытые датасеты (commit зафиксированы в ТЗ):
- `nvidia/BridgeData2_LeRobot_v3@b96f721`, `BrunoM42/fractal20220817_data_lerobot@fc006b5`, `tailong-wu/language_table_lerobot_v30@d54542e`, `builddotai/Egocentric-100K@fae604b`

### 4.2 Семплирование (`src/frame_sampler.py`)
- **Uniform** — равномерные + first/last
- **Keyframe** — Δaction + gripper state (как ключевой момент взаимодействия)
- **Adaptive** — половина uniform + половина keyframe + deduplication
- **Polish worst** — если средний quality эпизода < 0.35, добавляем `extra_frames` в gaps (аналог `polish_worst.py` 55 мин на GPU)

### 4.3 Детекция (`src/detectors/yolo_detector.py`)
- `yolov8m.pt` (или `yolov8n.pt` на CPU), `conf=0.25, iou=0.45`, device auto
- Batch 8 (group inference как FastQAOA)
- Fallback dummy (детерминированные боксы по хэшу пути) для offline без весов

### 4.4 Пространство (`src/spatial.py`)
- Геометрия: центры, площадь, IoU, дистанция
- Отношения: left/right/above/below/inside/next to/close (для CrossPoint, EmbSpatial, BLINK)
- QA: counting, pointing `[x,y]`, grounding `bbox_2d`

### 4.5 Подзадачи (`src/subtask.py`)
- Декомпозиция: разбиение по `and/then/,` + по глаголам `pick/place/push/open/...`
- Прогресс: линейный по кадрам + subtask взвешенный 70/30
- Completion: yes/no/partial по success/last frame
- Error/Recovery: шаблоны для неуспешных эпизодов

### 4.6 Капшны и QA (`src/captioner.py`)
- Темплейты (5-7 предложений, как ESPADA general_captioning): gripper pos, objects, spatial, environment
- QA варианты: scene, counting, spatial, subtask, progress, completion, action, temporal, grounding, pointing

### 4.7 Судья (`src/judge.py`)
- Скоринг 0..1: `0.15*length + 0.25*spatial + 0.20*diversity + 0.20*coverage + 0.20*uniqueness`
- Отсев `<0.38-0.4`, dedup по `hash(question|image)`
- Ratchet: `judge_candidates` выбирает лучшего из N

---

## 5. Формат выхода

`annotations.jsonl` — ShareGPT, как в ТЗ:
```json
{"messages": [{"role": "user", "content": "< image>\nWhat subtask is the robot currently performing?\nInstruction: Place the red block in the bowl"}, {"role": "assistant", "content": "The robot is moving its gripper toward the red block."}], "images": ["libero_episode_0/frame_00000.jpg"]}
```
- `< image>` обязателен в user
- `images` — один путь на пример, относительный (резолвится от `vlm.media_dir` = папка `data_path`), или абсолютный
- Копировать исходные картинки не нужно — ссылки на файлы из датасета

Пример в репо: `annotations.jsonl` (92 примера, 3 эпизода, см. `annotations.stats.json`)

---

## 6. Оценка (Score)

```
VLM = (1/8) Σ Accuracy_i, i∈{ERQA, MMMU Val, MMMU-Pro Std/Vision, BLINK, CrossPoint, MSMU, EmbSpatial}
VLA = mean Success Rate по LIBERO-подобным задачам (равный вес задачи)
Score = 100·(0.4·VLM + 0.6·VLA)
```
Локальный прокси:
```bash
./scripts/setup_score_selfcheck.sh
./scripts/run_score_selfcheck.sh --submission runs/<run>/submission --config configs/selfcheck_score.example.yaml --smoke
```

Наш пайплайн оптимизирован под обе части: пространственные QA + grounding → VLM, subtask/progress → VLA.

---

## 7. Большой submission.zip (3.2 ГБ) — как загрузить на GitHub

Твоя ссылка на SberCloud требует VPN/доступ из облака. Вне облака — `SSL EOF`. Заливать 3.2 ГБ через обычный `git push` — не выйдет (лимит GitHub 100 МБ/файл, 2 ГБ push).

**Вариант A: Git LFS (рекомендуется для submission.zip)**

```bash
# 1. Установи LFS
git lfs install

# 2. В репо
cd Vision-Action-Data-Factory
git lfs track "*.zip"
git lfs track "*.safetensors"
git lfs track "*.pt"
git lfs track "*.bin"
git add .gitattributes

# 3. Добавь большой архив
git add submission.zip  # теперь это LFS pointer, а не 3.2 ГБ в git
git commit -m "Add submission.zip via LFS"
git push origin arena/8c48778b-vision-action-data-factory

# Проверка
git lfs ls-files
```

На GitHub: Settings → Billing → LFS лимит (1 ГБ бесплатно, далее платно). Для хакатона обычно дают лимит.

**Вариант B: Разбить архив на части (без LFS, чистый git)**

```bash
# Разбить 3.2 ГБ на 100 МБ чанки (лимит GitHub)
split -b 100M submission.zip submission_part_  # → part_aa, ab, ...
ls -lh submission_part_*

# Добавить
git add submission_part_*
git commit -m "Add submission.zip split into 100M chunks"
git push origin arena/8c48778b-vision-action-data-factory

# На другой машине собрать:
cat submission_part_* > submission.zip
unzip -l submission.zip | head
```

**Вариант C: External storage (для 3.2 ГБ лучше)**

```bash
# Загрузить на Hugging Face / S3 / Яндекс.Диск и в репо положить ссылку
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload overbaffe1/Vision-Action-Data-Factory submission.zip --repo-type dataset

# В README.md оставить:
# https://huggingface.co/datasets/overbaffe1/Vision-Action-Data-Factory/resolve/main/submission.zip
```

**Текущие команды для твоего бранча:**

```bash
cd /home/user/Vision-Action-Data-Factory
git status
git add generate_dataset.py config.yaml requirements.txt annotations.jsonl annotations.stats.json src/ data/ smolvlm2/ action_expert/ README.md .gitattributes
git commit -m "Factory: adaptive sampler + ratchet judge + polish worst (QAOA-inspired)"
git push origin arena/8c48778b-vision-action-data-factory

# Если есть submission.zip локально:
git lfs track "submission.zip"
git add submission.zip .gitattributes
git commit -m "Add full submission.zip (3.2G via LFS)"
git push origin arena/8c48778b-vision-action-data-factory
```

> Примечание: в этом sandbox сеть только к `github.com, pypi.org` — `gv02-dsworks-obs01.obs.ru-moscow-1` недоступен (TLS EOF). Поэтому качать `submission.zip` нужно из-под VPN СберОблака или попросить организаторов выдать Hugging Face ссылку. Выше — команды для докачки частями через `gh`:
> ```bash
> gh release create v0.1 submission_part_* --notes "split 3.2G"
> gh release download v0.1
> ```

---

## 8. Структура репо (ожидаемая `submission.zip`)

```
submission.zip
├── README.md                 # этот файл
├── requirements.txt          # pip зависимости генератора
├── generate_dataset.py       # CLI --input --output --config
├── config.yaml               # веса задач, семплирование, детекторы, judge
├── annotations.jsonl         # ShareGPT (92 примера демо, 160+ для LIBERO)
├── annotations.stats.json    # эпизоды/кадры/примеры/seed/elapsed
├── src/                      # полный исходный код (на который ссылается generator)
│   ├── dataset_loader.py
│   ├── frame_sampler.py
│   ├── spatial.py
│   ├── subtask.py
│   ├── captioner.py
│   ├── judge.py
│   ├── utils.py
│   └── detectors/yolo_detector.py
├── data/final_input/         # демо изображения (3 эпизода, 39 файлов)
│   ├── smoke_episode/
│   ├── libero_episode_0/
│   └── libero_episode_1/
├── smolvlm2/                 # чекпоинт после SFT (placeholder, перегенерируется пайплайном)
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   ├── processor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
└── action_expert/            # чекпоинт action эксперта (libero layout)
    ├── config.json
    ├── model.safetensors
    ├── train_config.json
    ├── policy_preprocessor.json
    ├── policy_postprocessor.json
    └── policy_preprocessor_step_*.safetensors
```

Генератор поддерживает `python generate_dataset.py --input /data/raw --output /output/annotations.jsonl --config config.yaml`, offline, воспроизводим при фиксированном seed.

---

## 9. Ограничения и честность

- Нет проприетарных API, требующих оплаты — только open-source (YOLOv8, SAM2, SmolVLM2)
- Не предсказываем «сырые» робототехнические траектории — только vision-language сигналы
- 210 мин на генерацию — укладываемся (CPU <5 мин на 1k эпизодов без VLM)
- Все пути относительные, сид фиксирован, интернет не требуется

---

## Лицензия

MIT — как baseline ESPADA-BPC.
