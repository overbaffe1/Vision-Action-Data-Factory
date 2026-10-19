<div align="center">

# ESpaDa-BPC

### Extracting Spatial Data from Internet Videos — BBoxing, Pointing, Captioning

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white" alt="CUDA supported">
  <img src="https://img.shields.io/badge/output-LLaMA%20Factory-6C5CE7" alt="LLaMA Factory output">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

Автоматическая разметка изображений для задач visual grounding и обучения
мультимодальных моделей. 

<img src="docs/teaser.png" width="760" alt="ESpaDa-BPC pipeline preview">

</div>

## Возможности

ESpaDa-BPC строит три типа разметки для каждого изображения:

- **BBoxing** — локализация объектов ограничивающими рамками;
- **Pointing** — локализация объектов точками;
- **Captioning** — подробное описание сцены и действий.

Пайплайн объединяет YOLOv8, SAM2 и vision-language model, доступную через
OpenAI-compatible Chat Completions API. Результаты сохраняются в мультимодальном
JSONL-формате, который можно использовать в LLaMA Factory.

Основной CLI дополнительно поддерживает:

- техническую валидацию ответов VLM;
- повторные запросы с backoff;
- checkpoint и продолжение прерванного запуска;
- несколько VLM endpoints;
- несколько GPU;
- опциональный VLM Judge для проверки bboxing и pointing;
- сохранение логов и визуализаций.

## Требования

- Linux;
- Python 3.11;
- NVIDIA GPU с CUDA для производительного запуска;
- доступ к OpenAI-compatible VLM endpoint;
- свободное место для весов моделей и результатов разметки.

При первом запуске будут загружены веса:

- `facebook/sam2.1-hiera-large` из Hugging Face;
- `yolov8m.pt` для Ultralytics YOLOv8.

Запуск на CPU технически возможен при соответствующей настройке device, но будет
существенно медленнее.

## Установка

### Вариант с uv

```bash
git clone ssh://git@gitlab.frontierai.ru:8022/frontierai/teams/embodied-ai/cv-team/espada.git
cd espada

uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### Вариант со стандартным venv и pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Проверить установку CLI:

```bash
espada-bpc --help
```

Если editable-установка не использовалась, команды ниже можно запускать через
`python espada_bpc.py` вместо `espada-bpc`.

## Настройка VLM

Создайте `.env` из примера:

```bash
cp .env.example .env
```

Заполните переменные:

```dotenv
BASE_URL=http://localhost:8001
API_TOKEN=your-token
MODEL_VLM=your-vlm-model-name
```

| Переменная | Назначение |
| --- | --- |
| `BASE_URL` | Базовый URL OpenAI-compatible endpoint |
| `API_TOKEN` | Bearer-токен; может быть пустым для локального сервера |
| `MODEL_VLM` | Имя модели, передаваемое в Chat Completions API |

Клиент сначала обращается к `/v1/chat/completions`, а при ответе `404` пробует
`/chat/completions`.

Параметры VLM также можно задать напрямую в `configs/bpc.yaml`:

```yaml
vlm:
  workers: 8
  endpoints:
    - "http://vlm-1:8001"
    - "http://vlm-2:8001"
  max_concurrency_per_endpoint: 2
  model: "your-vlm-model-name"
  timeout_s: 60
```

Если `vlm.model` задан в YAML, переменная `MODEL_VLM` не обязательна. Настройки
`targets`, `router_url` и `models` позволяют использовать gateway с несколькими
model-routed backend-инстансами.

## Быстрый старт

### 1. Подготовить изображения

Исходные изображения могут быть в JPEG, PNG, WebP, BMP или TIFF. Команда
рекурсивно обходит входную директорию, применяет EXIF orientation, переводит
изображения в RGB, уменьшает длинную сторону и сохраняет JPEG:

```bash
espada-bpc prepare-images \
  --input data/raw_images \
  --output data/processed_images \
  --size 768 \
  --quality 85
```

Уже существующие файлы не перезаписываются, поэтому подготовку можно безопасно
запустить повторно.

### 2. Запустить разметку

```bash
espada-bpc run \
  --input data/processed_images \
  --output runs/example \
  --config configs/bpc.yaml \
  --gpus 0 \
  --log-level INFO
```

Минимально необходимы `--input` и `--output`:

```bash
espada-bpc run --input data/processed_images --output runs/example
```

### 3. Проверить результат

```bash
espada-bpc status --run runs/example
```

Команда покажет количество завершённых, выполняющихся, пропущенных и неуспешных
задач отдельно для bboxing, pointing, captions и Judge.

## Основные параметры запуска

```text
--input PATH                 директория с подготовленными JPEG
--output PATH                корень запуска: outputs/, logs/, visualizations/
--config PATH                YAML-конфигурация, по умолчанию configs/bpc.yaml
--gpus 0,1                   список GPU или all
--num-nodes INTEGER          число машин в distributed-запуске
--node-rank INTEGER          индекс текущей машины, начиная с 0
--batch-size INTEGER         размер checkpoint-батча, по умолчанию 1000
--input-recursive            рекурсивный поиск изображений, включён по умолчанию
--no-input-recursive         поиск только в указанной директории
--skip-preflight             пропустить предварительную проверку всех изображений
--log-level LEVEL            DEBUG, INFO, WARNING или ERROR
```

Полный список параметров всегда доступен через:

```bash
espada-bpc run --help
```

## Конфигурация пайплайна

Основной конфигурационный файл — `configs/bpc.yaml`.

| Раздел | Что настраивает |
| --- | --- |
| `input` | расширения, RGB-проверку и максимальный размер изображения |
| `yolo` | веса YOLO, batch size и device |
| `sam2` | workers, device, dtype и очистку CUDA cache |
| `vlm` | endpoints, модели, concurrency, workers и timeout |
| `judge` | включение Judge, endpoints и разрешённые изменения |
| `validation` | ограничения bbox, pointing и captions |
| `retry` | число попыток и backoff между запросами |
| `storage` | визуализации и дополнительные промежуточные артефакты |

Промпты для всех VLM-задач находятся в `configs/prompts.yaml`.

### Включение Judge

```yaml
judge:
  enabled: true
  tasks: ["bboxing", "pointing"]
  workers: 4
  endpoints:
    - "http://localhost:8011"
  max_concurrency_per_endpoint: 2
  allowed_edits: ["labels"]
  forbid_coordinate_changes: true
```

Judge может исправлять подписи объектов, но изменение координат блокируется
проверкой пайплайна.

## Multi-GPU

Использовать все доступные GPU:

```bash
espada-bpc run \
  --input data/processed_images \
  --output runs/example \
  --gpus all
```

Использовать выбранные GPU:

```bash
espada-bpc run \
  --input data/processed_images \
  --output runs/example \
  --gpus 0,1,2,3
```

Для каждого GPU создаётся отдельный процесс с собственными YOLO и SAM2. Входные
изображения делятся между процессами без пересечений, а результаты записываются
в общий output root.

## Продолжение прерванного запуска

Для продолжения достаточно повторить ту же команду с тем же `--output`:

```bash
espada-bpc run \
  --input data/processed_images \
  --output runs/example \
  --config configs/bpc.yaml \
  --gpus 0
```

Пайплайн прочитает `logs/state.jsonl`, checkpoint и существующие per-image
артефакты, после чего обработает только незавершённые задачи.

Посмотреть состояние запуска можно в любой момент:

```bash
espada-bpc status --run runs/example
```

## Повтор неуспешных задач

Поставить в очередь все задачи из `logs/failed.jsonl`:

```bash
espada-bpc retry-failed --run runs/example
```

Повторить только bboxing на стадии VLM:

```bash
espada-bpc retry-failed \
  --run runs/example \
  --stage vlm \
  --task bboxing
```

После этого повторно запустите `espada-bpc run` с исходными параметрами.

## Результаты

Структура завершённого запуска:

```text
runs/example/
├── config.yaml
├── outputs/
│   ├── bboxing.jsonl
│   ├── captions.jsonl
│   ├── pointing.jsonl
│   ├── bboxing-judged.jsonl
│   ├── pointing-judged.jsonl
│   ├── bboxing/              # атомарные per-image JSONL shards
│   ├── captions/
│   ├── pointing/
│   ├── bboxing-judged/
│   └── pointing-judged/
├── logs/
│   ├── pipeline.log
│   ├── state.jsonl
│   ├── failed.jsonl
│   ├── preflight.log
│   └── progress.json
└── visualizations/
    ├── boxes/
    └── points/
```

Файлы `bboxing-judged.jsonl` и `pointing-judged.jsonl` будут пустыми, если Judge
отключён. Некоторые log-файлы создаются только при наличии соответствующих
событий.

### Формат LLaMA Factory

Каждая строка итогового JSONL — отдельный мультимодальный пример:

```json
{"messages":[{"role":"user","content":"<image>\nWhat is the person doing?"},{"role":"assistant","content":"A person is holding an object with both hands."}],"images":["/absolute/path/to/image.jpg"]}
```

Для bboxing и pointing структурированный ответ сериализуется в строку внутри
`assistant.content`:

```json
{"messages":[{"role":"user","content":"<image>\nLocate every cup in the image. Return the answer in JSON."},{"role":"assistant","content":"[{\"bbox_2d\":[120,80,260,240],\"label\":\"cup\"}]"}],"images":["/absolute/path/to/image.jpg"]}
```

Абсолютный путь к изображению находится в `images`, а `<image>` в пользовательском
сообщении связывает изображение с диалогом.

## Упрощённый запуск через runner.py

`runner.py` оставлен для небольших локальных прогонов без расширенной
BPC-валидации, checkpoint, status и retry-команд.

Обработать директорию из `configs/config.yaml`:

```bash
python runner.py
```

Обработать указанную директорию:

```bash
python runner.py data/raw_images --output-dir outputs
```

Обработать одно изображение:

```bash
python runner.py --img-path data/raw_images/example.jpg --output-dir outputs
```

Доступные параметры:

```bash
python runner.py --help
```

Для долгих, возобновляемых и multi-GPU запусков рекомендуется использовать
`espada-bpc run`.

## Тесты

После установки тестовых зависимостей:

```bash
python -m pip install pytest
python -m pytest
```

Быстрые unit-тесты:

```bash
python -m pytest tests/unit -m unit
```

Smoke-тесты пайплайна используют fake-модели и mocked VLM, поэтому не выполняют
реальные сетевые запросы к модели.

## Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
