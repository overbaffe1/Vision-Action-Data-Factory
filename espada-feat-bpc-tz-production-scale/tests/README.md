# Тесты espada

Каркас тестов как «сетка безопасности» перед рефакторингом эффективности
(мульти-GPU / батчинг). Сначала фиксируем поведение, потом меняем код.

## Структура

```
tests/
├── conftest.py              # общие фикстуры + стабы тяжёлых деп + make_synthetic_dataset
├── unit/
│   ├── test_structure_checkers.py   # валидация 3 JSON-схем (чистая логика)
│   ├── test_vlm_parsing.py          # парсинг chat-completion ответов
│   ├── test_prompts_processing.py   # YAML + рендер {{ placeholders }} (+ реальный prompts.yaml)
│   ├── test_utils.py                # геометрия боксов, центроид, JSON, ресайз (needs cv2)
│   ├── test_yolov8.py               # обвязка YoloV8 + empty_cache (needs pandas)
│   ├── test_init_models_config.py   # проброс models-конфига в init_models
│   ├── test_yolo_worker_batched.py  # батч-воркер: группировка, task_done, sentinel (needs cv2)
│   ├── test_sharding.py             # ~99 кейсов: покрытие/дизъюнктность/edge шардинга
│   └── test_multigpu.py             # resolve_gpu_ids, план+spawn, пиннинг device
└── smoke/
    ├── test_pipeline_smoke.py       # e2e single-process (per-image и batched)
    ├── test_multigpu_e2e.py         # e2e мульти-GPU/мульти-нода (1/2/4 GPU, 2×2/4×2 ноды)
    └── test_cli_routing.py          # CLI: ветвление single/multi, флаги, валидация
```

Всего: **232 passed, 0 skipped**.

## Запуск

```bash
# всё
python -m pytest

# только быстрые юниты
python -m pytest tests/unit -m unit

# только smoke
python -m pytest tests/smoke
```

## Дизайн: работа без GPU/тяжёлых либ

Тяжёлые зависимости (`torch`, `transformers`, `ultralytics`) есть только в
полном venv `vvinokurov-markup-env`. Чтобы сюита гонялась и в минимальном
окружении:

- `tests/conftest.py` ставит **лёгкие стабы** этих модулей в `sys.modules`,
  если реальные не установлены (в полном venv реальные побеждают).
- `cv2`/`pandas` нужны по-настоящему для `utils`/`yolov8`/`smoke` — там
  `pytest.importorskip`. В минимальном окружении достаточно
  `pip install opencv-python-headless pandas`.
- Модели и сетевой VLM нигде не вызываются по-настоящему: YOLO/SAM2 — фейки,
  `chat_completion` — мок.

## Маркеры (pyproject.toml)

`unit`, `smoke`, `needs_cv2`, `needs_pandas`, `needs_torch`.

## Что покрыто

- **Валидаторы** bbox/pointing/captioning — позитив и негатив.
- **Парсинг VLM** — dict/str/bytes, list-контент, битый JSON, дефолты полей.
- **Промпты** — рендер, dot-path, JSON-сериализация, реальный `configs/prompts.yaml`.
- **utils** — площадь/фильтр боксов, центроид по distance transform, extract/build/save
  JSON, ресайз и пропуск уже обработанных, `_try_save_vlm_json` (валид/невалид/неизвестный таск).
- **YoloV8** — выбор device, пустой/непустой результат → DataFrame, батч-чанки.
- **smoke** — полный конвейер: 2 кадра → 3 JSON на кадр; и сценарий «нет детекций →
  pointing пропущен».
