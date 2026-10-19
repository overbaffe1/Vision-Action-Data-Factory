# План улучшения Score: VLA 9.909 → нормальный результат

## 1. Диагноз: почему сейчас 9.909

В `submission.zip` лежит **smoke-чекпоинт**, а не обученная модель. Доказательства —
файлы прямо из архива (`action_expert/train_config.json`):

| Поле | Значение в submission | Стартовый «настоящий» шаблон |
|------|----------------------|------------------------------|
| `dataset.repo_id` | `local/libero-mini-smoke` (5 эпизодов!) | `HuggingFaceVLA/libero` @ v3.0, **1693 эпизода** |
| `training.steps` | **10** | **30000** |
| `training.batch_size` | 1 | 4 |
| `job_name` | `smolvla-libero-smoke` | `smolvla-libero-hackathon` |
| scheduler warmup/decay | 1 / 20 | 1000 / 30000 |
| `chunk_size` / `n_action_steps` | 8 / 8 | 50 / 50 |
| чекпоинт-процессоры | `..._step_5_...`, `..._step_0_...` (save_freq=5) | save_freq=5000 |

То есть expert прошёл **10 шагов обучения на 5 эпизодах** — это валидационный прогон
пайплайна (`run_pipeline.sh … --smoke`), а не обучение. 9.9 на `libero_spatial` —
это уровень почти случайного policy; у реально обученного SmolVLA по этому рецепту
self-check даёт единицы-десятки процентов уже на ранних чекпоинтах и **0.6–0.95**
к финалу.

Дополнительные следы smoke-сборки:

- `annotations.jsonl` в архиве — **63 байта-заглушка** → VLM на данных фабрики
  вообще не обучался (в новой сборке zip заменён на 92 примера, но их всё равно мало
  для SFT — нужно генерировать на реальных источниках).
- `vlm_model_name: /home/ubuntu22/aij_robotics_judge/submission/smolvlm2` —
  абсолютный путь сборки на чужой машине (на машине судьи такой путь существует,
  поэтому eval не падал).
- `policy_preprocessor_step_5_normalizer_processor.safetensors` — нормализатор
  state/action, посчитанный по **mini-датасету** (5 эпизодов) — статистики враные
  даже для inference.

**Вывод: обе компоненты (VLM и action expert) нужно обучить заново по полному
рецепту.** Никакие инференс-кнуты (chunk_size, num_steps) smoke-модель не спасут.

## 2. Цели

- **VLA (вес 0.6):** полное обучение на libero v3.0 → self-check `libero_spatial`
  (10 задач, mean SR) в районе 0.6–0.95 с выбором лучшего чекпоинта.
- **VLM (вес 0.4):** full-SFT SmolVLM2 на аннотациях фабрики из 4 разрешённых
  источников (BridgeData2, Fractal, Language Table, Egocentric-100K).
- Итог: `Score = 100 × (0.4·VLM + 0.6·VLA)` — при VLA=0.7 и VLM=0.5 уже ≈ **48**
  вместо текущих ~6.

## 3. Что нужно для обучения

- **GPU** ≥ 16 GB (40xx/50xx ок, H100 идеально) — локально или в облаке (MLSPACE и т.п.).
- **Docker** + NVIDIA Container Toolkit (или `./scripts/setup_env.sh` без Docker).
- **Доступ к huggingface.co** — для `HuggingFaceVLA/libero` (v3.0), весов
  SmolVLM2-500M-Video-Instruct и датасетов-источников VLM.
- Стартовый kit `aij_robotics` (публичное зеркало уже лежит в этом репо как
  `aij_robotics-main/` для справки; на машине обучения клонируйте его).

## 4. Пошаговый план

### 4.1 Данные VLM (фабрика)

На машине с HF-доступом и скачанными 4 источниками:

```bash
python generate_dataset.py --config config.yaml   # параметры сэмплинга — в config.yaml
```

На выходе: ShareGPT `annotations.jsonl` + `images/`. Правила: **только** 4
разрешённых источника, **без** кадров LIBERO, **без** сторонних аннотаций (ECoT и
т.п.) — свои аннотации фабрики ок. Чем больше эпизодов/кадров — тем лучше VLM
(целевой порядок: сотни-тысячи примеров, не 92).

### 4.2 Полное обучение (одной командой)

```bash
# participant.yaml: vlm.data_path → ваш ShareGPT, vla.dataset_dir → корень LeRobot v3.0
docker run --gpus all --rm --shm-size=8g \
  -e CUDA_HOME=/usr/local/cuda -e MUJOCO_GL=egl \
  -v "$PWD/configs:/workspace/configs:ro" \
  -v /path/to/sharegpt:/data/vlm:ro \
  -v /path/to/HuggingFaceVLA-libero-v3:/data/vla:ro \
  -v "$PWD/runs:/workspace/runs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  aij-robotics:latest \
  ./scripts/run_pipeline.sh configs/participant.yaml
```

Флаги: `--skip-vlm` / `--skip-vla` — если одну половину уже обучили,
`--vlm-checkpoint PATH` — подставить готовый VLM.

**Рекомендуемые правки `configs/vla_libero.yaml` (правила разрешают):**

```yaml
training:
  save_freq: 2500        # было 5000 — больше чекпоинтов для отбора
  batch_size: 4          # 8–16, если VRAM позволяет (H100: 16 спокойно)
# всё остальное шаблона оставляем:
# steps 30000, lr 1e-4, warmup 1000, chunk 50/50,
# freeze_vision_encoder/train_expert_only/train_state_proj: true
```

`configs/vlm_sft.yaml` (полный SFT, 1 эпоха, lr 1e-5, full) — ок как есть; больше
всего решают **данные** (п. 4.1).

**Строго нельзя (правила):** аугментации изображений, парафразы инструкций,
libero_90/доп. демо, чужие SmolVLA-чекпоинты (init/teacher), LIBERO-кадры в VLM.

### 4.3 Отбор лучшего чекпоинта VLA

Из `runs/.../action_expert/checkpoints/*` для каждого кандидата (шаги 7.5k/15k/22.5k/30k…):

```bash
SUITE=libero_spatial TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' N_EPISODES=1 SEED=1000 \
  MUJOCO_GL=egl ./scripts/run_libero_selfcheck.sh runs/<run>/submission
# лучший по overall.pc_success → скопировать его веса в submission/action_expert/
```

(Можно добавить `episodes_per_task: 5` в `selfcheck_score.example.yaml` для
менее шумной оценки.)

### 4.4 Валидация и сборка submission

```bash
./scripts/validate_submission.sh runs/<run>/submission
```

Итоговая структура (README §5.1):

```text
submission/
├── smolvlm2/
├── action_expert/     # config.json + train_config.json перезапишутся реальным прогоном
├── metadata.json      # {"base_vlm": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "vla_suite": "libero"}
└── participant.yaml
```

Zip: **только эти 4 пункта, ≤5 ГБ, без датасетов/картинок/кэшей/логов**
(лимит inode уже убивал посылки).

### 4.5 Загрузка

- Прямая загрузка zip в форму (≤5 ГБ), либо
- наш workflow `push-parts-to-git.yml` (если канал с вашей машины слабый:
  URL → чанки в git → release), затем `finalize-submission.yml` соберёт
  готовый релиз со ссылками на 2 части.

Старый релиз `submission-final-20261018-2216` со смок-весами **не использовать**
для новых загрузок — он оставлен только как история.

## 5. Ожидаемый эффект

| Компонент | Сейчас (smoke) | После полного обучения |
|-----------|----------------|------------------------|
| VLA (`acc-vla-prom`) | 9.9 | десятки–90+ (self-check protocol) |
| VLM | не обучался на фабрике | SFT на полных аннотациях |
| Score = 100(0.4·VLM+0.6·VLA) | ~6–10 | зависит от обеих частей |

Официальный протокол VLA судьи закрыт (self-check — `libero_spatial`), поэтому
цифру обещать нельзя, но разрыв «10 шагов vs 30000» — это основной вклад.
