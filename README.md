# Vision-Action Data Factory

**Из эпизодов робота — в обучающие примеры «кадр, вопрос, ответ».**  
Хакатонный проект · Python · LeRobot · SmolVLM2 / SmolVLA

<p align="center"><a href="data/final_input/libero_episode_0/frame_00000.jpg"><img src="data/final_input/libero_episode_0/frame_00000.jpg" width="33%" alt="Начало эпизода LIBERO, кадр 0"></a><a href="data/final_input/libero_episode_0/frame_00007.jpg"><img src="data/final_input/libero_episode_0/frame_00007.jpg" width="33%" alt="Середина эпизода, кадр 7"></a><a href="data/final_input/libero_episode_0/frame_00015.jpg"><img src="data/final_input/libero_episode_0/frame_00015.jpg" width="33%" alt="Конец эпизода, кадр 15"></a></p>

*Три кадра одного эпизода симуляции LIBERO. Запись ниже относится к первому кадру.*

> **Задача:** взять банку супа и положить её в корзину.  
> **Что делать дальше?** Фрагмент ответа генератора: “Next subtask: Place it in the basket.”

**[Открыть запись JSONL](annotations.jsonl#L2)** · **[Запустить демо](#быстрый-старт)** · **[Скачать starter kit](https://github.com/overbaffe1/Vision-Action-Data-Factory/releases/tag/starter-kit-v1)**

## Что я сделал на хакатоне

Запись манипуляции даёт кадры и общую инструкцию, но не объясняет, что робот делает в конкретный момент. Я собрал генератор таких аннотаций и автоматизировал упаковку материалов для отправки.

1. [Семплер](src/frame_sampler.py) выбирает кадры по времени и, если доступна телеметрия, по изменениям действий и состояния захвата.
2. [Детектор](src/detectors/yolo_detector.py), [геометрия](src/spatial.py) и [правила разбиения инструкций](src/subtask.py) дают материал для вопросов о сцене, прогрессе и следующем шаге.
3. [Эвристический судья](src/judge.py) ранжирует варианты, а [генератор](generate_dataset.py) отбрасывает слабые записи и дубли. На выходе — ShareGPT JSONL с полями `messages` и `images`.

В [сохранённом демо](annotations.stats.json) — **3 эпизода, 23 кадра, 92 записи**. Это объём демо, не оценка качества модели и не место в рейтинге.

## Быстрый старт

Linux, macOS или WSL · Python 3.10+ · без GPU и весов. Частичный клон нужен, чтобы не скачивать архив хакатона.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/overbaffe1/Vision-Action-Data-Factory.git
cd Vision-Action-Data-Factory
git sparse-checkout set src data/final_input

python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'numpy>=1.24' 'Pillow>=10' 'PyYAML>=6'
PYTHONHASHSEED=0 python generate_dataset.py \
  --input data/final_input \
  --output /tmp/vision-action-demo/annotations.jsonl \
  --config config.yaml
python -m json.tool /tmp/vision-action-demo/annotations.stats.json
```

В этом режиме боксы синтетические: демо показывает путь от кадров до JSONL, а не точность детекции. Число записей может отличаться от сохранённого примера.

## Материалы хакатона

- [Исходный PR участия](https://github.com/overbaffe1/Vision-Action-Data-Factory/pull/1) — генератор, упаковка и разбор baseline.
- [План обучения](IMPROVE.md) и [комплект организаторов](aij_robotics-main/README.md) — отдельный пайплайн SmolVLM2 → SmolVLA.
- [Архивная сборка submission](https://github.com/overbaffe1/Vision-Action-Data-Factory/releases/tag/submission-final-20261018-2216) — две части и инструкция склейки. В архиве smoke-чекпоинт, а не результат полного переобучения.
