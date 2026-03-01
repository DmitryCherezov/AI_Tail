
# Что скрывает хвост кота? 🐈

Проект посвящён задаче сегментации кошачьего хвоста на изображениях с использованием архитектуры **YOLOv11n-seg**.

Проект реализует полный end-to-end pipeline:
- сбор данных из открытых источников
- подготовка и очистка изображений
- аннотирование
- конвертация в формат YOLO
- синтетическая аугментация
- обучение модели
- валидация и визуальный анализ результатов

Проект построен в инженерной логике reproducible ML-пайплайна с чётким разделением данных, скриптов обработки и этапов обучения.

---

# Архитектурная идея проекта

Задача рассматривается как **instance segmentation**, где требуется:
- обнаружить хвост
- корректно сегментировать его маску

Используется модель YOLOv11n-seg как лёгкая архитектура, подходящая для:
- небольшого датасета
- быстрого обучения
- возможности последующего деплоя на edge-устройства

Для небольшого датасета отключены агрессивные методы аугментации (mosaic, mixup), чтобы избежать переобучения на синтетических комбинациях.

---

# Структура проекта

```
train.py
predict.py
demo.py

data/
    annotations/
    dataset/
        data.yaml
        images/train
        images/val
        labels/train
        labels/val
    labels/
    png_augmentations/
    png_images/
    raw_augmentations/
    raw_images/

scripts/
    data_utils.py
    script_convert_images.py
    script_create_labels.py
    script_split_data.py
    script_copy_past_augmentation.py
    yolo_seg_viewer.py

task/
    cat_tails_project.docx
```

Проект разделён на три логических блока:
1. Data preparation
2. Training & evaluation
3. Visualization & demo

---

# Data pipeline

## raw_images/
Хранятся оригинальные изображения без изменений.

## script_convert_images.py
Конвертация изображений в PNG для унификации формата.

## png_images/
Изображения, используемые для аннотирования.

## annotations/
JSON-аннотации масок хвоста.

## script_create_labels.py
Конвертация JSON в формат YOLO segmentation.

## script_split_data.py
Случайное разбиение датасета (например 120 train / 30 val).

## dataset/
Финальная структура, читаемая YOLO через `data.yaml`.

---

# Синтетическая аугментация

## raw_augmentations/
Изображения без хвостов.

## script_copy_past_augmentation.py
Генерация синтетических данных путём copy-paste хвостов на изображения без хвостов.

Это позволяет:
- увеличить разнообразие позиций хвоста
- снизить риск переобучения
- повысить устойчивость модели

---

# Обучение модели (train.py)

Пример запуска:

```bash
python train.py --data data.yaml --models-dir yolov11n-seg.pt --save runs/project --epochs 50
```

Ключевые параметры:

- `--data` — путь к data.yaml
- `--models-dir` — предобученная модель
- `--save` — директория проекта
- `--epochs` — число эпох
- `--batch` — размер батча
- `--imgsz` — размер входного изображения
- Геометрические аугментации (degrees, translate, scale)
- Цветовые аугментации (HSV)
- Flip вероятности

Для небольшого датасета mosaic и mixup отключены.

---

# Метрики оценки

Валидация производится стандартными метриками YOLO:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

mAP@0.5:0.95 отражает усреднённую точность сегментации при различных IoU-порогах и является основной метрикой качества.

---

# Предсказание (predict.py)

```bash
python predict.py --images input_folder --out output_folder --weights runs/project/best.pt
```

Параметры:
- `--images` — входные изображения
- `--weights` — обученная модель
- `--out` — папка для сохранения YOLO txt
- `--conf` — порог уверенности
- `--device` — cuda / cpu

---

# Визуализация (demo.py)

```bash
streamlit run demo.py -- --images-dir images --labels-dir validation_prediction --classes "0:tail"
```

Позволяет:
- визуально проверить сегментацию
- анализировать ошибки
- сравнивать предсказания и ground truth

---

# Инженерные особенности

Проект построен по принципам:

- разделение данных и логики
- воспроизводимость экспериментов
- CLI-управление параметрами
- возможность расширения под другие классы
- адаптация под небольшие датасеты

Архитектура легко масштабируется на:
- multi-class segmentation
- object detection
- задачи edge-deployment

---

# Возможные улучшения

- Добавление cross-validation
- Автоматический логгинг метрик (MLflow)
- Анализ ошибок по IoU
- Балансировка датасета
- Экспорт модели в ONNX / TensorRT

---

Проект демонстрирует полный цикл разработки CV-модели: от сырых данных до работающего demo-интерфейса.
