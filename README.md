# Sign Language MNIST Classifier (PyTorch Lightning)

Классификатор жестов по датасету Sign Language MNIST. Реализация основана на учебном ноутбуке Kaggle (см. ссылку ниже), код перенесен в структуру проекта с конфигами Hydra, обучением в PyTorch Lightning и логированием в MLflow.

Ссылка на исходный ноутбук:
```
https://www.kaggle.com/code/vijaypro/cnn-pytorch-96/notebook
```

## Что внутри
- Свёрточная сеть для классификации 24 классов (буквы жестового алфавита).
- Датасет в CSV формате (28x28 пиксели, колонка `label` + `pixel1..pixel784`).
- Конфиги через Hydra (`config/`).
- Логирование метрик в MLflow.
- Скрипты: загрузка данных, обучение, инференс, экспорт в ONNX.

## Быстрый старт
1. Установить зависимости через `uv`:
```
uv venv -p 3.12 .venv
source .venv/bin/activate
uv sync
```

2. Скачать данные (архив с Google Drive, ссылка в `config/data_load/data_load.yaml`):
```
uv run python sign_language_classifier/data_load/download_data.py
```

3. Обучить модель:
```
uv run python sign_language_classifier/train/train.py
```

4. Инференс на тесте:
```
uv run python sign_language_classifier/infer/infer.py \
  --test_csv_path data/sign_mnist_test.csv \
  --checkpoint_path checkpoints/<your_checkpoint>.ckpt
```

## Конфигурация
Все параметры в `config/`:
- `config/data_load/data_load.yaml` — пути к CSV, размер изображений, batch size, ссылка на данные.
- `config/training/training.yaml` — число эпох, learning rate, workers.
- `config/model/model.yaml` — число классов, входные каналы, чекпоинты.
- `config/logging/logging.yaml` — MLflow параметры.

Переопределение параметров через Hydra:
```
uv run python sign_language_classifier/train/train.py training.num_epochs=30 training.lr=0.0005
```

## Структура проекта
- `sign_language_classifier/modules/model.py` — архитектура CNN.
- `sign_language_classifier/modules/data.py` — датасет и Lightning DataModule.
- `sign_language_classifier/modules/trainer.py` — LightningModule (loss, метрики).
- `sign_language_classifier/train/train.py` — запуск обучения.
- `sign_language_classifier/infer/infer.py` — инференс чекпоинта.
- `sign_language_classifier/modules/onnx_compilation.py` — экспорт в ONNX.
- `config/` — Hydra-конфиги.
- `data/` — CSV файлы датасета.

## MLflow
По умолчанию используется локальный MLflow (`tracking_uri` в `config/logging/logging.yaml`).
Запуск UI:
```
mlflow ui --port 8080
```

## Экспорт в ONNX
```
python sign_language_classifier/modules/onnx_compilation.py \
  --checkpoint_path checkpoints/<your_checkpoint>.ckpt \
  --onnx_output_path models/model.onnx
```

## Примечания
- Текущий код — чистая классификация, а не детектор YOLO.
- Датасет Sign Language MNIST содержит 24 класса (нет J и Z).
