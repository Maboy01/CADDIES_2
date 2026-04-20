# Caddies 2.0 - Golf Swing AI

Proyecto para entrenar modelos que clasifican swings de golf usando GolfDB.

## Scripts principales

- `train_csv_model.py`: baseline rapido con solo `GolfDB.csv`.
- `train_cnn_lstm.py`: modelo robusto con videos, CNN por frame y LSTM temporal.
- `train_model.py`: launcher pequeno; por defecto ejecuta CNN+LSTM.

## Entrenar CNN+LSTM

```powershell
.\venv\Scripts\python.exe train_cnn_lstm.py --max-videos 115 --epochs 25 --patience 6 --device cuda
```

Opciones utiles:

```powershell
.\venv\Scripts\python.exe train_cnn_lstm.py --max-videos 20 --epochs 2 --sequence-length 8 --frame-size 64 --device cuda
.\venv\Scripts\python.exe train_cnn_lstm.py --max-videos 100 --epochs 10 --sequence-length 16 --frame-size 96 --device cuda
.\venv\Scripts\python.exe train_cnn_lstm.py --max-videos 115 --epochs 25 --patience 6 --device cuda --use-bbox-crop --bbox-padding 0.45
```

## Entrenar baseline CSV

```powershell
.\venv\Scripts\python.exe train_csv_model.py --max-rows 50
```

Tambien se puede ejecutar desde el launcher:

```powershell
.\venv\Scripts\python.exe train_model.py --max-videos 50 --epochs 5 --device cuda
.\venv\Scripts\python.exe train_model.py --csv-baseline --max-rows 50
```

## Resultados

Los resultados se guardan separados:

- `model_results/cnn_lstm/`
- `model_results/csv_baseline/`

El modelo CNN+LSTM guarda:

- `cnn_lstm_model.pt`
- `training_history.png`
- `confusion_matrix.png`
- `metrics_by_class.png`
- `prediction_distribution.png`
- `confidence_distribution.png`
- `training_history.csv`
- `class_metrics.csv`
- `predictions.csv`
- `run_summary.csv`

## Entrenamiento CNN+LSTM actual

El flujo usa `70%` entrenamiento, `15%` validacion y `15%` test. Tambien usa:

- cache de tensores en `model_results/cnn_lstm/tensor_cache/`
- augmentations ligeras en entrenamiento
- `AdamW` con `weight_decay`
- scheduler de learning rate
- early stopping por accuracy de validacion

## Dependencias

El entorno local usa PyTorch CUDA:

```text
torch==2.11.0+cu128
```

La instalacion usa el indice CUDA de PyTorch definido en `requirements.txt`.
