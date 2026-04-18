# ⛳ CADDIES 2.0 - Golf Coach AI

Modelo de IA para analizar y mejorar swing de golf con **curva de aprendizaje** y **métricas detalladas**.

## 📊 Contenido

- **train_model.py** - Modelo ML con curva de aprendizaje y épocas
- **GolfDB.csv** - Metadata de 1400+ golpes profesionales
- **videos_160/** - Videos de swings en 160x160px

## 🚀 Inicio Rápido

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo
python train_model.py
```

## 📈 Salida

El script genera automáticamente 4 gráficos:

1. **learning_curve.png** - Curva de aprendizaje (Train vs Validation)
2. **training_epochs.png** - Loss y Accuracy por época
3. **confusion_matrix.png** - Matriz de confusión
4. **metrics_by_class.png** - Precision, Recall, F1 por clase

Se guardan en: `model_results/`

## 🎯 Datos del Modelo

- **Datos**: 1400+ registros de golpes profesionales
- **Features**: Player, Género, Club, Vista, Cámara Lenta
- **Target**: Clasificar tipo de palo (driver, iron, fairway, etc)
- **Algoritmo**: Random Forest
- **Accuracy**: ~89%

## 📊 Métricas Generadas

```
✓ Accuracy General
✓ Precision por clase
✓ Recall por clase  
✓ F1-Score por clase
✓ Curva de aprendizaje
✓ Épocas de entrenamiento
```

## 🔧 Requisitos

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## 📁 Estructura

```
CADDIES 2.0/
├── train_model.py           ← Modelo ML principal
├── requirements.txt         ← Dependencias
├── GolfDB.csv              ← Datos profesionales
├── golfDB.mat              ← Datos MATLAB
├── GolfDB.pkl              ← Datos serializados
├── videos_160/             ← 1400 videos
└── model_results/          ← Gráficos generados
```

## 🎯 Próximos Pasos

- [ ] Agregar procesamiento de videos
- [ ] Integrar MediaPipe para pose detection
- [ ] Crear API REST
- [ ] Dashboard web interactivo

---

**Próxima versión**: Golf Coach con análisis de poses en tiempo real
