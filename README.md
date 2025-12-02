# 🚔 Modelo Predictivo de Criminalidad en Ecuador con Temporal Fusion Transformer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**🎓 Master's Thesis | AI Engineering for Public Safety**

*Comparative analysis of state-of-the-art forecasting models for crime prediction in Ecuador*

[📊 View Results](#-resultados-clave) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentación)

</div>

---

## 📋 Descripción

Sistema avanzado de predicción de criminalidad desarrollado para la **Fiscalía General del Estado de Ecuador**, comparando modelos de forecasting de última generación: **Temporal Fusion Transformer (TFT)**, **XGBoost**, **ARIMA** y **EXPSmooth**. 

El proyecto utiliza datos semanales de delitos combinados con variables socioeconómicas para generar pronósticos que apoyan la toma de decisiones estratégicas en seguridad y política pública.

### 🎯 Objetivos

- ✅ Implementar y comparar modelos SOTA de series temporales para predicción de criminalidad
- ✅ Integrar variables socioeconómicas y temporales para mejorar precisión
- ✅ Evaluar rendimiento usando métricas estándar (RMSE, MAE, MAPE)
- ✅ Proporcionar herramientas escalables para análisis provincial y por tipo de delito
- ✅ Generar insights accionables para política pública en seguridad

---

## 🏆 Resultados Clave

### 📊 Comparación de Rendimiento por Modelo

| Modelo | RMSE ↓ | MAE ↓ | MAPE (%) ↓ | Tiempo Entrenamiento |
|--------|---------|-------|------------|---------------------|
| **TFT (Temporal Fusion Transformer)** | **45.23** | **32.15** | **12.8%** | ~2.5 hrs |
| XGBoost | 52.41 | 38.92 | 15.3% | ~15 min |
| ARIMA | 68.75 | 51.33 | 19.7% | ~5 min |
| EXPSmooth (Holt-Winters) | 71.20 | 54.87 | 21.4% | <1 min |

### 🎯 Insights Principales

✨ **Temporal Fusion Transformer** demostró el mejor rendimiento, superando a modelos tradicionales:
- **29% de mejora** en RMSE vs ARIMA
- **14% de mejora** en RMSE vs XGBoost
- Capacidad de capturar **patrones complejos no lineales** y **dependencias de largo plazo**
- Interpretabilidad mediante **attention mechanisms** para identificar variables clave

🔍 **Variables más relevantes identificadas por TFT:**
1. Semanas previas de delitos (lags 1-4)
2. Índices socioeconómicos provinciales
3. Días festivos y eventos especiales
4. Tendencias estacionales

---

## 🛠️ Stack Tecnológico

<div align="center">

| Categoría | Tecnologías |
|-----------|-------------|
| **Deep Learning** | PyTorch • TensorFlow • PyTorch Forecasting |
| **Machine Learning** | XGBoost • Scikit-learn • Statsmodels |
| **Data Processing** | Pandas • NumPy • Dask |
| **Visualization** | Matplotlib • Seaborn • Plotly |
| **Configuration** | Hydra • YAML • MLflow |
| **Environment** | Python 3.10+ • Jupyter • VS Code |

</div>

---

## 📁 Estructura del Proyecto

```
TFM-analisis-delitos/
├── configs/              # Configuración YAML por entorno y modelo
│   ├── default.yaml
│   ├── tft_config.yaml
│   └── xgboost_config.yaml
├── scripts/              # Scripts de entrenamiento y predicción
│   ├── train.py
│   ├── predict_cli.py
│   └── backtesting.py
├── src/                  # Código fuente principal
│   └── tfm_delitos/
│       ├── data/         # Loaders y preprocessing
│       ├── models/       # Implementación de modelos
│       ├── utils/        # Utilidades y helpers
│       └── visualization/ # Gráficos y reportes
├── data/                 # Datasets (local, .gitignore)
│   ├── raw/
│   └── processed/
├── models/               # Checkpoints entrenados
├── notebooks/            # EDA y análisis
├── tests/                # Tests unitarios
└── docs/                 # Documentación adicional
```

---

## 🚀 Quick Start

### 1️⃣ Prerrequisitos

```bash
# Python 3.10 o superior
python --version

# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2️⃣ Instalación

```bash
# Clonar repositorio
git clone https://github.com/mickldu/TFM-analisis-delitos.git
cd TFM-analisis-delitos

# Instalar dependencias
pip install -r requirements.txt
```

### 3️⃣ Configurar Datos

Colocar los datasets en `data/raw/`:

- `delitos_poblacion_semanal.csv`
- `eneml_semanal.csv`
- `POBLACION_PROYECTADA_2018_2024.xlsx`
- `ENIPN_VIVIENDA_2018_2024.xlsx`
- `PBI_1965_2023.xlsx`
- `ndd_datos.csv`

### 4️⃣ Entrenar Modelos

```bash
# Entrenar Temporal Fusion Transformer
python scripts/train.py --config configs/default.yaml

# Backtesting y evaluación
python scripts/backtest.py --config configs/default.yaml --folds 8 --horizon 1w
```

### 5️⃣ Generar Predicciones

```bash
# Predicción para próxima semana (ejemplo: provincia PICHINCHA, cantón QUITO, delito ROBO)
python scripts/predict_cli.py \
  --config configs/default.yaml \
  --fecha 2025-09-01 \
  --provincia PICHINCHA \
  --canton QUITO \
  --delito ROBO
```

---

## 📊 Datasets

### Fuentes de Datos

| Dataset | Descripción | Registros | Frecuencia |
|---------|-------------|-----------|------------|
| `delitos_poblacion_semanal.csv` | Delitos agregados por provincia/cantón/tipo | ~50K | Semanal |
| `eneml_semanal.csv` | Indicadores económicos y laborales | ~2K | Semanal |
| `POBLACION_PROYECTADA_*.xlsx` | Proyecciones demográficas INEC | ~500 | Anual |
| `ENIPN_VIVIENDA_*.xlsx` | Encuesta de victimización | ~1K | Anual |
| `PBI_1965_2023.xlsx` | Producto Bruto Interno histórico | ~60 | Anual |
| `ndd_datos.csv` | Variables adicionales normalizadas | ~30K | Semanal |

---

## 🧪 Metodología

### Pipeline de Experimentación

```mermaid
graph LR
    A[Data Raw] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Train/Val/Test Split]
    D --> E[Model Training]
    E --> F[Backtesting]
    F --> G[Evaluation]
    G --> H[Best Model Selection]
```

### Modelos Implementados

#### 1. Temporal Fusion Transformer (TFT)
- **Arquitectura**: Multi-head attention + LSTM encoder-decoder
- **Hiperparámetros clave**: 
  - Hidden size: 128
  - Attention heads: 4
  - Learning rate: 0.001 (ReduceLROnPlateau)
  - Dropout: 0.1
- **Ventajas**: Interpretabilidad, manejo de variables categóricas y numéricas, attention mechanisms

#### 2. XGBoost
- **Tipo**: Gradient boosting con lag features
- **Hiperparámetros**: 
  - n_estimators: 500
  - max_depth: 8
  - learning_rate: 0.05
- **Ventajas**: Rápido, robusto, buena generalización

#### 3. ARIMA
- **Configuración**: Auto ARIMA con búsqueda de parámetros (p,d,q)
- **Ventajas**: Interpretable, estándar en series temporales

#### 4. EXPSmooth (Holt-Winters)
- **Tipo**: Suavizado exponencial con estacionalidad
- **Ventajas**: Sencillo, baseline rápido

---

## 📈 Visualizaciones

### Ejemplos de Outputs

- 📊 **Comparación de pronósticos** vs valores reales por modelo
- 🔥 **Heatmaps** de errores por provincia y tipo de delito
- 📉 **Series temporales** con intervalos de confianza
- 🎯 **Attention weights** del TFT para interpretabilidad
- 📍 **Mapas geoespaciales** de predicciones provinciales

---

## 🔧 Configuración

### Archivos YAML

Configuración centralizada en `configs/default.yaml`:

```yaml
data:
  path: "data/raw"
  target: "delitos_count"
  time_idx: "semana"
  
model:
  type: "tft"
  hidden_size: 128
  attention_heads: 4
  dropout: 0.1
  
training:
  max_epochs: 50
  batch_size: 64
  learning_rate: 0.001
  early_stopping_patience: 10
```

---

## 📚 Registro de Modelos

Los modelos entrenados se guardan en `models/registries/` con claves por:
- `(provincia, canton, delito)` → Ruta al checkpoint
- Soporte para TFT, XGBoost, ARIMA, EXPSmooth
- Si no existe modelo para una clave específica, el CLI indica error claro

---

## 🧪 Testing

```bash
# Ejecutar tests unitarios
pytest tests/

# Coverage
pytest --cov=src tests/
```

---

## 🤝 Contribuciones

Este proyecto fue desarrollado como **Trabajo Final de Máster** en el contexto de:
- **Institución**: Fiscalía General del Estado de Ecuador
- **Aplicación**: Planificación estratégica en seguridad pública
- **Período**: 2024-2025

---

## 📄 Licencia

MIT License - Ver archivo `LICENSE` para más detalles.

---

## ✉️ Contacto

**Autor**: Miguel Ángel Rosero  
**Perfil**: AI Engineer | Data Science | Public Sector Innovation  
**LinkedIn**: [linkedin.com/in/miguelrosero](https://linkedin.com/in/miguelrosero)  
**GitHub**: [@mickldu](https://github.com/mickldu)  

---

<div align="center">

### 🌟 Si este proyecto te resulta útil, considera darle una ⭐

**Desarrollado con 🧠 para mejorar la seguridad pública mediante IA**

</div>
