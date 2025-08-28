TFM análisis de delitos en Ecuador

Objetivo
Desarrollar y comparar modelos de predicción de delitos por provincia y cantón con variables exógenas. Base: Temporal Fusion Transformer frente a ARIMAX y XGBoost. Integración futura con LangChain para consultas en lenguaje natural.

Estructura del proyecto
src/tfm_delitos       Código de la librería
scripts               Ejecutables de entrenamiento y predicción
configs               Configuración YAML por entorno
data                  Carpeta local, no versionar datos sensibles
models                Registries y checkpoints
notebooks             EDA y prototipos
tests                 Pruebas unitarias
docs                  Documentación

Datos esperados
Coloca tus fuentes en data/raw
Ejemplos usados en tus consultas previas
- delitos_poblacion_semanal.csv
- enemu_semanal.csv
- POBLACION_PROYECTADA.xlsx
- ENEMDU_VIVIENDA_2018_2024.xlsx
- ENEMDU_LABORAL_2018_2024.xlsx
- PBI_1965_2023.xlsx
- ndd_datos.csv
Acomódalos por nombre en data/raw y ajusta rutas en configs/default.yaml

Entorno
1. Instala Python 3.10 o 3.11
2. Crea entorno virtual
   python -m venv .venv
   .venv\Scripts\activate  en Windows
   source .venv/bin/activate en Linux/Mac
3. Instala dependencias
   pip install -r requirements.txt

Uso rápido
1. Configura el archivo configs/default.yaml con rutas de datos y llaves de serie
2. Entrena
   python scripts/train.py --config configs/default.yaml
3. Backtesting rolling-origin
   python scripts/backtest.py --config configs/default.yaml --folds 8 --horizon 1w
4. Predicción para la próxima semana por provincia, cantón y delito
   python scripts/predict_cli.py --config configs/default.yaml --fecha 2025-09-01 --provincia PICHINCHA --canton QUITO --delito ROBO

Registro de modelos
Se guardan en models/registries con claves por (provincia, cantón, delito).
Soporte para ARIMAX, XGBoost y TFT. Si no existe un modelo entrenado para una clave, el CLI lo indicará de forma clara.

Integración con LangChain
Se deja un módulo base en src/tfm_delitos/utils/langchain_gateway.py para orquestar preguntas simples a modelos ya entrenados.

Convenciones
- Código en src con import tfm_delitos
- Configuración en YAML
- Semillas y logs en outputs/logs
- Estilo PEP8 y docstrings cortos

Licencia
MIT por defecto. Ajusta según tu institución.

Contacto
Autor: Miguel Espinoza Rosero
