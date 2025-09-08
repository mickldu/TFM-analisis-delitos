Modelo Predictivo de Criminalidad en Ecuador mediante la comparación de Temporal Fusion Transformer (TFT), XGBoost, ARIMA, EXPSmooth

Objetivo
Desarrollar y comparar modelos de predicción de delitos en Ecuador utilizando ExpSmooth, ARIMA, XGBoost y Temporal Fusion Transformer con datos semanales y variables socioeconómicas, con el fin de generar pronósticos que apoyen la toma de decisiones en seguridad y política pública.

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
Fuentes en data/raw
DataSet Utilizados
- delitos_poblacion_semanal.csv
- enemu_semanal.csv
- POBLACION_PROYECTADA.xlsx
- ENEMDU_VIVIENDA_2018_2024.xlsx
- ENEMDU_LABORAL_2018_2024.xlsx
- PBI_1965_2023.xlsx
- ndd_datos.csv

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
Soporte para EXPSmooth, ARIMAX, XGBoost,TFT. Si no existe un modelo entrenado para una clave, el CLI lo indicará de forma clara.

Convenciones
- Código en src con import tfm_delitos
- Configuración en YAML
- Semillas y logs en outputs/logs
- Estilo PEP8 y docstrings cortos

Licencia
MIT por defecto.

Contacto
Autor: Miguel Angel Rosero
