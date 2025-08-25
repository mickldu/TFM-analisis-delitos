# Análisis del Tipo de Cambio DOP/USD con Modelos de Series Temporales

Este repositorio contiene el código y los análisis para el Trabajo de Fin de Máster (TFM) enfocado en la predicción del tipo de cambio entre el Peso Dominicano (DOP) y el Dólar Estadounidense (USD).

## Descripción del Proyecto

El objetivo principal de este proyecto es comparar el rendimiento de tres modelos populares de pronóstico de series temporales para estimar la tasa de cambio DOP/USD. El análisis completo, la implementación de los modelos y la evaluación de resultados se encuentran en el Jupyter Notebook principal.

## Modelos Utilizados

Se implementaron y evaluaron los siguientes modelos:
- **ARIMA** (Autoregressive Integrated Moving Average)
- **SARIMA** (Seasonal ARIMA)
- **Prophet** (desarrollado por Facebook)

## Fuente de Datos

Los datos para el análisis no se encuentran almacenados en este repositorio. En su lugar, se descargan dinámicamente desde **Yahoo Finance** utilizando la librería `yfinance` de Python. El ticker utilizado para el tipo de cambio DOP/USD es `USDDOP=X`.

## Estructura del Repositorio

- **/docs/**: Contiene el Jupyter Notebook con el análisis completo (`Análisis_del_tipo_de_cambio_DOP_USD...ipynb`) y otros documentos de referencia.
- **/src/**: Código fuente modular (si aplica).
- **/data/**: Datos brutos, intermedios y procesados (actualmente ignorado por Git).
- **requirements.txt**: Lista de dependencias de Python para recrear el entorno.

## Cómo Empezar

1.  Clona este repositorio en tu máquina local.
2.  Crea un entorno virtual e instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Abre y ejecuta el notebook principal que se encuentra en la carpeta `/docs/` para ver el análisis completo.