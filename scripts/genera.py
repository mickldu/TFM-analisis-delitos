
# crear_notebook.py
# Este script genera automáticamente el archivo Jupyter Notebook
# "tmf_model2_adaptado.ipynb" con el contenido comentado para tu TFM.

notebook_json = r'''{

 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TFM — Modelado con `del_master_tfm.csv` (Notebook Adaptado y Comentado)\n",
    "\n",
    "Este cuaderno está **adaptado** para consumir directamente el dataset unificado `del_master_tfm.csv` generado por tu script `tfm_unificacion_master.py`.\n",
    "\n",
    "## ¿Qué incluye?\n",
    "- **Carga** del CSV y **validación del esquema** (claves, targets, exógenas, features de ingeniería).\n",
    "- **Partición temporal** (último ~15% de semanas como validación).\n",
    "- **Construcción de features**: categóricas (Provincia, Cantón, Delito) y numéricas (población, PIB, ENEMDU, lags, rollings, calendario).\n",
    "- Entrenamiento de **modelos base** (Random Forest) para dos objetivos:\n",
    "  1. `Total_delitos`\n",
    "  2. `Tasa_delitos_100k`\n",
    "- **Métricas**: RMSE, MAE, R².\n",
    "- **Gráficos** simples real vs predicción para un ejemplo representativo.\n",
    "- **Exportación** de predicciones por serie y un CSV opcional para preparar datos para **TFT** (Temporal Fusion Transformer).\n",
    "\n",
    "## Requisitos\n",
    "- Python 3.9+ (ideal 3.10/3.11)\n",
    "- Paquetes: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.\n",
    "  - Si falta alguno: `pip install pandas numpy scikit-learn matplotlib`\n",
    "\n",
    "## Convenciones del dataset\n",
    "- Clave final: **`[Periodo (lunes ISO), Provincia, Canton, Delito]`**\n",
    "- Targets: `Total_delitos`, `Tasa_delitos_100k`\n",
    "- Exógenas ENEMDU: columnas con prefijos `edu_`, `lab_`, `pob_`, `viv_`.\n",
    "- Ingeniería de series: `*_lag{1,2,4}`, `*_roll{4,8,12}_{mean,std}`.\n",
    "- Calendario: `anio`, `semana_anio`, `mes`, `trimestre`, `fin_de_mes`, `es_feriado`, `vacaciones_escolares`.\n",
    "\n",
    "Si tu CSV final cumple con lo anterior, este cuaderno funciona sin cambios. En caso contrario, edita los **prefijos** o **nombres** en las celdas de detección automática."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================\n",
    "# 1) Imports y configuración\n",
    "# ==========================\n",
    "# Esta celda importa librerías, fija rutas y crea carpeta de salidas.\n",
    "\n",
    "import os\n",
    "import re\n",
    "import math\n",
    "import json\n",
    "import warnings\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from datetime import datetime\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Ruta del dataset generado por el pipeline de unificación\n",
    "DATA_PATH = Path('del_master_tfm.csv')  # Cambia si tu archivo está en otra ruta\n",
    "\n",
    "# Carpeta de salidas (predicciones, TFT-ready, etc.)\n",
    "SAVE_DIR = Path('model_outputs')\n",
    "SAVE_DIR.mkdir(exist_ok=True)\n",
    "\n",
    "print(f'DATA_PATH = {DATA_PATH.resolve()}')\n",
    "print(f'Outputs -> {SAVE_DIR.resolve()}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ======================\n",
    "# 2) Carga del CSV base\n",
    "# ======================\n",
    "# - Forzamos el parseo de 'Periodo' como fecha.\n",
    "# - Inspeccionamos tamaño y primeras filas para verificar estructura.\n",
    "\n",
    "assert DATA_PATH.exists(), f'No se encontró {DATA_PATH}. Verifica la ruta.'\n",
    "df = pd.read_csv(DATA_PATH, parse_dates=['Periodo'])\n",
    "print('Filas:', len(df), ' Columnas:', len(df.columns))\n",
    "display(df.head(5))\n",
    "\n",
    "# Opcional: Quick info para verificar tipos\n",
    "display(df.dtypes.head(20))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Validación de esquema\n",
    "Comprobamos que existan las **claves**, los **targets** y detectamos automáticamente las **exógenas** y features de ingeniería (lags/rollings) junto con columnas de **calendario**.\n",
    "\n",
    "> Si alguna columna no aparece porque tu CSV final usa otros nombres/prefijos, modifica las listas/patrones en esta sección."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================\n",
    "# 3) Chequeos de consistencia\n",
    "# =========================\n",
    "expected_keys = ['Periodo','Provincia','Canton','Delito']\n",
    "for k in expected_keys:\n",
    "    assert k in df.columns, f'Falta columna clave: {k}'\n",
    "\n",
    "# Targets esperados\n",
    "assert 'Total_delitos' in df.columns, 'Falta Total_delitos'\n",
    "assert 'Tasa_delitos_100k' in df.columns, 'Falta Tasa_delitos_100k'\n",
    "\n",
    "# Detectar exógenas ENEMDU por prefijos estándar (ajustar si difieren en tu CSV)\n",
    "EXO_PREFIXES = ('edu_','lab_','pob_','viv_')\n",
    "exo_cols = [c for c in df.columns if c.startswith(EXO_PREFIXES)]\n",
    "\n",
    "# Detectar features de ingeniería (lags y rollings) por patrón\n",
    "lag_cols = [c for c in df.columns if re.match(r'(Total_delitos|Tasa_delitos_100k)_lag\\\\d+$', c)]\n",
    "roll_cols = [c for c in df.columns if re.match(r'(Total_delitos|Tasa_delitos_100k)_roll\\\\d+_(mean|std)$', c)]\n",
    "\n",
    "# Columnas de calendario (si existen en tu CSV, se agregan como numéricas)\n",
    "calendar_cols = ['anio','semana_anio','mes','trimestre','fin_de_mes','es_feriado','vacaciones_escolares']\n",
    "calendar_cols = [c for c in calendar_cols if c in df.columns]\n",
    "\n",
    "# Candidatas numéricas: población, PIB y todo lo detectado arriba\n",
    "numeric_candidates = ['Poblacion','PIB_por_canton'] + exo_cols + lag_cols + roll_cols + calendar_cols\n",
    "numeric_candidates = [c for c in numeric_candidates if c in df.columns]\n",
    "\n",
    "print('Exógenas detectadas:', len(exo_cols))\n",
    "print('Lags:', len(lag_cols), 'Rollings:', len(roll_cols))\n",
    "print('Calendario:', calendar_cols)\n",
    "print('Numéricas totales usadas:', len(numeric_candidates))\n",
    "\n",
    "# Estadísticas rápidas para validación\n",
    "display(df[numeric_candidates].describe().T.head(15))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Partición temporal (train/valid)\n",
    "Para evitar fuga de información temporal (**data leakage**), separamos por **tiempo**. Aquí usamos el percentil 0.85 de `Periodo` como corte aproximado: el ~85% más temprano para **train** y el ~15% más reciente para **valid**.\n",
    "\n",
    "Puedes cambiar la regla de corte a un **fecha fija** (p. ej., `cutoff = pd.Timestamp('2023-01-02')`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================================\n",
    "# 4) Partición temporal train / valid por semana\n",
    "# ==========================================\n",
    "# - Ordenamos por Periodo (lunes), y por claves para consistencia.\n",
    "# - Calculamos cutoff como el percentil 0.85 de las fechas.\n",
    "\n",
    "df = df.sort_values(['Periodo','Provincia','Canton','Delito']).reset_index(drop=True)\n",
    "cutoff = df['Periodo'].quantile(0.85)\n",
    "train = df[df['Periodo'] <= cutoff].copy()\n",
    "valid = df[df['Periodo'] > cutoff].copy()\n",
    "\n",
    "print('Fechas:')\n",
    "print('  Train :', train['Periodo'].min().date(), '→', train['Periodo'].max().date(), f'({train.Periodo.nunique()} semanas)')\n",
    "print('  Valid :', valid['Periodo'].min().date(), '→', valid['Periodo'].max().date(), f'({valid.Periodo.nunique()} semanas)')\n",
    "print('  Series únicas (Provincia|Canton|Delito) - train:', train.groupby(['Provincia','Canton','Delito']).ngroups,\n",
    "      '| valid:', valid.groupby(['Provincia','Canton','Delito']).ngroups)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Ensamblado de features\n",
    "- **Categóricas**: `Provincia`, `Canton`, `Delito` → one-hot.\n",
    "- **Numéricas**: `Poblacion`, `PIB_por_canton`, ENEMDU (`edu_`, `lab_`, `pob_`, `viv_`), lags/rollings, calendario.\n",
    "\n",
    "Antes de entrenar, se rellenan **NaN residuales** (si quedara alguno) con mediana (numéricas) o con la categoría `'DESCONOCIDO'` (categóricas).\n",
    "\n",
    "> Como el pipeline de unificación ya hace imputación, aquí deberían ser mínimos o nulos los faltantes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================\n",
    "# 5) Ensamblado y preprocesamiento\n",
    "# ==============================\n",
    "cat_features = ['Provincia','Canton','Delito']\n",
    "num_features = numeric_candidates  # ya detectadas arriba\n",
    "\n",
    "# Validamos que existan las columnas esenciales en train y valid\n",
    "essential = ['Total_delitos','Tasa_delitos_100k'] + cat_features + num_features + ['Periodo']\n",
    "for subset_name, part in [('train', train), ('valid', valid)]:\n",
    "    missing = [c for c in essential if c not in part.columns]\n",
    "    assert not missing, f\"Faltan columnas en {subset_name}: {missing}\"\n",
    "    # Relleno defensivo (por si quedara algo suelto)\n",
    "    for c in num_features:\n",
    "        if part[c].isna().any():\n",
    "            part[c] = part[c].fillna(part[c].median())\n",
    "    for c in cat_features:\n",
    "        if part[c].isna().any():\n",
    "            part[c] = part[c].fillna('DESCONOCIDO')\n",
    "\n",
    "# Preprocesador: one-hot para categóricas, 'passthrough' para numéricas\n",
    "pre = ColumnTransformer([\n",
    "    ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_features),\n",
    "    ('passthrough', 'passthrough', num_features)\n",
    "])\n",
    "\n",
    "print('Categóricas:', cat_features)\n",
    "print('Numéricas  :', len(num_features))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Entrenamiento y evaluación — función reutilizable\n",
    "Definimos una función para entrenar un **RandomForestRegressor** y reportar **RMSE**, **MAE** y **R²**. Además, guarda las predicciones de validación en `model_outputs/pred_<target>.csv`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================================\n",
    "# 6) Entrenar y evaluar (función genérica)\n",
    "# ==========================================\n",
    "def train_and_eval(target_col: str, train_df: pd.DataFrame, valid_df: pd.DataFrame):\n",
    "    \"\"\"\n",
    "    Entrena RandomForestRegressor para 'target_col' y evalúa en el bloque de validación.\n",
    "    Devuelve el modelo entrenado y un DataFrame con predicciones para valid.\n",
    "    \"\"\"\n",
    "    X_train = train_df[cat_features + num_features]\n",
    "    y_train = train_df[target_col].astype(float)\n",
    "    X_valid = valid_df[cat_features + num_features]\n",
    "    y_valid = valid_df[target_col].astype(float)\n",
    "\n",
    "    model = Pipeline(steps=[\n",
    "        ('pre', pre),\n",
    "        ('rf', RandomForestRegressor(\n",
    "            n_estimators=300,\n",
    "            max_depth=None,\n",
    "            random_state=42,\n",
    "            n_jobs=-1\n",
    "        ))\n",
    "    ])\n",
    "\n",
    "    model.fit(X_train, y_train)\n",
    "    pred = model.predict(X_valid)\n",
    "\n",
    "    rmse = mean_squared_error(y_valid, pred, squared=False)\n",
    "    mae  = mean_absolute_error(y_valid, pred)\n",
    "    r2   = r2_score(y_valid, pred)\n",
    "    print(f'[{target_col}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}')\n",
    "\n",
    "    # Guardar predicciones con claves para trazabilidad\n",
    "    out = valid_df[['Periodo','Provincia','Canton','Delito']].copy()\n",
    "    out[f'y_true_{target_col}'] = y_valid.values\n",
    "    out[f'y_pred_{target_col}'] = pred\n",
    "    out_path = SAVE_DIR / f'pred_{target_col}.csv'\n",
    "    out.to_csv(out_path, index=False, encoding='utf-8')\n",
    "    print('Predicciones ->', out_path.resolve())\n",
    "    return model, out\n",
    "\n",
    "print('Entrenando modelo para Total_delitos...')\n",
    "rf_total, pred_total = train_and_eval('Total_delitos', train, valid)\n",
    "\n",
    "print('Entrenando modelo para Tasa_delitos_100k...')\n",
    "rf_tasa, pred_tasa = train_and_eval('Tasa_delitos_100k', train, valid)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualización — real vs predicción en un ejemplo\n",
    "Trazamos un ejemplo sencillo de serie temporal (`Total_delitos` y `Tasa_delitos_100k`) para verificar tendencia y ajuste visualmente.\n",
    "\n",
    "> Si quieres ver un **cantón/delito específico**, puedes filtrar por `Provincia`, `Canton`, `Delito`. Si no, el gráfico elige una serie con valores altos (representativa)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ====================================\n",
    "# 7) Gráficos simples real vs predicho\n",
    "# ====================================\n",
    "def plot_example(pred_df: pd.DataFrame, target_col: str, prov=None, cant=None, delito=None, top_n=1):\n",
    "    df_plot = pred_df.copy()\n",
    "    if prov:   df_plot = df_plot[df_plot['Provincia'] == prov]\n",
    "    if cant:   df_plot = df_plot[df_plot['Canton'] == cant]\n",
    "    if delito: df_plot = df_plot[df_plot['Delito'] == delito]\n",
    "    if df_plot.empty:\n",
    "        print('No hay datos con ese filtro. Prueba con otros valores.')\n",
    "        return\n",
    "    # Tomar top_n series por valor real máximo para ver casos “representativos”\n",
    "    col_true = f'y_true_{target_col}'\n",
    "    df_plot = df_plot.sort_values(col_true, ascending=False).groupby(['Provincia','Canton','Delito']).head(top_n)\n",
    "    df_plot = df_plot.sort_values('Periodo')\n",
    "\n",
    "    plt.figure(figsize=(10,4))\n",
    "    plt.plot(df_plot['Periodo'], df_plot[col_true], label='Real')\n",
    "    plt.plot(df_plot['Periodo'], df_plot[f'y_pred_{target_col}'], label='Predicción')\n",
    "    plt.title(f'Ejemplo {target_col}')\n",
    "    plt.xlabel('Periodo (lunes)')\n",
    "    plt.ylabel(target_col)\n",
    "    plt.legend()\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Dos gráficos rápidos sin filtros (elige internamente series representativas)\n",
    "plot_example(pred_total, 'Total_delitos', top_n=1)\n",
    "plot_example(pred_tasa, 'Tasa_delitos_100k', top_n=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## (Opcional) Exportar dataset mínimo para TFT\n",
    "Si vas a entrenar un **Temporal Fusion Transformer (TFT)**, normalmente se utiliza un formato *long* con:\n",
    "- `Periodo` como eje temporal (o `time_idx` si conviertes a entero secuencial).\n",
    "- `group_id` que identifique la serie (p. ej., concatenar `Provincia|Canton|Delito`).\n",
    "- `target` elegido (usa uno de los dos: `Total_delitos` o `Tasa_delitos_100k`).\n",
    "- **known_future**: variables conocidas a futuro (calendario, feriados, población proyectada, PIB si dispones de proyecciones, etc.).\n",
    "- **observed_past**: variables que solo observas al pasado (lags y rollings del target, por ejemplo).\n",
    "\n",
    "Abajo exportamos un CSV con columnas mínimas para que puedas conectarlo a tu pipeline de TFT. Edita las listas si quieres cambiar el set de features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================================\n",
    "# 8) Export: datos mínimos listos para TFT\n",
    "# ==========================================\n",
    "# Creamos un id de serie (group_id) combinando Provincia|Canton|Delito\n",
    "df['group_id'] = df['Provincia'].astype(str) + '|' + df['Canton'].astype(str) + '|' + df['Delito'].astype(str)\n",
    "\n",
    "# Columnas base y conjuntos opcionales\n",
    "tft_cols_base = ['Periodo','group_id','Total_delitos','Tasa_delitos_100k']\n",
    "known_future = [c for c in ['mes','trimestre','semana_anio','es_feriado','vacaciones_escolares'] if c in df.columns]\n",
    "observed_past = [c for c in df.columns if re.match(r'(Total_delitos|Tasa_delitos_100k)_(lag\\\\d+|roll\\\\d+_(mean|std))', c)]\n",
    "\n",
    "# Armamos el subset y exportamos\n",
    "tft_df = df[tft_cols_base + known_future + observed_past].copy()\n",
    "tft_path = SAVE_DIR / 'tft_ready.csv'\n",
    "tft_df.to_csv(tft_path, index=False, encoding='utf-8')\n",
    "print('TFT-ready CSV ->', tft_path.resolve())\n",
    "\n",
    "# Nota: Si prefieres usar un solo target para TFT, puedes duplicar la columna elegida como 'target'\n",
    "# p.ej.: tft_df['target'] = tft_df['Total_delitos'] y eliminar 'Tasa_delitos_100k'."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Próximos pasos sugeridos (para tu TFM)\n",
    "1. **Hipótesis y justificación**: documenta por qué elegiste `Total_delitos` vs `Tasa_delitos_100k` como target principal.\n",
    "2. **Selección de features**: delimita qué exógenas resultaron más importantes (usa `feature_importances_` de RF como orientación inicial).\n",
    "3. **Validación temporal adicional**: usa *time series cross-validation* por bloques (por ejemplo, **expanding window**).\n",
    "4. **Modelos avanzados**: prueba Gradient Boosting (XGBoost/LightGBM) y, si apuntas a deep learning, prepara el pipeline para **TFT** (PyTorch Forecasting).\n",
    "5. **Métricas por subgrupos**: evalúa por provincia/cantón/delito para detectar sesgos o zonas donde el modelo rinde menos.\n",
    "6. **Backtesting**: simula predicciones retroactivas con diferentes horizontes (h=1, h=4, h=8 semanas) para medir robustez.\n",
    "\n",
    "Con este notebook tienes una base **reproducible y comentada** para tu TFM. Si quieres, puedo crear una versión **específica para TFT** (con `time_idx`, escalado por grupo, embeddings de categóricas, etc.)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5

}'''

# Guardar archivo .ipynb
with open("tmf_model2_adaptado.ipynb", "w", encoding="utf-8") as f:
    f.write(notebook_json)

print("✅ Archivo 'tmf_model2_adaptado.ipynb' creado correctamente.")
print("Ahora puedes abrirlo con Jupyter Notebook o VS Code.")
