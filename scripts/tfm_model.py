import pandas as pd

# Cargar el archivo (ajusta la ruta si es necesario)
df = pd.read_csv("data_tfm.csv")

# ------ 1. Valores únicos de 'Canton' en tabla -------
valores_canton = df['Canton'].unique()
df_cantones = pd.DataFrame(valores_canton, columns=['Cantón'])
display(df_cantones)

# ------ 2. Frecuencia (cuenta) de registros por cantón -------
cantones_freq = df['Canton'].value_counts().reset_index()
cantones_freq.columns = ['Cantón', 'Frecuencia']
display(cantones_freq)

# ------ 3. Lo mismo para cualquier columna categórica (ejemplo: 'Provincia') -------
valores_provincia = df['Provincia'].unique()
df_provincias = pd.DataFrame(valores_provincia, columns=['Provincia'])
display(df_provincias)

provincias_freq = df['Provincia'].value_counts().reset_index()
provincias_freq.columns = ['Provincia', 'Frecuencia']
display(provincias_freq)

# ------ 4. Puedes hacerlo para otras columnas relevantes (ejemplo: 'd_delito') -------
if 'd_delito' in df.columns:
    valores_delito = df['d_delito'].unique()
    df_delitos = pd.DataFrame(valores_delito, columns=['Delito'])
    display(df_delitos)

    delitos_freq = df['d_delito'].value_counts().reset_index()
    delitos_freq.columns = ['Delito', 'Frecuencia']
    display(delitos_freq)
