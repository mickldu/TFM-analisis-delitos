import pandas as pd
import numpy as np

# Cargar los archivos CSV
delitos_poblacion_semanal_df = pd.read_csv('delitos_poblacion_semanal.csv')
pib_semanal_df = pd.read_csv('PIB_semanal.csv')

# Limpiar los nombres de las columnas para eliminar espacios y caracteres especiales
delitos_poblacion_semanal_df.columns = delitos_poblacion_semanal_df.columns.str.strip()
pib_semanal_df.columns = pib_semanal_df.columns.str.strip()

# Acortar los nombres de las columnas de PIB (eliminar espacios y hacer más concisos)
pib_semanal_df.columns = pib_semanal_df.columns.str.replace(' ', '_')  # Reemplaza los espacios por guiones bajos

# Convertir 'Periodo' a formato datetime para ambos datasets
delitos_poblacion_semanal_df['Periodo'] = pd.to_datetime(delitos_poblacion_semanal_df['Periodo'], errors='coerce')
pib_semanal_df['Fecha'] = pd.to_datetime(pib_semanal_df['Fecha'], errors='coerce')  # Usar 'Fecha' en lugar de 'Periodo'

# Asegurarnos que las fechas de PIB sean de tipo datetime
delitos_poblacion_semanal_df['Año'] = delitos_poblacion_semanal_df['Periodo'].dt.year
delitos_poblacion_semanal_df['Semana'] = delitos_poblacion_semanal_df['Periodo'].dt.isocalendar().week

# Crear columnas de Año y Semana en PIB para asegurarnos de que estén alineados
pib_semanal_df['Año'] = pib_semanal_df['Fecha'].dt.year
pib_semanal_df['Semana'] = pib_semanal_df['Fecha'].dt.isocalendar().week

# Filtrar ambos datasets para incluir solo los años coincidentes
comunes_anos = delitos_poblacion_semanal_df['Año'].unique()
pib_semanal_df = pib_semanal_df[pib_semanal_df['Año'].isin(comunes_anos)]
delitos_poblacion_semanal_df = delitos_poblacion_semanal_df[delitos_poblacion_semanal_df['Año'].isin(comunes_anos)]

# Rellenar valores nulos numéricos usando interpolación lineal
delitos_poblacion_semanal_df[delitos_poblacion_semanal_df.select_dtypes(include=['float64', 'int64']).columns] = delitos_poblacion_semanal_df.select_dtypes(include=['float64', 'int64']).interpolate(method='linear', axis=0)
pib_semanal_df.fillna(pib_semanal_df.median(), inplace=True)

# Rellenar valores nulos categóricos (Provincias y Cantones) usando forward fill
delitos_poblacion_semanal_df['Provincia'] = delitos_poblacion_semanal_df['Provincia'].fillna(method='ffill')
delitos_poblacion_semanal_df['Canton'] = delitos_poblacion_semanal_df['Canton'].fillna(method='ffill')

# Dividir el PIB nacional uniformemente entre los cantones:
num_cantones = len(delitos_poblacion_semanal_df['Canton'].unique())  # Número de cantones en delitos_poblacion_semanal
pib_semanal_df['PIB_por_canton'] = pib_semanal_df['TOTAL_PIB'] / num_cantones

# Eliminar columnas de PIB que están completamente vacías
pib_semanal_df = pib_semanal_df.dropna(axis=1, how='all')

# Proyectar PIB para los años faltantes (2023, 2024, 2025)
# Calcular la tasa de crecimiento anual del PIB (si hay datos históricos)
pib_semanal_df = pib_semanal_df.sort_values(by='Año')  # Asegurarnos de que el PIB esté ordenado por año
pib_semanal_df['Tasa_crecimiento'] = pib_semanal_df['TOTAL_PIB'].pct_change()  # Cambio porcentual año a año

# Calcular la tasa promedio de crecimiento
tasa_crecimiento_promedio = pib_semanal_df['Tasa_crecimiento'].mean()

# Proyectar PIB para los años faltantes: 2023, 2024, 2025
años_faltantes = [2023, 2024, 2025]
proyecciones = []

for año in años_faltantes:
    # Proyectar el PIB usando el PIB más reciente y la tasa de crecimiento promedio
    pib_reciente = pib_semanal_df[pib_semanal_df['Año'] == pib_semanal_df['Año'].max()]['TOTAL_PIB'].values[0]
    pib_proyectado = pib_reciente * (1 + tasa_crecimiento_promedio) ** (año - pib_semanal_df['Año'].max())
    proyecciones.append({'Año': año, 'TOTAL_PIB': pib_proyectado, 'PIB_por_canton': pib_proyectado / num_cantones})

# Convertir las proyecciones en un DataFrame y concatenarlas al DataFrame original
if proyecciones:  # Solo agregar si hay proyecciones
    proyecciones_df = pd.DataFrame(proyecciones)
    pib_semanal_df = pd.concat([pib_semanal_df, proyecciones_df], ignore_index=True)

# Realizar la fusión final
final_merged_df = pd.merge(delitos_poblacion_semanal_df, pib_semanal_df, on=['Año', 'Semana'], how='left')

# Verificar el resultado
print(f"Primeros registros del archivo unificado: \n{final_merged_df.head()}")

# Guardar el archivo unificado en un nuevo CSV
final_merged_df.to_csv('del_pob_pib.csv', index=False)

# Mostrar las primeras filas para revisión
final_merged_df.head()
