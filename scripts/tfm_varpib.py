import pandas as pd
import numpy as np

# Cargar el archivo Excel
pib_data = pd.read_excel('./PBI_1965_2023.xlsx')

# Verificar si la columna 'Año' contiene valores no numéricos
pib_data['Año'] = pd.to_numeric(pib_data['Año'], errors='coerce')

# Eliminar filas donde 'Año' sea NaN (en caso de valores no válidos)
pib_data = pib_data.dropna(subset=['Año'])

# Crear un rango de fechas semanal basado en los años
years = pib_data['Año']
start_date = f'{int(years.min())}-01-01'  # Convertir a entero para asegurarnos de que es un año válido
end_date = f'{int(years.max())}-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='W')

# Interpolación de los datos anuales a valores semanales
pib_weekly = pd.DataFrame(date_range, columns=['Fecha'])

# Interpolar los valores de PIB para cada columna
for column in pib_data.columns[1:]:  # Excluir la columna 'Año'
    pib_weekly[column] = np.interp(
        np.linspace(years.min(), years.max(), len(date_range)),  # Rango de años
        years,  # Años existentes
        pib_data[column]  # Valores anuales del PIB
    )

# Mostrar los primeros resultados del dataframe con PIB semanal
print(pib_weekly.head())

# Guardar el archivo con los datos semanales
pib_weekly.to_csv('PIB_semanal.csv', index=False)
