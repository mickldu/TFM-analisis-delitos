import pandas as pd
from datetime import datetime, timedelta

# Función para calcular la fecha de inicio de la semana (lunes)
def get_start_date(year, month, week):
    # El primer día del mes
    first_day_of_month = datetime(year, month, 1)
    
    # Calcular el primer lunes del mes
    first_monday = first_day_of_month + timedelta(days=(7 - first_day_of_month.weekday()) % 7)
    
    # Calcular el inicio de la semana
    start_of_week = first_monday + timedelta(weeks=week - 1)
    
    return start_of_week.strftime('%d/%m/%Y')

# Función para ajustar al lunes de la semana
def adjust_to_monday(date_str):
    # Convertir el string a datetime
    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
    
    # Calcular la diferencia con el lunes anterior
    days_diff = date_obj.weekday()
    if days_diff != 0:  # Si no es lunes
        date_obj -= timedelta(days=days_diff)
    
    # Retornar la fecha ajustada en el formato deseado
    return date_obj.strftime('%d/%m/%Y')

# Cargar el archivo CSV
file_path_new = 'ENEMDU_UNIFICADO_SEMANAL.csv'  # Asegúrate de usar la ruta correcta en tu entorno
df_new = pd.read_csv(file_path_new)

# Crear la columna 'PeriodoEMU' basada en Año, Mes y Semana
df_new['PeriodoEMU'] = df_new.apply(lambda row: get_start_date(row['Periodo'], row['Mes'], row['Semana']), axis=1)

# Ajustar la columna 'Fecha_Lunes' al lunes de la semana
df_new['PeriodoEMU'] = df_new['PeriodoEMU'].apply(adjust_to_monday)

# Renombrar las columnas de manera coherente y concisa, manteniendo los prefijos edu_, lab_, pob_ y viv_
df_new_cleaned_renamed = df_new.rename(columns={ 
    'Periodo': 'Periodo',
    'Provincia': 'Provincia',
    'edu_Años promedio de escolaridad': 'edu_Años_Educacion',
    'edu_Tasa bruta de asistencia a  Educación General Básica': 'edu_Asistencia_EGB',
    'edu_Tasa bruta de asistencia a bachillerato': 'edu_Asistencia_Bachillerato',
    'edu_Tasa bruta de asistencia a primaria': 'edu_Asistencia_Primaria',
    'edu_Tasa bruta de asistencia a secundaria': 'edu_Asistencia_Secundaria',
    'edu_Tasa de analfabetismo': 'edu_Analfabetismo',
    'edu_Tasa neta de asistencia a  primaria': 'edu_Asistencia_Primaria_Neta',
    'edu_Tasa neta de asistencia a bachillerato': 'edu_Asistencia_Bachillerato_Neta',
    'edu_Tasa neta de asistencia a secundaria': 'edu_Asistencia_Secundaria_Neta',
    'edu_Tasa neta de asistencias Educación General Básica': 'edu_Asistencia_EGB_Neta',
    'lab_Sector informal': 'lab_Sector_Informal',
    'lab_Tasa de desempleo': 'lab_Tasa_Desempleo',
    'lab_Tasa de empleo adecuado': 'lab_Tasa_Empleo_Adecuado',
    'lab_Tasa de empleo bruto': 'lab_Tasa_Empleo_Bruto',
    'lab_Tasa de empleo global': 'lab_Tasa_Empleo_Global',
    'lab_Tasa de empleo no remunerado': 'lab_Tasa_Empleo_No_Remunerado',
    'lab_Tasa de otro empleo no pleno': 'lab_Tasa_Otro_Empleo_No_Pleno',
    'lab_Tasa de participación bruta': 'lab_Tasa_Participacion_Bruta',
    'lab_Tasa de participación global': 'lab_Tasa_Participacion_Global',
    'lab_Tasa de subempleo': 'lab_Tasa_Subempleo',
    'pob_Coeficiente de desigualdad de Gini': 'pob_Desigualdad_Gini',
    'pob_Pobreza extrema por ingresos': 'pob_Pobreza_Extrema_Ingresos',
    'pob_Pobreza por Necesidades Básicas Insatisfechas (NBI)': 'pob_Pobreza_NBI',
    'pob_Pobreza por ingresos': 'pob_Pobreza_Ingresos',
    'pob_Tasa de pobreza multidimensional': 'pob_Pobreza_Multidimensional',
    'viv_Déficit habitacional cualitativo': 'viv_Deficit_Habitacional_Cualitativo',
    'viv_Déficit habitacional cuantitativo': 'viv_Deficit_Habitacional_Cuantitativo',
    'viv_Porcentaje de hogares con acceso a electricidad': 'viv_Acceso_Electricidad',
    'viv_Porcentaje de hogares con acceso a red publica de agua': 'viv_Acceso_Agua_Red_Publica',
    'viv_Porcentaje de hogares con acceso a servicios básicos': 'viv_Acceso_Servicios_Basicos',
    'viv_Porcentaje de hogares con agua de red publica (SENAGUA)': 'viv_Acceso_Agua_SENAGUA',
    'viv_Porcentaje de hogares con recolección adecuada de desechos sólidos': 'viv_Acceso_Recogida_Desechos',
    'viv_Porcentaje de hogares que cuentan alumbrado público': 'viv_Alumbrado_Publico',
    'viv_Porcentaje de hogares que cuentan con un sistema adecuado de eliminación de excretas': 'viv_Eliminacion_Excretas',
    'viv_Porcentaje de hogares que cuentan con un sistema adecuado de eliminación de excretas (SENAGUA)': 'viv_Eliminacion_Excretas_SENAGUA',
    'viv_Porcentaje de hogares que viven en hacinamiento': 'viv_Hacinamiento',
    'Fecha_Lunes': 'Fecha_Lunes'
})

# Guardar el archivo final con todas las columnas, la nueva columna 'Fecha_Lunes' y las columnas renombradas
output_path_final_with_date = 'enemu_semanal.csv'  # Ruta del archivo en tu entorno local
df_new_cleaned_renamed.to_csv(output_path_final_with_date, index=False)

output_path_final_with_date  # Ruta del archivo guardado
