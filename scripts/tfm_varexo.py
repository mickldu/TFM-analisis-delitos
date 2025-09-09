import pandas as pd
import os
import unidecode
from functools import reduce

def normalizar_col(nombre):
    return unidecode.unidecode(nombre.strip().lower().replace('\n', ' '))

def transformar_enemdu(ruta, prefijo):
    if not os.path.exists(ruta):
        print(f"Archivo no encontrado: {ruta}")
        return None

    df = pd.read_excel(ruta)
    df.columns = [normalizar_col(col) for col in df.columns]
    col_periodo = [c for c in df.columns if 'period' in c][0]
    col_indicador = [c for c in df.columns if 'indicador' in c][0]
    provincias = [c for c in df.columns if c not in [col_periodo, col_indicador]]
    df_long = df.melt(id_vars=[col_periodo, col_indicador], value_vars=provincias,
                      var_name='Provincia', value_name='Valor')
    df_long = df_long.rename(columns={col_periodo: 'Periodo', col_indicador: 'Indicador'})
    df_long['Indicador'] = prefijo + "_" + df_long['Indicador']
    df_pivot = df_long.pivot_table(index=['Periodo', 'Provincia'], columns='Indicador', values='Valor').reset_index()

    # Interpolación semanal
    df_pivot['Fecha'] = pd.to_datetime(df_pivot['Periodo'].astype(str) + "-01-01")
    provincias = df_pivot['Provincia'].unique()
    dfs_semana = []
    for prov in provincias:
        sub = df_pivot[df_pivot['Provincia'] == prov].set_index('Fecha').sort_index()
        fechas = pd.date_range(start=sub.index.min(), end=sub.index.max(), freq='W')
        sub = sub.reindex(fechas)
        sub.interpolate(method='linear', inplace=True, limit_direction='both')
        sub['Provincia'] = prov.upper()   # <--- AQUÍ SE AJUSTA A MAYÚSCULAS
        sub = sub.reset_index().rename(columns={'index': 'Fecha'})
        sub['Periodo'] = sub['Fecha'].dt.year
        sub['Mes'] = sub['Fecha'].dt.month
        sub['Semana'] = sub['Fecha'].dt.isocalendar().week
        dfs_semana.append(sub)
    df_semana = pd.concat(dfs_semana, ignore_index=True)
    # Convierte a mayúsculas por si acaso quedó alguna en minúscula
    df_semana['Provincia'] = df_semana['Provincia'].str.upper()
    out_cols = ['Periodo', 'Mes', 'Semana', 'Provincia'] + [c for c in df_semana.columns if c.startswith(prefijo)]
    df_semana = df_semana[out_cols]
    return df_semana

archivos = [
    {'archivo': 'ENEMDU_EDUCACION_2018_2024.xlsx', 'prefijo': 'edu'},
    {'archivo': 'ENEMDU_LABORAL_2018_2024.xlsx', 'prefijo': 'lab'},
    {'archivo': 'ENEMDU_POBREZA_2018_2024.xlsx', 'prefijo': 'pob'},
    {'archivo': 'ENEMDU_VIVIENDA_2018_2024.xlsx', 'prefijo': 'viv'}
]

dfs = []
for a in archivos:
    print(f"Procesando: {a['archivo']}")
    df_arch = transformar_enemdu(a["archivo"], a["prefijo"])
    if df_arch is not None:
        dfs.append(df_arch)

if not dfs:
    raise ValueError("No se pudo cargar ningún archivo. Verifica nombres y ubicación.")

print("Uniendo todos los archivos por Periodo, Mes, Semana y Provincia...")
df_final = reduce(lambda left, right: pd.merge(left, right, on=["Periodo", "Mes", "Semana", "Provincia"], how="outer"), dfs)
df_final = df_final.sort_values(["Periodo", "Mes", "Semana", "Provincia"])
df_final.to_csv("ENEMDU_UNIFICADO_SEMANAL.csv", index=False)
print("✅ Archivo final generado: ENEMDU_UNIFICADO_SEMANAL.csv")
