import pandas as pd

# Cargar archivo
df_pob = pd.read_excel("POBLACION_PROYECTADA.xlsx")

# Normalizar encabezados
df_pob.columns = df_pob.columns.str.strip().str.upper()

# Identificar columnas clave para provincia y cantón
col_cod_prov = "COD_PROVINCIA"
col_nom_prov = "PROVINCIA"
col_cod_cant = "COD_CANTON"
col_nom_cant = "CANTON"
cols_anios = [col for col in df_pob.columns if col.isdigit()]

# Transformar a formato largo
df_largo = df_pob[[col_cod_prov, col_nom_prov, col_cod_cant, col_nom_cant] + cols_anios].melt(
    id_vars=[col_cod_prov, col_nom_prov, col_cod_cant, col_nom_cant],
    var_name="AÑO",
    value_name="POBLACION"
)

# Convertir año a entero
df_largo["AÑO"] = df_largo["AÑO"].astype(int)

# Crear fechas semanales (lunes de cada semana)
fechas = pd.date_range(
    start=f"{df_largo['AÑO'].min()}-01-01",
    end=f"{df_largo['AÑO'].max()}-12-31",
    freq="W-MON"
)
df_semanas = pd.DataFrame({"Semana": fechas})
df_semanas["AÑO"] = df_semanas["Semana"].dt.year

# Repetir población para cada semana del año correspondiente
df_final = pd.merge(df_semanas, df_largo, on="AÑO", how="left")

# Ajustar formatos de código y nombres
df_final[col_cod_prov] = df_final[col_cod_prov].astype(str).str.zfill(2)
df_final[col_cod_cant] = df_final[col_cod_cant].astype(str).str.zfill(4)
df_final[col_nom_prov] = df_final[col_nom_prov].astype(str).str.strip().str.upper()
df_final[col_nom_cant] = df_final[col_nom_cant].astype(str).str.strip().str.upper()

# Renombrar y seleccionar columnas finales
df_final = df_final.rename(columns={
    col_cod_prov: "Codigo_Provincia",
    col_nom_prov: "Provincia",
    col_cod_cant: "Codigo_Canton",
    col_nom_cant: "Canton"
})

# Seleccionar columnas finales
df_final = df_final[["Semana", "Codigo_Provincia", "Provincia", "Codigo_Canton", "Canton", "POBLACION"]]

# Guardar archivo final
df_final.to_csv("poblacion_proyectada_semanal.csv", index=False)
print("✅ Archivo generado: poblacion_proyectada_semanal.csv")
