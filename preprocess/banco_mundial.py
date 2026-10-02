import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata
from collections import Counter


csv_path = Path(r"C:\Users\otero\OneDrive\Desktop\DB\inflation-data-largo-v2.csv")
df = pd.read_csv(csv_path, sep=",")
# display(df.head())

# Quitar las columnas que son redundantes
df = df.drop(
    columns=["Frecuencia", "Indicador", "IMF Country Code", "Country", "Nota", "Fuente"]
)

# Datos faltantes
porcent_missing = df.isnull().sum() / len(df) * 100
# print(porcent_missing)

# Quitar las filas donde se toman valores mensuales o trimestrales
valores_a_quitar = [
    "hcpi_m",
    "hcpi_q",
    "ccpi_m",
    "ccpi_q",
    "ppi_m",
    "ppi_q",
    "def_q",
    "ecpi_m",
    "ecpi_q",
    "fcpi_m",
    "fcpi_q",
]
valores_nulos = [np.nan, "", " ", "N/A", "NA", "na", "n/a", "nA", "NaN"]
df = df[~df["Hoja"].isin(valores_a_quitar)]
df = df[~df["Indicator Type"].isnull()]
porcent_missing = df.isnull().sum() / len(df) * 100
# print(porcent_missing)

# Dato atípicos
df_numeric = df.select_dtypes(include=[np.number])
df_numeric = df_numeric.dropna()
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

atipicos_por_columna = (
    (df_numeric < limite_inferior) | (df_numeric > limite_superior)
).sum()
# print(atipicos_por_columna)
#
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df_numeric)
# plt.xticks(rotation=45)
# plt.show()

# Atípicos por serie temporal
claves = ["Hoja", "Country Code", "Indicator Type", "Series Name"]

resultados = []

for _, serie in df.groupby(claves, dropna=False):
    # Excluir faltantes solo durante el cálculo de esta serie
    serie = serie.dropna(subset=["Valor"]).copy()
    serie["Valor"] = pd.to_numeric(serie["Valor"], errors="coerce")
    serie = serie.dropna(subset=["Valor"])

    if len(serie) < 8:
        continue

    q1 = serie["Valor"].quantile(0.25)
    q3 = serie["Valor"].quantile(0.75)
    iqr = q3 - q1

    serie["Limite inferior"] = q1 - 1.5 * iqr
    serie["Limite superior"] = q3 + 1.5 * iqr
    serie["Es atípico"] = (serie["Valor"] < serie["Limite inferior"]) | (
        serie["Valor"] > serie["Limite superior"]
    )

    resultados.append(serie)

analisis = pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()
atipicos = analisis[analisis["Es atípico"]].copy()

# print(f"Series evaluadas: {analisis.groupby(claves, dropna=False).ngroups:,}")
# print(f"Posibles atípicos: {len(atipicos):,}")

# Datos duplicados
df_duplicates = df[df.duplicated(subset=["Valor"], keep=False)]
# if len(df_duplicates) > 0:
#    print("\nMuestras redundantes o inconsistentes:")
#    display(df_duplicates)
# else:
#    print("No existen valores duplicados")

# Caácteres especiales
# Signos habituales que no se marcarán como inusuales.
permitidos = set(" -_.,;:/()[]{}'\"%&+@")

conteos = Counter()
ejemplos = []

# for columna in df.select_dtypes(include=["object", "string"]).columns:
#    for indice, valor in df[columna].dropna().items():
#        texto = str(valor)
#
#        for caracter in texto:
#            categoria = unicodedata.category(caracter)
#
#            # Cuenta signos/símbolos que no están en la lista permitida.
#            # Letras con tilde, ñ y otros alfabetos se consideran alfanuméricos.
#            if (
#                not caracter.isalnum()
#                and not caracter.isspace()
#                and caracter not in permitidos
#            ):
#                conteos[(columna, caracter)] += 1
#                if len(ejemplos) < 100:
#                    ejemplos.append(
#                        {
#                            "fila": indice,
#                            "columna": columna,
#                            "valor": texto,
#                            "caracter": caracter,
#                            "categoria_unicode": categoria,
#                        }
#                    )
#
# print("Caracteres especiales encontrados:")
# for (columna, caracter), cantidad in conteos.most_common():
#    #print(f"{columna}: {caracter!r} — {cantidad:,} apariciones")
