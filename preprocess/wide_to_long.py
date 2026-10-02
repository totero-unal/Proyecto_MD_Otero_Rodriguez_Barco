"""Convierte las hojas de Inflation-data (1).xlsx a tablas largas limpias.

Genera dos archivos para no mezclar esquemas distintos:
  * Inflation-data-largo-v2.csv: series por país y frecuencia.
  * Inflation-data-aggregate-largo.csv: agregados globales de la hoja Aggregate.

Las hojas Intro y top son informativas y no se convierten. Las columnas vacías,
sin nombre (Unnamed) y las columnas que no son metadatos ni períodos se omiten.
"""

import re
from pathlib import Path

import pandas as pd


ARCHIVO = Path(r"C:\Users\otero\OneDrive\Desktop\DB\Inflation-data (1).xlsx")
CARPETA_SALIDA = ARCHIVO.parent
SALIDA_SERIES = CARPETA_SALIDA / "Inflation-data-largo-v2.csv"
SALIDA_AGREGADOS = CARPETA_SALIDA / "Inflation-data-aggregate-largo.csv"

HOJAS_INFORMATIVAS = {"Intro", "top"}

# Solo se retienen estos campos como identificadores/metadatos. Los nombres
# se buscan por encabezado, así que su posición puede variar entre hojas.
METADATOS_SERIES = {
    "Country Code": "Country Code",
    "IMF Country Code": "IMF Country Code",
    "Country": "Country",
    "Indicator Type": "Indicator Type",
    "Series Name": "Series Name",
}
METADATOS_AGREGADOS = {
    "Inflation measures": "Inflation measures",
    "Series Name": "Series Name",
    "Description": "Description",
    "Aggregation": "Aggregation",
    "Sample country": "Sample country",
}


def nombre_encabezado(valor):
    """Normaliza encabezados de texto y descarta los que pandas llama Unnamed."""
    if valor is None or pd.isna(valor):
        return ""
    nombre = str(valor).strip()
    if not nombre or nombre.casefold().startswith("unnamed:"):
        return ""
    return nombre


def codigo_periodo(valor):
    """Devuelve el código entero del período, o None si no parece un período."""
    if valor is None or pd.isna(valor) or isinstance(valor, bool):
        return None
    texto = str(valor).strip()
    # Excel/pandas puede representar un encabezado entero como 1970.0.
    if re.fullmatch(r"\d+\.0", texto):
        texto = texto[:-2]
    if not texto.isdigit():
        return None

    if len(texto) == 4 and 1900 <= int(texto) <= 2100:
        return texto
    if len(texto) == 5:
        anio, trimestre = int(texto[:4]), int(texto[4])
        if 1900 <= anio <= 2100 and 1 <= trimestre <= 4:
            return texto
    if len(texto) == 6:
        anio, mes = int(texto[:4]), int(texto[4:])
        if 1900 <= anio <= 2100 and 1 <= mes <= 12:
            return texto
    return None


def etiqueta_periodo(codigo):
    """Convierte AAAAMM/AAAAQ/AAAA en etiquetas cronológicas legibles."""
    if len(codigo) == 4:
        return codigo
    if len(codigo) == 5:
        return f"{codigo[:4]}-T{codigo[4]}"
    return f"{codigo[:4]}-{codigo[4:]}"


def frecuencia_hoja(nombre):
    sufijo = nombre.rsplit("_", 1)[-1].lower()
    return {"m": "mensual", "q": "trimestral", "a": "anual"}.get(sufijo, "")


def combinar_columnas_metadatos(datos, mapa):
    """Selecciona campos conocidos; combina encabezados repetidos si existen."""
    salida = pd.DataFrame(index=datos.index)
    nombres = [nombre_encabezado(c) for c in datos.columns]
    for encabezado, destino in mapa.items():
        posiciones = [i for i, nombre in enumerate(nombres) if nombre == encabezado]
        if not posiciones:
            continue
        valores = datos.iloc[:, posiciones].copy()
        # Si el libro repite el mismo encabezado, usa el primer valor no vacío.
        if len(posiciones) > 1:
            valores = valores.replace(r"^\s*$", pd.NA, regex=True)
            serie = valores.bfill(axis=1).iloc[:, 0]
        else:
            serie = valores.iloc[:, 0]
        salida[destino] = serie
    return salida


def convertir_hoja(nombre, datos, agregados=False):
    nombres = [nombre_encabezado(c) for c in datos.columns]
    columnas_periodo = {}
    for posicion, encabezado in enumerate(nombres):
        codigo = codigo_periodo(encabezado)
        if codigo is not None:
            columnas_periodo[posicion] = codigo

    if not columnas_periodo:
        print(f"Se omite '{nombre}': no se detectaron encabezados de período.")
        return None

    mapa = METADATOS_AGREGADOS if agregados else METADATOS_SERIES
    ids = combinar_columnas_metadatos(datos, mapa)

    # Fuente y notas pueden llamarse distinto según la hoja; se normalizan.
    if not agregados:
        fuentes = [i for i, c in enumerate(nombres) if c in {"Data source", "National sources"}]
        notas = [i for i, c in enumerate(nombres) if c.casefold() == "note"]
        if fuentes:
            vals = datos.iloc[:, fuentes].replace(r"^\s*$", pd.NA, regex=True)
            ids["Fuente"] = vals.bfill(axis=1).iloc[:, 0]
        if notas:
            vals = datos.iloc[:, notas].replace(r"^\s*$", pd.NA, regex=True)
            ids["Nota"] = vals.bfill(axis=1).iloc[:, 0]

    posiciones = list(columnas_periodo)
    periodo_a_posiciones = {}
    for pos, codigo in columnas_periodo.items():
        periodo_a_posiciones.setdefault(etiqueta_periodo(codigo), []).append(pos)

    # Evita perder valores si por error hay dos encabezados que representan
    # el mismo período: los deja identificados para que se detecten después.
    periodos = []
    bloques = []
    for periodo, posiciones_periodo in periodo_a_posiciones.items():
        for repeticion, pos in enumerate(posiciones_periodo, start=1):
            etiqueta = periodo if len(posiciones_periodo) == 1 else f"{periodo} [columna {repeticion}]"
            periodos.append(etiqueta)
            bloques.append(datos.iloc[:, pos])

    valores = pd.concat(bloques, axis=1)
    valores.columns = periodos
    largo = pd.concat([ids, valores], axis=1).melt(
        id_vars=list(ids.columns), var_name="Periodo", value_name="Valor"
    )
    largo.insert(0, "Hoja", nombre)

    if agregados:
        largo.insert(1, "Frecuencia", "anual")
    else:
        largo.insert(1, "Frecuencia", frecuencia_hoja(nombre))
        largo.insert(2, "Indicador", nombre.rsplit("_", 1)[0].upper())

    columnas_clave = list(ids.columns)
    largo = largo.dropna(subset=columnas_clave, how="all")
    return largo


def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {ARCHIVO}")

    libro = pd.ExcelFile(ARCHIVO, engine="openpyxl")
    series_largas = []
    agregados_largos = []

    for nombre in libro.sheet_names:
        if nombre in HOJAS_INFORMATIVAS:
            continue
        datos = pd.read_excel(libro, sheet_name=nombre, engine="openpyxl")
        if datos.empty:
            continue

        es_agregado = nombre == "Aggregate"
        largo = convertir_hoja(nombre, datos, agregados=es_agregado)
        if largo is None:
            continue
        if es_agregado:
            agregados_largos.append(largo)
        else:
            series_largas.append(largo)
        print(f"{nombre}: {len(largo):,} observaciones largas")
        del datos, largo

    if not series_largas:
        raise ValueError("No se encontraron hojas de series con períodos reconocibles.")

    base = pd.concat(series_largas, ignore_index=True, sort=False)
    base.to_csv(SALIDA_SERIES, index=False, encoding="utf-8-sig")
    print(f"\nSeries guardadas en: {SALIDA_SERIES}")
    print(f"Filas: {len(base):,}; columnas: {list(base.columns)}")

    if agregados_largos:
        base_agregados = pd.concat(agregados_largos, ignore_index=True, sort=False)
        base_agregados.to_csv(SALIDA_AGREGADOS, index=False, encoding="utf-8-sig")
        print(f"Agregados guardados en: {SALIDA_AGREGADOS}")
        print(f"Filas: {len(base_agregados):,}; columnas: {list(base_agregados.columns)}")


if __name__ == "__main__":
    main()
