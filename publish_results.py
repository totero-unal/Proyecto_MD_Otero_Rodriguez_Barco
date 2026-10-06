"""Publica únicamente resultados agregados, tablas resumen y gráficos.

Los CSV con filas individuales se conservan en reports/ y no se copian a
resultados/. El README principal muestra las figuras y estadísticas agregadas.
"""

from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
FIGURES_DIR = RESULTS / "graficos"
START = "<!-- AUTO-RESULTS-START -->"
END = "<!-- AUTO-RESULTS-END -->"

TABLES = {
    "EAC": [
        ("analisis_EAC_descriptivo", "faltantes.csv"),
        ("analisis_EAC_descriptivo", "descriptivos_variables_numericas.csv"),
        ("analisis_EAC_descriptivo", "resumen_atipicos_por_variable.csv"),
        ("analisis_EAC_descriptivo", "resumen_atipicos_variable_dominio.csv"),
        ("analisis_EAC_descriptivo", "reglas_calidad_sugeridas.csv"),
    ],
    "Inflación y meta": [
        ("analisis_inflacion_meta", "descriptivos.csv"),
        ("analisis_inflacion_meta", "faltantes.csv"),
        ("analisis_inflacion_meta", "cobertura_y_calendario.csv"),
        ("analisis_inflacion_meta", "resumen_anual.csv"),
        ("analisis_inflacion_meta", "resumen_iqr_por_decada.csv"),
        ("analisis_inflacion_meta", "reglas_calidad_sugeridas.csv"),
    ],
    "Inflación global": [
        ("analisis_inflacion_global", "descriptivos_por_indicador_tipo.csv"),
        ("analisis_inflacion_global", "faltantes_por_indicador_tipo.csv"),
        ("analisis_inflacion_global", "cobertura_por_anio.csv"),
        ("analisis_inflacion_global", "resumen_atipicos_por_indicador.csv"),
        ("analisis_inflacion_global", "reglas_calidad_sugeridas.csv"),
    ],
}

FIGURE_FILES = {
    "EAC": ["distribuciones_montos_log.png", "distribucion_personal.png", "frecuencia_idoj1.png"],
    "Inflación y meta": ["serie_inflacion_meta.png", "distribucion_inflacion.png", "comparacion_anual_meta.png"],
    "Inflación global": ["mediana_anual_por_indicador.png", "cobertura_paises_por_anio.png", "distribuciones_signed_log.png"],
}


def copy_public_tables():
    for dataset, names in TABLES.items():
        destination = RESULTS / dataset.lower().replace(" ", "_").replace("ó", "o")
        destination.mkdir(parents=True, exist_ok=True)
        for folder, filename in names:
            source = ROOT / "reports" / folder / filename
            if source.exists():
                shutil.copy2(source, destination / filename)


def copy_public_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, names in FIGURE_FILES.items():
        folder = {
            "EAC": "analisis_EAC_descriptivo",
            "Inflación y meta": "analisis_inflacion_meta",
            "Inflación global": "analisis_inflacion_global",
        }[dataset]
        for filename in names:
            source = ROOT / "reports" / folder / filename
            if source.exists():
                prefix = {"EAC": "eac", "Inflación y meta": "inflacion_meta", "Inflación global": "inflacion_global"}[dataset]
                shutil.copy2(source, FIGURES_DIR / f"{prefix}_{filename}")


def headline_stats():
    eac_stats = pd.read_csv(ROOT / "reports/analisis_EAC_descriptivo/descriptivos_variables_numericas.csv")
    eac_missing = pd.read_csv(ROOT / "reports/analisis_EAC_descriptivo/faltantes.csv")
    eac_outliers = pd.read_csv(ROOT / "reports/analisis_EAC_descriptivo/resumen_atipicos_por_variable.csv")
    meta_stats = pd.read_csv(ROOT / "reports/analisis_inflacion_meta/descriptivos.csv")
    meta_missing = pd.read_csv(ROOT / "reports/analisis_inflacion_meta/faltantes.csv")
    global_stats = pd.read_csv(ROOT / "reports/analisis_inflacion_global/descriptivos_por_indicador_tipo.csv")

    eac_rows = len(pd.read_csv(ROOT / "data/raw/EAC_CIFRAS_2024_ANONIMIZADA_FINAL.csv", sep=";", decimal=",", encoding="utf-8-sig", low_memory=False))
    eac_nmiss = int(eac_missing["faltantes"].sum())
    eac_iqr = int(eac_outliers["observaciones_atipicas"].sum())
    meta_n = int(meta_stats.loc[meta_stats["variable"].eq("Inflación total anual"), "n"].iloc[0])
    meta_missing_count = int(meta_missing.loc[meta_missing["variable"].eq("Meta de inflación"), "faltantes"].iloc[0])
    meta_missing_pct = float(meta_missing.loc[meta_missing["variable"].eq("Meta de inflación"), "porcentaje"].iloc[0])
    global_missing = int(global_stats["faltantes_valor"].sum())
    global_rows = int(global_stats["n"].sum() + global_missing)
    global_pct = 100 * global_missing / global_rows if global_rows else 0
    global_outliers = int(pd.read_csv(ROOT / "reports/analisis_inflacion_global/resumen_atipicos_por_indicador.csv")["eventos_atipicos"].sum())

    return [
        f"- **EAC 2024:** {eac_rows:,} registros; {eac_nmiss:,} faltantes reportados; {eac_iqr:,} eventos candidatos IQR agregados por variable.",
        f"- **Inflación y meta:** {meta_n:,} valores observados de inflación; meta faltante en {meta_missing_count:,} observaciones ({meta_missing_pct:.2f}%).",
        f"- **Inflación global:** {global_rows:,} observaciones anuales posibles entre grupos; {global_missing:,} valores faltantes ({global_pct:.2f}%); {global_outliers:,} candidatos IQR por serie temporal.",
    ]


def render_markdown(image_prefix="resultados/graficos", table_prefix="resultados/"):
    lines = [
        f"_Última ejecución publicada (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}_",
        "",
        "Las cifras siguientes son resúmenes de la ejecución más reciente. Los candidatos IQR son señales para revisión, no errores confirmados.",
        "",
        *headline_stats(),
        "",
        "Las tablas agregadas descargables están en las subcarpetas de `resultados/`. Los reportes de filas individuales se generan localmente durante el flujo y no se publican.",
    ]
    images = {
        "EAC": ["eac_distribuciones_montos_log.png", "eac_distribucion_personal.png", "eac_frecuencia_idoj1.png"],
        "Inflación y meta": ["inflacion_meta_serie_inflacion_meta.png", "inflacion_meta_distribucion_inflacion.png", "inflacion_meta_comparacion_anual_meta.png"],
        "Inflación global": ["inflacion_global_mediana_anual_por_indicador.png", "inflacion_global_cobertura_paises_por_anio.png", "inflacion_global_distribuciones_signed_log.png"],
    }
    for dataset, names in images.items():
        lines.extend(["", f"### {dataset}", ""])
        for image_name in names:
            if (FIGURES_DIR / image_name).exists():
                lines.extend([f"![{dataset}: {image_name}]({image_prefix}/{image_name})", ""])
    lines.extend(["", "#### Tablas de resultados", ""])
    for dataset in TABLES:
        sub = dataset.lower().replace(" ", "_").replace("ó", "o")
        lines.append(f"- **{dataset}:** [{sub}/]({table_prefix}{sub}/)")
    return "\n".join(lines).strip() + "\n"


def update_root_readme(section):
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.find(START)
    end = text.find(END)
    if start < 0 or end < start:
        raise ValueError("README.md debe incluir los marcadores AUTO-RESULTS-START y AUTO-RESULTS-END.")
    updated = text[: start + len(START)] + "\n\n" + section + "\n" + text[end:]
    readme.write_text(updated, encoding="utf-8")


def main():
    copy_public_tables()
    copy_public_figures()
    section = render_markdown()
    update_root_readme(section)
    nested_section = render_markdown(image_prefix="graficos", table_prefix="")
    (RESULTS / "README.md").write_text(nested_section, encoding="utf-8")
    print(f"Resumen público y tablas agregadas preparados en: {RESULTS}")


if __name__ == "__main__":
    main()
