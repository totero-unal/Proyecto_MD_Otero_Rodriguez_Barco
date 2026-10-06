# Proyecto MD: revisión y análisis de bases

Este repositorio contiene scripts de preprocesamiento y análisis descriptivo para la Encuesta Anual de Comercio 2024, la serie mensual de inflación y meta del Banco de la República, y la base global de inflación del FMI.

## Resultados de la ejecución más reciente

<!-- AUTO-RESULTS-START -->

_Última ejecución publicada (UTC): 2026-10-06 03:54_

Las cifras siguientes son resúmenes de la ejecución más reciente. Los candidatos IQR son señales para revisión, no errores confirmados.

- **EAC 2024:** 10,238 registros; 2 faltantes reportados; 73,196 eventos candidatos IQR agregados por variable.
- **Inflación y meta:** 854 valores observados de inflación; meta faltante en 426 observaciones (49.88%).
- **Inflación global:** 53,845 observaciones anuales posibles entre grupos; 12,838 valores faltantes (23.84%); 3,197 candidatos IQR por serie temporal.

Las tablas agregadas descargables están en las subcarpetas de `resultados/`. Los reportes de filas individuales se generan localmente durante el flujo y no se publican.

### EAC

![EAC: eac_distribuciones_montos_log.png](resultados/graficos/eac_distribuciones_montos_log.png)

![EAC: eac_distribucion_personal.png](resultados/graficos/eac_distribucion_personal.png)

![EAC: eac_frecuencia_idoj1.png](resultados/graficos/eac_frecuencia_idoj1.png)


### Inflación y meta

![Inflación y meta: inflacion_meta_serie_inflacion_meta.png](resultados/graficos/inflacion_meta_serie_inflacion_meta.png)

![Inflación y meta: inflacion_meta_distribucion_inflacion.png](resultados/graficos/inflacion_meta_distribucion_inflacion.png)

![Inflación y meta: inflacion_meta_comparacion_anual_meta.png](resultados/graficos/inflacion_meta_comparacion_anual_meta.png)


### Inflación global

![Inflación global: inflacion_global_mediana_anual_por_indicador.png](resultados/graficos/inflacion_global_mediana_anual_por_indicador.png)

![Inflación global: inflacion_global_cobertura_paises_por_anio.png](resultados/graficos/inflacion_global_cobertura_paises_por_anio.png)

![Inflación global: inflacion_global_distribuciones_signed_log.png](resultados/graficos/inflacion_global_distribuciones_signed_log.png)


#### Tablas de resultados

- **EAC:** [eac/](resultados/eac/)
- **Inflación y meta:** [inflacion_y_meta/](resultados/inflacion_y_meta/)
- **Inflación global:** [inflacion_global/](resultados/inflacion_global/)

<!-- AUTO-RESULTS-END -->

## Reproducir el análisis

Las bases de entrada se encuentran en `data/raw/`. Para ejecutar el flujo localmente:

```bash
python -m pip install -r requirements.txt
python preprocess/EAC.py
python preprocess/inflacion_meta.py
python preprocess/wide_to_long.py
python preprocess/banco_mundial.py
python AD/analisis_descriptivo_eac.py
python AD/analisis_descriptivo_inflacion_meta.py
python AD/analisis_descriptivo_inflacion_global.py
python publish_results.py
```

Los reportes detallados se generan en `reports/`. El flujo automático solo publica tablas agregadas y gráficos en `resultados/` y actualiza esta página. Los resultados IQR son candidatos estadísticos a revisión; por sí solos no indican errores.

## Organización

- `preprocess/`: diagnósticos de calidad y conversión del libro del FMI a formato largo.
- `AD/`: análisis descriptivo, tablas y gráficos.
- `data/raw/`: bases fuente necesarias para ejecutar los scripts.
- `data/processed/`: bases derivadas que se regeneran y no se versionan.
- `reports/`: salidas detalladas locales de los scripts; no se publican automáticamente.
- `resultados/`: resúmenes agregados y gráficos publicados por GitHub Actions.

## Fuentes

- Encuesta Anual de Comercio 2024 anonimizada: [microdatos DANE](https://microdatos.dane.gov.co/index.php/catalog/901).
- Inflación y meta: [series históricas del Banco de la República](https://www.banrep.gov.co/es/estadisticas-economicas/series-historicas/precios-inflacion).
- Inflación global: libro de datos del FMI distribuido como `Inflation-data (1).xlsx`.
