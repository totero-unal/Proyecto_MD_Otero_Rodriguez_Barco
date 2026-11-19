_Última ejecución publicada (UTC): 2026-11-19 14:35_

Las cifras siguientes son resúmenes de la ejecución más reciente. Los candidatos IQR son señales para revisión, no errores confirmados.

- **EAC 2024:** 10,238 registros; 2 faltantes reportados; 73,196 eventos candidatos IQR agregados por variable.
- **Inflación y meta:** 854 valores observados de inflación; meta faltante en 426 observaciones (49.88%).
- **Inflación global:** 53,845 observaciones anuales posibles entre grupos; 12,838 valores faltantes (23.84%); 3,197 candidatos IQR por serie temporal.
- **Preparación EAC (`CORRE_9`):** 9 clases, 8,190 registros en train y 2,048 en test. Se comparan tres técnicas de remuestreo y se seleccionan predictores por reducción de impureza en un árbol de decisión.

Las tablas agregadas descargables están en las subcarpetas de `resultados/`. Los reportes de filas individuales se generan localmente durante el flujo y no se publican.

### EAC

![EAC: eac_distribuciones_montos_log.png](graficos/eac_distribuciones_montos_log.png)

![EAC: eac_distribucion_personal.png](graficos/eac_distribucion_personal.png)

![EAC: eac_frecuencia_idoj1.png](graficos/eac_frecuencia_idoj1.png)


### Inflación y meta

![Inflación y meta: inflacion_meta_serie_inflacion_meta.png](graficos/inflacion_meta_serie_inflacion_meta.png)

![Inflación y meta: inflacion_meta_distribucion_inflacion.png](graficos/inflacion_meta_distribucion_inflacion.png)

![Inflación y meta: inflacion_meta_comparacion_anual_meta.png](graficos/inflacion_meta_comparacion_anual_meta.png)


### Inflación global

![Inflación global: inflacion_global_mediana_anual_por_indicador.png](graficos/inflacion_global_mediana_anual_por_indicador.png)

![Inflación global: inflacion_global_cobertura_paises_por_anio.png](graficos/inflacion_global_cobertura_paises_por_anio.png)

![Inflación global: inflacion_global_distribuciones_signed_log.png](graficos/inflacion_global_distribuciones_signed_log.png)


### Preparación EAC

![Preparación EAC: preparacion_eac_01_clases_originales.png](graficos/preparacion_eac_01_clases_originales.png)

![Preparación EAC: preparacion_eac_02_particion_estratificada.png](graficos/preparacion_eac_02_particion_estratificada.png)

![Preparación EAC: preparacion_eac_03_clases_por_tecnica.png](graficos/preparacion_eac_03_clases_por_tecnica.png)

![Preparación EAC: preparacion_eac_04_cambio_estructura.png](graficos/preparacion_eac_04_cambio_estructura.png)

![Preparación EAC: preparacion_eac_05_importancia_impureza.png](graficos/preparacion_eac_05_importancia_impureza.png)


#### Tablas de resultados

- **EAC:** [eac/](eac/)
- **Inflación y meta:** [inflacion_y_meta/](inflacion_y_meta/)
- **Inflación global:** [inflacion_global/](inflacion_global/)
- **Preparación EAC:** [preparacion_eac/](preparacion_eac/)
