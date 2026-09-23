# Editor de parámetros — Reemplazo de revestimientos SAG (v7)

App Streamlit para ajustar a mano los parámetros del modelo (Cuadro 1 de la
Formulación v7) y ver su efecto en vivo, antes de pasarlos al MILP.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
streamlit run app.py
```

Se abre en el navegador en `http://localhost:8501`.

- **Calendario de detenciones programadas y Flujo de caja (Pestaña 1)**:
  generador rápido de paradas periódicas (presets de 8, 12 o 16 semanas), edición
  manual de ventanas $W_k$ (h), línea de tiempo interactiva de 156 semanas (3 años),
  y análisis económico completo de flujo de caja (ingresos por producción efectiva,
  costos de indisponibilidad $C^D$, costos de recambio $CS_i$, flujo neto semanal y
  acumulado, con tabla exportable a CSV).
- **Parámetros por sección**: tabla editable con `e_i, L_i, R_i, β_i, η_i,
  CS_i, τ_i` para las 9 secciones (Cuadro 1). Avisa si alguna sección queda
  infactible (`e_i > L_i+R_i`, pendiente "Factibilidad inicial", Sección 7.2).
- **Globales de planta**: `ρ, τ_c, C^D, CF`, y precio/margen neto por tonelada ($/t).
- **Riesgo y costo de arco**: curva de riesgo Weibull `P_i(Δ)` (Ec. 3) y costo
  de arco `g_i(Δ)` (Ec. 5), con la zona admisible/podada marcada como en la
  Figura 1 del documento.
- **Resumen y exportar**: tamaño resultante de la red `|A_i|` por sección
  (usa `generar_arcos.py`), y descarga/carga de los parámetros como JSON.

## Archivos

- `app.py` — la app.
- `generar_arcos.py` — generador de arcos (mismo del paso anterior); `app.py`
  lo usa para mostrar `|A_i|` en vivo. Debe quedar en la misma carpeta.
- `parametros_sag.json` — se genera al usar "Descargar parámetros" desde la
  app; es el archivo pensado para alimentar el MILP más adelante.
