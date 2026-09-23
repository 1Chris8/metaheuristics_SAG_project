"""
Editor visual de parámetros — Reemplazo oportunista de revestimientos SAG (v7)
================================================================================
App Streamlit para configurar el calendario de detenciones programadas,
analizar el flujo de caja del horizonte de 156 semanas, y calibrar los
parámetros del modelo (Cuadro 1 de la Formulación v7) antes de correr el MILP.

Incluye:
  - 📅 Calendario de detenciones programadas (E) y flujo de caja en vivo
  - 🧩 Parámetros por sección (e_i, L_i, R_i, β_i, η_i, CS_i, τ_i)
  - 🌐 Parámetros globales de planta (ρ, τ_c, C^D, CF, precio $/t)
  - 📈 Curvas de riesgo Weibull P_i(Δ) y costo de arco g_i(Δ)
  - 📊 Resumen del tamaño de la red |A_i| y exportación JSON

Ejecutar con:  streamlit run app.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from generar_arcos import generar_arcos_seccion

# ---------------------------------------------------------------------------
# Constantes del modelo (Sección 1.1)
# ---------------------------------------------------------------------------
H = 156  # horizonte, semanas (3 años de 52 semanas)
SECCIONES = list(range(1, 10))  # N = {1, ..., 9}

COLUMNAS_SECCION = ["e_i", "L_i", "R_i", "beta_i", "eta_i", "CS_i", "tau_i"]
AYUDA_COLUMNAS = {
    "e_i": "Edad acumulada al inicio del horizonte (sem).",
    "L_i": "Vida nominal de diseño (sem).",
    "R_i": "† Sobreuso máximo admisible sobre L_i (sem).",
    "beta_i": "† Parámetro de forma de la Weibull.",
    "eta_i": "† Parámetro de escala de la Weibull (sem).",
    "CS_i": "Costo de adquisición y recambio ($).",
    "tau_i": "Tiempo marginal de intervención (h).",
}

st.set_page_config(
    page_title="SAG — Calendario, Flujo de Caja y Parámetros",
    page_icon="⚙️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Parámetros por defecto
# ---------------------------------------------------------------------------
def parametros_por_defecto() -> dict:
    # Paradas típicas cada 12 semanas (48 h c/u) sobre el horizonte de 156 semanas
    E_inicial = {w: 48.0 for w in range(12, H + 1, 12)}

    return {
        "global": {
            "rho": 250000.0,    # t/sem
            "tau_c": 12.0,      # h
            "C_D": 600.0,       # $/h
            "CF": 120000.0,     # $ (†, costo de falla)
            "precio_ton": 35.0, # $/t (margen/ingreso neto de tratamiento)
        },
        "secciones": {
            i: {
                "e_i": 0.0,
                "L_i": 40.0,
                "R_i": 8.0,     # †, pendiente
                "beta_i": 2.2,  # †, pendiente
                "eta_i": 48.0,  # †, pendiente
                "CS_i": 22000.0,
                "tau_i": 4.0,
            }
            for i in SECCIONES
        },
        "E": E_inicial,
    }


if "params" not in st.session_state:
    st.session_state.params = parametros_por_defecto()


# ---------------------------------------------------------------------------
# Conversiones de datos
# ---------------------------------------------------------------------------
def secciones_a_df(secciones: dict) -> pd.DataFrame:
    filas = [{"Sección": i, **secciones[i]} for i in SECCIONES]
    return pd.DataFrame(filas).set_index("Sección")


def df_a_secciones(df: pd.DataFrame) -> dict:
    out = {}
    for i, fila in df.iterrows():
        out[int(i)] = {c: float(fila[c]) for c in COLUMNAS_SECCION}
    return out


def E_a_df(E: dict) -> pd.DataFrame:
    if not E:
        return pd.DataFrame({"semana": pd.Series(dtype="int"), "W_k": pd.Series(dtype="float")})
    filas = [{"semana": int(s), "W_k": float(w)} for s, w in sorted(E.items())]
    return pd.DataFrame(filas)


def df_a_E(df: pd.DataFrame) -> dict:
    out = {}
    for _, fila in df.iterrows():
        if pd.isna(fila.get("semana")) or pd.isna(fila.get("W_k")):
            continue
        semana = int(fila["semana"])
        if 1 <= semana <= H:
            out[semana] = float(fila["W_k"])
    return out


# ---------------------------------------------------------------------------
# Weibull y costo de arco
# ---------------------------------------------------------------------------
def weibull_P(x: float, beta: float, eta: float) -> float:
    if x < 0 or eta <= 0:
        return 0.0
    return 1.0 - math.exp(-((x / eta) ** beta))


def curva_riesgo_costo(p: dict, CF: float, x_max: int) -> pd.DataFrame:
    xs = list(range(0, x_max + 1))
    Ps = [weibull_P(x, p["beta_i"], p["eta_i"]) for x in xs]
    g_intermedio = [p["CS_i"] + CF * P for P in Ps]
    g_terminal = [CF * P for P in Ps]
    return pd.DataFrame(
        {"delta": xs, "P": Ps, "g_intermedio": g_intermedio, "g_terminal": g_terminal}
    )


# ---------------------------------------------------------------------------
# Simulación y cálculo del Flujo de Caja
# ---------------------------------------------------------------------------
def calcular_flujo_caja(params: dict, precio_ton: float) -> pd.DataFrame:
    g = params["global"]
    rho = float(g.get("rho", 250000.0))
    C_D = float(g.get("C_D", 600.0))
    E = params.get("E", {})
    secciones = params["secciones"]

    edades = {i: float(secciones[i]["e_i"]) for i in SECCIONES}
    semanas_E_ordenadas = sorted(E.keys())

    filas = []
    flujo_acumulado = 0.0

    for t in range(1, H + 1):
        es_E = t in E
        W_k = float(E[t]) if es_E else 0.0

        horas_detencion = min(168.0, W_k)
        horas_operacion = max(0.0, 168.0 - horas_detencion)
        disp_pct = (horas_operacion / 168.0) * 100.0

        prod_ton = rho * (horas_operacion / 168.0)
        ingreso = prod_ton * precio_ton

        costo_indisp = C_D * horas_detencion

        # Evaluación de recambio de revestimientos en paradas programadas E
        costo_revestimientos = 0.0
        secciones_reemplazadas = []

        proximas_paradas = [s for s in semanas_E_ordenadas if s > t]
        proxima_parada = proximas_paradas[0] if proximas_paradas else (H + 1)
        delta_prox = proxima_parada - t

        for i in SECCIONES:
            L_i = float(secciones[i]["L_i"])
            R_i = float(secciones[i]["R_i"])
            cota = L_i + R_i
            edad_actual = edades[i]

            reemplazar = False
            if es_E:
                # Se reemplaza si ya alcanzó su vida nominal o si no aguantará hasta la próxima parada
                if edad_actual >= L_i:
                    reemplazar = True
                elif (edad_actual + delta_prox) > cota:
                    reemplazar = True

            if reemplazar:
                costo_revestimientos += float(secciones[i]["CS_i"])
                secciones_reemplazadas.append(i)
                edades[i] = 0.0  # pieza nueva
            else:
                edades[i] += 1.0  # envejece

        costos_totales = costo_indisp + costo_revestimientos
        flujo_neto = ingreso - costos_totales
        flujo_acumulado += flujo_neto

        filas.append(
            {
                "semana": t,
                "año": (t - 1) // 52 + 1,
                "mes": (t - 1) // 4 + 1,
                "clase": "E" if es_E else "P",
                "W_k": W_k,
                "horas_operacion": horas_operacion,
                "disponibilidad_pct": disp_pct,
                "prod_ton": prod_ton,
                "ingreso": ingreso,
                "costo_indisp": costo_indisp,
                "costo_revestimientos": costo_revestimientos,
                "costos_totales": costos_totales,
                "flujo_neto": flujo_neto,
                "flujo_acumulado": flujo_acumulado,
                "secciones_reemplazadas": secciones_reemplazadas,
                "reemplazos_txt": ", ".join(f"S{s}" for s in secciones_reemplazadas) if secciones_reemplazadas else "-",
            }
        )

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Sidebar: Gestión de parámetros (JSON)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💾 Archivo de Parámetros")

    json_actual = json.dumps(st.session_state.params, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Descargar parámetros (JSON)",
        data=json_actual,
        file_name="parametros_sag.json",
        mime="application/json",
        width="stretch",
    )

    archivo = st.file_uploader("Cargar parámetros (JSON)", type=["json"])
    if archivo is not None:
        if st.button("Aplicar archivo cargado", width="stretch"):
            try:
                cargado = json.load(archivo)
                cargado["secciones"] = {int(k): v for k, v in cargado["secciones"].items()}
                cargado["E"] = {int(k): v for k, v in cargado.get("E", {}).items()}
                st.session_state.params = cargado
                st.success("Parámetros cargados exitosamente.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al leer archivo: {e}")

    if st.button("↺ Restaurar valores por defecto", width="stretch"):
        st.session_state.params = parametros_por_defecto()
        st.rerun()

    st.caption("Configuración activa para molino SAG (H = 156 sem, 9 secciones).")


# ---------------------------------------------------------------------------
# Título principal y pestañas
# ---------------------------------------------------------------------------
st.title("⚙️ Reemplazo de Revestimientos SAG — Planificación y Parámetros")
st.caption("Formulación v7 · Horizonte H = 156 semanas (3 años) · N = 9 secciones de revestimiento")

tab_calendario_flujo, tab_secciones, tab_global, tab_riesgo, tab_resumen = st.tabs(
    [
        "📅 Calendario de Detenciones y Flujo de Caja",
        "🧩 Parámetros por sección",
        "🌐 Globales de planta",
        "📈 Riesgo y costo de arco",
        "📊 Resumen y exportar",
    ]
)

# ===========================================================================
# TAB 1 — Calendario de Detenciones y Flujo de Caja
# ===========================================================================
with tab_calendario_flujo:
    # -----------------------------------------------------------------------
    # 1. Configuración y Generador Rápido de Detenciones Programadas
    # -----------------------------------------------------------------------
    st.subheader("1. Calendario de Detenciones Programadas (E)")
    st.caption(
        "Define las detenciones exógenas del molino con su ventana disponible W_k (horas). "
        "Usa el generador automático o edita la tabla directamente."
    )

    with st.expander("⚡ Generador de Paradas Periódicas y Presets", expanded=False):
        c_p1, c_p2, c_p3 = st.columns([2, 2, 3])
        with c_p1:
            freq_gen = st.number_input("Intervalo (cada N semanas)", min_value=1, max_value=52, value=12, step=1)
        with c_p2:
            dur_gen = st.number_input("Duración W_k (horas)", min_value=1.0, max_value=168.0, value=48.0, step=6.0)
        with c_p3:
            st.write("Acciones rápidas:")
            c_b1, c_b2 = st.columns(2)
            if c_b1.button("Generar periódicas", width="stretch"):
                st.session_state.params["E"] = {w: float(dur_gen) for w in range(freq_gen, H + 1, freq_gen)}
                st.rerun()
            if c_b2.button("Limpiar paradas", width="stretch"):
                st.session_state.params["E"] = {}
                st.rerun()

        st.markdown("**Atajos típicos de mantención:**")
        b1, b2, b3 = st.columns(3)
        if b1.button("Típico: Cada 12 sem (48 h)", width="stretch"):
            st.session_state.params["E"] = {w: 48.0 for w in range(12, H + 1, 12)}
            st.rerun()
        if b2.button("Mayor: Cada 16 sem (72 h)", width="stretch"):
            st.session_state.params["E"] = {w: 72.0 for w in range(16, H + 1, 16)}
            st.rerun()
        if b3.button("Frecuente: Cada 8 sem (36 h)", width="stretch"):
            st.session_state.params["E"] = {w: 36.0 for w in range(8, H + 1, 8)}
            st.rerun()

    # Pre-cálculo del flujo de caja con los parámetros actuales
    precio_ton_actual = float(st.session_state.params["global"].get("precio_ton", 35.0))
    df_fc = calcular_flujo_caja(st.session_state.params, precio_ton_actual)

    col_izq, col_der = st.columns([3, 2])

    with col_izq:
        # Gráfico visual del calendario / timeline de las 156 semanas
        E_actual = st.session_state.params["E"]
        colores_h = ["#e74c3c" if t in E_actual else "#2ecc71" for t in range(1, H + 1)]
        alturas_h = [float(E_actual.get(t, 0.0)) if t in E_actual else 8.0 for t in range(1, H + 1)]

        hover_h = []
        for t in range(1, H + 1):
            if t in E_actual:
                w = E_actual[t]
                rec = df_fc.loc[df_fc["semana"] == t, "reemplazos_txt"].values[0]
                hover_h.append(
                    f"<b>Semana {t} (Detención E)</b><br>"
                    f"Duración: {w:g} h<br>"
                    f"Disponibilidad: {((168-w)/168)*100:.1f}%<br>"
                    f"Reemplazos previstos: {rec}"
                )
            else:
                hover_h.append(f"<b>Semana {t} (Operación P)</b><br>100% disponible (168 h)")

        fig_cal = go.Figure()
        fig_cal.add_trace(
            go.Bar(
                x=list(range(1, H + 1)),
                y=alturas_h,
                marker_color=colores_h,
                hovertext=hover_h,
                hoverinfo="text",
                name="Semanas",
            )
        )

        # Separadores de años
        fig_cal.add_vline(x=52.5, line_dash="dash", line_color="rgba(120,120,120,0.5)", annotation_text="Año 1 | Año 2")
        fig_cal.add_vline(x=104.5, line_dash="dash", line_color="rgba(120,120,120,0.5)", annotation_text="Año 2 | Año 3")

        fig_cal.update_layout(
            title="Línea de Tiempo del Horizonte (156 Semanas)",
            height=260,
            showlegend=False,
            xaxis=dict(title="Semana", range=[0.5, H + 0.5], tickmode="linear", tick0=0, dtick=12),
            yaxis=dict(title="Horas Parada W_k", range=[0, max(max(alturas_h, default=10), 50) * 1.15]),
            margin=dict(t=40, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_cal, width="stretch")
        st.markdown(
            "🔴 **Rojo (E):** Semana con detención programada (altura = duración $W_k$ en horas) &nbsp;|&nbsp; "
            "🟢 **Verde (P):** Semana normal de producción (168 h)"
        )

    with col_der:
        st.write("**Tabla de Paradas Programadas E:**")
        df_E_actual = E_a_df(st.session_state.params["E"])
        df_E_editado = st.data_editor(
            df_E_actual,
            num_rows="dynamic",
            width="stretch",
            height=230,
            column_config={
                "semana": st.column_config.NumberColumn("Semana", min_value=1, max_value=H, step=1),
                "W_k": st.column_config.NumberColumn("W_k (horas)", min_value=0.5, max_value=168.0, step=1.0),
            },
            key="editor_E_tab1",
        )
        # Sincronizar cambios si el usuario edita la tabla
        st.session_state.params["E"] = df_a_E(df_E_editado)

        total_paradas = len(st.session_state.params["E"])
        total_horas_p = sum(st.session_state.params["E"].values())
        st.info(f"📊 **{total_paradas}** detenciones programadas · **{total_horas_p:,.1f} h** totales de detención.")

    st.divider()

    # -----------------------------------------------------------------------
    # 2. Flujo de Caja de Todo el Horizonte Planificado
    # -----------------------------------------------------------------------
    st.subheader("2. Flujo de Caja de Todo el Horizonte Planificado (156 semanas)")

    # Parámetros económicos interactivos
    with st.expander("⚙️ Parámetros Económicos del Flujo de Caja", expanded=False):
        c_eco1, c_eco2, c_eco3, c_eco4 = st.columns(4)
        precio_input = c_eco1.number_input(
            "Precio/Margen neto ($/t tratada)",
            min_value=0.0,
            value=float(st.session_state.params["global"].get("precio_ton", 35.0)),
            step=1.0,
            key="input_precio_ton",
        )
        st.session_state.params["global"]["precio_ton"] = precio_input

        c_eco2.metric("Tasa Tratamiento (ρ)", f"{st.session_state.params['global']['rho']:,.0f} t/sem")
        c_eco3.metric("Costo Indisponibilidad (C^D)", f"${st.session_state.params['global']['C_D']:,.0f} / h")
        c_eco4.metric("Costo Falla Molino (CF)", f"${st.session_state.params['global']['CF']:,.0f}")

    # Recalcular el flujo con el precio actualizado
    df_fc = calcular_flujo_caja(st.session_state.params, precio_input)

    # Métricas clave del horizonte
    flujo_total = df_fc["flujo_neto"].sum()
    ingresos_totales = df_fc["ingreso"].sum()
    costo_indisp_total = df_fc["costo_indisp"].sum()
    costo_rev_total = df_fc["costo_revestimientos"].sum()
    horas_totales_op = df_fc["horas_operacion"].sum()
    disp_global = (horas_totales_op / (H * 168.0)) * 100.0
    prod_total_ton = df_fc["prod_ton"].sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💰 Flujo de Caja Neto Total", f"${flujo_total:,.0f}")
    m2.metric("📈 Ingresos por Producción", f"${ingresos_totales:,.0f}")
    m3.metric("⏱️ Costo Indisponibilidad", f"-${costo_indisp_total:,.0f}")
    m4.metric("🔩 Inversión Revestimientos", f"-${costo_rev_total:,.0f}")
    m5.metric("⚙️ Disponibilidad Promedio", f"{disp_global:.2f}%", f"{prod_total_ton/1e6:.2f} Mt tratadas")

    # Gráfico de Flujo de Caja: Semanal y Acumulado
    fig_fc = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Flujo Semanal: Ingresos, Costos y Flujo Neto ($)",
            "Flujo de Caja Acumulado del Horizonte ($)",
        ),
        row_heights=[0.55, 0.45],
    )

    # Row 1: Ingresos, Costos Totales y Flujo Neto semanal
    fig_fc.add_trace(
        go.Bar(
            x=df_fc["semana"],
            y=df_fc["ingreso"],
            name="Ingresos semanales",
            marker_color="rgba(46, 204, 113, 0.6)",
            hoverinfo="x+y+name",
        ),
        row=1,
        col=1,
    )
    fig_fc.add_trace(
        go.Bar(
            x=df_fc["semana"],
            y=-df_fc["costos_totales"],
            name="Costos semanales (Indisp. + Revest.)",
            marker_color="rgba(231, 76, 60, 0.6)",
            hoverinfo="x+y+name",
        ),
        row=1,
        col=1,
    )
    fig_fc.add_trace(
        go.Scatter(
            x=df_fc["semana"],
            y=df_fc["flujo_neto"],
            mode="lines+markers",
            name="Flujo Neto Semanal",
            line=dict(color="#2980b9", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Row 2: Flujo Acumulado
    fig_fc.add_trace(
        go.Scatter(
            x=df_fc["semana"],
            y=df_fc["flujo_acumulado"],
            mode="lines",
            fill="tozeroy",
            name="Flujo Neto Acumulado",
            line=dict(color="#27ae60", width=3),
            fillcolor="rgba(46, 204, 113, 0.15)",
        ),
        row=2,
        col=1,
    )

    # Líneas de años en ambos subplots
    for r in (1, 2):
        fig_fc.add_vline(x=52.5, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=1)
        fig_fc.add_vline(x=104.5, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=1)

    fig_fc.update_xaxes(title_text="Semana del Horizonte", row=2, col=1, range=[0.5, H + 0.5])
    fig_fc.update_yaxes(title_text="$ / semana", row=1, col=1)
    fig_fc.update_yaxes(title_text="$ acumulado", row=2, col=1)
    fig_fc.update_layout(
        height=580,
        barmode="relative",
        legend=dict(orientation="h", y=1.08, x=0.1),
        margin=dict(t=50, b=40, l=60, r=30),
    )
    st.plotly_chart(fig_fc, width="stretch")

    # Resumen y Tabla Detallada del Flujo de Caja
    with st.expander("📋 Ver Tabla Detallada de Flujo de Caja (Exportable a CSV)", expanded=False):
        c_ag1, c_ag2 = st.columns([2, 2])
        modo_vista = c_ag1.radio("Agrupación de datos:", ["Por Semana (156)", "Por Año (1 a 3)"], horizontal=True)

        if modo_vista == "Por Año (1 a 3)":
            df_anual = (
                df_fc.groupby("año")
                .agg(
                    Semanas=("semana", "count"),
                    Horas_Detencion=("W_k", "sum"),
                    Disponibilidad_Prom=("disponibilidad_pct", "mean"),
                    Toneladas_Tratadas=("prod_ton", "sum"),
                    Ingresos=("ingreso", "sum"),
                    Costo_Indisponibilidad=("costo_indisp", "sum"),
                    Inversion_Revestimientos=("costo_revestimientos", "sum"),
                    Costos_Totales=("costos_totales", "sum"),
                    Flujo_Neto=("flujo_neto", "sum"),
                )
                .reset_index()
            )
            st.dataframe(df_anual.style.format({
                "Disponibilidad_Prom": "{:.2f}%",
                "Toneladas_Tratadas": "{:,.0f} t",
                "Ingresos": "${:,.0f}",
                "Costo_Indisponibilidad": "${:,.0f}",
                "Inversion_Revestimientos": "${:,.0f}",
                "Costos_Totales": "${:,.0f}",
                "Flujo_Neto": "${:,.0f}",
            }), width="stretch")
        else:
            columnas_mostrar = [
                "semana",
                "año",
                "clase",
                "W_k",
                "disponibilidad_pct",
                "prod_ton",
                "ingreso",
                "costo_indisp",
                "costo_revestimientos",
                "reemplazos_txt",
                "flujo_neto",
                "flujo_acumulado",
            ]
            df_tabla = df_fc[columnas_mostrar].copy()
            st.dataframe(
                df_tabla.style.format({
                    "W_k": "{:.1f} h",
                    "disponibilidad_pct": "{:.1f}%",
                    "prod_ton": "{:,.0f}",
                    "ingreso": "${:,.0f}",
                    "costo_indisp": "${:,.0f}",
                    "costo_revestimientos": "${:,.0f}",
                    "flujo_neto": "${:,.0f}",
                    "flujo_acumulado": "${:,.0f}",
                }),
                width="stretch",
                height=350,
            )

        csv_fc = df_fc.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar Flujo de Caja Completo (CSV)",
            data=csv_fc,
            file_name="flujo_caja_sag_156_semanas.csv",
            mime="text/csv",
        )


# ===========================================================================
# TAB 2 — Parámetros por sección
# ===========================================================================
with tab_secciones:
    st.subheader("Parámetros por sección (Cuadro 1, 'Por sección i')")
    st.caption("Edita directamente en la tabla. Pasa el mouse sobre un encabezado para ver su significado.")

    df_secciones = secciones_a_df(st.session_state.params["secciones"])
    column_config = {
        c: st.column_config.NumberColumn(c, help=AYUDA_COLUMNAS[c], min_value=0.0, step=0.5)
        for c in COLUMNAS_SECCION
    }
    df_editado = st.data_editor(
        df_secciones,
        column_config=column_config,
        num_rows="fixed",
        width="stretch",
        key="editor_secciones",
    )
    st.session_state.params["secciones"] = df_a_secciones(df_editado)

    # Aviso de factibilidad inicial (Sección 7.2): e_i > L_i + R_i
    infactibles = [
        i for i, p in st.session_state.params["secciones"].items()
        if p["e_i"] > p["L_i"] + p["R_i"]
    ]
    if infactibles:
        st.error(
            f"Secciones infactibles (e_i > L_i+R_i, ningún arco sale de la "
            f"fuente 0): {infactibles}. Ver pendiente 'Factibilidad inicial', "
            f"Sección 7.2."
        )
    else:
        st.success("Todas las secciones tienen al menos un arco factible desde la fuente 0.")


# ===========================================================================
# TAB 3 — Globales de planta
# ===========================================================================
with tab_global:
    st.subheader("Parámetros globales de planta (Cuadro 1, 'Globales de planta')")
    g = st.session_state.params["global"]
    c1, c2, c3, c4 = st.columns(4)
    g["rho"] = c1.number_input("ρ — tasa de tratamiento (t/sem)", value=float(g["rho"]), min_value=0.0, step=1000.0)
    g["tau_c"] = c2.number_input("τ_c — tiempo común de detención (h)", value=float(g["tau_c"]), min_value=0.0, step=0.5)
    g["C_D"] = c3.number_input("C^D — costo horario de indisponibilidad ($/h)", value=float(g["C_D"]), min_value=0.0, step=10.0)
    g["CF"] = c4.number_input("CF — costo de falla del molino ($) †", value=float(g["CF"]), min_value=0.0, step=1000.0)

    c5, c6 = st.columns(2)
    g["precio_ton"] = c5.number_input("Precio / Margen neto de producción ($/t)", value=float(g.get("precio_ton", 35.0)), min_value=0.0, step=1.0)
    st.session_state.params["global"] = g


# ===========================================================================
# TAB 4 — Riesgo y costo de arco (Ecs. 3, 5, 6 / Figura 1 del documento)
# ===========================================================================
with tab_riesgo:
    st.subheader("Riesgo de falla y costo de arco por sección")
    seccion_sel = st.selectbox("Sección", SECCIONES, key="seccion_riesgo")
    p = st.session_state.params["secciones"][seccion_sel]
    CF = st.session_state.params["global"]["CF"]

    cota = p["L_i"] + p["R_i"]
    x_max = max(int(cota * 1.6), int(p["L_i"] * 1.6), 20)
    df_curva = curva_riesgo_costo(p, CF, x_max)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            f"P_{seccion_sel}(Δ) — riesgo incondicional de falla (Ec. 3)",
            f"g_{seccion_sel}(Δ) — costo del arco (Ec. 5)",
        ),
    )

    # Zona admisible / poda topológica (Fig. 1 del documento)
    for row in (1, 2):
        fig.add_vrect(x0=0, x1=cota, fillcolor="LightGreen", opacity=0.15,
                       line_width=0, row=row, col=1)
        fig.add_vrect(x0=cota, x1=x_max, fillcolor="LightCoral", opacity=0.12,
                       line_width=0, row=row, col=1)
        fig.add_vline(x=p["L_i"], line_dash="dash", line_color="gray", row=row, col=1)
        fig.add_vline(x=cota, line_dash="dot", line_color="firebrick", row=row, col=1)

    fig.add_trace(go.Scatter(x=df_curva["delta"], y=df_curva["P"], mode="lines",
                              name="P(Δ)", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_curva["delta"], y=df_curva["g_intermedio"], mode="lines",
                              name="g(Δ), t∈T (con CS_i)", line=dict(color="darkorange")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_curva["delta"], y=df_curva["g_terminal"], mode="lines",
                              name="g(Δ), t=∞ (solo riesgo)", line=dict(color="seagreen", dash="dash")),
                  row=2, col=1)

    fig.update_yaxes(title_text="Probabilidad", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Costo ($)", row=2, col=1)
    fig.update_xaxes(title_text="Edad del ciclo Δ (semanas)", row=2, col=1)
    fig.update_layout(height=650, legend=dict(orientation="h", y=-0.15))

    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Verde: zona admisible Δ ≤ L_i+R_i = {cota:g} sem (Ec. 6). "
        f"Rojo: poda topológica (arcos que no existen). "
        f"Línea gris punteada: L_i = {p['L_i']:g}. Línea roja punteada: L_i+R_i."
    )


# ===========================================================================
# TAB 5 — Resumen y exportación
# ===========================================================================
with tab_resumen:
    st.subheader("Resumen del tamaño de la red (Sección 6.1)")

    filas_resumen = []
    total_arcos = 0
    for i in SECCIONES:
        p = st.session_state.params["secciones"][i]
        A_i = generar_arcos_seccion(i, p["e_i"], p["L_i"], p["R_i"], H)
        cota_teorica = (H + 1) * (p["L_i"] + p["R_i"] + 1)
        filas_resumen.append(
            {"Sección": i, "|A_i|": len(A_i), "Cota (H+1)(L_i+R_i+1)": round(cota_teorica)}
        )
        total_arcos += len(A_i)

    df_resumen = pd.DataFrame(filas_resumen).set_index("Sección")
    st.dataframe(df_resumen, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de variables x_ij", f"{total_arcos:,}")
    c2.metric("Variables z_it (9·H)", f"{9 * H:,}")
    c3.metric("Variables y_t + w_k (H)", f"{H:,}")

    st.divider()
    st.subheader("Parámetros actuales (JSON)")
    st.json(st.session_state.params, expanded=False)
