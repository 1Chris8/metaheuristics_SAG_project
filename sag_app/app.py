"""
Sistema de Planificación de Reemplazo de Revestimientos SAG — Formulación v7
=============================================================================
Aplicación para configuración de parámetros del modelo, programación de
detenciones del molino y simulación del flujo de caja en el horizonte de 156 semanas.
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
H = 156  # horizonte de planificación (semanas, 3 años de 52 semanas)
SECCIONES = list(range(1, 10))  # N = {1, ..., 9}

COLUMNAS_SECCION = ["e_i", "L_i", "R_i", "beta_i", "eta_i", "CS_i", "tau_i"]
AYUDA_COLUMNAS = {
    "e_i": "Edad acumulada al inicio del horizonte (semanas).",
    "L_i": "Vida nominal de diseño del revestimiento (semanas).",
    "R_i": "Sobreuso máximo admisible sobre L_i (semanas).",
    "beta_i": "Parámetro de forma de la distribución Weibull.",
    "eta_i": "Parámetro de escala de la distribución Weibull (semanas).",
    "CS_i": "Costo directo de adquisición y montaje del revestimiento ($).",
    "tau_i": "Tiempo marginal requerido de intervención (horas).",
}

st.set_page_config(
    page_title="Planificación Revestimientos SAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inyección de estilos CSS para apariencia web técnica y limpia
st.markdown(
    """
    <style>
      /* Tipografía y espaciado general */
      html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      }
      
      /* Ocultar barra superior decorativa de Streamlit */
      header[data-testid="stHeader"] {
        background-color: transparent;
      }

      /* Contenedor principal */
      .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
      }

      /* Estilo de pestañas */
      button[data-baseweb="tab"] {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        padding-top: 8px;
        padding-bottom: 8px;
        border-radius: 0px;
      }
      button[data-baseweb="tab"][aria-selected="true"] {
        color: #0f172a !important;
        border-bottom-color: #0f172a !important;
      }

      /* Botones planos estilo HTML corporativo */
      div.stButton > button {
        border-radius: 4px;
        font-size: 13px;
        font-weight: 500;
        border: 1px solid #cbd5e1;
        transition: all 0.15s ease-in-out;
      }
      div.stButton > button:hover {
        border-color: #0f172a;
        color: #0f172a;
        background-color: #f8fafc;
      }

      /* Tablas y editores de datos */
      div[data-testid="stDataEditor"] {
        border: 1px solid #e2e8f0;
        border-radius: 4px;
      }

      /* Tarjeta de métrica */
      .html-card {
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        padding: 14px 16px;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
      }
      .html-card-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-bottom: 4px;
      }
      .html-card-value {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      }
      .html-card-sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 3px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Parámetros por defecto
# ---------------------------------------------------------------------------
def parametros_por_defecto() -> dict:
    # 13 paradas típicas cada 12 semanas (48 h cada una) en el horizonte de 156 semanas
    E_inicial = {w: 48.0 for w in range(12, H + 1, 12)}

    return {
        "global": {
            "rho": 250000.0,    # t/semana
            "tau_c": 12.0,      # h
            "C_D": 600.0,       # $/h
            "CF": 120000.0,     # $ (costo de falla catastrófica)
            "precio_ton": 35.0, # $/t (margen o valor neto por tonelada tratada)
        },
        "secciones": {
            i: {
                "e_i": 0.0,
                "L_i": 40.0,
                "R_i": 8.0,
                "beta_i": 2.2,
                "eta_i": 48.0,
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
# Cálculo del Flujo de Caja
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

        # Regla de reemplazo oportunista de revestimientos en detenciones E
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
                if edad_actual >= L_i:
                    reemplazar = True
                elif (edad_actual + delta_prox) > cota:
                    reemplazar = True

            if reemplazar:
                costo_revestimientos += float(secciones[i]["CS_i"])
                secciones_reemplazadas.append(i)
                edades[i] = 0.0
            else:
                edades[i] += 1.0

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
    st.markdown(
        """
        <div style="font-size: 14px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">
          Gestión de Parámetros
        </div>
        """,
        unsafe_allow_html=True,
    )

    json_actual = json.dumps(st.session_state.params, indent=2, ensure_ascii=False)
    st.download_button(
        "Descargar parámetros (JSON)",
        data=json_actual,
        file_name="parametros_sag.json",
        mime="application/json",
        width="stretch",
    )

    archivo = st.file_uploader("Cargar archivo JSON", type=["json"])
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

    if st.button("Restaurar valores de referencia", width="stretch"):
        st.session_state.params = parametros_por_defecto()
        st.rerun()

    st.markdown(
        """
        <div style="font-size: 11px; color: #64748b; line-height: 1.4; margin-top: 16px;">
          Configuración activa para molino SAG.<br>
          Horizonte: 156 semanas.<br>
          Secciones: 9 componentes.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Encabezado HTML principal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background-color: #0f172a; color: #f8fafc; padding: 22px 28px; border-radius: 4px; margin-bottom: 22px;">
      <div style="font-size: 11px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;">
        Optimización de Operaciones y Mantenimiento · Formulación v7
      </div>
      <div style="font-size: 24px; font-weight: 700; letter-spacing: -0.4px; color: #ffffff; margin-bottom: 10px;">
        Planificación de Reemplazo de Revestimientos — Molino SAG
      </div>
      <div style="font-size: 13px; color: #cbd5e1; display: flex; gap: 24px; flex-wrap: wrap;">
        <span>Horizonte de Planificación: <strong style="color: #ffffff;">156 semanas (3 años)</strong></span>
        <span>Componentes: <strong style="color: #ffffff;">9 secciones de revestimiento</strong></span>
        <span>Modelo: <strong style="color: #ffffff;">Optimización MILP de flujo en redes</strong></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_calendario_flujo, tab_secciones, tab_global, tab_riesgo, tab_resumen = st.tabs(
    [
        "Calendario y Flujo de Caja",
        "Parámetros por Sección",
        "Parámetros Globales",
        "Curvas de Riesgo y Costos",
        "Resumen del Modelo",
    ]
)


# ===========================================================================
# TAB 1 — Calendario de Detenciones y Flujo de Caja
# ===========================================================================
with tab_calendario_flujo:
    # -----------------------------------------------------------------------
    # Sección 1: Calendario de Detenciones Programadas
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 700; color: #0f172a;">1. Calendario de Detenciones Programadas (Conjunto E)</div>
          <div style="font-size: 13px; color: #64748b;">Configuración de semanas con parada de planta exógena y duración disponible W_k en horas.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Generador de Paradas Periódicas y Programas Típicos", expanded=False):
        c_p1, c_p2, c_p3 = st.columns([2, 2, 3])
        with c_p1:
            freq_gen = st.number_input("Intervalo (cada N semanas)", min_value=1, max_value=52, value=12, step=1)
        with c_p2:
            dur_gen = st.number_input("Duración W_k (horas)", min_value=1.0, max_value=168.0, value=48.0, step=6.0)
        with c_p3:
            st.write("Acciones:")
            c_b1, c_b2 = st.columns(2)
            if c_b1.button("Generar paradas periódicas", width="stretch"):
                st.session_state.params["E"] = {w: float(dur_gen) for w in range(freq_gen, H + 1, freq_gen)}
                st.rerun()
            if c_b2.button("Limpiar paradas", width="stretch"):
                st.session_state.params["E"] = {}
                st.rerun()

        st.markdown("<div style='font-size: 12px; font-weight: 600; color: #475569; margin-top: 10px; margin-bottom: 6px;'>Programas típicos de mantenimiento de planta:</div>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        if b1.button("Estándar: Cada 12 sem (48 h)", width="stretch"):
            st.session_state.params["E"] = {w: 48.0 for w in range(12, H + 1, 12)}
            st.rerun()
        if b2.button("Mayor: Cada 16 sem (72 h)", width="stretch"):
            st.session_state.params["E"] = {w: 72.0 for w in range(16, H + 1, 16)}
            st.rerun()
        if b3.button("Frecuente: Cada 8 sem (36 h)", width="stretch"):
            st.session_state.params["E"] = {w: 36.0 for w in range(8, H + 1, 8)}
            st.rerun()

    # Cálculo inicial del flujo de caja
    precio_ton_actual = float(st.session_state.params["global"].get("precio_ton", 35.0))
    df_fc = calcular_flujo_caja(st.session_state.params, precio_ton_actual)

    col_izq, col_der = st.columns([3, 2])

    with col_izq:
        E_actual = st.session_state.params["E"]
        colores_h = ["#b91c1c" if t in E_actual else "#15803d" for t in range(1, H + 1)]
        alturas_h = [float(E_actual.get(t, 0.0)) if t in E_actual else 6.0 for t in range(1, H + 1)]

        hover_h = []
        for t in range(1, H + 1):
            if t in E_actual:
                w = E_actual[t]
                rec = df_fc.loc[df_fc["semana"] == t, "reemplazos_txt"].values[0]
                hover_h.append(
                    f"Semana {t} (Detención E)<br>"
                    f"Duración: {w:g} h<br>"
                    f"Disponibilidad: {((168-w)/168)*100:.1f}%<br>"
                    f"Reemplazos previstos: {rec}"
                )
            else:
                hover_h.append(f"Semana {t} (Operación P)<br>Disponibilidad: 100% (168 h)")

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

        fig_cal.add_vline(x=52.5, line_dash="dash", line_color="#94a3b8", annotation_text="Año 1 | Año 2")
        fig_cal.add_vline(x=104.5, line_dash="dash", line_color="#94a3b8", annotation_text="Año 2 | Año 3")

        fig_cal.update_layout(
            template="plotly_white",
            height=250,
            showlegend=False,
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12),
            xaxis=dict(title="Semana del Horizonte", range=[0.5, H + 0.5], tickmode="linear", tick0=0, dtick=12),
            yaxis=dict(title="Duración Parada W_k (h)", range=[0, max(max(alturas_h, default=10), 50) * 1.15]),
            margin=dict(t=20, b=35, l=45, r=20),
        )
        st.plotly_chart(fig_cal, width="stretch")

        st.markdown(
            """
            <div style="display: flex; gap: 24px; font-size: 12px; color: #475569; margin-top: 4px;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; background-color: #b91c1c; border-radius: 2px;"></span>
                <span><strong>Clase E:</strong> Detención programada (altura = duración W_k en horas)</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; background-color: #15803d; border-radius: 2px;"></span>
                <span><strong>Clase P:</strong> Operación continua (168 h disponibles)</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_der:
        st.markdown("<div style='font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 6px;'>Edición de Paradas Programadas (Semana y Horas W_k):</div>", unsafe_allow_html=True)
        df_E_actual = E_a_df(st.session_state.params["E"])
        df_E_editado = st.data_editor(
            df_E_actual,
            num_rows="dynamic",
            width="stretch",
            height=215,
            column_config={
                "semana": st.column_config.NumberColumn("Semana", min_value=1, max_value=H, step=1),
                "W_k": st.column_config.NumberColumn("W_k (horas)", min_value=0.5, max_value=168.0, step=1.0),
            },
            key="editor_E_tab1",
        )
        st.session_state.params["E"] = df_a_E(df_E_editado)

        total_paradas = len(st.session_state.params["E"])
        total_horas_p = sum(st.session_state.params["E"].values())
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 8px 12px; border-radius: 4px; font-size: 12px; color: #334155; margin-top: 8px;">
              Detenciones configuradas: <strong>{total_paradas}</strong> &nbsp;|&nbsp; Horas acumuladas de parada: <strong>{total_horas_p:,.1f} h</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 24px; margin-bottom: 24px; border-top: 1px solid #e2e8f0;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Sección 2: Flujo de Caja del Horizonte Planificado
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 700; color: #0f172a;">2. Flujo de Caja del Horizonte Planificado (156 semanas)</div>
          <div style="font-size: 13px; color: #64748b;">Simulación económica de ingresos operacionales, costos por indisponibilidad e inversión en revestimientos.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Parámetros Económicos del Flujo de Caja", expanded=False):
        c_eco1, c_eco2, c_eco3, c_eco4 = st.columns(4)
        precio_input = c_eco1.number_input(
            "Margen neto de tratamiento ($/t)",
            min_value=0.0,
            value=float(st.session_state.params["global"].get("precio_ton", 35.0)),
            step=1.0,
            key="input_precio_ton",
        )
        st.session_state.params["global"]["precio_ton"] = precio_input

        c_eco2.metric("Tasa Nominal (ρ)", f"{st.session_state.params['global']['rho']:,.0f} t/sem")
        c_eco3.metric("Costo Indisponibilidad (C^D)", f"${st.session_state.params['global']['C_D']:,.0f} / h")
        c_eco4.metric("Costo Falla Catastrófica (CF)", f"${st.session_state.params['global']['CF']:,.0f}")

    df_fc = calcular_flujo_caja(st.session_state.params, precio_input)

    flujo_total = df_fc["flujo_neto"].sum()
    ingresos_totales = df_fc["ingreso"].sum()
    costo_indisp_total = df_fc["costo_indisp"].sum()
    costo_rev_total = df_fc["costo_revestimientos"].sum()
    horas_totales_op = df_fc["horas_operacion"].sum()
    disp_global = (horas_totales_op / (H * 168.0)) * 100.0
    prod_total_ton = df_fc["prod_ton"].sum()

    # Tarjetas HTML de Métricas
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    with c_m1:
        st.markdown(
            f"""
            <div class="html-card" style="border-top: 3px solid #2563eb;">
              <div class="html-card-title">Flujo de Caja Neto Total</div>
              <div class="html-card-value">${flujo_total:,.0f}</div>
              <div class="html-card-sub">Margen operacional 156 sem</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m2:
        st.markdown(
            f"""
            <div class="html-card" style="border-top: 3px solid #16a34a;">
              <div class="html-card-title">Ingresos por Tratamiento</div>
              <div class="html-card-value">${ingresos_totales:,.0f}</div>
              <div class="html-card-sub">{prod_total_ton/1e6:.2f} Mt procesadas</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m3:
        st.markdown(
            f"""
            <div class="html-card" style="border-top: 3px solid #dc2626;">
              <div class="html-card-title">Costo por Indisponibilidad</div>
              <div class="html-card-value">-${costo_indisp_total:,.0f}</div>
              <div class="html-card-sub">Horas no disponibles × C^D</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m4:
        st.markdown(
            f"""
            <div class="html-card" style="border-top: 3px solid #f59e0b;">
              <div class="html-card-title">Inversión en Revestimientos</div>
              <div class="html-card-value">-${costo_rev_total:,.0f}</div>
              <div class="html-card-sub">Recambios de secciones CS_i</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m5:
        st.markdown(
            f"""
            <div class="html-card" style="border-top: 3px solid #475569;">
              <div class="html-card-title">Disponibilidad Global</div>
              <div class="html-card-value">{disp_global:.2f}%</div>
              <div class="html-card-sub">{horas_totales_op:,.0f} h operadas / 26,208 h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # Gráfico de Flujo de Caja
    fig_fc = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Flujo Semanal: Ingresos, Costos y Flujo Neto ($ / semana)",
            "Flujo de Caja Acumulado del Horizonte de 156 Semanas ($ acumulado)",
        ),
        row_heights=[0.55, 0.45],
    )

    fig_fc.add_trace(
        go.Bar(
            x=df_fc["semana"],
            y=df_fc["ingreso"],
            name="Ingresos semanales",
            marker_color="#86efac",
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
            marker_color="#fca5a5",
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
            line=dict(color="#1d4ed8", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    fig_fc.add_trace(
        go.Scatter(
            x=df_fc["semana"],
            y=df_fc["flujo_acumulado"],
            mode="lines",
            fill="tozeroy",
            name="Flujo Neto Acumulado",
            line=dict(color="#047857", width=2.5),
            fillcolor="rgba(16, 185, 129, 0.12)",
        ),
        row=2,
        col=1,
    )

    for r in (1, 2):
        fig_fc.add_vline(x=52.5, line_dash="dash", line_color="#94a3b8", opacity=0.6, row=r, col=1)
        fig_fc.add_vline(x=104.5, line_dash="dash", line_color="#94a3b8", opacity=0.6, row=r, col=1)

    fig_fc.update_xaxes(title_text="Semana del Horizonte", row=2, col=1, range=[0.5, H + 0.5])
    fig_fc.update_yaxes(title_text="USD / sem", row=1, col=1)
    fig_fc.update_yaxes(title_text="USD acumulado", row=2, col=1)
    fig_fc.update_layout(
        template="plotly_white",
        height=560,
        barmode="relative",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12),
        legend=dict(orientation="h", y=1.07, x=0.05),
        margin=dict(t=50, b=40, l=60, r=25),
    )
    st.plotly_chart(fig_fc, width="stretch")

    # Tabla detallada y exportación
    with st.expander("Tabla Detallada de Flujo de Caja", expanded=False):
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
            st.dataframe(
                df_anual.style.format({
                    "Disponibilidad_Prom": "{:.2f}%",
                    "Toneladas_Tratadas": "{:,.0f} t",
                    "Ingresos": "${:,.0f}",
                    "Costo_Indisponibilidad": "${:,.0f}",
                    "Inversion_Revestimientos": "${:,.0f}",
                    "Costos_Totales": "${:,.0f}",
                    "Flujo_Neto": "${:,.0f}",
                }),
                width="stretch",
            )
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
            "Descargar datos completos de flujo de caja (CSV)",
            data=csv_fc,
            file_name="flujo_caja_sag_156_semanas.csv",
            mime="text/csv",
        )


# ===========================================================================
# TAB 2 — Parámetros por sección
# ===========================================================================
with tab_secciones:
    st.markdown(
        """
        <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Parámetros por Sección (Cuadro 1, Formulación v7)</div>
          <div style="font-size: 13px; color: #64748b;">Valores técnicos y económicos por cada sección de revestimiento i = 1, ..., 9.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    infactibles = [
        i for i, p in st.session_state.params["secciones"].items()
        if p["e_i"] > p["L_i"] + p["R_i"]
    ]
    if infactibles:
        st.error(
            f"Alerta de factibilidad inicial (Sección 7.2): Las secciones {infactibles} "
            f"tienen e_i > L_i + R_i. No existen arcos admisibles desde el nodo fuente 0."
        )
    else:
        st.success("Condición de factibilidad inicial verificada: todas las secciones poseen arcos admisibles desde la fuente 0.")


# ===========================================================================
# TAB 3 — Globales de planta
# ===========================================================================
with tab_global:
    st.markdown(
        """
        <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Parámetros Globales de Planta (Cuadro 1)</div>
          <div style="font-size: 13px; color: #64748b;">Tasas de tratamiento, tiempos base de intervención y coeficientes de costo.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g = st.session_state.params["global"]
    c1, c2, c3, c4 = st.columns(4)
    g["rho"] = c1.number_input("ρ — Tasa de tratamiento nominal (t/sem)", value=float(g["rho"]), min_value=0.0, step=1000.0)
    g["tau_c"] = c2.number_input("τ_c — Tiempo común de detención (h)", value=float(g["tau_c"]), min_value=0.0, step=0.5)
    g["C_D"] = c3.number_input("C^D — Costo horario de indisponibilidad ($/h)", value=float(g["C_D"]), min_value=0.0, step=10.0)
    g["CF"] = c4.number_input("CF — Costo por falla del molino ($)", value=float(g["CF"]), min_value=0.0, step=1000.0)

    c5, c6 = st.columns(2)
    g["precio_ton"] = c5.number_input("Margen neto de producción ($/t tratada)", value=float(g.get("precio_ton", 35.0)), min_value=0.0, step=1.0)
    st.session_state.params["global"] = g


# ===========================================================================
# TAB 4 — Riesgo y costo de arco
# ===========================================================================
with tab_riesgo:
    st.markdown(
        """
        <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Curvas de Riesgo de Falla y Costo de Arco por Sección</div>
          <div style="font-size: 13px; color: #64748b;">Evaluación de la distribución Weibull P_i(Δ) y de la función de costo g_i(Δ) (Ecuaciones 3 y 5).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    seccion_sel = st.selectbox("Seleccionar Sección:", SECCIONES, key="seccion_riesgo")
    p = st.session_state.params["secciones"][seccion_sel]
    CF = st.session_state.params["global"]["CF"]

    cota = p["L_i"] + p["R_i"]
    x_max = max(int(cota * 1.6), int(p["L_i"] * 1.6), 20)
    df_curva = curva_riesgo_costo(p, CF, x_max)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            f"P_{seccion_sel}(Δ) — Probabilidad incondicional de falla acumulada (Weibull)",
            f"g_{seccion_sel}(Δ) — Costo esperado del arco",
        ),
    )

    for row in (1, 2):
        fig.add_vrect(x0=0, x1=cota, fillcolor="#dcfce7", opacity=0.35, line_width=0, row=row, col=1)
        fig.add_vrect(x0=cota, x1=x_max, fillcolor="#fee2e2", opacity=0.35, line_width=0, row=row, col=1)
        fig.add_vline(x=p["L_i"], line_dash="dash", line_color="#64748b", row=row, col=1)
        fig.add_vline(x=cota, line_dash="dot", line_color="#b91c1c", row=row, col=1)

    fig.add_trace(go.Scatter(x=df_curva["delta"], y=df_curva["P"], mode="lines",
                              name="P(Δ)", line=dict(color="#1d4ed8", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_curva["delta"], y=df_curva["g_intermedio"], mode="lines",
                              name="g(Δ), nodo intermedio t ∈ T (con CS_i)", line=dict(color="#d97706", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_curva["delta"], y=df_curva["g_terminal"], mode="lines",
                              name="g(Δ), nodo terminal t = ∞ (solo riesgo)", line=dict(color="#059669", width=2, dash="dash")),
                  row=2, col=1)

    fig.update_yaxes(title_text="Probabilidad", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Costo ($)", row=2, col=1)
    fig.update_xaxes(title_text="Edad del ciclo Δ (semanas)", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=620,
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=40, l=50, r=20),
    )

    st.plotly_chart(fig, width="stretch")
    st.markdown(
        f"""
        <div style="font-size: 12px; color: #475569; margin-top: 6px;">
          Zona verde: Región admisible Δ ≤ L_i + R_i = {cota:g} semanas (Ec. 6). &nbsp;|&nbsp; 
          Zona roja: Poda topológica de arcos. &nbsp;|&nbsp; 
          Línea punteada gris: L_i = {p['L_i']:g} sem. &nbsp;|&nbsp; Línea punteada roja: L_i + R_i.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# TAB 5 — Resumen del Modelo
# ===========================================================================
with tab_resumen:
    st.markdown(
        """
        <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;">
          <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Dimensión y Variables de la Red de Optimización (Sección 6.1)</div>
          <div style="font-size: 13px; color: #64748b;">Recuento de arcos admisibles |A_i| y variables de decisión para el resolvedor MILP.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filas_resumen = []
    total_arcos = 0
    for i in SECCIONES:
        p = st.session_state.params["secciones"][i]
        A_i = generar_arcos_seccion(i, p["e_i"], p["L_i"], p["R_i"], H)
        cota_teorica = (H + 1) * (p["L_i"] + p["R_i"] + 1)
        filas_resumen.append(
            {"Sección": i, "|A_i|": len(A_i), "Cota teórica (H+1)(L_i+R_i+1)": round(cota_teorica)}
        )
        total_arcos += len(A_i)

    df_resumen = pd.DataFrame(filas_resumen).set_index("Sección")
    st.dataframe(df_resumen, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Variables de flujo x_ij", f"{total_arcos:,}")
    c2.metric("Variables de intervención z_it (9 × 156)", f"{9 * H:,}")
    c3.metric("Variables de detención y_t + w_k", f"{H:,}")

    st.markdown("<div style='margin-top: 24px; margin-bottom: 16px; border-top: 1px solid #e2e8f0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Estructura Actual de Parámetros (JSON)</div>", unsafe_allow_html=True)
    st.json(st.session_state.params, expanded=False)
