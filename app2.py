import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO VISUAL (CSS FUTURISTA)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HubSpot Galactic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌌"
)

# Paleta de colores solicitada
COLOR_PALETTE = ["#38bdf8", "#0ea5e9", "#6366f1", "#22d3ee", "#8b5cf6", "#ec4899"]

# Inyección de CSS
st.markdown(
    """
    <style>
    /* Fondo general */
    .stApp {
        background-color: #0B0F19;
    }
    
    /* Estilo para las métricas (KPI Capsules) */
    div[data-testid="metric-container"] {
        display: none; 
    }

    .kpi-card {
        background: linear-gradient(145deg, #111827, #1f2937);
        border: 1px solid #6366f1;
        border-radius: 20px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.15);
        margin-bottom: 10px;
        height: 100%;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .kpi-value {
        color: #f0f9ff;
        font-size: 1.6rem;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(255,255,255,0.2);
    }
    .kpi-sub {
        color: #38bdf8;
        font-size: 0.75rem;
    }

    /* Títulos */
    h1, h2, h3, h4 {
        color: #f0f9ff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    p, label, .stMarkdown, .stRadio label {
        color: #cbd5e1 !important;
    }

    /* Tablas */
    div[data-testid="stDataFrame"] {
        background-color: #111827;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e2937;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. FUNCIONES HELPERS
# -----------------------------------------------------------------------------

def clasificar_estado_etapa(etapa: str) -> str:
    if not isinstance(etapa, str):
        return "Abierto"
    e = etapa.lower()
    if "ganad" in e or "closed won" in e or "cierre ganado" in e:
        return "Ganado"
    if "perd" in e or "lost" in e or "closed lost" in e or "cierre perdido" in e:
        return "Perdido"
    if "descart" in e:
        return "Descartado"
    return "Abierto"

def inferir_moneda(monto: float, moneda_raw: str) -> str:
    """
    Aplica la lógica del usuario:
    1. Si la moneda está explícita (USD/MXN), se respeta.
    2. Si está vacía:
       - Si el monto > 50,000 -> Asumimos MXN (Monto grande).
       - Si el monto <= 50,000 -> Asumimos USD (Monto pequeño).
    """
    m = str(moneda_raw).upper().strip()
    if m in ["USD", "MXN"]:
        return m
    
    # Lógica heurística para vacíos
    if monto > 50000:
        return "MXN"
    else:
        return "USD"

def normalizar_unidad(unidad_raw: str, pipeline_label: str) -> str:
    texto_base = str(unidad_raw) if unidad_raw and unidad_raw.lower() not in ["nan", "sin dato", ""] else str(pipeline_label)
    texto = texto_base.lower()

    if any(x in texto for x in ["cloud", "aws", "ai ", "artificial"]):
        return "Cloud & AI Solutions"
    if any(x in texto for x in ["data", "analytics"]):
        return "Data & Analytics"
    if any(x in texto for x in ["enterprise", "enterprises", "usa", "calls", "government", "pdm"]):
        return "Enterprise Solutions"
    
    if unidad_raw and unidad_raw.lower() not in ["nan", "sin dato", ""]:
        return unidad_raw
    return "Sin Unidad"

def display_kpi(label, value, sub_text=""):
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub_text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. CARGA Y PROCESAMIENTO
# -----------------------------------------------------------------------------
CSV_FILE = "bd_final.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {path}")
        return pd.DataFrame()

    # Fechas
    for col in ["origen_created_date", "deal_created_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Montos
    for col in ["origen_amount", "deal_amount", "origen_duracion_meses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Texto
    df["pipeline_marketing"] = df.get("origen_pipeline_label", "").fillna("").astype(str)
    df["pipeline_comercial"] = df.get("deal_pipeline_label", "").fillna("").astype(str)
    df["etapa_marketing"] = df.get("origen_dealstage_label", "").fillna("").astype(str)
    df["etapa_comercial"] = df.get("deal_dealstage_label", "").fillna("").astype(str)

    # Estados
    df["estado_marketing"] = df["etapa_marketing"].apply(clasificar_estado_etapa)
    df["estado_comercial"] = df["etapa_comercial"].apply(clasificar_estado_etapa)

    # --- LÓGICA DE MONEDA CORREGIDA ---
    # Procesamos row por row para aplicar la heurística
    df["origen_currency_raw"] = df.get("origen_currency", "").fillna("").astype(str)
    df["deal_currency_raw"] = df.get("deal_currency", "").fillna("").astype(str)

    df["origen_currency"] = df.apply(lambda x: inferir_moneda(x["origen_amount"], x["origen_currency_raw"]), axis=1)
    df["deal_currency"] = df.apply(lambda x: inferir_moneda(x["deal_amount"], x["deal_currency_raw"]), axis=1)

    # Normalización Unidades
    df["origen_unidad_raw"] = df.get("origen_unidad_de_negocio_asignada", "").fillna("").astype(str)
    df["deal_unidad_raw"] = df.get("deal_unidad_de_negocio_asignada", "").fillna("").astype(str)
    
    df["origen_unidad_norm"] = df.apply(
        lambda row: normalizar_unidad(row["origen_unidad_raw"], row["pipeline_marketing"]), axis=1
    )
    df["deal_unidad_norm"] = df.apply(
        lambda row: normalizar_unidad(row["deal_unidad_raw"], row["pipeline_comercial"]), axis=1
    )

    # Rellenar textos
    cols_text = ["origen_origen_del_negocio", "origen_producto_catalogo"]
    for c in cols_text:
        if c in df.columns:
            df[c] = df[c].fillna("Sin dato").astype(str)
            
    return df

df = load_data(CSV_FILE)

if df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# 4. FILTROS
# -----------------------------------------------------------------------------
df_origen_all = df[df["tipo_negocio"] == "origen_marketing"].copy()
df_post_all = df[df["tipo_negocio"] == "posterior_contacto"].copy()

df_origen_unique_all = df_origen_all.sort_values("origen_created_date").drop_duplicates(subset=["origen_deal_id"])

st.sidebar.title("🚀 Filtros")

# Filtro Fecha
min_d, max_d = df_origen_unique_all["origen_created_date"].min(), df_origen_unique_all["origen_created_date"].max()
if pd.isna(min_d): min_d, max_d = pd.Timestamp.now(), pd.Timestamp.now()

dates = st.sidebar.date_input("Fecha Creación (Mkt)", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_date, end_date = dates if isinstance(dates, tuple) and len(dates) == 2 else (min_d, max_d)

# Filtros Categoría
unidades_opts = sorted(df_origen_unique_all["origen_unidad_norm"].unique())
sel_unidades = st.sidebar.multiselect("Unidad de Negocio", options=unidades_opts, default=unidades_opts)

origen_opts = sorted(df_origen_unique_all["origen_origen_del_negocio"].unique())
sel_origenes = st.sidebar.multiselect("Origen del Negocio", options=origen_opts, default=origen_opts)

# Aplicar Filtros
mask_origen = (
    (df_origen_unique_all["origen_created_date"].dt.date >= start_date) &
    (df_origen_unique_all["origen_created_date"].dt.date <= end_date) &
    (df_origen_unique_all["origen_unidad_norm"].isin(sel_unidades)) &
    (df_origen_unique_all["origen_origen_del_negocio"].isin(sel_origenes))
)
df_origen_f = df_origen_unique_all[mask_origen].copy()

ids_origen_validos = df_origen_f["origen_deal_id"].unique()
df_post_f = df_post_all[df_post_all["origen_deal_id"].isin(ids_origen_validos)].copy()
df_post_f_unique = df_post_f.sort_values("deal_created_date").drop_duplicates(subset=["deal_id"])

# -----------------------------------------------------------------------------
# 5. DASHBOARD - SECCIÓN SUPERIOR (PROTAGONISMO A GANADOS)
# -----------------------------------------------------------------------------

st.title("🌌 iNBest.marketing | Galactic Dashboard")

# Subconjuntos Ganados
df_origen_ganados = df_origen_f[df_origen_f["estado_marketing"] == "Ganado"].copy()
ids_ganados = df_origen_ganados["origen_deal_id"].unique()
df_post_de_ganados = df_post_f_unique[df_post_f_unique["origen_deal_id"].isin(ids_ganados)].copy()

# Cálculos KPI Impacto
w_count = df_origen_ganados["origen_deal_id"].nunique()
w_post_count = df_post_de_ganados["deal_id"].nunique()

# Suma separada USD/MXN para derivados de ganados
w_post_usd = df_post_de_ganados[df_post_de_ganados["deal_currency"] == "USD"]["deal_amount"].sum()
w_post_mxn = df_post_de_ganados[df_post_de_ganados["deal_currency"] == "MXN"]["deal_amount"].sum()

st.subheader("🏆 Impacto Comercial (Origen Ganado)")
c_imp1, c_imp2, c_imp3, c_imp4 = st.columns(4)

with c_imp1: display_kpi("Deals Ganados (Mkt)", f"{w_count}", "Cierre Ganado")
with c_imp2: display_kpi("Derivados Comerciales", f"{w_post_count}", "Deals generados")
with c_imp3: display_kpi("Monto Generado (USD)", f"${w_post_usd:,.2f}", "De Ganados Mkt")
with c_imp4: display_kpi("Monto Generado (MXN)", f"${w_post_mxn:,.2f}", "De Ganados Mkt")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. SECCIÓN METRICAS GENERALES (Volumen)
# -----------------------------------------------------------------------------
st.subheader("📡 Volumen General del Periodo")

kpi_mkt_count = df_origen_f["origen_deal_id"].nunique()
kpi_post_total_unique = df_post_f_unique["deal_id"].nunique()
# Sumas totales generales (independiente si viene de ganado o no)
total_usd = df_post_f_unique[df_post_f_unique["deal_currency"] == "USD"]["deal_amount"].sum()
total_mxn = df_post_f_unique[df_post_f_unique["deal_currency"] == "MXN"]["deal_amount"].sum()

col_gen1, col_gen2, col_gen3, col_gen4 = st.columns(4)
with col_gen1: display_kpi("Total Marketing", f"{kpi_mkt_count:,}", "Todos los estados")
with col_gen2: display_kpi("Total Posteriores", f"{kpi_post_total_unique:,}", "Todos los estados")
with col_gen3: display_kpi("Total Pipeline (USD)", f"${total_usd:,.2f}", "Suma General")
with col_gen4: display_kpi("Total Pipeline (MXN)", f"${total_mxn:,.2f}", "Suma General")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. GRÁFICAS DE ESTADOS (TODOS LOS PIPELINES)
# -----------------------------------------------------------------------------
st.subheader("📊 Distribución de Estados")
t_est1, t_est2 = st.tabs(["Estados Marketing", "Estados Comerciales"])

with t_est1:
    if not df_origen_f.empty:
        fig_mkt = px.bar(
            df_origen_f, x="pipeline_marketing", color="estado_marketing",
            title="Estados por Pipeline de Marketing",
            color_discrete_sequence=COLOR_PALETTE, barmode="group"
        )
        fig_mkt.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mkt, use_container_width=True)
    else:
        st.info("Sin datos de marketing.")

with t_est2:
    if not df_post_f_unique.empty:
        # Aquí mostramos TODOS los pipelines comerciales encontrados
        count_comercial = df_post_f_unique.groupby(["pipeline_comercial", "estado_comercial"]).size().reset_index(name="count")
        
        fig_com = px.bar(
            count_comercial, x="pipeline_comercial", y="count", color="estado_comercial",
            title="Estados por Pipeline Comercial (Posteriores)",
            color_discrete_sequence=COLOR_PALETTE, barmode="group"
        )
        fig_com.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_com, use_container_width=True)
    else:
        st.info("Sin datos comerciales posteriores.")

# -----------------------------------------------------------------------------
# 8. SECCIÓN DESGLOSE (RESTAURADA)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Desglose por pipeline y etapa comercial")

if df_post_f.empty:
    st.info("No hay datos posteriores con los filtros actuales.")
else:
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Top pipelines comerciales por monto posterior**")
        # Agrupamos por pipeline comercial, sumamos amounts sin distinguir moneda para el ranking 'bruto'
        # Ojo: Sumar USD y MXN directos es raro, pero seguimos la lógica del bloque original
        top_pipelines = (
            df_post_f.groupby("pipeline_comercial")
            .agg(
                num_deals=("deal_id", "nunique"),
                monto_total=("deal_amount", "sum"),
                monto_promedio=("deal_amount", "mean"),
            )
            .reset_index()
            .sort_values("monto_total", ascending=False)
        )
        st.dataframe(
            top_pipelines,
            use_container_width=True,
            hide_index=True,
            column_config={
                "monto_total": st.column_config.NumberColumn("Monto Mix", format="$%.2f"),
                "monto_promedio": st.column_config.NumberColumn("Promedio Mix", format="$%.2f"),
            }
        )

    with col_t2:
        st.markdown("**Detalle de etapas dentro de un pipeline comercial**")
        pipelines_disp = sorted(df_post_f["pipeline_comercial"].unique())
        
        if pipelines_disp:
            pipeline_sel = st.selectbox(
                "Selecciona pipeline comercial",
                options=pipelines_disp,
            )

            df_etapas = df_post_f[df_post_f["pipeline_comercial"] == pipeline_sel]

            etapas = (
                df_etapas.groupby("etapa_comercial")
                .agg(
                    num_deals=("deal_id", "nunique"),
                    monto_total=("deal_amount", "sum"),
                )
                .reset_index()
                .sort_values("monto_total", ascending=False)
            )

            st.dataframe(
                etapas,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "monto_total": st.column_config.NumberColumn("Monto Total", format="$%.2f")
                }
            )
        else:
            st.warning("No hay pipelines comerciales disponibles.")

# -----------------------------------------------------------------------------
# 9. GRÁFICAS ADICIONALES (Mix y Evolución)
# -----------------------------------------------------------------------------
st.markdown("---")
c_mix1, c_mix2 = st.columns(2)

with c_mix1:
    st.subheader("📅 Evolución Mensual (Mkt)")
    if not df_origen_f.empty:
        df_origen_f["mes"] = df_origen_f["origen_created_date"].dt.to_period("M").dt.to_timestamp()
        evol = df_origen_f.groupby("mes").size().reset_index(name="count")
        fig_ev = px.line(evol, x="mes", y="count", markers=True, title="Leads por Mes")
        fig_ev.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        fig_ev.update_traces(line_color="#ec4899")
        st.plotly_chart(fig_ev, use_container_width=True)

with c_mix2:
    st.subheader("🥧 Mix por Unidad (Marketing)")
    if not df_origen_f.empty:
        fig_pie = px.pie(df_origen_f, names="origen_unidad_norm", hole=0.5, color_discrete_sequence=COLOR_PALETTE)
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------------------------------
# 10. SANKEY
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔀 Flujo: Origen ➡ Unidad Destino")

check_sankey_mkt = st.checkbox("Solo origen iNBest.marketing", value=True)
df_sankey = df_post_f.copy()
if check_sankey_mkt:
    df_sankey = df_sankey[df_sankey["pipeline_marketing"] == "iNBest.marketing"]

if not df_sankey.empty:
    sankey_g = df_sankey.groupby(["origen_origen_del_negocio", "deal_unidad_norm"])["deal_id"].nunique().reset_index(name="value")
    
    all_sources = list(sankey_g["origen_origen_del_negocio"].unique())
    all_targets = list(sankey_g["deal_unidad_norm"].unique())
    all_nodes = all_sources + all_targets
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    link_source = sankey_g["origen_origen_del_negocio"].map(node_map).tolist()
    link_target = sankey_g["deal_unidad_norm"].map(node_map).tolist()
    link_value = sankey_g["value"].tolist()
    
    fig_san = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes, color=["#38bdf8"] * len(all_nodes)
        ),
        link=dict(
            source=link_source, target=link_target, value=link_value,
            color="rgba(99, 102, 241, 0.3)"
        )
    )])
    fig_san.update_layout(template="plotly_dark", height=500, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_san, use_container_width=True)
else:
    st.info("No hay datos suficientes para el diagrama de flujo.")
