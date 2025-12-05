import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO VISUAL (CSS FUTURISTA)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Reporte Marketing 2025",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Paleta de colores vibrante
COLOR_PALETTE = ["#38bdf8", "#0ea5e9", "#6366f1", "#22d3ee", "#8b5cf6", "#ec4899", "#f472b6"]

# Inyección de CSS
st.markdown(
    """
    <style>
    /* VARIABLES GLOBALES PARA ELIMINAR EL ROJO POR DEFECTO */
    :root {
        --primary-color: #38bdf8;
        --background-color: #0B0F19;
        --secondary-background-color: #111827;
        --text-color: #f0f9ff;
        --font: 'Inter', sans-serif;
    }

    /* Fondo general */
    .stApp {
        background-color: #0B0F19;
    }
    
    /* SOBRESCRIBIR EL ACENTO ROJO/NARANJA DE STREAMLIT EN WIDGETS */
    div.stDateInput > div > div > input {
        color: #38bdf8;
    }
    div.stMultiSelect span[data-baseweb="tag"] {
        background-color: #1e293b !important;
        border: 1px solid #38bdf8 !important;
    }
    div.stMultiSelect div[data-baseweb="select"] {
        border-color: #38bdf8 !important;
    }
    /* Checkbox y Radios */
    .stCheckbox div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }
    span[data-baseweb="checkbox"] div {
        background-color: #38bdf8 !important;
    }

    /* Estilo para las métricas (KPI Cards) */
    div[data-testid="metric-container"] {
        display: none; 
    }

    .kpi-card {
        background: linear-gradient(145deg, #111827, #1f2937);
        border: 1px solid #6366f1;
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.1), 0 2px 4px -1px rgba(99, 102, 241, 0.06);
        margin-bottom: 10px;
        height: 100%;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #8b5cf6;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .kpi-value {
        color: #f0f9ff;
        font-size: 1.8rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
    .kpi-sub {
        color: #38bdf8;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* Títulos */
    h1, h2, h3, h4 {
        color: #f0f9ff !important;
        font-family: 'Inter', sans-serif;
    }
    p, label, .stMarkdown, .stRadio label, .stCheckbox label {
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

    # Normalización Unidades
    df["origen_unidad_raw"] = df.get("origen_unidad_de_negocio_asignada", "").fillna("").astype(str)
    df["deal_unidad_raw"] = df.get("deal_unidad_de_negocio_asignada", "").fillna("").astype(str)
    
    df["origen_unidad_norm"] = df.apply(
        lambda row: normalizar_unidad(row["origen_unidad_raw"], row["pipeline_marketing"]), axis=1
    )
    df["deal_unidad_norm"] = df.apply(
        lambda row: normalizar_unidad(row["deal_unidad_raw"], row["pipeline_comercial"]), axis=1
    )

    # Rellenar textos clave
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

# Deduplicar origen para filtros y métricas base
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
# 5. HEADER Y KPI'S IMPACTO (SOLO CONTEOS Y DURACIÓN)
# -----------------------------------------------------------------------------
st.title("🚀 Reporte de Leads Marketing 2025")

# --- Lógica KPIs Ganados ---
df_origen_ganados = df_origen_f[df_origen_f["estado_marketing"] == "Ganado"].copy()
ids_ganados = df_origen_ganados["origen_deal_id"].unique()
df_post_de_ganados = df_post_f_unique[df_post_f_unique["origen_deal_id"].isin(ids_ganados)].copy()

# Cálculos
w_count = df_origen_ganados["origen_deal_id"].nunique()

# Monto USD (Marketing Ganado - Dato seguro)
w_amount_usd = df_origen_ganados["origen_amount"].sum()

# KPI solicitado: "Negocios posteriores creados" = Suma de 'origen_duracion_meses'
val_kpi_posterior = df_origen_ganados["origen_duracion_meses"].sum()

st.subheader("🏆 Impacto Comercial (Origen Ganado)")
c_imp1, c_imp2, c_imp3 = st.columns(3)

with c_imp1: display_kpi("Deals Ganados (Mkt)", f"{w_count}", "Cierre Ganado")
with c_imp2: display_kpi("Monto Ganado (USD)", f"${w_amount_usd:,.2f}", "Total Pipeline Marketing")
with c_imp3: display_kpi("Negocios posteriores creados", f"{val_kpi_posterior:,.1f}", "Suma Negocios Posteriores")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. KPI'S GENERALES (SOLO VOLUMEN)
# -----------------------------------------------------------------------------
st.subheader("📡 Métricas Generales (Todo el Pipeline)")

kpi_mkt_count = df_origen_f["origen_deal_id"].nunique()
kpi_post_total_unique = df_post_f_unique["deal_id"].nunique()

col_gen1, col_gen2 = st.columns(2)
with col_gen1: display_kpi("Total Marketing", f"{kpi_mkt_count:,}", "Todos los estados")
with col_gen2: display_kpi("Total Posteriores", f"{kpi_post_total_unique:,}", "Todos los estados")

# -----------------------------------------------------------------------------
# 7. GRÁFICA: NEGOCIOS POR ETAPA MKT (FUNNEL)
# -----------------------------------------------------------------------------
st.markdown("### 🧬 Negocios de Marketing por Etapa")
if not df_origen_f.empty:
    etapa_counts = (
        df_origen_f.groupby("etapa_marketing")["origen_deal_id"]
        .nunique()
        .reset_index(name="num_deals")
        .sort_values("num_deals", ascending=False)
    )
    
    # Escala
    max_val = etapa_counts["num_deals"].max()
    
    fig_etapas = px.bar(
        etapa_counts, x="etapa_marketing", y="num_deals",
        text="num_deals",
        color_discrete_sequence=[COLOR_PALETTE[0]]
    )
    
    # Configuración de etiquetas fuera
    fig_etapas.update_traces(
        textposition='outside', 
        textfont=dict(color='white'),
        cliponaxis=False
    )
    
    fig_etapas.update_layout(
        xaxis_title="Etapa", 
        yaxis_title="Deals",
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, max_val * 1.2]) 
    )
    st.plotly_chart(fig_etapas, use_container_width=True)
else:
    st.info("Sin datos de marketing.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 8. DISTRIBUCIÓN DE ESTADOS (TODOS LOS PIPES)
# -----------------------------------------------------------------------------
st.subheader("🧩 Distribución de Estados")

col_est1, col_est2 = st.columns(2)

with col_est1:
    st.markdown("**Estados de Marketing (Pipeline Marketing)**")
    if not df_origen_f.empty:
        mkt_estado = (
            df_origen_f.groupby(["pipeline_marketing", "estado_marketing"])["origen_deal_id"]
            .nunique()
            .reset_index(name="num_deals")
        )
        fig_mkt = px.bar(
            mkt_estado, x="pipeline_marketing", y="num_deals",
            color="estado_marketing", barmode="stack",
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_mkt.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Pipeline", yaxis_title="Deals"
        )
        fig_mkt.update_traces(marker=dict(opacity=1.0))
        st.plotly_chart(fig_mkt, use_container_width=True)
    else:
        st.info("No hay negocios de origen.")

with col_est2:
    st.markdown("**Estados Comerciales (Todos los Pipelines Comerciales)**")
    if not df_post_f_unique.empty:
        # Agrupamos por pipeline comercial y estado
        com_estado = (
            df_post_f_unique.groupby(["pipeline_comercial", "estado_comercial"])["deal_id"]
            .nunique()
            .reset_index(name="num_deals")
        )
        
        fig_com = px.bar(
            com_estado, x="pipeline_comercial", y="num_deals",
            color="estado_comercial", barmode="stack",
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_com.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Pipeline", yaxis_title="Deals"
        )
        fig_com.update_traces(marker=dict(opacity=1.0))
        st.plotly_chart(fig_com, use_container_width=True)
    else:
        st.info("No hay negocios posteriores.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 9. EVOLUCIÓN TEMPORAL
# -----------------------------------------------------------------------------
st.subheader("📅 Evolución Temporal")

col_time1, col_time2 = st.columns(2)

with col_time1:
    st.markdown("**Negocios de marketing por mes**")
    if not df_origen_f.empty:
        tmp = df_origen_f.copy()
        tmp["mes"] = tmp["origen_created_date"].dt.to_period("M").dt.to_timestamp()
        evo = tmp.groupby("mes")["origen_deal_id"].nunique().reset_index(name="num_negocios")
        
        fig_evo = px.bar(
            evo, x="mes", y="num_negocios",
            color_discrete_sequence=[COLOR_PALETTE[1]]
        )
        fig_evo.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Mes", yaxis_title="Deals Mkt")
        st.plotly_chart(fig_evo, use_container_width=True)
    else:
        st.info("Sin datos.")

with col_time2:
    st.markdown("**Negocios posteriores por mes**")
    if not df_post_f_unique.empty:
        tmp = df_post_f_unique.copy()
        tmp["mes"] = tmp["deal_created_date"].dt.to_period("M").dt.to_timestamp()
        evo2 = tmp.groupby("mes")["deal_id"].nunique().reset_index(name="num_negocios")

        fig_evo2 = px.bar(
            evo2, x="mes", y="num_negocios",
            color_discrete_sequence=[COLOR_PALETTE[3]]
        )
        fig_evo2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Mes", yaxis_title="Deals Post")
        st.plotly_chart(fig_evo2, use_container_width=True)
    else:
        st.info("Sin datos.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 10. POR UNIDAD DE NEGOCIO (SIN MIX / SIN PRODUCTO)
# -----------------------------------------------------------------------------
st.subheader("🥧 Distribución por Unidad de Negocio (Marketing)")

if not df_origen_f.empty:
    mix_unidad = df_origen_f.groupby("origen_unidad_norm")["origen_deal_id"].nunique().reset_index(name="count")
    fig_mix_u = px.pie(
        mix_unidad, names="origen_unidad_norm", values="count",
        hole=0.4, color_discrete_sequence=COLOR_PALETTE
    )
    fig_mix_u.update_layout(template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig_mix_u, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 11. TABLA RESUMEN POR NEGOCIO ORIGEN (SIN MONTOS)
# -----------------------------------------------------------------------------
st.subheader("📌 Resumen Detallado por Negocio Marketing")

if not df_origen_f.empty:
    # Preparar tabla base
    base = df_origen_f[["origen_deal_id", "origen_deal_name", "origen_created_date", "pipeline_marketing", 
                        "etapa_marketing", "estado_marketing", "origen_unidad_norm"]].copy()
    
    # Agregar datos posteriores (solo conteo)
    agg_post = df_post_f.groupby("origen_deal_id").agg(
        num_post=("deal_id", "nunique")
    ).reset_index()

    resumen = base.merge(agg_post, on="origen_deal_id", how="left")
    resumen["num_post"] = resumen["num_post"].fillna(0).astype(int)

    st.dataframe(
        resumen.sort_values("num_post", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "origen_created_date": st.column_config.DateColumn("Fecha"),
        }
    )
else:
    st.info("Sin datos.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 12. INSIGHTS VISUALES (SIN CLASIFICACIÓN MONEDA)
# -----------------------------------------------------------------------------
st.subheader("📈 Insights Visuales")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Deals Posteriores por Unidad Destino**")
    if not df_post_f_unique.empty:
        # Agrupación solo por unidad, sin moneda
        deals_u = df_post_f_unique.groupby("deal_unidad_norm")["deal_id"].nunique().reset_index(name="count").sort_values("count", ascending=False)
        fig_ins1 = px.bar(
            deals_u, x="deal_unidad_norm", y="count", 
            text="count",
            color_discrete_sequence=[COLOR_PALETTE[2]]
        )
        fig_ins1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Unidad", yaxis_title="Num Deals")
        st.plotly_chart(fig_ins1, use_container_width=True)
    else:
        st.info("Sin datos.")

with col_g2:
    st.markdown("**Cantidad de Deals Posteriores por Pipeline**")
    if not df_post_f_unique.empty:
        deals_pipe = df_post_f_unique.groupby("pipeline_comercial")["deal_id"].nunique().reset_index(name="count").sort_values("count", ascending=False)
        fig_ins2 = px.bar(
            deals_pipe, x="pipeline_comercial", y="count",
            text="count",
            color_discrete_sequence=[COLOR_PALETTE[4]]
        )
        fig_ins2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Pipeline", yaxis_title="Num Deals")
        st.plotly_chart(fig_ins2, use_container_width=True)
    else:
        st.info("Sin datos.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 13. SANKEY
# -----------------------------------------------------------------------------
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

st.markdown("---")

# -----------------------------------------------------------------------------
# 14. DESGLOSE (SOLO CONTEO DE DEALS - SIN MONTOS)
# -----------------------------------------------------------------------------
st.subheader("📊 Desglose por pipeline y etapa comercial")

if df_post_f.empty:
    st.info("No hay datos posteriores con los filtros actuales.")
else:
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Top pipelines comerciales (por cantidad)**")
        top_pipelines = (
            df_post_f.groupby("pipeline_comercial")
            .agg(
                num_deals=("deal_id", "nunique"),
            )
            .reset_index()
            .sort_values("num_deals", ascending=False)
        )
        st.dataframe(
            top_pipelines,
            use_container_width=True,
            hide_index=True,
        )

    with col_t2:
        st.markdown("**Detalle de etapas dentro de un pipeline comercial**")
        pipelines_disp = sorted(df_post_f["pipeline_comercial"].unique())
        pipeline_sel = st.selectbox(
            "Selecciona pipeline comercial",
            options=pipelines_disp,
        )

        df_etapas = df_post_f[df_post_f["pipeline_comercial"] == pipeline_sel]

        etapas = (
            df_etapas.groupby("etapa_comercial")
            .agg(
                num_deals=("deal_id", "nunique"),
            )
            .reset_index()
            .sort_values("num_deals", ascending=False)
        )

        st.dataframe(
            etapas,
            use_container_width=True,
            hide_index=True,
        )

st.markdown("<br><br><div style='text-align: center; color: #475569;'>Desarrollado por Héctor Plascencia | 2025 🚀</div>", unsafe_allow_html=True)


