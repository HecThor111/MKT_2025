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
BG_COLOR = "#0e1117"  # Fondo oscuro de Streamlit por defecto

# Inyección de CSS para estilo "Capsula Espacial" y Dark Mode avanzado
st.markdown(
    """
    <style>
    /* Fondo general */
    .stApp {
        background-color: #0B0F19;
    }
    
    /* Estilo para las métricas (KPI Capsules) */
    div[data-testid="metric-container"] {
        display: none; /* Ocultamos la métrica nativa para usar nuestras cards HTML */
    }

    .kpi-card {
        background: linear-gradient(145deg, #111827, #1f2937);
        border: 1px solid #38bdf8;
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .kpi-value {
        color: #f0f9ff;
        font-size: 1.8rem;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(255,255,255,0.3);
    }
    .kpi-sub {
        color: #38bdf8;
        font-size: 0.8rem;
    }

    /* Títulos y textos */
    h1, h2, h3 {
        color: #f0f9ff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    p, label, .stMarkdown {
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
# 2. FUNCIONES HELPERS Y NORMALIZACIÓN
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
    """
    Normaliza la unidad de negocio basada en palabras clave en la unidad o el pipeline.
    """
    # Usar texto de unidad si existe, sino pipeline
    texto_base = str(unidad_raw) if unidad_raw and unidad_raw.lower() not in ["nan", "sin dato", ""] else str(pipeline_label)
    texto = texto_base.lower()

    if any(x in texto for x in ["cloud", "aws", "ai ", "artificial"]):
        return "Cloud & AI Solutions"
    if any(x in texto for x in ["data", "analytics"]):
        return "Data & Analytics"
    if any(x in texto for x in ["enterprise", "enterprises", "usa", "calls", "government", "pdm"]):
        return "Enterprise Solutions"
    
    # Si no matchea y había unidad original, devolverla, si no "Sin Unidad"
    if unidad_raw and unidad_raw.lower() not in ["nan", "sin dato", ""]:
        return unidad_raw
    return "Sin Unidad"

def display_kpi(label, value, sub_text=""):
    """
    Genera una tarjeta HTML personalizada para los KPIs.
    """
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub_text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. CARGA Y PROCESAMIENTO DE DATOS
# -----------------------------------------------------------------------------
CSV_FILE = "bd_final.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {path}")
        return pd.DataFrame()

    # --- Limpieza de Fechas ---
    for col in ["origen_created_date", "deal_created_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Limpieza de Montos y Numéricos ---
    for col in ["origen_amount", "deal_amount", "origen_duracion_meses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --- Limpieza de Textos y Normalización de Pipeline/Etapa ---
    df["pipeline_marketing"] = df.get("origen_pipeline_label", "").fillna("").astype(str)
    df["pipeline_comercial"] = df.get("deal_pipeline_label", "").fillna("").astype(str)
    
    df["etapa_marketing"] = df.get("origen_dealstage_label", "").fillna("").astype(str)
    df["etapa_comercial"] = df.get("deal_dealstage_label", "").fillna("").astype(str)

    # --- Estados ---
    df["estado_marketing"] = df["etapa_marketing"].apply(clasificar_estado_etapa)
    df["estado_comercial"] = df["etapa_comercial"].apply(clasificar_estado_etapa)

    # --- Monedas (Limpieza estricta) ---
    # Asumimos que si está vacío es MXN o lo marcamos para no sumar erróneamente.
    # Normalizamos a mayúsculas y quitamos espacios
    if "origen_currency" in df.columns:
        df["origen_currency"] = df["origen_currency"].fillna("USD").astype(str).str.upper().str.strip()
    else:
        df["origen_currency"] = "USD" # Default marketing

    if "deal_currency" in df.columns:
        df["deal_currency"] = df["deal_currency"].fillna("MXN").astype(str).str.upper().str.strip()
    else:
        df["deal_currency"] = "MXN"

    # --- Normalización de Unidades de Negocio ---
    # Rellenar nulos previos a la función
    df["origen_unidad_raw"] = df.get("origen_unidad_de_negocio_asignada", "").fillna("").astype(str)
    df["deal_unidad_raw"] = df.get("deal_unidad_de_negocio_asignada", "").fillna("").astype(str)
    
    # Aplicar función row by row
    df["origen_unidad_norm"] = df.apply(
        lambda row: normalizar_unidad(row["origen_unidad_raw"], row["pipeline_marketing"]), axis=1
    )
    df["deal_unidad_norm"] = df.apply(
        lambda row: normalizar_unidad(row["deal_unidad_raw"], row["pipeline_comercial"]), axis=1
    )

    # --- Relleno de columnas de texto clave ---
    cols_text = ["origen_origen_del_negocio", "origen_producto_catalogo"]
    for c in cols_text:
        if c in df.columns:
            df[c] = df[c].fillna("Sin dato").astype(str)
            
    return df

df = load_data(CSV_FILE)

if df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# 4. LÓGICA DE FILTRADO (Separación Origen vs Posteriores)
# -----------------------------------------------------------------------------
# Dataframes base
df_origen_all = df[df["tipo_negocio"] == "origen_marketing"].copy()
df_post_all = df[df["tipo_negocio"] == "posterior_contacto"].copy()

# Versión deduplicada de Origen (para conteos correctos de marketing)
df_origen_unique_all = df_origen_all.sort_values("origen_created_date").drop_duplicates(subset=["origen_deal_id"])

# --- SIDEBAR ---
st.sidebar.title("🚀 Filtros de Misión")

# 1. Filtro Fecha (Origen)
min_d, max_d = df_origen_unique_all["origen_created_date"].min(), df_origen_unique_all["origen_created_date"].max()
if pd.isna(min_d): min_d, max_d = pd.Timestamp.now(), pd.Timestamp.now()

dates = st.sidebar.date_input(
    "Fecha Creación (Marketing)",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)
start_date, end_date = dates if isinstance(dates, tuple) and len(dates) == 2 else (min_d, max_d)

# 2. Unidad de Negocio (Marketing - Normalizada)
unidades_opts = sorted(df_origen_unique_all["origen_unidad_norm"].unique())
sel_unidades = st.sidebar.multiselect("Unidad de Negocio (Marketing)", options=unidades_opts, default=unidades_opts)

# 3. Origen del Negocio
origen_opts = sorted(df_origen_unique_all["origen_origen_del_negocio"].unique())
sel_origenes = st.sidebar.multiselect("Origen del Negocio", options=origen_opts, default=origen_opts)

# --- APLICACIÓN DE FILTROS ---
# Filtramos Origen Primero
mask_origen = (
    (df_origen_unique_all["origen_created_date"].dt.date >= start_date) &
    (df_origen_unique_all["origen_created_date"].dt.date <= end_date) &
    (df_origen_unique_all["origen_unidad_norm"].isin(sel_unidades)) &
    (df_origen_unique_all["origen_origen_del_negocio"].isin(sel_origenes))
)
df_origen_f = df_origen_unique_all[mask_origen].copy()

# Filtramos Posteriores basados en los IDs de origen resultantes
ids_origen_validos = df_origen_f["origen_deal_id"].unique()
df_post_f = df_post_all[df_post_all["origen_deal_id"].isin(ids_origen_validos)].copy()

# Deduplicamos posteriores para cálculos de monto (para evitar doble contabilidad si hubiera joins raros)
df_post_f_unique = df_post_f.sort_values("deal_created_date").drop_duplicates(subset=["deal_id"])

# -----------------------------------------------------------------------------
# 5. DASHBOARD LAYOUT
# -----------------------------------------------------------------------------

st.title("🌌 iNBest.marketing | Galactic Dashboard")
st.markdown(f"**Periodo:** {start_date} al {end_date} | **Registros Marketing:** {len(df_origen_f)}")

# --- FILA 1: KPIs GENERALES ---
st.markdown("### 📡 Métricas Generales")
c1, c2, c3, c4 = st.columns(4)

# Cálculos
kpi_mkt_count = df_origen_f["origen_deal_id"].nunique()
kpi_post_count = df_post_f_unique["deal_id"].nunique()

# Suma separada de monedas (Posteriores)
kpi_post_usd = df_post_f_unique[df_post_f_unique["deal_currency"] == "USD"]["deal_amount"].sum()
kpi_post_mxn = df_post_f_unique[df_post_f_unique["deal_currency"] == "MXN"]["deal_amount"].sum()

with c1: display_kpi("Negocios Marketing", f"{kpi_mkt_count:,}", "Pipeline iNBest")
with c2: display_kpi("Negocios Posteriores", f"{kpi_post_count:,}", "Asociados únicos")
with c3: display_kpi("Monto Posterior (USD)", f"${kpi_post_usd:,.2f}", "Solo deals en USD")
with c4: display_kpi("Monto Posterior (MXN)", f"${kpi_post_mxn:,.2f}", "Solo deals en MXN")

# --- FILA 2: NEGOCIOS DE MARKETING POR ETAPA ---
st.markdown("### 🧬 Funnel de Marketing")
if not df_origen_f.empty:
    etapas_count = df_origen_f["etapa_marketing"].value_counts().reset_index()
    etapas_count.columns = ["etapa", "count"]
    
    fig_funnel = px.bar(
        etapas_count, x="etapa", y="count", 
        color_discrete_sequence=COLOR_PALETTE,
        text_auto=True,
        title="Negocios por Etapa de Marketing"
    )
    fig_funnel.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_funnel, use_container_width=True)
else:
    st.info("No hay datos de marketing para mostrar el funnel.")

st.markdown("---")

# --- FILA 3: PERFORMANCE GANADOS (MARKETING) ---
st.markdown("### 🏆 Performance: Negocios GANADOS en Marketing (Base USD)")

# Filtrar solo ganados en origen
df_origen_ganados = df_origen_f[df_origen_f["estado_marketing"] == "Ganado"].copy()
ids_ganados = df_origen_ganados["origen_deal_id"].unique()
df_post_de_ganados = df_post_f_unique[df_post_f_unique["origen_deal_id"].isin(ids_ganados)].copy()

col_w1, col_w2, col_w3, col_w4, col_w5, col_w6 = st.columns(6)

# Cálculos Ganados
w_count = df_origen_ganados["origen_deal_id"].nunique()
w_amount_usd = df_origen_ganados[df_origen_ganados["origen_currency"] == "USD"]["origen_amount"].sum()
w_post_count = df_post_de_ganados["deal_id"].nunique()

# Factor Multiplicación (Solo comparamos USD con USD posterior para consistencia matemática o sumamos todo convertido?
# El prompt dice: "(suma posterior USD para esos ganados) / (suma origen USD ganados)"
w_post_amt_usd = df_post_de_ganados[df_post_de_ganados["deal_currency"] == "USD"]["deal_amount"].sum()
w_factor = (w_post_amt_usd / w_amount_usd) if w_amount_usd > 0 else 0

# Conversión: Cuántos ganados tienen al menos 1 posterior
ids_con_posterior = df_post_de_ganados["origen_deal_id"].unique()
w_con_hijo = len(set(ids_ganados).intersection(set(ids_con_posterior)))
w_conversion_pct = (w_con_hijo / w_count * 100) if w_count > 0 else 0

# Tiempo promedio
# Merge fechas
df_min_fecha_post = df_post_de_ganados.groupby("origen_deal_id")["deal_created_date"].min().reset_index()
df_tiempos = df_origen_ganados[["origen_deal_id", "origen_created_date"]].merge(df_min_fecha_post, on="origen_deal_id")
df_tiempos["dias_diff"] = (df_tiempos["deal_created_date"] - df_tiempos["origen_created_date"]).dt.days
w_dias_prom = df_tiempos["dias_diff"].mean()
w_dias_str = f"{w_dias_prom:.1f} días" if not pd.isna(w_dias_prom) else "N/A"

with col_w1: display_kpi("Mkt Ganados", f"{w_count}", "Count")
with col_w2: display_kpi("Monto Mkt (USD)", f"${w_amount_usd:,.0f}", "Suma origen")
with col_w3: display_kpi("Derivados", f"{w_post_count}", "Deals posteriores")
with col_w4: display_kpi("Multiplicador", f"{w_factor:.2f}x", "Posterior USD / Origen USD")
with col_w5: display_kpi("Conv. Efectiva", f"{w_conversion_pct:.1f}%", "Con ≥1 posterior")
with col_w6: display_kpi("Velocidad", w_dias_str, "Promedio 1er posterior")

st.markdown("---")

# --- FILA 4: GRÁFICAS COMPARATIVAS Y DISTRIBUCIONES ---
c_g1, c_g2 = st.columns(2)

with c_g1:
    st.subheader("📊 Estados de Marketing por Pipeline")
    if not df_origen_f.empty:
        fig_stack = px.histogram(
            df_origen_f, x="pipeline_marketing", color="estado_marketing",
            barmode="group", color_discrete_sequence=COLOR_PALETTE,
            title="Distribución de Estados"
        )
        fig_stack.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Sin datos.")

with c_g2:
    st.subheader("💰 Distribución Estado Comercial (Monto Global)")
    if not df_post_f_unique.empty:
        # Sumamos montos sin importar moneda solo para ver la "torta" de estados (warning: mixing currencies visually but logically separate aggregation)
        # El prompt dice: "suma de deal_amount (independientemente de la moneda)"
        fig_pie = px.pie(
            df_post_f_unique, names="estado_comercial", values="deal_amount",
            color_discrete_sequence=COLOR_PALETTE, hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Sin negocios posteriores.")

# --- FILA 5: EVOLUCIÓN TEMPORAL ---
st.subheader("📅 Evolución Mensual")
t1, t2 = st.columns(2)

with t1:
    if not df_origen_f.empty:
        df_origen_f["mes"] = df_origen_f["origen_created_date"].dt.to_period("M").dt.to_timestamp()
        evol_mkt = df_origen_f.groupby("mes")["origen_deal_id"].nunique().reset_index(name="count")
        fig_ev1 = px.bar(evol_mkt, x="mes", y="count", title="Marketing: Nuevos Negocios", color_discrete_sequence=[COLOR_PALETTE[0]])
        fig_ev1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ev1, use_container_width=True)

with t2:
    if not df_post_f_unique.empty:
        df_post_f_unique["mes"] = df_post_f_unique["deal_created_date"].dt.to_period("M").dt.to_timestamp()
        evol_post = df_post_f_unique.groupby("mes")["deal_id"].nunique().reset_index(name="count")
        fig_ev2 = px.bar(evol_post, x="mes", y="count", title="Posterior: Nuevos Negocios", color_discrete_sequence=[COLOR_PALETTE[2]])
        fig_ev2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ev2, use_container_width=True)

# --- FILA 6: MIX & INSIGHTS ---
st.subheader("🔎 Insights de Unidades y Pipelines")
i1, i2 = st.columns(2)

with i1:
    st.markdown("**Monto Posterior por Unidad Destino (Separado por Moneda)**")
    if not df_post_f_unique.empty:
        # Agrupación estricta por moneda
        ins_unidad = df_post_f_unique.groupby(["deal_unidad_norm", "deal_currency"])["deal_amount"].sum().reset_index()
        fig_ins1 = px.bar(
            ins_unidad, x="deal_unidad_norm", y="deal_amount", color="deal_currency",
            barmode="group", color_discrete_sequence=["#38bdf8", "#ec4899"], # Azul para uno, Rosa para otro
            title="Ingresos generados (USD vs MXN)"
        )
        fig_ins1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ins1, use_container_width=True)
    else:
        st.info("No hay data posterior.")

with i2:
    st.markdown("**Mix de Origen (Marketing)**")
    if not df_origen_f.empty:
        col_mix_pie = "origen_unidad_norm"
        fig_mix = px.pie(
            df_origen_f, names=col_mix_pie, title="Marketing por Unidad Normalizada",
            color_discrete_sequence=COLOR_PALETTE, hole=0.3
        )
        fig_mix.update_layout(template="plotly_dark")
        st.plotly_chart(fig_mix, use_container_width=True)

# --- FILA 7: SANKEY DIAGRAM ---
st.markdown("---")
st.subheader("🔀 Flujo: Origen del Negocio ➡ Unidad Destino")

# Opciones Sankey
check_sankey_mkt = st.checkbox("Filtrar solo origen iNBest.marketing (estricto)", value=True)
metric_sankey = st.radio("Métrica del flujo:", ["Cantidad Negocios", "Monto Total (Mix)"], horizontal=True)

# Preparar datos Sankey
df_sankey = df_post_f.copy() # Usamos la versión con duplicados de origen permitidos para mapear flujo completo
if check_sankey_mkt:
    df_sankey = df_sankey[df_sankey["pipeline_marketing"] == "iNBest.marketing"]

if not df_sankey.empty:
    # Source: Origen del negocio marketing
    # Target: Unidad normalizada posterior
    
    # Agrupar
    if metric_sankey == "Cantidad Negocios":
        sankey_g = df_sankey.groupby(["origen_origen_del_negocio", "deal_unidad_norm"])["deal_id"].nunique().reset_index(name="value")
    else:
        sankey_g = df_sankey.groupby(["origen_origen_del_negocio", "deal_unidad_norm"])["deal_amount"].sum().reset_index(name="value")
    
    # Crear índices para nodos
    all_sources = list(sankey_g["origen_origen_del_negocio"].unique())
    all_targets = list(sankey_g["deal_unidad_norm"].unique())
    all_nodes = all_sources + all_targets
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Colores nodos
    node_colors = ["#38bdf8"] * len(all_sources) + ["#ec4899"] * len(all_targets)
    
    link_source = sankey_g["origen_origen_del_negocio"].map(node_map).tolist()
    link_target = sankey_g["deal_unidad_norm"].map(node_map).tolist()
    link_value = sankey_g["value"].tolist()
    
    fig_san = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes, color=node_colors
        ),
        link=dict(
            source=link_source, target=link_target, value=link_value,
            color="rgba(99, 102, 241, 0.2)" # Morado translúcido
        )
    )])
    fig_san.update_layout(title_text="Flujo de Conversión", font_size=12, template="plotly_dark", height=600)
    st.plotly_chart(fig_san, use_container_width=True)

else:
    st.warning("No hay datos suficientes para generar el diagrama de flujo.")

# --- FILA 8: TABLAS DETALLADAS ---
st.markdown("---")
st.subheader("📑 Detalles de Negocios")

tab1, tab2 = st.tabs(["Resumen por Marketing Deal", "Pipelines Comerciales"])

with tab1:
    if not df_origen_f.empty:
        # Preparamos tabla resumen
        # Agregamos data de posteriores
        agg_post = df_post_f.groupby("origen_deal_id").agg(
            cant_post=("deal_id", "nunique"),
            suma_post_amt=("deal_amount", "sum"), # Nota: esto suma mix monedas solo para ref rápida
            post_monedas=("deal_currency", lambda x: ", ".join(x.unique()))
        ).reset_index()
        
        tbl_resumen = df_origen_f[[
            "origen_deal_id", "pipeline_marketing", "etapa_marketing", "estado_marketing", 
            "origen_amount", "origen_currency", "origen_unidad_norm", "origen_created_date"
        ]].copy()
        
        tbl_final = tbl_resumen.merge(agg_post, on="origen_deal_id", how="left").fillna(0)
        
        # Formato visual simple
        st.dataframe(
            tbl_final.sort_values("cant_post", ascending=False),
            use_container_width=True,
            column_config={
                "origen_created_date": st.column_config.DateColumn("Fecha"),
                "origen_amount": st.column_config.NumberColumn("Monto Mkt", format="$%.2f"),
                "suma_post_amt": st.column_config.NumberColumn("Total Post (Mix)", format="$%.2f"),
            }
        )
    else:
        st.info("Sin datos.")

with tab2:
    if not df_post_f_unique.empty:
        st.markdown("**Top Pipelines Comerciales (por Monto)**")
        top_pipe = df_post_f_unique.groupby("pipeline_comercial").agg(
            deals=("deal_id", "nunique"),
            total_amount=("deal_amount", "sum")
        ).reset_index().sort_values("total_amount", ascending=False)
        
        st.dataframe(top_pipe, use_container_width=True, hide_index=True)
    else:
        st.info("Sin datos.")

# Footer
st.markdown("<br><br><div style='text-align: center; color: #475569;'>Desarrollado para iNBest.marketing | 2025 Edition 🚀</div>", unsafe_allow_html=True)
