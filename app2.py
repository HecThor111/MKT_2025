import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------
CSV_FILE = "bd_final.csv"

st.set_page_config(
    page_title="HubSpot 2025 – Marketing → Negocios posteriores",
    layout="wide",
)

# ---- Estilo futurista / tech en azules ----
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #000000 100%);
        color: #e5e7eb;
    }
    .main > div {
        padding-top: 0.5rem;
    }
    h1, h2, h3, h4 {
        color: #e5e7eb !important;
    }
    .metric-card {
        padding: 0.9rem 1.1rem;
        border-radius: 14px;
        border: 1px solid #1e293b;
        background: linear-gradient(135deg, rgba(15,23,42,.95), rgba(37,99,235,.35));
        box-shadow: 0 18px 45px rgba(15,23,42,.85);
    }
    .metric-card .stMetric {
        text-align: left;
    }
    .metric-card label, .stMetric > label {
        color: #93c5fd !important;
        font-size: 0.75rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .metric-card [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .section-label {
        font-size: 0.8rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: #60a5fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 HubSpot – Marketing → Negocios posteriores (2025)")
st.caption(f"Origen de datos: {CSV_FILE}")
st.caption("💱 Todos los montos se muestran en la moneda original del negocio (USD o MXN).")


# -------------------------
# HELPERS
# -------------------------
def clasificar_estado_etapa(etapa: str) -> str:
    """Clasifica una etapa textual en: Ganado / Perdido / Descartado / Abierto."""
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


UNIDAD_LABEL_MAP = {
    "AWS": "Cloud & AI Solutions",
    "Cloud & AI Solutions": "Cloud & AI Solutions",
    "Data Analytics": "Data & Analytics",
    "Data & Analytics": "Data & Analytics",
    "Enterprises": "Enterprise Solutions",
    "Enterprise Solutions": "Enterprise Solutions",
}

PIPE_UNIDAD_FALLBACK = {
    "iNBest.Cloud": "Cloud & AI Solutions",
    "iNBest.Data&Analytics": "Data & Analytics",
    "iNBest.Enterprises": "Enterprise Solutions",
    "iNBest.USA": "Enterprise Solutions",
}


def normalizar_unidad(unidad, pipeline) -> str:
    """Normaliza la unidad de negocio al catálogo (Cloud & AI / D&A / Enterprise)."""
    u = str(unidad or "").strip()
    p = str(pipeline or "").strip()

    if not u or u == "Sin dato":
        return PIPE_UNIDAD_FALLBACK.get(p, "Sin dato")

    return UNIDAD_LABEL_MAP.get(u, u)


PLOT_TEMPLATE = "plotly_dark"


# -------------------------
# CARGA DE DATOS
# -------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fechas
    if "origen_created_date" in df.columns:
        df["origen_created_date"] = pd.to_datetime(
            df["origen_created_date"], errors="coerce"
        )
    if "deal_created_date" in df.columns:
        df["deal_created_date"] = pd.to_datetime(
            df["deal_created_date"], errors="coerce"
        )

    # Montos
    for col in ["origen_amount", "deal_amount", "origen_duracion_meses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Pipelines y etapas "bonitos"
    df["pipeline_marketing"] = df.get("origen_pipeline_label", "").fillna("").astype(str)
    df["pipeline_comercial"] = df.get("deal_pipeline_label", "").fillna("").astype(str)

    df["etapa_marketing"] = df.get("origen_dealstage_label", "").fillna("").astype(str)
    df["etapa_comercial"] = df.get("deal_dealstage_label", "").fillna("").astype(str)

    # Estados
    df["estado_marketing"] = df["etapa_marketing"].apply(clasificar_estado_etapa)
    df["estado_comercial"] = df["etapa_comercial"].apply(clasificar_estado_etapa)

    # Texto importante
    text_cols = [
        "tipo_negocio",
        "pipeline_marketing",
        "pipeline_comercial",
        "etapa_marketing",
        "etapa_comercial",
        "estado_marketing",
        "estado_comercial",
        "origen_origen_del_negocio",
        "origen_unidad_de_negocio_asignada",
        "origen_producto_catalogo",
        "origen_due_o_del_deal",
        "deal_unidad_de_negocio_asignada",
        "deal_producto_catalogo",
        "deal_due_o_del_deal",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Sin dato").astype(str)

    # Monedas (en tu CSV vienen vacías, pero dejamos preparado)
    for col in ["origen_currency", "deal_currency"]:
        if col in df.columns:
            df[col] = df[col].fillna("Sin moneda").astype(str)

    # Unidades normalizadas
    df["origen_unidad_norm"] = df.apply(
        lambda r: normalizar_unidad(
            r.get("origen_unidad_de_negocio_asignada"),
            r.get("origen_pipeline_label"),
        ),
        axis=1,
    )
    df["deal_unidad_norm"] = df.apply(
        lambda r: normalizar_unidad(
            r.get("deal_unidad_de_negocio_asignada"),
            r.get("deal_pipeline_label"),
        ),
        axis=1,
    )

    return df


df = load_data(CSV_FILE)

if df.empty:
    st.error("El CSV está vacío o no se pudo cargar.")
    st.stop()

# -------------------------
# SEPARAR ORIGEN Y POSTERIORES
# -------------------------
df_origen = df[df["tipo_negocio"] == "origen_marketing"].copy()
df_post = df[df["tipo_negocio"] == "posterior_contacto"].copy()

df_origen_unique = (
    df_origen.sort_values("origen_created_date")
    .drop_duplicates(subset=["origen_deal_id"])
    .copy()
)
df_post_unique = (
    df_post.sort_values("deal_created_date")
    .drop_duplicates(subset=["deal_id"])
    .copy()
)

# -------------------------
# SIDEBAR – FILTROS (simplificado)
# -------------------------
st.sidebar.title("⚙️ Filtros globales")
st.sidebar.caption("Afectan a todo el dashboard.")

min_date = df_origen_unique["origen_created_date"].min()
max_date = df_origen_unique["origen_created_date"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("No hay fechas válidas en 'origen_created_date'.")
    st.stop()

date_range = st.sidebar.date_input(
    "Rango de creación del negocio de marketing",
    value=(min_date.date(), max_date.date()),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

with st.sidebar.expander("Filtros por unidad de negocio (opcional)", expanded=False):
    unidades_mkt = sorted(df_origen_unique["origen_unidad_norm"].unique())
    unidad_filter = st.multiselect(
        "Unidad de negocio de marketing",
        options=unidades_mkt,
        default=unidades_mkt,
    )

# Filtro aplicado a negocios de origen
mask_origen = (
    (df_origen_unique["origen_created_date"].dt.date >= start_date)
    & (df_origen_unique["origen_created_date"].dt.date <= end_date)
)

if unidad_filter:
    mask_origen &= df_origen_unique["origen_unidad_norm"].isin(unidad_filter)

df_origen_f = df_origen_unique[mask_origen].copy()

# Filtrar también los posteriores en función de los origen filtrados
origen_ids_filtrados = df_origen_f["origen_deal_id"].astype(str).unique()
df_post_f = df_post[df_post["origen_deal_id"].astype(str).isin(origen_ids_filtrados)].copy()
df_post_f_unique = (
    df_post_f.sort_values("deal_created_date")
    .drop_duplicates(subset=["deal_id"])
    .copy()
)

# Subconjuntos de GANADOS en marketing
df_origen_g = df_origen_f[df_origen_f["estado_marketing"] == "Ganado"].copy()
df_post_from_g = df_post_f[
    df_post_f["origen_deal_id"].isin(df_origen_g["origen_deal_id"])
].copy()
df_post_from_g_unique = (
    df_post_from_g.sort_values("deal_created_date")
    .drop_duplicates(subset=["deal_id"])
    .copy()
)

# -------------------------
# KPIs GENERALES
# -------------------------
st.markdown('<p class="section-label">Resumen del pipeline de marketing</p>', unsafe_allow_html=True)
st.markdown("### 🔢 KPIs generales del pipeline iNBest.marketing")

col1, col2, col3, col4 = st.columns(4)

num_origen = df_origen_f["origen_deal_id"].nunique()
num_post_unicos = df_post_f_unique["deal_id"].nunique()

total_origen_amount = df_origen_f["origen_amount"].sum()
total_post_amount = df_post_f_unique["deal_amount"].sum()

deals_post_por_origen = num_post_unicos / num_origen if num_origen > 0 else 0

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Negocios totales en iNBest.marketing", f"{num_origen:,}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Negocios posteriores únicos asociados", f"{num_post_unicos:,}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Monto posterior total (USD/MXN)", f"{total_post_amount:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Negocios posteriores por negocio de marketing", f"{deals_post_por_origen:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Negocios por etapa (SQL, MQL, Localizando, etc)
st.markdown("#### 📍 Negocios de marketing por etapa")
if not df_origen_f.empty:
    etapa_counts = (
        df_origen_f.groupby("etapa_marketing")["origen_deal_id"]
        .nunique()
        .reset_index(name="num_deals")
        .sort_values("num_deals", ascending=False)
    )
    fig_etapas = px.bar(
        etapa_counts,
        x="etapa_marketing",
        y="num_deals",
        template=PLOT_TEMPLATE,
    )
    fig_etapas.update_layout(
        xaxis_title="Etapa del pipeline de marketing",
        yaxis_title="Número de negocios",
        margin=dict(l=10, r=10, t=30, b=80),
    )
    st.plotly_chart(fig_etapas, use_container_width=True)
else:
    st.info("No hay negocios de marketing con los filtros actuales.")

st.markdown("---")

# -------------------------
# KPIs ESPECIALES (GANADOS)
# -------------------------
st.markdown('<p class="section-label">Foco en resultados</p>', unsafe_allow_html=True)
st.markdown("### 🏆 Negocios ganados en iNBest.marketing")

colg1, colg2, colg3, colg4 = st.columns(4)

num_origen_g = df_origen_g["origen_deal_id"].nunique()
total_origen_g_amount = df_origen_g["origen_amount"].sum()
num_post_from_g_unicos = df_post_from_g_unique["deal_id"].nunique()
total_duracion_meses_g = df_origen_g["origen_duracion_meses"].sum()

with colg1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Negocios de marketing ganados", f"{num_origen_g:,}")
    st.markdown("</div>", unsafe_allow_html=True)

with colg2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Monto total ganado (USD/MXN)", f"{total_origen_g_amount:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with colg3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Negocios posteriores desde ganados", f"{num_post_from_g_unicos:,}")
    st.markdown("</div>", unsafe_allow_html=True)

with colg4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Meses totales de contratos ganados",
        f"{total_duracion_meses_g:,.1f}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# MÉTRICAS AVANZADAS (GANADOS)
# -------------------------
st.markdown("### 🧠 Métricas avanzadas (solo ganados en marketing)")

col5, col6, col7, col8 = st.columns(4)

agg_post_g = (
    df_post_from_g.groupby("origen_deal_id")["deal_id"]
    .nunique()
    .reset_index(name="posterior_deals")
)
num_origen_con_post_g = agg_post_g[agg_post_g["posterior_deals"] > 0][
    "origen_deal_id"
].nunique()
conversion_rate_g = (
    num_origen_con_post_g / num_origen_g * 100 if num_origen_g > 0 else 0
)

total_post_amount_g = df_post_from_g_unique["deal_amount"].sum()
roi_factor_g = (
    total_post_amount_g / total_origen_g_amount if total_origen_g_amount > 0 else 0
)

primer_posterior_g = (
    df_post_from_g.groupby("origen_deal_id")["deal_created_date"]
    .min()
    .reset_index(name="fecha_primer_posterior")
)
tmp_g = df_origen_g.merge(primer_posterior_g, on="origen_deal_id", how="inner")
tmp_g["dias_a_primer_posterior"] = (
    tmp_g["fecha_primer_posterior"] - tmp_g["origen_created_date"]
).dt.days
dias_prom_g = tmp_g["dias_a_primer_posterior"].mean() if not tmp_g.empty else np.nan

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Ganados con ≥1 negocio posterior",
        f"{num_origen_con_post_g:,}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col6:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Tasa de conversión (ganados → posterior)",
        f"{conversion_rate_g:.1f}%" if num_origen_g > 0 else "N/A",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col7:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Factor de multiplicación ingresos (posterior/origen)",
        f"{roi_factor_g:.2f}x" if total_origen_g_amount > 0 else "N/A",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col8:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Días promedio al primer negocio posterior",
        f"{dias_prom_g:.1f} días" if not np.isnan(dias_prom_g) else "N/A",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# DISTRIBUCIÓN DE ESTADOS
# -------------------------
st.markdown("### 🧩 Distribución de estados comerciales y de marketing")
col_est1, col_est2 = st.columns(2)

with col_est1:
    st.markdown("**Negocios posteriores por estado comercial (monto)**")
    if not df_post_f_unique.empty:
        estado_counts_amt = (
            df_post_f_unique.groupby("estado_comercial")["deal_amount"]
            .sum()
            .reset_index()
        )
        fig_estado = px.pie(
            estado_counts_amt,
            names="estado_comercial",
            values="deal_amount",
            hole=0.4,
            template=PLOT_TEMPLATE,
        )
        fig_estado.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_estado, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

with col_est2:
    st.markdown("**Estados comerciales por pipeline**")
    if not df_post_f_unique.empty:
        mkt_estado = (
            df_post_f_unique.groupby(["pipeline_comercial", "estado_comercial"])[
                "deal_id"
            ]
            .nunique()
            .reset_index(name="num_deals")
        )
        fig_mkt = px.bar(
            mkt_estado,
            x="pipeline_comercial",
            y="num_deals",
            color="estado_comercial",
            barmode="stack",
            template=PLOT_TEMPLATE,
        )
        fig_mkt.update_layout(
            xaxis_title="Pipeline comercial",
            yaxis_title="Número de negocios",
            margin=dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig_mkt, use_container_width=True)
    else:
        st.info("No hay negocios comerciales con los filtros actuales.")

st.markdown("---")

# -------------------------
# EVOLUCIÓN TEMPORAL
# -------------------------
st.markdown("### 📆 Evolución temporal")

col_time1, col_time2 = st.columns(2)

with col_time1:
    st.markdown("**Negocios de marketing por mes (cantidad y monto)**")
    if not df_origen_f.empty:
        tmp = df_origen_f.copy()
        tmp["mes"] = tmp["origen_created_date"].dt.to_period("M").dt.to_timestamp()
        evo = (
            tmp.groupby("mes")
            .agg(
                num_negocios=("origen_deal_id", "nunique"),
                monto_origen=("origen_amount", "sum"),
            )
            .reset_index()
        )
        fig_evo = px.bar(
            evo,
            x="mes",
            y="num_negocios",
            hover_data=["monto_origen"],
            template=PLOT_TEMPLATE,
        )
        fig_evo.update_layout(
            xaxis_title="Mes",
            yaxis_title="Negocios de marketing",
            margin=dict(l=10, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_evo, use_container_width=True)
    else:
        st.info("No hay negocios de marketing con los filtros actuales.")

with col_time2:
    st.markdown("**Negocios posteriores por mes (cantidad y monto)**")
    if not df_post_f_unique.empty:
        tmp = df_post_f_unique.copy()
        tmp["mes"] = tmp["deal_created_date"].dt.to_period("M").dt.to_timestamp()
        evo = (
            tmp.groupby("mes")
            .agg(
                num_negocios=("deal_id", "nunique"),
                monto_posterior=("deal_amount", "sum"),
            )
            .reset_index()
        )
        fig_evo2 = px.bar(
            evo,
            x="mes",
            y="num_negocios",
            hover_data=["monto_posterior"],
            template=PLOT_TEMPLATE,
        )
        fig_evo2.update_layout(
            xaxis_title="Mes",
            yaxis_title="Negocios posteriores",
            margin=dict(l=10, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_evo2, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

st.markdown("---")

# -------------------------
# MIX DE MARKETING POR UNIDAD / ORIGEN
# -------------------------
st.subheader("🥧 Mix del pipeline iNBest.marketing")

col_mix1, col_mix2 = st.columns(2)

with col_mix1:
    st.markdown("**Distribución de negocios de marketing por unidad de negocio**")
    if not df_origen_f.empty:
        mix_unidad = (
            df_origen_f.groupby("origen_unidad_norm")["origen_deal_id"]
            .nunique()
            .reset_index(name="num_deals")
        )
        fig_mix_unidad = px.pie(
            mix_unidad,
            names="origen_unidad_norm",
            values="num_deals",
            hole=0.3,
            template=PLOT_TEMPLATE,
        )
        fig_mix_unidad.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mix_unidad, use_container_width=True)
    else:
        st.info("No hay negocios de marketing con los filtros actuales.")

# En vez de producto catálogo (que viene vacío), usamos origen_del_negocio
with col_mix2:
    st.markdown("**Distribución de negocios de marketing por origen del negocio**")
    if not df_origen_f.empty:
        mix_origen = (
            df_origen_f.groupby("origen_origen_del_negocio")["origen_deal_id"]
            .nunique()
            .reset_index(name="num_deals")
            .sort_values("num_deals", ascending=False)
        )
        fig_mix_origen = px.pie(
            mix_origen,
            names="origen_origen_del_negocio",
            values="num_deals",
            hole=0.3,
            template=PLOT_TEMPLATE,
        )
        fig_mix_origen.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mix_origen, use_container_width=True)
    else:
        st.info("No hay negocios de marketing con los filtros actuales.")

st.markdown("---")

# -------------------------
# TABLA RESUMEN POR NEGOCIO ORIGEN
# -------------------------
st.subheader("📌 Resumen por negocio de marketing")

if df_origen_f.empty:
    st.info("No hay negocios de marketing con los filtros actuales.")
else:
    base_origen = (
        df_origen_f.sort_values("origen_created_date")
        .drop_duplicates(subset=["origen_deal_id"])
        .loc[
            :,
            [
                "origen_deal_id",
                "origen_deal_name",
                "origen_created_date",
                "pipeline_marketing",
                "etapa_marketing",
                "estado_marketing",
                "origen_origen_del_negocio",
                "origen_unidad_norm",
                "origen_amount",
                "origen_duracion_meses",
            ],
        ]
    )

    agg_post_count = (
        df_post_f.groupby("origen_deal_id")["deal_id"]
        .nunique()
        .reset_index(name="posterior_deals")
    )

    agg_post_monto = (
        df_post_f.groupby("origen_deal_id")["deal_amount"]
        .agg(posterior_monto_total="sum", posterior_monto_promedio="mean")
        .reset_index()
    )

    resumen = base_origen.merge(agg_post_count, on="origen_deal_id", how="left")
    resumen = resumen.merge(agg_post_monto, on="origen_deal_id", how="left")

    resumen["posterior_deals"] = resumen["posterior_deals"].fillna(0).astype(int)
    resumen["posterior_monto_total"] = resumen["posterior_monto_total"].fillna(0.0)
    resumen["posterior_monto_promedio"] = resumen["posterior_monto_promedio"].fillna(0.0)

    st.dataframe(
        resumen.sort_values("posterior_monto_total", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# -------------------------
# INSIGHTS VISUALES
# -------------------------
st.subheader("📈 Insights visuales")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Monto total posterior por unidad de negocio y moneda (deals únicos)**")
    if not df_post_f_unique.empty:
        tmp = df_post_f_unique.copy()
        tmp["deal_unidad_norm"] = tmp["deal_unidad_norm"].replace({"": "Sin dato"})
        tmp["deal_currency"] = tmp.get("deal_currency", "Sin moneda")

        monto_por_unidad = (
            tmp.groupby(["deal_unidad_norm", "deal_currency"])["deal_amount"]
            .sum()
            .reset_index()
            .sort_values("deal_amount", ascending=False)
        )
        fig_owner = px.bar(
            monto_por_unidad,
            x="deal_unidad_norm",
            y="deal_amount",
            color="deal_currency",
            barmode="group",
            template=PLOT_TEMPLATE,
        )
        fig_owner.update_layout(
            xaxis_title="Unidad de negocio (posterior)",
            yaxis_title="Monto posterior (USD/MXN)",
            margin=dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig_owner, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

with col_g2:
    st.markdown("**Cantidad de negocios posteriores por pipeline comercial**")
    if not df_post_f_unique.empty:
        deals_por_pipeline = (
            df_post_f_unique.groupby("pipeline_comercial")["deal_id"]
            .nunique()
            .reset_index()
            .rename(columns={"deal_id": "num_deals"})
            .sort_values("num_deals", ascending=False)
        )

        fig_pipe = px.bar(
            deals_por_pipeline,
            x="pipeline_comercial",
            y="num_deals",
            template=PLOT_TEMPLATE,
        )
        fig_pipe.update_layout(
            xaxis_title="Pipeline comercial",
            yaxis_title="Número de negocios",
            margin=dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig_pipe, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

st.markdown("---")

# -------------------------
# SANKEY
# -------------------------
st.subheader("🔀 Flujo: Origen del negocio (marketing) → Unidad de negocio asignada (posterior)")

if df_post_f.empty:
    st.info("No hay negocios posteriores para construir el diagrama de flujo con los filtros actuales.")
else:
    st.markdown("Ajustes del diagrama:")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        metrica_flujo = st.radio(
            "Métrica para el ancho del flujo",
            ("Monto posterior total", "Número de negocios posteriores"),
            horizontal=True,
        )
    with col_s2:
        solo_mkt = st.checkbox(
            "Incluir solo negocios nacidos en iNBest.marketing",
            value=True,
        )

    sankey_base = (
        df_post_f
        .drop_duplicates(subset=["origen_deal_id", "deal_id"])
        .copy()
    )

    if solo_mkt:
        sankey_base = sankey_base[sankey_base["pipeline_marketing"] == "iNBest.marketing"]

    if sankey_base.empty:
        st.info("No hay datos suficientes para el Sankey con los filtros seleccionados.")
    else:
        sankey_base["origen_label"] = sankey_base["origen_origen_del_negocio"].replace(
            {"": "Sin origen", "Sin dato": "Sin origen"}
        )
        sankey_base["destino_label"] = sankey_base["deal_unidad_norm"].replace(
            {"": "Sin unidad", "Sin dato": "Sin unidad"}
        )

        sankey_group = (
            sankey_base.groupby(["origen_label", "destino_label"])
            .agg(
                total_amount=("deal_amount", "sum"),
                num_deals=("deal_id", "nunique"),
            )
            .reset_index()
        )

        if sankey_group.empty:
            st.info("No hay datos suficientes para el Sankey después de agrupar.")
        else:
            if metrica_flujo == "Monto posterior total":
                values = sankey_group["total_amount"].values
            else:
                values = sankey_group["num_deals"].values

            origen_labels = sankey_group["origen_label"].unique().tolist()
            destino_labels = sankey_group["destino_label"].unique().tolist()

            origen_index = {label: i for i, label in enumerate(origen_labels)}
            destino_index = {
                label: i + len(origen_labels) for i, label in enumerate(destino_labels)
            }

            labels = origen_labels + destino_labels
            sources = sankey_group["origen_label"].map(origen_index).values
            targets = sankey_group["destino_label"].map(destino_index).values

            n_origen = len(origen_labels)
            n_destino = len(destino_labels)
            colors = (
                ["rgba(59,130,246,0.9)"] * n_origen  # origen_del_negocio
                + ["rgba(16,185,129,0.9)"] * n_destino  # unidad_de_negocio_asignada
            )

            fig = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=20,
                            thickness=20,
                            line=dict(width=0.5),
                            label=labels,
                            color=colors,
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                        ),
                    )
                ]
            )

            fig.update_layout(
                height=550,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                **Cómo leer el gráfico:**

                - Cada bloque del lado izquierdo es el **origen del negocio** del pipeline de marketing.
                - Cada bloque del lado derecho es la **unidad de negocio asignada** de los negocios posteriores
                  (Cloud & AI Solutions, Data & Analytics, Enterprise Solutions).
                - El grosor de cada cinta representa la métrica seleccionada (monto o número de negocios).
                """
            )

st.markdown("---")

# -------------------------
# DESGLOSE POR PIPELINE Y ETAPA COMERCIAL
# -------------------------
st.subheader("📊 Desglose por pipeline y etapa comercial")

if df_post_f.empty:
    st.info("No hay datos posteriores con los filtros actuales.")
else:
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Top pipelines comerciales por monto posterior**")
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
                monto_total=("deal_amount", "sum"),
            )
            .reset_index()
            .sort_values("monto_total", ascending=False)
        )

        st.dataframe(
            etapas,
            use_container_width=True,
            hide_index=True,
        )
