import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------
# Ajusta el nombre si tu archivo se llama distinto
CSV_FILE = "bd_final.csv"
TASA_CAMBIO = 18.19  # Tasa solicitada

st.set_page_config(
    page_title="Dashboard HubSpot – Marketing → Negocios posteriores (2025)",
    layout="wide",
)

st.title("📊 HubSpot – Marketing → Negocios posteriores (2025)")
st.caption(f"Origen de datos: {CSV_FILE} | Visualizando todo en USD (Tasa conversión: {TASA_CAMBIO} MXN/USD)")


# -------------------------
# HELPERS
# -------------------------
def clasificar_estado_etapa(etapa: str) -> str:
    """
    Clasifica una etapa textual en:
    - Ganado
    - Perdido
    - Descartado
    - Abierto
    """
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

    # -----------------------------------------------------------
    # LÓGICA DE CONVERSIÓN DE MONEDA (MXN -> USD)
    # -----------------------------------------------------------
    # Convertimos montos de Origen
    if "origen_currency" in df.columns and "origen_amount" in df.columns:
        # Detectamos MXN
        mask_mxn_orig = df["origen_currency"].astype(str).str.contains("MXN", case=False, na=False)
        # Convertimos
        df.loc[mask_mxn_orig, "origen_amount"] = df.loc[mask_mxn_orig, "origen_amount"] / TASA_CAMBIO
    
    # Convertimos montos de Posteriores
    if "deal_currency" in df.columns and "deal_amount" in df.columns:
        # Detectamos MXN
        mask_mxn_deal = df["deal_currency"].astype(str).str.contains("MXN", case=False, na=False)
        # Convertimos
        df.loc[mask_mxn_deal, "deal_amount"] = df.loc[mask_mxn_deal, "deal_amount"] / TASA_CAMBIO
    # -----------------------------------------------------------

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
        "deal_name", # Nos aseguramos que lea el nombre real del negocio
        "origen_deal_name"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Sin dato").astype(str)

    # Monedas
    for col in ["origen_currency", "deal_currency"]:
        if col in df.columns:
            df[col] = df[col].fillna("Sin moneda").astype(str)

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

# Versión deduplicada por negocio (para métricas más limpias)
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
# SIDEBAR – FILTROS
# -------------------------
st.sidebar.header("🔍 Filtros")

# Rango de fechas del negocio origen (usando negocios únicos)
min_date = df_origen_unique["origen_created_date"].min()
max_date = df_origen_unique["origen_created_date"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("No hay fechas válidas en 'origen_created_date'.")
    st.stop()

date_range = st.sidebar.date_input(
    "Rango de fecha (creación del negocio origen)",
    value=(min_date.date(), max_date.date()),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

# Filtro por unidad de negocio asignada (marketing)
unidades_mkt = sorted(
    df_origen_unique["origen_unidad_de_negocio_asignada"].replace(
        {"": "Sin dato"}
    ).unique()
)
unidad_filter = st.sidebar.multiselect(
    "Unidad de negocio (marketing)",
    options=unidades_mkt,
    default=unidades_mkt,
)

# Filtro por producto catálogo (marketing)
productos_mkt = sorted(
    df_origen_unique["origen_producto_catalogo"].replace({"": "Sin dato"}).unique()
)
producto_filter = st.sidebar.multiselect(
    "Producto catálogo (marketing)",
    options=productos_mkt,
    default=productos_mkt,
)

# Aplicar filtros sobre los negocios de origen (unique)
mask_origen = (
    (df_origen_unique["origen_created_date"].dt.date >= start_date)
    & (df_origen_unique["origen_created_date"].dt.date <= end_date)
    & (df_origen_unique["origen_unidad_de_negocio_asignada"].isin(unidad_filter))
    & (df_origen_unique["origen_producto_catalogo"].isin(producto_filter))
)

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
# NUEVA SECCION PRINCIPAL: POSTERIORES ABIERTOS Y PERDIDOS
# -------------------------
# Calculamos subsets
df_post_abiertos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Abierto"]
df_post_perdidos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Perdido"]

# KPIs Nuevos
kpi_abiertos_cant = df_post_abiertos["deal_id"].nunique()
kpi_abiertos_monto = df_post_abiertos["deal_amount"].sum()
kpi_perdidos_cant = df_post_perdidos["deal_id"].nunique()
kpi_perdidos_monto = df_post_perdidos["deal_amount"].sum()

st.markdown("### 🎯 Estado de Negocios Posteriores (Abiertos y Perdidos)")
col_new1, col_new2, col_new3, col_new4 = st.columns(4)
col_new1.metric("Negocios Posteriores Abiertos", f"{kpi_abiertos_cant}")
col_new2.metric("Monto Abierto (USD)", f"${kpi_abiertos_monto:,.2f}")
col_new3.metric("Negocios Posteriores Perdidos", f"{kpi_perdidos_cant}")
col_new4.metric("Monto Perdido (USD)", f"${kpi_perdidos_monto:,.2f}")

# Tabla Detalle (Expandible)
with st.expander("Ver detalle de empresas (Abiertos y Perdidos)", expanded=True):
    # Unimos para la tabla
    df_detalle = pd.concat([df_post_abiertos, df_post_perdidos])
    if not df_detalle.empty:
        # Seleccionamos columnas y usamos deal_name que viene del CSV
        cols_table = ["deal_name", "estado_comercial", "etapa_comercial", "deal_amount", "pipeline_comercial"]
        
        # Renombrar para que se vea limpio en pantalla
        df_view = df_detalle[cols_table].rename(columns={
            "deal_name": "Empresa / Negocio",
            "estado_comercial": "Estado",
            "etapa_comercial": "Etapa",
            "deal_amount": "Monto (USD)",
            "pipeline_comercial": "Pipeline"
        }).sort_values(["Estado", "Monto (USD)"], ascending=[True, False])

        st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Monto (USD)": st.column_config.NumberColumn(format="$%.2f")
            }
        )
    else:
        st.info("No hay negocios abiertos o perdidos en la selección.")

st.markdown("---")


# -------------------------
# MÉTRICAS GENERALES (fila 2 - Antes era fila 1)
# -------------------------
st.markdown("### 🔢 KPIs generales del pipeline iNBest.marketing (y Ganados)")

col1, col2, col3, col4 = st.columns(4)

num_origen = df_origen_f["origen_deal_id"].nunique()
num_post_unicos = df_post_f_unique["deal_id"].nunique()
total_post_amount = df_post_f_unique["deal_amount"].sum()
deals_post_por_origen = num_post_unicos / num_origen if num_origen > 0 else 0

col1.metric("Negocios totales iNBest.marketing", f"{num_origen:,}")
col2.metric("Negocios posteriores únicos", f"{num_post_unicos:,}")
col3.metric("Monto posterior total (USD)", f"{total_post_amount:,.2f}")
col4.metric("Deals posteriores por negocio origen", f"{deals_post_por_origen:.2f}")

# --- KPIs Especiales Ganados (Ahora abajo, más pequeños) ---
st.markdown("#### Impacto desde Origen Ganado")
colg1, colg2, colg3, colg4 = st.columns(4)

num_origen_g = df_origen_g["origen_deal_id"].nunique()
total_origen_g_amount = df_origen_g["origen_amount"].sum()
num_post_from_g_unicos = df_post_from_g_unique["deal_id"].nunique()
total_duracion_meses_g = df_origen_g["origen_duracion_meses"].sum()

colg1.metric("Negocios GANADOS (marketing)", f"{num_origen_g:,}")
colg2.metric("Monto total GANADO (USD)", f"{total_origen_g_amount:,.2f}")
colg3.metric("Negocios posteriores (desde GANADOS)", f"{num_post_from_g_unicos:,}")
colg4.metric("Suma duración contratos (meses)", f"{total_duracion_meses_g:,.1f}")


# --- Negocios por etapa (SQL, MQL, Localizando, etc)
# NOTA: Mantenemos esta, eliminamos la de "Estados" como pediste
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
    )
    fig_etapas.update_layout(
        xaxis_title="Etapa de marketing",
        yaxis_title="Núm. negocios",
        margin=dict(l=10, r=10, t=30, b=80),
    )
    st.plotly_chart(fig_etapas, use_container_width=True)
else:
    st.info("No hay negocios de marketing con los filtros actuales.")

st.markdown("---")

# -------------------------
# MÉTRICAS AVANZADAS (GANADOS)
# -------------------------
st.markdown("### 🧠 Métricas avanzadas (solo GANADOS en marketing)")

col5, col6, col7, col8 = st.columns(4)

# Conversión: GANADOS con al menos 1 posterior
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

# ROI / factor de multiplicación (solo USD si quieres luego filtrar por moneda)
total_post_amount_g = df_post_from_g_unique["deal_amount"].sum()
roi_factor_g = (
    total_post_amount_g / total_origen_g_amount if total_origen_g_amount > 0 else 0
)

# Tiempo medio marketing → primer negocio posterior (para GANADOS)
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

col5.metric(
    "GANADOS con ≥1 negocio posterior",
    f"{num_origen_con_post_g:,}",
)
col6.metric(
    "Tasa de conversión (GANADOS → posterior)",
    f"{conversion_rate_g:.1f}%" if num_origen_g > 0 else "N/A",
)
col7.metric(
    "Factor de multiplicación (posterior/origen, GANADOS)",
    f"{roi_factor_g:.2f}x" if total_origen_g_amount > 0 else "N/A",
)
col8.metric(
    "Días promedio a primer posterior (GANADOS)",
    f"{dias_prom_g:.1f} días" if not np.isnan(dias_prom_g) else "N/A",
)

st.markdown("---")

# -------------------------
# DISTRIBUCIÓN DE ESTADOS
# -------------------------
# AQUI ELIMINE LA GRAFICA "Estados de Marketing" QUE PEDISTE QUITAR
st.markdown("### 🧩 Distribución de estados comerciales")
col_est1, col_est2 = st.columns([1,1]) # Solo dejo columnas para equilibrar

with col_est1:
    st.markdown("**Distribución de negocios posteriores por estado comercial (monto USD)**")
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
        )
        fig_estado.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_estado, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

# La columna 2 se queda vacía o para futuro uso porque se eliminó la gráfica solicitada

st.markdown("---")

# -------------------------
# EVOLUCIÓN TEMPORAL
# -------------------------
st.markdown("### 📆 Evolución temporal")

col_time1, col_time2 = st.columns(2)

with col_time1:
    st.markdown("**Negocios de marketing por mes (cantidad y monto USD)**")
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
    st.markdown("**Negocios posteriores por mes (cantidad y monto USD)**")
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
# MIX DE MARKETING POR UNIDAD / PRODUCTO
# -------------------------
st.subheader("🥧 Mix del pipeline iNBest.marketing")

col_mix1, col_mix2 = st.columns(2)

with col_mix1:
    st.markdown("**Distribución de negocios de marketing por unidad de negocio**")
    if not df_origen_f.empty:
        mix_unidad = (
            df_origen_f.groupby("origen_unidad_de_negocio_asignada")["origen_deal_id"]
            .nunique()
            .reset_index(name="num_deals")
        )
        fig_mix_unidad = px.pie(
            mix_unidad,
            names="origen_unidad_de_negocio_asignada",
            values="num_deals",
            hole=0.3,
        )
        fig_mix_unidad.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mix_unidad, use_container_width=True)
    else:
        st.info("No hay negocios de marketing con los filtros actuales.")

with col_mix2:
    st.markdown("**Distribución de negocios de marketing por producto catálogo**")
    if not df_origen_f.empty:
        mix_prod = (
            df_origen_f.groupby("origen_producto_catalogo")["origen_deal_id"]
            .nunique()
            .reset_index(name="num_deals")
            .sort_values("num_deals", ascending=False)
        )
        fig_mix_prod = px.pie(
            mix_prod,
            names="origen_producto_catalogo",
            values="num_deals",
            hole=0.3,
        )
        fig_mix_prod.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mix_prod, use_container_width=True)
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
                "origen_unidad_de_negocio_asignada",
                "origen_producto_catalogo",
                "origen_amount",
                "origen_duracion_meses",
            ],
        ]
    )

    # Agregados de posteriores por negocio origen
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
# SECCIÓN INSIGHTS VISUALES (barras)
# -------------------------
st.subheader("📈 Insights visuales")

col_g1, col_g2 = st.columns(2)

# 1) Monto posterior por unidad de negocio (deals únicos) y moneda
with col_g1:
    st.markdown("**Monto total posterior por unidad de negocio (USD)**")
    if not df_post_f_unique.empty:
        tmp = df_post_f_unique.copy()
        tmp["deal_unidad_de_negocio_asignada"] = tmp[
            "deal_unidad_de_negocio_asignada"
        ].replace({"": "Sin dato"})
        # tmp["deal_currency"]  <- Ya no es necesario mostrar el desglose de moneda porque todo es USD

        monto_por_unidad = (
            tmp.groupby(["deal_unidad_de_negocio_asignada"])["deal_amount"]
            .sum()
            .reset_index()
            .sort_values("deal_amount", ascending=False)
        )
        fig_owner = px.bar(
            monto_por_unidad,
            x="deal_unidad_de_negocio_asignada",
            y="deal_amount",
            # color="deal_currency", -> Eliminado para unificar color
            barmode="group",
        )
        fig_owner.update_layout(
            xaxis_title="Unidad de negocio (posterior)",
            yaxis_title="Monto posterior",
            margin=dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig_owner, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

# 2) Cantidad de deals posteriores por pipeline comercial
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
        )
        fig_pipe.update_layout(
            xaxis_title="Pipeline comercial",
            yaxis_title="Núm. negocios",
            margin=dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig_pipe, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

st.markdown("---")

# -------------------------
# SANKEY: ORIGEN_DEL_NEGOCIO → UNIDAD_DE_NEGOCIO_ASIGNADA
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
            "Incluir solo negocios donde el pipeline de origen es iNBest.marketing",
            value=True,
        )

    # BASE DEL SANKEY: UNA FILA POR (origen_deal_id, deal_id)
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
        # Etiquetas de origen (origen_del_negocio marketing) y destino (unidad_de_negocio_asignada posterior)
        sankey_base["origen_label"] = sankey_base["origen_origen_del_negocio"].replace({"": "Sin origen"})
        sankey_base["destino_label"] = sankey_base["deal_unidad_de_negocio_asignada"].replace({"": "Sin unidad"})

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
                ["rgba(33, 150, 243, 0.8)"] * n_origen  # origen_del_negocio
                + ["rgba(76, 175, 80, 0.8)"] * n_destino  # unidad_de_negocio_asignada
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
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                **Cómo leer el gráfico:**

                - Cada bloque del lado izquierdo es el **origen del negocio** del pipeline de marketing.
                - Cada bloque del lado derecho es la **unidad de negocio asignada** de los negocios posteriores.
                - El grosor de la cinta representa la métrica seleccionada:
                  - **Monto posterior total**: suma del `deal_amount` de los negocios posteriores.
                  - **Número de negocios posteriores**: cantidad de deals distintos.
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
