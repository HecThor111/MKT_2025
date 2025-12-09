import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------
CSV_FILE = "bd_final.csv"
TC_USD_MXN = 18.19  # Tasa de cambio fija solicitada

st.set_page_config(
    page_title="Dashboard HubSpot – Marketing → Negocios posteriores (USD)",
    layout="wide",
)

st.title("📊 HubSpot – Marketing → Negocios posteriores (USD)")
st.caption(f"Origen de datos: {CSV_FILE} | Tasa de cambio aplicada: 1 USD ≈ {TC_USD_MXN} MXN")


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

    # Limpieza inicial de columnas de texto para evitar errores
    text_cols = [
        "origen_currency", "deal_currency", "origen_deal_name", "deal_deal_name"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Sin dato").astype(str)

    # Montos y Conversión a USD
    # -------------------------
    # 1. Limpiar a numérico
    for col in ["origen_amount", "deal_amount", "origen_duracion_meses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    # 2. Convertir ORIGEN a USD si es MXN
    if "origen_currency" in df.columns and "origen_amount" in df.columns:
        mask_mxn_origen = df["origen_currency"].str.upper() == "MXN"
        df.loc[mask_mxn_origen, "origen_amount"] = df.loc[mask_mxn_origen, "origen_amount"] / TC_USD_MXN
        # Actualizamos la etiqueta de moneda para reflejar que ya es USD (o equivalente)
        df.loc[mask_mxn_origen, "origen_currency"] = "USD (Conv)"

    # 3. Convertir POSTERIOR a USD si es MXN
    if "deal_currency" in df.columns and "deal_amount" in df.columns:
        mask_mxn_deal = df["deal_currency"].str.upper() == "MXN"
        df.loc[mask_mxn_deal, "deal_amount"] = df.loc[mask_mxn_deal, "deal_amount"] / TC_USD_MXN
        df.loc[mask_mxn_deal, "deal_currency"] = "USD (Conv)"

    # Pipelines y etapas "bonitos"
    df["pipeline_marketing"] = df.get("origen_pipeline_label", "").fillna("").astype(str)
    df["pipeline_comercial"] = df.get("deal_pipeline_label", "").fillna("").astype(str)

    df["etapa_marketing"] = df.get("origen_dealstage_label", "").fillna("").astype(str)
    df["etapa_comercial"] = df.get("deal_dealstage_label", "").fillna("").astype(str)

    # Estados
    df["estado_marketing"] = df["etapa_marketing"].apply(clasificar_estado_etapa)
    df["estado_comercial"] = df["etapa_comercial"].apply(clasificar_estado_etapa)

    # Texto importante
    cols_extra = [
        "tipo_negocio",
        "pipeline_marketing", "pipeline_comercial",
        "etapa_marketing", "etapa_comercial",
        "estado_marketing", "estado_comercial",
        "origen_origen_del_negocio", "origen_unidad_de_negocio_asignada",
        "origen_producto_catalogo", "origen_due_o_del_deal",
        "deal_unidad_de_negocio_asignada", "deal_producto_catalogo",
        "deal_due_o_del_deal", "origen_deal_name", "deal_deal_name"
    ]
    for col in cols_extra:
        if col in df.columns:
            df[col] = df[col].fillna("Sin dato").astype(str)

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

# Versión deduplicada por negocio
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

unidades_mkt = sorted(df_origen_unique["origen_unidad_de_negocio_asignada"].replace({"": "Sin dato"}).unique())
unidad_filter = st.sidebar.multiselect("Unidad de negocio (marketing)", options=unidades_mkt, default=unidades_mkt)

productos_mkt = sorted(df_origen_unique["origen_producto_catalogo"].replace({"": "Sin dato"}).unique())
producto_filter = st.sidebar.multiselect("Producto catálogo (marketing)", options=productos_mkt, default=productos_mkt)

# Aplicar filtros
mask_origen = (
    (df_origen_unique["origen_created_date"].dt.date >= start_date)
    & (df_origen_unique["origen_created_date"].dt.date <= end_date)
    & (df_origen_unique["origen_unidad_de_negocio_asignada"].isin(unidad_filter))
    & (df_origen_unique["origen_producto_catalogo"].isin(producto_filter))
)

df_origen_f = df_origen_unique[mask_origen].copy()

# Filtrar posteriores basado en origen filtrado
origen_ids_filtrados = df_origen_f["origen_deal_id"].astype(str).unique()
df_post_f = df_post[df_post["origen_deal_id"].astype(str).isin(origen_ids_filtrados)].copy()
df_post_f_unique = (
    df_post_f.sort_values("deal_created_date")
    .drop_duplicates(subset=["deal_id"])
    .copy()
)

# -------------------------
# CALCULOS PREVIOS (Filtros de Estado)
# -------------------------
# Negocios Posteriores por Estado (Abierto, Perdido, Ganado)
df_post_abiertos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Abierto"]
df_post_perdidos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Perdido"]
df_post_ganados = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Ganado"]

# -------------------------
# SECCIÓN 1: KPIs PRINCIPALES (POSTERIORES ABIERTOS Y PERDIDOS)
# -------------------------
st.markdown("### 🎯 Foco Actual: Negocios Posteriores (Abiertos vs Perdidos)")
col_main1, col_main2, col_main3, col_main4 = st.columns(4)

# Abiertos
total_abierto_usd = df_post_abiertos["deal_amount"].sum()
count_abierto = df_post_abiertos["deal_id"].nunique()

# Perdidos
total_perdido_usd = df_post_perdidos["deal_amount"].sum()
count_perdido = df_post_perdidos["deal_id"].nunique()

col_main1.metric("💰 Monto Abierto (Pipeline)", f"${total_abierto_usd:,.2f} USD")
col_main2.metric("📂 Cantidad Negocios Abiertos", f"{count_abierto}")
col_main3.metric("💸 Monto Perdido Total", f"${total_perdido_usd:,.2f} USD")
col_main4.metric("❌ Cantidad Negocios Perdidos", f"{count_perdido}")

# --- DETALLE DE ESTOS NEGOCIOS (TABLA) ---
st.markdown("#### 🔎 Detalle de Negocios (Abiertos y Perdidos)")
with st.expander("Ver lista detallada de empresas y etapas (Abiertos y Perdidos)", expanded=True):
    # Unimos ambos dataframes para mostrar en una sola tabla
    df_detail = pd.concat([df_post_abiertos, df_post_perdidos])
    
    if not df_detail.empty:
        # Seleccionamos columnas relevantes
        display_cols = [
            "deal_deal_name",
            "estado_comercial",
            "etapa_comercial",
            "deal_amount",
            "deal_created_date",
            "pipeline_comercial",
            "deal_unidad_de_negocio_asignada"
        ]
        
        # Renombrar para que se vea bonito
        df_show = df_detail[display_cols].rename(columns={
            "deal_deal_name": "Nombre del Negocio / Empresa",
            "estado_comercial": "Estado",
            "etapa_comercial": "Etapa Actual",
            "deal_amount": "Monto (USD)",
            "deal_created_date": "Fecha Creación",
            "pipeline_comercial": "Pipeline",
            "deal_unidad_de_negocio_asignada": "Unidad Negocio"
        }).sort_values(["Estado", "Monto (USD)"], ascending=[True, False])
        
        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Monto (USD)": st.column_config.NumberColumn(format="$%.2f")
            }
        )
    else:
        st.info("No hay negocios abiertos o perdidos con los filtros actuales.")

st.markdown("---")

# -------------------------
# SECCIÓN 2: KPIs SECUNDARIOS (ORIGEN GANADO Y TOTALES)
# -------------------------
st.markdown("### 📊 Métricas de Origen Ganado y Totales")

# Subconjuntos de GANADOS en marketing para cálculos
df_origen_g = df_origen_f[df_origen_f["estado_marketing"] == "Ganado"].copy()
df_post_from_g = df_post_f[df_post_f["origen_deal_id"].isin(df_origen_g["origen_deal_id"])].copy()

# Cálculos
num_origen_g = df_origen_g["origen_deal_id"].nunique()
total_origen_g_amount = df_origen_g["origen_amount"].sum()
total_post_general_usd = df_post_f_unique["deal_amount"].sum() # Total de todo (Ganado+Abierto+Perdido)
total_deals_post_general = df_post_f_unique["deal_id"].nunique()

col_sec1, col_sec2, col_sec3, col_sec4 = st.columns(4)

col_sec1.metric("Origen Ganado (Mkt)", f"{num_origen_g}")
col_sec2.metric("Monto Ganado (Mkt)", f"${total_origen_g_amount:,.2f} USD")
col_sec3.metric("Total Negocios Posteriores (Todos)", f"{total_deals_post_general}")
col_sec4.metric("Monto Total Posterior (Todos)", f"${total_post_general_usd:,.2f} USD")

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
            labels={"monto_origen": "Monto USD"}
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
            labels={"monto_posterior": "Monto USD"}
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
# MIX DE MARKETING
# -------------------------
st.subheader("🥧 Mix del pipeline iNBest.marketing")

col_mix1, col_mix2 = st.columns(2)

with col_mix1:
    st.markdown("**Distribución por Unidad de Negocio**")
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
        st.info("No datos.")

with col_mix2:
    st.markdown("**Distribución por Producto Catálogo**")
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
        st.info("No datos.")

st.markdown("---")

# -------------------------
# SANKEY
# -------------------------
st.subheader("🔀 Flujo: Origen (Mkt) → Destino (Posterior)")

if df_post_f.empty:
    st.info("No hay negocios posteriores para construir el diagrama.")
else:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        metrica_flujo = st.radio(
            "Métrica flujo",
            ("Monto posterior total", "Número de negocios posteriores"),
            horizontal=True,
        )
    with col_s2:
        solo_mkt = st.checkbox("Solo pipeline iNBest.marketing", value=True)

    sankey_base = df_post_f.drop_duplicates(subset=["origen_deal_id", "deal_id"]).copy()

    if solo_mkt:
        sankey_base = sankey_base[sankey_base["pipeline_marketing"] == "iNBest.marketing"]

    if sankey_base.empty:
        st.info("Datos insuficientes para Sankey.")
    else:
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

        if not sankey_group.empty:
            values = sankey_group["total_amount"].values if metrica_flujo == "Monto posterior total" else sankey_group["num_deals"].values
            origen_labels = sankey_group["origen_label"].unique().tolist()
            destino_labels = sankey_group["destino_label"].unique().tolist()
            
            origen_index = {label: i for i, label in enumerate(origen_labels)}
            destino_index = {label: i + len(origen_labels) for i, label in enumerate(destino_labels)}
            
            labels = origen_labels + destino_labels
            sources = sankey_group["origen_label"].map(origen_index).values
            targets = sankey_group["destino_label"].map(destino_index).values
            
            colors = ["rgba(33, 150, 243, 0.8)"] * len(origen_labels) + ["rgba(76, 175, 80, 0.8)"] * len(destino_labels)

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=20, thickness=20, line=dict(width=0.5), label=labels, color=colors),
                link=dict(source=sources, target=targets, value=values)
            )])
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
