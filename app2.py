import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------
CSV_FILE = "bd_final.csv"
TASA_USD_MXN = 18.19  # Tasa de conversión fija

st.set_page_config(
    page_title="Dashboard HubSpot – Marketing → Negocios posteriores (2025)",
    layout="wide",
)

st.title("📊 HubSpot – Marketing → Negocios posteriores (2025)")
st.caption(f"Origen de datos: {CSV_FILE} | Montos convertidos a USD (Tasa: {TASA_USD_MXN})")


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
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

    # Fechas
    if "origen_created_date" in df.columns:
        df["origen_created_date"] = pd.to_datetime(
            df["origen_created_date"], errors="coerce"
        )
    if "deal_created_date" in df.columns:
        df["deal_created_date"] = pd.to_datetime(
            df["deal_created_date"], errors="coerce"
        )

    # Limpieza básica de numéricos
    for col in ["origen_amount", "deal_amount", "origen_duracion_meses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # -----------------------------------------------
    # CONVERSIÓN DE MONEDA (MXN -> USD)
    # -----------------------------------------------
    # 1. Origen
    if "origen_currency" in df.columns and "origen_amount" in df.columns:
        # Normalizar texto moneda
        df["origen_currency"] = df["origen_currency"].astype(str).str.strip().str.upper()
        mask_mxn = df["origen_currency"] == "MXN"
        # Aplicar conversión
        df.loc[mask_mxn, "origen_amount"] = df.loc[mask_mxn, "origen_amount"] / TASA_USD_MXN
        # Etiquetar todo como USD (o equivalente) para evitar confusión visual posterior
        df.loc[mask_mxn, "origen_currency"] = "USD"
    
    # 2. Posterior (Deal)
    if "deal_currency" in df.columns and "deal_amount" in df.columns:
        # Normalizar texto moneda
        df["deal_currency"] = df["deal_currency"].astype(str).str.strip().str.upper()
        mask_mxn_deal = df["deal_currency"] == "MXN"
        # Aplicar conversión
        df.loc[mask_mxn_deal, "deal_amount"] = df.loc[mask_mxn_deal, "deal_amount"] / TASA_USD_MXN
        df.loc[mask_mxn_deal, "deal_currency"] = "USD"

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
        "deal_name",         # Aseguramos que existan como texto
        "origen_deal_name"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Sin dato").astype(str)

    # Monedas (rellenar nulos)
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
# CÁLCULOS PARA NUEVA SECCIÓN PRINCIPAL (Abiertos y Perdidos)
# -------------------------
# Posteriores ABIERTOS (excluye ganados, perdidos y descartados)
df_post_abiertos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Abierto"]
count_abiertos = df_post_abiertos["deal_id"].nunique()
monto_abiertos = df_post_abiertos["deal_amount"].sum()

# Posteriores PERDIDOS
df_post_perdidos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Perdido"]
count_perdidos = df_post_perdidos["deal_id"].nunique()
monto_perdidos = df_post_perdidos["deal_amount"].sum()

# -------------------------
# VISUALIZACIÓN: FILA 1 (NUEVOS PROTAGONISTAS)
# -------------------------
st.markdown("### 🎯 Foco: Estado de Negocios Posteriores")

kp1, kp2, kp3, kp4 = st.columns(4)
kp1.metric("Negocios Posteriores Abiertos", f"{count_abiertos}")
kp2.metric("Monto Abierto (USD)", f"${monto_abiertos:,.2f}")
kp3.metric("Negocios Posteriores Perdidos", f"{count_perdidos}")
kp4.metric("Monto Perdido (USD)", f"${monto_perdidos:,.2f}")

# --- TABLA DETALLE (NUEVA SOLICITUD) ---
st.markdown("#### 📝 Detalle de Negocios (Abiertos y Perdidos)")
with st.expander("Ver lista de Empresas y Etapas", expanded=False):
    # Juntar ambos dataframes
    df_foco = pd.concat([df_post_abiertos, df_post_perdidos])
    if not df_foco.empty:
        # Seleccionar columnas clave. OJO: Usamos 'deal_name' que es la correcta
        cols_mostrar = ["deal_name", "etapa_comercial", "estado_comercial", "deal_amount", "pipeline_comercial"]
        
        # Renombrar para presentación
        df_show = df_foco[cols_mostrar].rename(columns={
            "deal_name": "Empresa / Negocio",
            "etapa_comercial": "Etapa Actual",
            "estado_comercial": "Estado",
            "deal_amount": "Monto (USD)",
            "pipeline_comercial": "Pipeline"
        }).sort_values(by=["Estado", "Monto (USD)"], ascending=[True, False])

        st.dataframe(
            df_show, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Monto (USD)": st.column_config.NumberColumn(format="$%.2f")
            }
        )
    else:
        st.info("No hay negocios abiertos o perdidos en la selección actual.")

st.markdown("---")

# -------------------------
# FILA 2: MÉTRICAS GENERALES Y "KPIs ESPECIALES" (MOVIDOS AQUÍ)
# -------------------------
st.markdown("### 📉 Impacto Comercial y Métricas Generales")

# KPIs Generales (Original Fila 1)
num_origen = df_origen_f["origen_deal_id"].nunique()
num_post_unicos = df_post_f_unique["deal_id"].nunique()
total_origen_amount = df_origen_f["origen_amount"].sum()
total_post_amount = df_post_f_unique["deal_amount"].sum() # Esto incluye todo (Ganado + Abierto + Perdido)

# KPIs Especiales (Original Fila 2 - Ganados)
num_origen_g = df_origen_g["origen_deal_id"].nunique()
total_origen_g_amount = df_origen_g["origen_amount"].sum()
num_post_from_g_unicos = df_post_from_g_unique["deal_id"].nunique()
monto_post_from_g = df_post_from_g_unique["deal_amount"].sum()

# Organizamos en dos filas de 4 columnas para que quepa todo sin saturar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Negocios Origen (Mkt)", f"{num_origen:,}")
c2.metric("Total Posteriores (Todos)", f"{num_post_unicos:,}")
c3.metric("Monto Total Posterior (USD)", f"${total_post_amount:,.2f}")
c4.metric("Ticket Prom. Posterior", f"${(total_post_amount/num_post_unicos if num_post_unicos>0 else 0):,.2f}")

st.markdown("##### 🏆 Desempeño desde Origen Ganado")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Deals Ganados (Mkt)", f"{num_origen_g}")
g2.metric("Monto Ganado (Mkt USD)", f"${total_origen_g_amount:,.2f}")
g3.metric("Posteriores creados (de Ganados)", f"{num_post_from_g_unicos}")
g4.metric("Monto Generado (de Ganados USD)", f"${monto_post_from_g:,.2f}")


# -------------------------
# MÉTRICAS AVANZADAS (Mantenemos igual pero más abajo)
# -------------------------
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

# ROI / factor de multiplicación
roi_factor_g = (
    monto_post_from_g / total_origen_g_amount if total_origen_g_amount > 0 else 0
)

st.markdown("---")

# -------------------------
# DISTRIBUCIÓN DE ESTADOS
# -------------------------
st.markdown("### 🧩 Distribución")
# NOTA: Se eliminó la gráfica "Estados de Marketing" a petición.
# Dejamos solo la distribución de estados comerciales, centrada.

col_est_centrada = st.columns([1, 2, 1]) # Columnas para centrar
with col_est_centrada[1]:
    st.markdown("**Distribución de negocios posteriores por estado comercial (Monto USD)**")
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

# 1) Monto posterior por unidad de negocio (deals únicos)
with col_g1:
    st.markdown("**Monto total posterior por unidad de negocio (USD)**")
    if not df_post_f_unique.empty:
        tmp = df_post_f_unique.copy()
        tmp["deal_unidad_de_negocio_asignada"] = tmp[
            "deal_unidad_de_negocio_asignada"
        ].replace({"": "Sin dato"})

        # Agrupamos solo por unidad (ya convertimos todo a USD)
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
            # quitamos color="deal_currency" porque todo es USD ahora
        )
        fig_owner.update_layout(
            xaxis_title="Unidad de negocio (posterior)",
            yaxis_title="Monto posterior (USD)",
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
        st.markdown("**Top pipelines comerciales por monto posterior (USD)**")
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
