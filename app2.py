import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO VISUAL
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
    :root {
        --primary-color: #38bdf8;
        --background-color: #0B0F19;
        --secondary-background-color: #111827;
        --text-color: #f0f9ff;
        --font: 'Inter', sans-serif;
    }
    .stApp { background-color: #0B0F19; }
    
    /* Widgets Style Overrides */
    div.stDateInput > div > div > input { color: #38bdf8; }
    div.stMultiSelect span[data-baseweb="tag"] { background-color: #1e293b !important; border: 1px solid #38bdf8 !important; }
    div.stMultiSelect div[data-baseweb="select"] { border-color: #38bdf8 !important; }
    .stCheckbox div[data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }
    
    /* KPI Cards */
    div[data-testid="metric-container"] { display: none; }
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
    .kpi-card:hover { transform: translateY(-2px); border-color: #8b5cf6; }
    .kpi-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; font-weight: 600; }
    .kpi-value { color: #f0f9ff; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px rgba(56, 189, 248, 0.3); }
    .kpi-sub { color: #38bdf8; font-size: 0.75rem; margin-top: 4px; }

    /* Headers & Text */
    h1, h2, h3, h4 { color: #f0f9ff !important; font-family: 'Inter', sans-serif; }
    p, label, .stMarkdown, .stRadio label, .stCheckbox label { color: #cbd5e1 !important; }
    
    /* Tables & Sidebar */
    div[data-testid="stDataFrame"] { background-color: #111827; border-radius: 10px; border: 1px solid #374151; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e2937; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. FUNCIONES HELPERS & DICCIONARIOS
# -----------------------------------------------------------------------------

# Diccionario de coordenadas para ciudades principales (lat, lon)
CITY_COORDS = {
    "ciudad de méxico": (19.4326, -99.1332),
    "cdmx": (19.4326, -99.1332),
    "mexico city": (19.4326, -99.1332),
    "tlalpan": (19.2889, -99.1623),
    "cuauthemoc": (19.4326, -99.1332),
    "naucalpan": (19.4785, -99.2329),
    "estado de méxico": (19.3562, -99.6462),
    "monterrey": (25.6866, -100.3161),
    "guadalajara": (20.6597, -103.3496),
    "gdl": (20.6597, -103.3496),
    "zapopan": (20.7166, -103.4005),
    "querétaro": (20.5888, -100.3899),
    "león": (21.1221, -101.6664),
    "león, gto": (21.1221, -101.6664),
    "guanajuato": (21.0190, -101.2574),
    "merida": (20.9674, -89.5926),
    "mérida": (20.9674, -89.5926),
    "puebla": (19.0414, -98.2063),
    "tijuana": (32.5149, -117.0382),
    "cancún": (21.1619, -86.8515),
    "bogotá": (4.7110, -74.0721),
    "san jose": (9.9281, -84.0907), # Asumiendo Costa Rica por contexto LATAM, o San Jose CA.
    "manzanillo": (19.0522, -104.3159),
    "hgo": (20.1011, -98.7591) # Pachuca/Hidalgo
}

def obtener_coords(ciudad):
    if not isinstance(ciudad, str):
        return None, None
    c_clean = ciudad.lower().strip()
    return CITY_COORDS.get(c_clean, (None, None))

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

def calcular_origen_limpio(nombre_deal: str) -> str:
    """Extrae el texto después del guion si existe."""
    if not isinstance(nombre_deal, str):
        return "Desconocido"
    if "-" in nombre_deal:
        parts = nombre_deal.split("-", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return nombre_deal

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
CSV_FILE = "final.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {path}")
        return pd.DataFrame()

    # 1. Limpieza automática de etapas "Localizando"
    cols_etapa = ["origen_dealstage_label", "deal_dealstage_label"]
    for col in cols_etapa:
        if col in df.columns:
            df[col] = df[col].replace("Localizando", "Acercamiento", regex=True)

    # 2. Creación de columna 'Origen' calculada
    if "origen_deal_name" in df.columns:
        df["Origen"] = df["origen_deal_name"].apply(calcular_origen_limpio)
    else:
        df["Origen"] = "Desconocido"

    # Fechas
    for col in ["origen_created_date", "deal_created_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Montos
    for col in ["origen_amount", "deal_amount", "origen_duracion_meses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Conversión de moneda (MXN -> USD estimado)
    if "origen_currency" in df.columns and "origen_amount" in df.columns:
        mask_mxn_orig = df["origen_currency"] == "MXN"
        df.loc[mask_mxn_orig, "origen_amount"] = df.loc[mask_mxn_orig, "origen_amount"] / 18.19

    if "deal_currency" in df.columns and "deal_amount" in df.columns:
        mask_mxn_deal = df["deal_currency"] == "MXN"
        df.loc[mask_mxn_deal, "deal_amount"] = df.loc[mask_mxn_deal, "deal_amount"] / 18.19

    # Textos y Mapeos
    df["pipeline_marketing"] = df.get("origen_pipeline_label", "").fillna("").astype(str)
    df["pipeline_comercial"] = df.get("deal_pipeline_label", "").fillna("").astype(str)
    df["etapa_marketing"] = df.get("origen_dealstage_label", "").fillna("").astype(str)
    df["etapa_comercial"] = df.get("deal_dealstage_label", "").fillna("").astype(str)
    
    # Nuevas columnas requeridas
    df["contact_name"] = df.get("Contact_Nombre_Completo", "").fillna("Sin Nombre").astype(str)
    df["contact_company"] = df.get("Contact_Empresa", "").fillna("Sin Empresa").astype(str)
    df["contact_city"] = df.get("Contact_Ciudad", "").fillna("").astype(str)
    
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
    st.error("El DataFrame está vacío. Verifica que 'final.csv' tenga datos.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. FILTROS
# -----------------------------------------------------------------------------
df_origen_all = df[df["tipo_negocio"] == "origen_marketing"].copy()
df_post_all = df[df["tipo_negocio"] == "posterior_contacto"].copy()

# Deduplicar origen para filtros
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

# Filtro por Columna 'Origen' calculada en lugar de 'origen_origen_del_negocio' si se prefiere, 
# pero mantengo el original para consistencia, agregando el nuevo como opción extra si se desea.
origen_opts = sorted(df_origen_unique_all["origen_origen_del_negocio"].unique())
sel_origenes = st.sidebar.multiselect("Fuente del Negocio", options=origen_opts, default=origen_opts)

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
# HEADER DEL REPORTE
# -----------------------------------------------------------------------------
st.title("🚀 Reporte de Leads Marketing 2025")

# -----------------------------------------------------------------------------
# 5. KPI'S & TABLAS DETALLE
# -----------------------------------------------------------------------------
st.subheader("🎯 Negocios Posteriores (Pipeline)")

# Filtros para KPIs
post_abiertos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Abierto"]
post_perdidos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Perdido"]
post_ganados = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Ganado"]

# Métricas
c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
with c_kpi1: display_kpi("Negocios Abiertos", f"{len(post_abiertos)}", "En Pipeline")
with c_kpi2: display_kpi("Negocios Ganados", f"{len(post_ganados)}", "Ventas Cerradas")
with c_kpi3: display_kpi("Monto Ganado", f"${post_ganados['deal_amount'].sum():,.0f}", "USD")
with c_kpi4: display_kpi("Tasa Conversión", f"{(len(post_ganados)/len(df_origen_f)*100 if len(df_origen_f)>0 else 0):.1f}%", "De Lead a Venta")

# TABLA 1: ABIERTOS Y PERDIDOS
with st.expander("🔍 Ver Detalle de Negocios (Abiertos y Perdidos)"):
    cols_required = ["origen_deal_name", "Origen", "contact_name", "contact_company", "deal_dealtype", "deal_amount", "etapa_comercial"]
    
    # Combinar y filtrar
    df_detail_1 = pd.concat([post_abiertos, post_perdidos])
    
    if not df_detail_1.empty:
        # Asegurar que existan las columnas, si no rellenar
        for c in cols_required:
            if c not in df_detail_1.columns: df_detail_1[c] = "-"
            
        st.dataframe(
            df_detail_1[cols_required],
            use_container_width=True,
            hide_index=True,
            column_config={
                "origen_deal_name": "Nombre Lead (Original)",
                "Origen": "Origen (Campaña/Evento)",
                "contact_name": "Contacto",
                "contact_company": "Empresa",
                "deal_dealtype": "Tipo Deal",
                "deal_amount": st.column_config.NumberColumn("Monto (USD)", format="$%.2f"),
                "etapa_comercial": "Etapa Actual"
            }
        )
    else:
        st.info("No hay negocios abiertos o perdidos con los filtros actuales.")

# TABLA 2: GANADOS
with st.expander("🏆 Ver Detalle de Deals Ganados (Ventas)"):
    if not post_ganados.empty:
        for c in cols_required:
            if c not in post_ganados.columns: post_ganados[c] = "-"
            
        st.dataframe(
            post_ganados[cols_required],
            use_container_width=True,
            hide_index=True,
            column_config={
                "origen_deal_name": "Nombre Lead (Original)",
                "Origen": "Origen (Campaña/Evento)",
                "contact_name": "Contacto",
                "contact_company": "Empresa",
                "deal_dealtype": "Tipo Deal",
                "deal_amount": st.column_config.NumberColumn("Monto (USD)", format="$%.2f"),
                "etapa_comercial": "Etapa Final"
            }
        )
    else:
        st.info("No hay ventas registradas con los filtros actuales.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. MAPA INTERACTIVO DE CIUDADES
# -----------------------------------------------------------------------------
st.subheader("🌎 Distribución Geográfica de Leads")

# Preparar datos para el mapa
if not df_origen_f.empty:
    # Agrupar por ciudad
    city_counts = df_origen_f["contact_city"].value_counts().reset_index()
    city_counts.columns = ["ciudad", "conteo"]
    
    # Obtener coordenadas
    coords = city_counts["ciudad"].apply(obtener_coords)
    city_counts["lat"] = coords.apply(lambda x: x[0])
    city_counts["lon"] = coords.apply(lambda x: x[1])
    
    # Filtrar ciudades sin coordenadas conocidas
    map_data = city_counts.dropna(subset=["lat", "lon"])
    
    if not map_data.empty:
        col_map1, col_map2 = st.columns([3, 1])
        
        with col_map1:
            # Mapa con Plotly Scatter Mapbox/Geo
            fig_map = px.scatter_mapbox(
                map_data,
                lat="lat",
                lon="lon",
                size="conteo",
                hover_name="ciudad",
                hover_data={"conteo": True, "lat": False, "lon": False},
                color="conteo",
                color_continuous_scale=px.colors.sequential.Plasma,
                size_max=30,
                zoom=3.5,
                center={"lat": 23.6345, "lon": -102.5528} # Centro de México
            )
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                margin={"r":0,"t":0,"l":0,"b":0},
                height=500,
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        with col_map2:
            st.markdown("**Top Ciudades**")
            st.dataframe(
                map_data[["ciudad", "conteo"]].head(10),
                hide_index=True,
                use_container_width=True
            )
            st.caption(f"*Se muestran {len(map_data)} ciudades identificadas.")
    else:
        st.warning("No se pudieron geolocalizar las ciudades disponibles.")
else:
    st.info("Sin datos para mostrar en el mapa.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. GRÁFICAS ADICIONALES (VOLUMEN Y FUNNEL)
# -----------------------------------------------------------------------------
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("### 🧬 Funnel Marketing")
    if not df_origen_f.empty:
        etapa_counts = df_origen_f["etapa_marketing"].value_counts().reset_index()
        etapa_counts.columns = ["etapa", "count"]
        
        fig_funnel = px.funnel(
            etapa_counts, x="count", y="etapa",
            color_discrete_sequence=[COLOR_PALETTE[0]]
        )
        fig_funnel.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_funnel, use_container_width=True)

with col_g2:
    st.markdown("### 🥧 Origen (Campaña)")
    if not df_origen_f.empty:
        # Usamos la nueva columna calculada 'Origen'
        origen_counts = df_origen_f["Origen"].value_counts().reset_index().head(10)
        origen_counts.columns = ["origen_clean", "count"]
        
        fig_pie = px.pie(
            origen_counts, values="count", names="origen_clean",
            hole=0.4, color_discrete_sequence=COLOR_PALETTE
        )
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<br><div style='text-align: center; color: #475569;'>Desarrollado para actualización 2025 🚀</div>", unsafe_allow_html=True)
