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
    /* VARIABLES GLOBALES */
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
    
    /* WIDGETS */
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
    .stCheckbox div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }
    span[data-baseweb="checkbox"] div {
        background-color: #38bdf8 !important;
    }

    /* KPI Cards */
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

    /* Textos */
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

# --- FUNCIÓN DE LIMPIEZA MODIFICADA (FUSIÓN STEP IN & INNOVATE) ---
def limpiar_origen_sankey(texto_origen):
    if not isinstance(texto_origen, str):
        return None
    
    txt = texto_origen.lower().strip()
    
    # 1. FILTRO: Eliminar leads genéricos
    if txt == "lead de aws":
        return None

    # 2. AGRUPACIÓN DE EVENTOS
    
    # Modernización (Online y Presencial agrupados)
    if "modernización" in txt and "infraestructura" in txt:
        return "Modernización Infraestructura AWS"
    
    # Eventos Driven (MSFT, AWS, Fabric, MTY agrupados)
    if "driven" in txt:
        return "Eventos Driven"
    
    # --- Fusionar Agnostico y Step In & Innovate en uno solo ---
    if "agnostico" in txt or "agnóstico" in txt:
        return "Step In & Innovate Workshops"
    
    if "step in & innovate" in txt:
        return "Step In & Innovate Workshops"

    # 3. Retornar texto original si no cae en grupos (truncado)
    return texto_origen[:50] + "..." if len(texto_origen) > 50 else texto_origen

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

    # --- Limpieza de "Localizando" -> "Acercamiento" ---
    cols_to_clean = ["origen_dealstage_label", "deal_dealstage_label"]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].replace("Localizando", "Acercamiento", regex=True)

    # --- Cálculo de columna "Origen" MEJORADO (Split por guion) ---
    def calcular_origen(val):
        s_val = str(val)
        if "-" not in s_val:
            return s_val.strip()
        
        # Dividir por guiones
        parts = [p.strip() for p in s_val.split("-")]
        
        # Lista de palabras a ignorar si están al final
        ignore_list = [
            "GDL", "CDMX", "MX", "USA", "LATAM", "MTY", "Bajio", "Occidente", 
            "Agnostico", "Upselling", "Crosselling", "Base Instalada", 
            "General", "Página Web"
        ]
        
        candidate = parts[-1]
        
        # Si la última parte está en la lista negra y hay más partes, tomamos la penúltima
        if candidate in ignore_list and len(parts) > 1:
            return parts[-2]
        
        return candidate
    
    if "origen_deal_name" in df.columns:
        df["Origen"] = df["origen_deal_name"].apply(calcular_origen)
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

    # Moneda (MXN -> USD)
    if "origen_currency" in df.columns and "origen_amount" in df.columns:
        mask_mxn_orig = df["origen_currency"] == "MXN"
        df.loc[mask_mxn_orig, "origen_amount"] = df.loc[mask_mxn_orig, "origen_amount"] / 18.19

    if "deal_currency" in df.columns and "deal_amount" in df.columns:
        mask_mxn_deal = df["deal_currency"] == "MXN"
        df.loc[mask_mxn_deal, "deal_amount"] = df.loc[mask_mxn_deal, "deal_amount"] / 18.19

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
        lambda row: normalizar_unidad(row["deal_unidad_raw"], row["deal_pipeline_label"]), axis=1
    )

    # Rellenar textos clave
    cols_text = ["origen_origen_del_negocio", "origen_producto_catalogo", "Contact_Nombre_Completo", "Contact_Empresa", "deal_dealtype", "Contact_Ciudad"]
    for c in cols_text:
        if c in df.columns:
            df[c] = df[c].fillna("Sin dato").astype(str)
        else:
            df[c] = "Sin dato"
            
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

min_d, max_d = df_origen_unique_all["origen_created_date"].min(), df_origen_unique_all["origen_created_date"].max()
if pd.isna(min_d): min_d, max_d = pd.Timestamp.now(), pd.Timestamp.now()

dates = st.sidebar.date_input("Fecha Creación (Mkt)", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_date, end_date = dates if isinstance(dates, tuple) and len(dates) == 2 else (min_d, max_d)

unidades_opts = sorted(df_origen_unique_all["origen_unidad_norm"].unique())
sel_unidades = st.sidebar.multiselect("Unidad de Negocio", options=unidades_opts, default=unidades_opts)

origen_opts = sorted(df_origen_unique_all["origen_origen_del_negocio"].unique())
sel_origenes = st.sidebar.multiselect("Origen del Negocio", options=origen_opts, default=origen_opts)

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
# 5. FILA 1: MÉTRICAS GENERALES
# -----------------------------------------------------------------------------
st.subheader("📡 Métricas Generales (Todo el Pipeline)")

kpi_mkt_count = df_origen_f["origen_deal_id"].nunique()
kpi_post_total_unique = df_post_f_unique["deal_id"].nunique()

col_gen1, col_gen2 = st.columns(2)
with col_gen1: display_kpi("Total Leads Marketing", f"{kpi_mkt_count:,}", "Todos los estados")
with col_gen2: display_kpi("Total Negocios Posteriores", f"{kpi_post_total_unique:,}", "Todos los estados")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. FILA 2: IMPACTO COMERCIAL
# -----------------------------------------------------------------------------
df_post_ganados_real = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Ganado"].copy()

deals_ganados_count = len(df_post_ganados_real)
monto_ganado_total = df_post_ganados_real["deal_amount"].sum()
unique_origins_count = df_post_ganados_real["origen_deal_id"].nunique()
tasa_multiplicacion = deals_ganados_count / unique_origins_count if unique_origins_count > 0 else 0

avg_dias_creacion = 0
if not df_post_ganados_real.empty:
    df_primeros = df_post_ganados_real.groupby("origen_deal_id").agg({
        "deal_created_date": "min",
        "origen_created_date": "first"
    }).reset_index()
    if not df_primeros.empty:
        df_primeros["delta_days"] = (df_primeros["deal_created_date"] - df_primeros["origen_created_date"]).dt.days
        avg_dias_creacion = df_primeros["delta_days"].mean()

st.subheader("🏆 Impacto Comercial (Cierres Reales)")
c_imp1, c_imp2, c_imp3, c_imp4 = st.columns(4)

with c_imp1: display_kpi("Deals Ganados (Ventas)", f"{deals_ganados_count}", "Cierre Ganado Real")
with c_imp2: display_kpi("Monto Ganado (USD)", f"${monto_ganado_total:,.2f}", "Total Cerrado")
with c_imp3: display_kpi("Tasa de Multiplicación", f"{tasa_multiplicacion:.2f}", "Deals por Lead Exitoso")
with c_imp4: display_kpi("Tiempo Promedio 1er Deal", f"{avg_dias_creacion:.0f} días", "Desde creación Lead")

with st.expander("🔍 Ver Detalle de Deals Ganados (Ventas)"):
    cols_show_ganados = ["deal_name", "Origen", "Contact_Nombre_Completo", "Contact_Empresa", "deal_dealtype", "deal_amount", "pipeline_comercial"]
    st.dataframe(
        df_post_ganados_real[cols_show_ganados],
        use_container_width=True,
        hide_index=True,
        column_config={
            "deal_amount": st.column_config.NumberColumn("Monto (USD)", format="$%.2f")
        }
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. FILA 3: NEGOCIOS POSTERIORES (MODIFICADO - 3 COLUMNAS)
# -----------------------------------------------------------------------------
st.subheader("🎯 Negocios Posteriores (Abiertos y Perdidos)")

post_abiertos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Abierto"]
post_perdidos = df_post_f_unique[df_post_f_unique["estado_comercial"] == "Perdido"]

abiertos_count = len(post_abiertos)
abiertos_amount = post_abiertos["deal_amount"].sum()
perdidos_count = len(post_perdidos)
# perdidos_amount no se muestra, pero se calcula si fuera necesario
perdidos_amount = post_perdidos["deal_amount"].sum()

# Cambio a 3 columnas para eliminar KPI de monto perdido
c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
with c_kpi1: display_kpi("Negocios Abiertos", f"{abiertos_count}", "Pipeline Comercial")
with c_kpi2: display_kpi("Monto Abierto (USD)", f"${abiertos_amount:,.2f}", "Potencial Activo")
with c_kpi3: display_kpi("Negocios Perdidos", f"{perdidos_count}", "Cierre Perdido")

with st.expander("🔍 Ver Detalle de Negocios (Abiertos y Perdidos)"):
    cols_show = ["deal_name", "Origen", "Contact_Nombre_Completo", "Contact_Empresa", "deal_dealtype", "etapa_comercial", "deal_amount", "estado_comercial"]
    detail_df = pd.concat([post_abiertos, post_perdidos])[cols_show].copy()
    
    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "deal_amount": st.column_config.NumberColumn("Monto (USD)", format="$%.2f")
        }
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 8. FILA 4: TIPOS DE DEAL
# -----------------------------------------------------------------------------
st.subheader("🧬 Clasificación por Tipo de Deal")

count_new_business = 0
count_existing_business = 0
count_renewals = 0

if not df_post_f_unique.empty:
    s_types = df_post_f_unique["deal_dealtype"].str.lower()
    count_new_business = s_types.apply(lambda x: 1 if "new" in x or "nuevo" in x else 0).sum()
    count_existing_business = s_types.apply(lambda x: 1 if "existing" in x or "existente" in x else 0).sum()
    count_renewals = s_types.apply(lambda x: 1 if "renewal" in x or "renovación" in x or "renovacion" in x else 0).sum()

c_type1, c_type2, c_type3 = st.columns(3)
with c_type1: display_kpi("Negocios Nuevos", f"{count_new_business}", "New Business")
with c_type2: display_kpi("Negocios Existentes", f"{count_existing_business}", "Existing Business")
with c_type3: display_kpi("Renovaciones", f"{count_renewals}", "Renewals")

st.markdown("---")

# -----------------------------------------------------------------------------
# 9. FILA 5: FINANZAS ACTUALIZADO
# -----------------------------------------------------------------------------
st.subheader("💰 Métricas Financieras y ROI")

# --- VALORES MANUALES ACTUALIZADOS ---
costo_eventos = 0 # Placeholder por compatibilidad si se necesitara calcular
costo_campanas = 0
cpl_val = "$95,92"
cac_val = "$1.932,95"
roi_val = "744,1%"

# Sección superior: ROI y CAC Global
c_fin1, c_fin2, c_fin3 = st.columns(3)

with c_fin1: display_kpi("CPL (Cost Per Lead)", cpl_val, "Global")
with c_fin2: display_kpi("CAC", cac_val, "Costo Adq. Cliente")
with c_fin3: display_kpi("ROI", roi_val, "Retorno Inversión")

st.markdown("#### 📉 Sección: Gastos")

gastos_mkt = "$63.787,35"
gasto_prop_evento = "$2.039,05"
gasto_prop_campana = "$2.970,03"
gasto_prop_webinar = "$0.0"

c_gasto1, c_gasto2, c_gasto3, c_gasto4 = st.columns(4)

with c_gasto1: display_kpi("Gastos Mkt", gastos_mkt, "Total")
with c_gasto2: display_kpi("Gasto Prop x Evento", gasto_prop_evento, "Promedio")
with c_gasto3: display_kpi("Gasto Prop x Campaña", gasto_prop_campana, "Promedio")
with c_gasto4: display_kpi("Gasto Prop x Webinar", gasto_prop_webinar, "Promedio")

st.markdown("---")

# -----------------------------------------------------------------------------
# 9.1 RESUMEN POR CATEGORIA
# -----------------------------------------------------------------------------
st.subheader("📑 Resumen por Categoría")

data_cat = {
    "Categoría": ["Eventos", "Campañas", "Webinars"],
    "Leads": [468, 100, 97],
    "Cost": ["$48.937,20", "$14.850,15", "$-"],
    "Won $": ["$298.369,58", "$66.320,55", "$135.855,65"],
    "CPL": ["$104,57", "$148,50", "$-"],
    "CAC": ["$2.039,05", "$7.425,08", "$-"],
    "G. Prom leads": ["$12.432,07", "$33.160,28", "$19.407,95"],
    "Won %": ["245%", "22%", "77%"]
}

df_cat = pd.DataFrame(data_cat)
st.dataframe(df_cat, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 9.2 CONVERSIÓN DETALLADA
# -----------------------------------------------------------------------------
st.subheader("📊 Conversión por Eventos / Campaña / Webinar")

# Construcción manual de la tabla con los datos proporcionados
data_det = [
    ["Acuna", "Campaña", "3", "$-", "3", "$-", "25", "$5.000,00", "$200,00", "$-", "$-", "0%", "-100%"],
    ["Data-Driven AWS CDMX", "Evento", "3", "$112.001,52", "$-", "$-", "10", "$1.083,50", "$108,35", "$361,17", "$37.333,84", "30%", "10237%"],
    ["Data-Driven AWS GDL", "Evento", "3", "$11.885,40", "$-", "$-", "18", "$1.250,00", "$69,44", "$416,67", "$3.961,80", "17%", "851%"],
    ["Data-Driven MSFT CDMX", "Evento", "1", "$-", "$-", "6", "$1.083,50", "$180,58", "$-", "$-", "0%", "-100%", ""], 
    ["Data-Driven MSFT GDL", "Evento", "1", "$735,40", "$-", "$-", "14", "$1,00", "$0,07", "$1,00", "$735,40", "7%", "73440%"],
    ["Data-Driven MSFT MTY", "Evento", "3", "$7.285,92", "1", "1", "17", "$1.500,00", "$88,24", "$1.500,00", "$7.285,92", "6%", "386%"],
    ["DPL Protección Inteligente de Datos", "Evento", "6", "$7.102,16", "4", "$16.637,50", "1", "$-", "17", "$1.300,00", "$76,47", "$650,00", "$3.551,08"],
    ["Modernización de Infraestructura AWS (ONLINE)", "Webinar", "3", "$49.151,00", "1", "$-", "$-", "33", "$1,00", "$0,03", "$0,50", "$24.575,50", "6%"],
    ["Modernización de Infraestructura AWS (PRESENCIAL)", "Evento", "5", "$90.151,68", "$-", "$-", "20", "$1,00", "$0,05", "$0,20", "$18.030,34", "25%", "9015068%"],
    ["Orgánico", "Campaña", "18", "$66.320,55", "5", "1", "197", "$1.500,00", "$15,46", "$750,00", "$33.160,28", "2%", "4321%"],
    ["Padel Tech Connect", "Evento", "1", "$-", "1", "$-", "$-", "59", "$1.000,00", "$16,95", "$-", "$-", "0%"],
    ["Ruta Tequila", "Evento", "1", "$-", "1", "$-", "$-", "24", "$5.000,00", "$208,33", "$-", "$-", "0%"],
    ["Seguridad con Microsoft 365 y Copilot", "Evento", "3", "$5.852,01", "$-", "$-", "13", "$5.000,00", "$384,62", "$1.666,67", "$1.950,67", "23%", "17%"],
    ["Smart City Expo", "Evento", "3", "$19.852,85", "$-", "2", "$-", "35", "$10.000,00", "$285,71", "$10.000,00", "$19.852,85", "3%"],
    ["Step In & Innovate: Workshop (León)", "Evento", "8", "$69.396,00", "4", "$-", "$-", "26", "$2.253,70", "$86,68", "$563,43", "$17.349,00", "15%"],
    ["Step In & Innovate: Workshop Morelia", "Evento", "1", "$12.000,00", "$-", "$-", "25", "$750,00", "$30,00", "$750,00", "$12.000,00", "4%", "1500%"],
    ["Webinar Data-Driven MSFT", "Webinar", "9", "$86.700,75", "1", "$-", "3", "$-", "64", "$1,00", "$0,02", "$0,20", "$17.340,15"],
    ["TOTALES", "", "72", "$538.435,24", "22", "$16.637,50", "18", "$-", "503", "$36.724,70", "$103,00", "$979,99", "$11.595,70"]
]

# Normalizamos longitudes de filas para evitar errores en DataFrame
max_len = 13
data_norm = []
for row in data_det:
    if len(row) < max_len:
        row = row + [""] * (max_len - len(row))
    data_norm.append(row[:max_len])

cols_det = ["Fuente", "Type", "Leads (Won)", "Ganados $", "SQL", "MQL", "Leads Total", "Costo", "CPL", "CAC", "Ganancia Prom", "% Conv", "ROI"]
df_det = pd.DataFrame(data_norm, columns=cols_det)

# --- FUNCIÓN DE ESTILO PARA RESALTAR TOTALES ---
def highlight_totals(row):
    # Verificamos si la primera columna es 'TOTALES'
    if row["Fuente"] == "TOTALES":
        # Retornamos un estilo gris oscuro/azulado (tipo header) y texto en negrita
        return ['background-color: #1f2937; color: #ffffff; font-weight: bold; border-top: 2px solid #38bdf8'] * len(row)
    else:
        return [''] * len(row)

# Aplicamos el estilo y mostramos
st.dataframe(
    df_det.style.apply(highlight_totals, axis=1),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# -----------------------------------------------------------------------------
# 10. GRÁFICAS: FUNNEL MKT + DEAL TYPE (COLORES EDITADOS)
# -----------------------------------------------------------------------------
st.markdown("### 🧬 Análisis de Etapas y Tipos")

col_graph_1, col_graph_2 = st.columns(2)

with col_graph_1:
    st.markdown("**Embudo de Marketing (Funnel)**")
    if not df_origen_f.empty:
        etapa_counts = (
            df_origen_f.groupby("etapa_marketing")["origen_deal_id"]
            .nunique()
            .reset_index(name="num_deals")
            .sort_values("num_deals", ascending=False)
        )
        
        # --- FILTRO: ELIMINA 'Lead' y 'Acercamiento' SI LO DESEAS ---
        etapa_counts = etapa_counts[~etapa_counts["etapa_marketing"].isin(["Acercamiento", "Lead", "lead"])]
        
        # --- LÓGICA DE COLORES PERSONALIZADA (PETICIÓN USUARIO) ---
        # 1. El Turquesa vibrante para GANADOS
        color_ganado = "#22d3ee" 
        # 2. Paleta de Azules y Morados para el resto (SIN ROSAS)
        palette_others = ["#38bdf8", "#0ea5e9", "#6366f1", "#8b5cf6", "#818cf8"]
        
        # Creamos un mapa de colores explícito
        color_map_funnel = {}
        unique_stages = etapa_counts["etapa_marketing"].unique()
        
        for i, stage in enumerate(unique_stages):
            s_lower = stage.lower()
            # Si es Ganado -> Turquesa
            if "ganad" in s_lower or "won" in s_lower or "cierre" in s_lower or "cliente" in s_lower:
                color_map_funnel[stage] = color_ganado
            # Si es Perdido -> Un morado/azul fuerte (o lo que prefieras)
            elif "perdid" in s_lower or "lost" in s_lower:
                color_map_funnel[stage] = "#6366f1" # Indigo
            else:
                # El resto (MQL, SQL, etc) -> Ciclo de azules
                color_map_funnel[stage] = palette_others[i % len(palette_others)]

        fig_etapas = px.funnel(
            etapa_counts, 
            y="etapa_marketing", 
            x="num_deals",
            color="etapa_marketing", # Importante: Colorear por etapa
            color_discrete_map=color_map_funnel # Usamos nuestro mapa personalizado
        )
        fig_etapas.update_traces(textinfo="value+percent initial")
        fig_etapas.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=20, b=20),
            showlegend=False 
        )
        st.plotly_chart(fig_etapas, use_container_width=True)
    else:
        st.info("Sin datos de marketing.")

with col_graph_2:
    st.markdown("**Distribución por Tipo de Deal (Deal Type)**")
    if not df_post_f_unique.empty:
        temp_dtype = df_post_f_unique["deal_dealtype"].replace({"Sin dato": "No Definido", "": "No Definido"})
        dtype_counts = temp_dtype.value_counts().reset_index()
        dtype_counts.columns = ["Tipo de Deal", "Conteo"]
        
        # Paleta segura también para el Pie Chart (Azules/Turquesas)
        safe_pie_colors = ["#38bdf8", "#22d3ee", "#6366f1", "#0ea5e9"]
        
        fig_dtype = px.pie(
            dtype_counts, 
            names="Tipo de Deal", 
            values="Conteo", 
            hole=0.4,
            color_discrete_sequence=safe_pie_colors
        )
        fig_dtype.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_dtype, use_container_width=True)
    else:
        st.info("Sin datos de deals posteriores.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 11. DISTRIBUCIÓN DE ESTADOS
# -----------------------------------------------------------------------------
st.subheader("🧩 Distribución de Estados Comerciales")
if not df_post_f_unique.empty:
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
# 12. EVOLUCIÓN TEMPORAL
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
# 13. POR UNIDAD DE NEGOCIO
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
# 14. TABLA RESUMEN POR NEGOCIO ORIGEN
# -----------------------------------------------------------------------------
st.subheader("📌 Resumen Detallado por Negocio Marketing")
if not df_origen_f.empty:
    base = df_origen_f[["origen_deal_id", "origen_deal_name", "origen_created_date", "pipeline_marketing", 
                        "etapa_marketing", "estado_marketing", "origen_unidad_norm"]].copy()
    
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
# 15. INSIGHTS VISUALES
# -----------------------------------------------------------------------------
st.subheader("📈 Insights Visuales")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Deals Posteriores por Unidad Destino**")
    if not df_post_f_unique.empty:
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
# 16. MAPA INTERACTIVO DE CIUDADES
# -----------------------------------------------------------------------------
st.subheader("🗺️ Origen Geográfico de Leads (Por Ciudad)")

CITY_COORDS = {
    "Ciudad de México": [19.4326, -99.1332],
    "Estado de México": [19.3582, -99.6453],
    "Monterrey": [25.6866, -100.3161],
    "Guadalajara": [20.6597, -103.3496],
    "Manzanillo": [19.0522, -104.3159],
    "Mérida": [20.9674, -89.5926],
    "León": [21.1221, -101.6664],
    "Guanajuato": [21.0190, -101.2574],
    "HGO": [20.0911, -98.7624],
    "San Jose": [37.3382, -121.8863],
}

def normalizar_ciudad_mapa(nombre):
    n = str(nombre).lower().strip()
    if n in ['cdmx', 'ciudad de mexico', 'ciudad de méxico', 'cuauthemoc', 'tlalpan', 'benito juarez', 'miguel hidalgo', 'coyoacan']:
        return 'Ciudad de México'
    if n in ['gdl', 'guadalajara', 'zapopan', 'tlaquepaque']:
        return 'Guadalajara'
    if n in ['monterrey', 'mty', 'san pedro garza garcia', 'san pedro']:
        return 'Monterrey'
    if 'león' in n or 'leon' in n:
        return 'León'
    if 'estado de' in n or 'naucalpan' in n or 'tlalnepantla' in n:
        return 'Estado de México'
    if 'merida' in n or 'mérida' in n:
        return 'Mérida'
    if 'san jose' in n or 'san josé' in n:
        return 'San Jose'
    return nombre

if not df.empty and "Contact_Ciudad" in df.columns:
    df["Ciudad_Norm"] = df["Contact_Ciudad"].apply(normalizar_ciudad_mapa)
    df_geo = df[df["Ciudad_Norm"].isin(CITY_COORDS.keys())].copy()
    
    if not df_geo.empty:
        geo_counts = df_geo["Ciudad_Norm"].value_counts().reset_index()
        geo_counts.columns = ["ciudad", "conteo"]
        geo_counts["lat"] = geo_counts["ciudad"].map(lambda x: CITY_COORDS[x][0])
        geo_counts["lon"] = geo_counts["ciudad"].map(lambda x: CITY_COORDS[x][1])
        
        fig_map = px.scatter_mapbox(
            geo_counts, lat="lat", lon="lon",
            hover_name="ciudad", hover_data={"conteo": True, "lat": False, "lon": False},
            size="conteo", color="conteo",
            color_continuous_scale=px.colors.sequential.Plasma,
            size_max=35, zoom=4,
            mapbox_style="carto-darkmatter"
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No hay datos geográficos suficientes tras la normalización.")
else:
    st.info("Columna de ciudad no disponible.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 17. SANKEY (MEJORADO CON AGRUPACIÓN Y FILTRO)
# -----------------------------------------------------------------------------
st.subheader("🔀 Flujo: Origen Campaña ➡ Unidad Destino")

check_sankey_mkt = st.checkbox("Solo origen iNBest.marketing", value=True)
df_sankey = df_post_f.copy()
if check_sankey_mkt:
    df_sankey = df_sankey[df_sankey["pipeline_marketing"] == "iNBest.marketing"]

if not df_sankey.empty:
    df_sankey["Origen_Clean"] = df_sankey["Origen"].apply(limpiar_origen_sankey)

    # --- FILTRO IMPORTANTE: Eliminar los que devolvieron None (como "Lead de AWS") ---
    df_sankey = df_sankey.dropna(subset=["Origen_Clean"])

    sankey_g = df_sankey.groupby(["Origen_Clean", "deal_unidad_norm"])["deal_id"].nunique().reset_index(name="value")
    
    all_sources = list(sankey_g["Origen_Clean"].unique())
    all_targets = list(sankey_g["deal_unidad_norm"].unique())
    all_nodes = all_sources + all_targets
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    link_source = sankey_g["Origen_Clean"].map(node_map).tolist()
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
    fig_san.update_layout(template="plotly_dark", height=600, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_san, use_container_width=True)
else:
    st.info("No hay datos suficientes para el diagrama de flujo.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 18. DESGLOSE FINAL
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
            .agg(num_deals=("deal_id", "nunique"))
            .reset_index()
            .sort_values("num_deals", ascending=False)
        )
        st.dataframe(top_pipelines, use_container_width=True, hide_index=True)

    with col_t2:
        st.markdown("**Detalle de etapas dentro de un pipeline comercial**")
        pipelines_disp = sorted(df_post_f["pipeline_comercial"].unique())
        if pipelines_disp:
            pipeline_sel = st.selectbox("Selecciona pipeline comercial", options=pipelines_disp)
            df_etapas = df_post_f[df_post_f["pipeline_comercial"] == pipeline_sel]
            etapas = (
                df_etapas.groupby("etapa_comercial")
                .agg(num_deals=("deal_id", "nunique"))
                .reset_index()
                .sort_values("num_deals", ascending=False)
            )
            st.dataframe(etapas, use_container_width=True, hide_index=True)

st.markdown("<br><br><div style='text-align: center; color: #475569;'>Desarrollado por Héctor Plascencia | 2025 🚀</div>", unsafe_allow_html=True)





