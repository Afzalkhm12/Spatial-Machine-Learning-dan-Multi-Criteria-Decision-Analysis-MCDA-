import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import plotly.graph_objects as go
import numpy as np
import json
import random

# ==============================================================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==============================================================================
st.set_page_config(page_title="Urban SDSS | Semarang", page_icon="🏙️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50;}
    .stTabs [data-baseweb="tab-list"] {gap: 15px;}
    .stTabs [data-baseweb="tab"] {background-color: #ffffff; color: #2c3e50; border-radius: 4px; padding: 10px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .stTabs [aria-selected="true"] {background-color: #2980b9; color: white !important; font-weight: bold;}
    .insight-box {background-color: #e8f4f8; padding: 20px; border-left: 5px solid #2980b9; border-radius: 5px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. STATE MANAGEMENT (UNTUK FITUR RANDOM INPUT)
# ==============================================================================
if 'w_heat' not in st.session_state:
    st.session_state.w_heat = 50
if 'w_pop' not in st.session_state:
    st.session_state.w_pop = 50
if 'w_transit' not in st.session_state:
    st.session_state.w_transit = 50

def randomize_weights():
    st.session_state.w_heat = random.randint(0, 100)
    st.session_state.w_pop = random.randint(0, 100)
    st.session_state.w_transit = random.randint(0, 100)

# ==============================================================================
# 3. DATA PIPELINE (CACHE & JSON METRICS)
# ==============================================================================
@st.cache_data
def load_infrastructure():
    try:
        df = pd.read_csv("Semarang_Final_Master_600DPI.csv")
        gdf = gpd.read_file("Semarang_Final_Spatial_600DPI.geojson")
        boundary_gdf = gpd.read_file("Semarang_Boundary.geojson")
        
        if gdf.crs != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)
        if boundary_gdf.crs != "EPSG:4326": boundary_gdf = boundary_gdf.to_crs(epsg=4326)
        
        with open("metrics.json", "r") as f:
            ml_metrics = json.load(f)
            
        return df, gdf, boundary_gdf, ml_metrics
    except Exception as e:
        return None, None, None, None

df, gdf, boundary_gdf, ml_metrics = load_infrastructure()

if df is None:
    st.error("🚨 CRITICAL PIPELINE FAILURE: Spatial data, Boundary, or metrics.json missing.")
    st.stop()

# ==============================================================================
# 4. DYNAMIC TOPSIS ENGINE (ON-THE-FLY MCDA)
# ==============================================================================
def run_dynamic_topsis(data, w_heat, w_pop, w_transit):
    X = data[['ML_Risk_Prediction', 'Population', 'Network_Dist_Transit_m']].copy()
    
    norm = pd.DataFrame()
    norm['ML_Risk_Prediction'] = (X['ML_Risk_Prediction'] - X['ML_Risk_Prediction'].min()) / (X['ML_Risk_Prediction'].max() - X['ML_Risk_Prediction'].min())
    norm['Population'] = (X['Population'] - X['Population'].min()) / (X['Population'].max() - X['Population'].min())
    norm['Network_Dist_Transit_m'] = (X['Network_Dist_Transit_m'].max() - X['Network_Dist_Transit_m']) / (X['Network_Dist_Transit_m'].max() - X['Network_Dist_Transit_m'].min())
    
    weights = np.array([w_heat, w_pop, w_transit])
    if weights.sum() == 0: weights = np.array([1, 1, 1])
    weights = weights / weights.sum()
    
    V = norm * weights
    ideal = V.max()
    anti_ideal = V.min()
    
    S_plus = np.sqrt(((V - ideal)**2).sum(axis=1))
    S_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
    
    score = S_minus / (S_plus + S_minus)
    return score.fillna(0)

# ==============================================================================
# 5. SIDEBAR: INTERACTIVE DECISION CONTROLS
# ==============================================================================
st.sidebar.title("🎛️ Dynamic Policy Sliders")
st.sidebar.markdown("Adjust criteria weights to simulate urban planning scenarios on-the-fly.")

# Tombol Randomize untuk Presentasi
st.sidebar.button("🎲 Randomize Scenario (Demo)", on_click=randomize_weights, use_container_width=True)
st.sidebar.caption("Generates random weights to prove real-time TOPSIS recalculation.")
st.sidebar.markdown("---")

st.sidebar.subheader("Multi-Criteria Weights")
w_heat = st.sidebar.slider("🔥 Heat Risk (UHI) Importance:", 0, 100, key="w_heat")
w_pop = st.sidebar.slider("👥 Population Density Importance:", 0, 100, key="w_pop")
w_transit = st.sidebar.slider("🚶 Transit Accessibility Importance:", 0, 100, key="w_transit")

st.sidebar.markdown("---")
top_n = st.sidebar.slider("🎯 Select Top N Targets:", 5, 100, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 AI Model Telemetry")
st.sidebar.info(
    f"**Champion Model:** {ml_metrics.get('Model_Name', 'XGBoost (Optuna Tuned)')}\n\n"
    f"**Accuracy ($R^2$):** {ml_metrics.get('R2_Score', 0.8088):.4f}\n\n"
    f"**Error (RMSE):** {ml_metrics.get('RMSE', 1.4758):.4f} °C\n\n"
    f"**Spatial Lags:** Micro & Macro Context Enabled"
)

# EKSEKUSI TOPSIS DINAMIS
gdf['Dynamic_Score'] = run_dynamic_topsis(gdf, w_heat, w_pop, w_transit)
gdf['Dynamic_Rank'] = gdf['Dynamic_Score'].rank(ascending=False).astype(int)

top_sites = gdf[gdf['Dynamic_Rank'] <= top_n].copy()

# ==============================================================================
# 6. MAIN DASHBOARD HEADER
# ==============================================================================
st.title("🏙️ Real-Time Spatial Decision Support System")
st.markdown("**Dynamic Multi-Criteria Decision Analysis powered by XGBoost Inference.**")

avg_heat = top_sites['ML_Risk_Prediction'].mean()
avg_pop = top_sites['Population'].mean()

st.markdown(f"""
<div class="insight-box">
    <h4>💡 Real-Time Target Acquisition Analysis</h4>
    <p>Based on your current weight configuration <b>(Heat: {w_heat}, Pop: {w_pop}, Transit: {w_transit})</b>, the AI has isolated <b>{top_n} optimal zones</b>. These priority areas face an average predicted heat risk of <b>{avg_heat:.2f}°C</b> and serve an average local population of <b>{avg_pop:,.0f} people</b>.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🗺️ Folium Spatial Masterplan", "📉 Pareto Trade-off Analytics", "🗄️ Processed Matrix"])

# ------------------------------------------------------------------------------
# TAB 1: FOLIUM RENDERING (WITH BOUNDARY)
# ------------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processed Grids", f"{len(gdf):,}")
    col2.metric("Highest Suitability Score", f"{gdf['Dynamic_Score'].max():.4f}")
    col3.metric("Selected Targets", str(top_n))

    # Inisialisasi Peta
    center_lat = boundary_gdf.geometry.centroid.y.mean()
    center_lon = boundary_gdf.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11.5, tiles="CartoDB positron")

    # LAYER 1: BATAS ADMINISTRASI SEMARANG
    folium.GeoJson(
        boundary_gdf,
        name="Semarang City Boundary",
        style_function=lambda feature: {
            'fillColor': 'none',
            'color': '#2c3e50',
            'weight': 3,
            'dashArray': '5, 5',
            'fillOpacity': 0
        }
    ).add_to(m)

    # Colormap
    colormap = cm.LinearColormap(colors=['#f7f7f7', '#41b6c4', '#253494'], vmin=gdf['Dynamic_Score'].min(), vmax=gdf['Dynamic_Score'].max())
    colormap.caption = 'Dynamic Suitability Score'
    m.add_child(colormap)

    # LAYER 2: GRID URBAN
    folium.GeoJson(
        gdf,
        name="Urban Grids",
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Dynamic_Score']),
            'color': 'black',
            'weight': 0.2,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['Dynamic_Rank', 'Dynamic_Score', 'ML_Risk_Prediction', 'Population'],
            aliases=['Rank:', 'Suitability Score:', 'Heat Risk (°C):', 'Population:'],
            localize=True
        )
    ).add_to(m)

    # LAYER 3: MARKER LOKASI PRIORITAS
    for idx, row in top_sites.iterrows():
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=folium.Popup(f"<b>Rank: {row['Dynamic_Rank']}</b><br>Score: {row['Dynamic_Score']:.4f}", max_width=200),
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st.caption("⚠️ Rendering Engine: Folium (Leaflet.js). Rendering thousands of HTML polygons may cause browser lag when adjusting sliders.")
    st_folium(m, width="100%", height=600, returned_objects=[])

# ------------------------------------------------------------------------------
# TAB 2: PARETO FRONTIER ANALYTICS
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("### 🎯 Pareto Optimization Frontier")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gdf['ML_Risk_Prediction'], y=gdf['Population'], mode='markers',
        marker=dict(color='rgba(150, 150, 150, 0.3)', size=6),
        name='All Urban Zones', hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=top_sites['ML_Risk_Prediction'], y=top_sites['Population'], mode='markers',
        marker=dict(color='#e74c3c', size=12, line=dict(color='white', width=1)),
        text=top_sites['Dynamic_Rank'],
        hovertemplate="<b>Rank %{text}</b><br>Heat Risk: %{x:.2f}°C<br>Pop: %{y:.0f}<extra></extra>",
        name=f'Top {top_n} Selected Targets'
    ))
    fig.update_layout(
        title="AI Selection Proof: Heat Risk vs Population Density",
        xaxis_title="Predicted UHI Risk (°C) -> Higher requires intervention",
        yaxis_title="Population Density -> Higher serves more people",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: DYNAMIC RAW MATRIX
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("### 🗄️ On-the-fly Processed Matrix")
    display_cols = ['Dynamic_Rank', 'Dynamic_Score', 'ML_Risk_Prediction', 'Population', 'Network_Dist_Transit_m', 'NDBI', 'NDVI']
    df_display = gdf.sort_values(by='Dynamic_Rank')[display_cols]
    df_display.columns = ['Priority Rank', 'Suitability Score', 'Predicted Heat (°C)', 'Population Density', 'Transit Dist. (m)', 'Concrete (NDBI)', 'Greenery (NDVI)']
    
    st.dataframe(df_display.head(top_n).style.background_gradient(subset=['Suitability Score'], cmap='Blues'), use_container_width=True, height=500)