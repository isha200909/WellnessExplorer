"""
Wellness Neighbourhoods Streamlit App V2
=========================================
Enhanced interactive web app with:
- Cosine Similarity-based Synergy Score (replaces rules-based)
- Folium map showing selected planning area
- Gauge chart for Synergy Score
- User profile overlay on radar chart

Input: normalized_clustering_master_table_v2.csv
Output: Interactive web app at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Singapore Wellness Neighbourhoods",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("normalized_clustering_master_table_v2.csv")
    return df

@st.cache_data
def load_geojson():
    """Load planning area boundaries for map visualization"""
    geojson_path = "MasterPlan2019PlanningAreaBoundaryNoSea.geojson"
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            return json.load(f)
    return None

df = load_data()
geojson = load_geojson()

# ============================================================
# CONSTANTS & MAPPINGS
# ============================================================
AGE_GROUPS = {
    "Young Adult (18-35)": "young",
    "Mid-Career (36-50)": "mid",
    "Mature Adult (51-64)": "mature",
    "Senior (65+)": "senior"
}

LIFESTYLE_OPTIONS = [
    "Jogging/Walking",
    "Cycling", 
    "Gym & Fitness",
    "Swimming/Sports",
    "Healthy Eating",
    "Clinics",
    "Eldercare"
]

# Map lifestyles to category weights for user vector
# Each lifestyle contributes to one or more wellness categories
LIFESTYLE_WEIGHTS = {
    "Jogging/Walking": {"active_commute": 1.0, "structured_fitness": 0.2},
    "Cycling": {"active_commute": 1.0},
    "Gym & Fitness": {"structured_fitness": 1.0},
    "Swimming/Sports": {"structured_fitness": 0.8, "active_commute": 0.2},
    "Healthy Eating": {"healthy_nutrition": 1.0},
    "Clinics": {"preventative_care": 0.7},
    "Eldercare": {"preventative_care": 1.0}
}

# Age-based preference adjustments
AGE_WEIGHTS = {
    "young": {"active_commute": 0.3, "structured_fitness": 0.4, "healthy_nutrition": 0.25, "preventative_care": 0.05},
    "mid": {"active_commute": 0.3, "structured_fitness": 0.3, "healthy_nutrition": 0.25, "preventative_care": 0.15},
    "mature": {"active_commute": 0.25, "structured_fitness": 0.25, "healthy_nutrition": 0.25, "preventative_care": 0.25},
    "senior": {"active_commute": 0.2, "structured_fitness": 0.15, "healthy_nutrition": 0.25, "preventative_care": 0.4}
}

# Map lifestyles to raw metrics for nudges
LIFESTYLE_TO_METRICS = {
    "Jogging/Walking": ["park_area_sqm", "pcn_length_km"],
    "Cycling": ["pcn_length_km"],
    "Gym & Fitness": ["count_community_gyms", "count_private_fitness"],
    "Swimming/Sports": ["count_major_infra", "count_school_facilities"],
    "Healthy Eating": ["count_healthier_eateries"],
    "Clinics": ["count_chas_clinics"],
    "Eldercare": ["count_eldercare_facilities"]
}

METRIC_LABELS = {
    "park_area_sqm": "Park Area",
    "pcn_length_km": "Park Connector Length",
    "count_major_infra": "Sports Facilities",
    "count_school_facilities": "School Sports",
    "count_community_gyms": "Community Gyms",
    "count_private_fitness": "Private Fitness",
    "count_healthier_eateries": "Healthier Eateries",
    "count_chas_clinics": "CHAS Clinics",
    "count_eldercare_facilities": "Eldercare Centres"
}

ARCHETYPE_DESCRIPTIONS = {
    "Mature Active Heartland": "A well-established residential area with strong public sports infrastructure and eldercare support. Ideal for families and seniors who prefer community-based wellness options.",
    "Green Haven": "Exceptionally green with vast park areas and extensive park connector networks. Perfect for nature lovers who enjoy outdoor activities and serene environments.",
    "Urban Wellness Hub": "A bustling urban centre with abundant private gyms, fitness studios, and healthy dining options. Best suited for young professionals seeking convenience and variety.",
    "Quiet Living Enclave": "A low-density residential area with a quieter pace of life. May have fewer public facilities but offers privacy and access to private estates."
}

ARCHETYPE_COLORS = {
    "Mature Active Heartland": "#1f77b4",
    "Green Haven": "#2ca02c",
    "Urban Wellness Hub": "#ff7f0e",
    "Quiet Living Enclave": "#9467bd"
}

# ============================================================
# HELPER FUNCTIONS: COSINE SIMILARITY SYNERGY
# ============================================================

def create_user_vector(selected_lifestyles, age_group):
    """
    Create a normalized user preference vector based on selected lifestyles and age.
    Returns a 4-dimensional vector: [active_commute, structured_fitness, healthy_nutrition, preventative_care]
    """
    # Initialize vector
    user_vector = {
        "active_commute": 0.0,
        "structured_fitness": 0.0,
        "healthy_nutrition": 0.0,
        "preventative_care": 0.0
    }
    
    # Add lifestyle preferences
    for lifestyle in selected_lifestyles:
        if lifestyle in LIFESTYLE_WEIGHTS:
            for category, weight in LIFESTYLE_WEIGHTS[lifestyle].items():
                user_vector[category] += weight
    
    # If no lifestyles selected, use age-based defaults
    if not selected_lifestyles:
        user_vector = AGE_WEIGHTS[age_group].copy()
    else:
        # Normalize to 0-1 scale
        max_val = max(user_vector.values()) if max(user_vector.values()) > 0 else 1
        for key in user_vector:
            user_vector[key] = user_vector[key] / max_val
        
        # Apply age adjustment (blend 70% lifestyle, 30% age)
        age_weights = AGE_WEIGHTS[age_group]
        for key in user_vector:
            user_vector[key] = 0.7 * user_vector[key] + 0.3 * age_weights[key]
    
    return np.array([
        user_vector["active_commute"],
        user_vector["structured_fitness"],
        user_vector["healthy_nutrition"],
        user_vector["preventative_care"]
    ])


def create_town_vector(row):
    """
    Create a town vector from the 4 category scores.
    """
    return np.array([
        row['cat_active_commute'],
        row['cat_structured_fitness'],
        row['cat_healthy_nutrition'],
        row['cat_preventative_care']
    ])


def compute_synergy_score_cosine(user_vector, town_vector):
    """
    Compute Synergy Score using Cosine Similarity.
    Returns a score from 0-100.
    """
    # Reshape for sklearn
    user_vec = user_vector.reshape(1, -1)
    town_vec = town_vector.reshape(1, -1)
    
    # Cosine similarity returns value between -1 and 1
    # For our normalized positive vectors, it will be 0 to 1
    similarity = cosine_similarity(user_vec, town_vec)[0][0]
    
    # Scale to 0-100
    synergy_score = similarity * 100
    
    return max(0, min(100, synergy_score))


def generate_nudges(row, selected_lifestyles, age_group):
    """
    Generate personalised wellness nudges based on user profile.
    """
    nudges = {"strengths": [], "adequate": [], "gaps": []}
    
    # Check metrics relevant to selected lifestyles
    for lifestyle in selected_lifestyles:
        metrics = LIFESTYLE_TO_METRICS.get(lifestyle, [])
        for metric in metrics:
            if metric not in row:
                continue
            
            value = row[metric]
            avg = df[metric].mean()
            ratio = value / avg if avg > 0 else 0
            label = METRIC_LABELS.get(metric, metric)
            
            if ratio >= 1.5:
                nudges["strengths"].append(f"✅ **{label}**: {ratio:.1f}× above average")
            elif ratio >= 0.8:
                nudges["adequate"].append(f"👍 **{label}**: adequately served")
            else:
                nudges["gaps"].append(f"⚠️ **{label}**: below average — consider exploring neighbouring areas")
    
    # Age-specific nudges
    if age_group == "senior":
        if row.get('count_eldercare_facilities', 0) == 0:
            nudges["gaps"].append("⚠️ No eldercare centres in this area — check adjacent planning areas")
        if row.get('pop_aged_65_over_pct', 0) > 20:
            nudges["strengths"].append("✅ Active senior community in this area")
    
    elif age_group == "young":
        if row.get('count_private_fitness', 0) > df['count_private_fitness'].mean():
            nudges["strengths"].append("✅ Great variety of private fitness studios nearby")
        nudges["adequate"].append("💡 Tip: Check out ActiveSG facilities for affordable gym access")
    
    # Archetype-specific tips
    archetype = row.get('archetype', '')
    if archetype == "Green Haven":
        nudges["strengths"].append("🌿 Leverage the extensive park connectors for cycling and jogging")
    elif archetype == "Urban Wellness Hub":
        nudges["strengths"].append("🏙️ Take advantage of the diverse healthy dining options")
    
    return nudges


def find_best_match_towns(user_vector, current_town, top_n=5):
    """
    Find top N towns that best match the user's lifestyle preferences using cosine similarity.
    """
    scores = []
    for _, row in df.iterrows():
        if row['planning_area'] == current_town:
            continue
        town_vector = create_town_vector(row)
        score = compute_synergy_score_cosine(user_vector, town_vector)
        scores.append({
            'planning_area': row['planning_area'],
            'synergy_score': score,
            'archetype': row['archetype'],
            'Active_Living_Score': row['Active_Living_Score']
        })
    
    scores_df = pd.DataFrame(scores)
    return scores_df.nlargest(top_n, 'synergy_score')


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_synergy_gauge(synergy_score):
    """
    Create a gauge chart for the Synergy Score.
    """
    # Determine color based on score
    if synergy_score >= 75:
        color = "#2ca02c"  # Green
    elif synergy_score >= 50:
        color = "#ff7f0e"  # Orange
    else:
        color = "#d62728"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=synergy_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Synergy Score", 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffebee'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': synergy_score
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_radar_chart_with_user(row, user_vector, national_avg):
    """
    Create radar chart comparing town's category scores with user preferences and national average.
    """
    categories = ['Active Commute', 'Structured Fitness', 'Healthy Nutrition', 'Preventative Care']
    
    town_values = [
        row['cat_active_commute'],
        row['cat_structured_fitness'],
        row['cat_healthy_nutrition'],
        row['cat_preventative_care']
    ]
    
    avg_values = [
        national_avg['cat_active_commute'],
        national_avg['cat_structured_fitness'],
        national_avg['cat_healthy_nutrition'],
        national_avg['cat_preventative_care']
    ]
    
    # User preferences (already 0-1 scale)
    user_values = user_vector.tolist()
    
    fig = go.Figure()
    
    # Town profile
    fig.add_trace(go.Scatterpolar(
        r=town_values + [town_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=row['planning_area'],
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    # User preferences
    fig.add_trace(go.Scatterpolar(
        r=user_values + [user_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Your Preferences',
        line_color='#2ca02c',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(dash='dash')
    ))
    
    # National average
    fig.add_trace(go.Scatterpolar(
        r=avg_values + [avg_values[0]],
        theta=categories + [categories[0]],
        fill='none',
        name='National Average',
        line_color='#ff7f0e',
        opacity=0.6,
        line=dict(dash='dot')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        width=600,
        height=500,
        margin=dict(l=120, r=120, t=50, b=80)
    )
    
    return fig


def create_planning_area_map(planning_area, geojson_data):
    """
    Create a Plotly map highlighting the selected planning area.
    """
    if geojson_data is None:
        return None
    
    # Find the selected planning area feature
    selected_feature = None
    all_features = []
    
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        name = props.get('PLN_AREA_N', props.get('Name', '')).upper()
        
        if name == planning_area.upper():
            selected_feature = feature
        all_features.append({
            'name': name,
            'is_selected': name == planning_area.upper()
        })
    
    if selected_feature is None:
        return None
    
    # Create a GeoJSON with all areas but highlight selected
    fig = go.Figure()
    
    # Collect centroids for hover labels
    centroids = []
    
    # Add all planning areas (light gray)
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        name = props.get('PLN_AREA_N', props.get('Name', '')).upper()
        geom = feature.get('geometry', {})
        
        if geom.get('type') == 'Polygon':
            coords_list = [geom['coordinates']]
        elif geom.get('type') == 'MultiPolygon':
            coords_list = geom['coordinates']
        else:
            continue
        
        is_selected = name == planning_area.upper()
        
        # Calculate centroid from the first polygon ring
        all_lons = []
        all_lats = []
        
        for coords in coords_list:
            if isinstance(coords[0][0], list):
                # Handle nested coordinates
                for ring in coords:
                    lons = [c[0] for c in ring]
                    lats = [c[1] for c in ring]
                    all_lons.extend(lons)
                    all_lats.extend(lats)
                    
                    fig.add_trace(go.Scattermap(
                        lon=lons,
                        lat=lats,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(31, 119, 180, 0.6)' if is_selected else 'rgba(200, 200, 200, 0.3)',
                        line=dict(
                            color='#1f77b4' if is_selected else '#888888',
                            width=3 if is_selected else 1
                        ),
                        name='',
                        showlegend=False,
                        hoverinfo='skip'  # Disable polygon boundary hover
                    ))
            else:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                all_lons.extend(lons)
                all_lats.extend(lats)
                
                fig.add_trace(go.Scattermap(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.6)' if is_selected else 'rgba(200, 200, 200, 0.3)',
                    line=dict(
                        color='#1f77b4' if is_selected else '#888888',
                        width=3 if is_selected else 1
                    ),
                    name='',
                    showlegend=False,
                    hoverinfo='skip'  # Disable polygon boundary hover
                ))
        
        # Calculate centroid for this planning area
        if all_lons and all_lats:
            centroid_lon = sum(all_lons) / len(all_lons)
            centroid_lat = sum(all_lats) / len(all_lats)
            centroids.append({
                'name': name,
                'lon': centroid_lon,
                'lat': centroid_lat,
                'is_selected': is_selected
            })
    
    # Add invisible centroid markers for hover labels
    for c in centroids:
        marker_color = '#1f77b4' if c['is_selected'] else 'rgba(0,0,0,0)'
        marker_size = 12 if c['is_selected'] else 1
        fig.add_trace(go.Scattermap(
            lon=[c['lon']],
            lat=[c['lat']],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=marker_color,
                opacity=0.01 if not c['is_selected'] else 0.8
            ),
            hoverinfo='text',
            hovertext=f"<b>{c['name']}</b>" + (" (Selected)" if c['is_selected'] else ""),
            showlegend=False
        ))
    
    # Calculate center from selected feature
    if selected_feature:
        geom = selected_feature.get('geometry', {})
        if geom.get('type') == 'Polygon':
            coords = geom['coordinates'][0]
        elif geom.get('type') == 'MultiPolygon':
            coords = geom['coordinates'][0][0]
        else:
            coords = [[103.8, 1.35]]
        
        center_lon = sum(c[0] for c in coords) / len(coords)
        center_lat = sum(c[1] for c in coords) / len(coords)
    else:
        center_lon, center_lat = 103.8, 1.35
    
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lon=center_lon, lat=center_lat),
            zoom=11
        ),
        showlegend=False,  # Hide legend to avoid clutter
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        # Move modebar to bottom left to avoid overlap
        modebar=dict(
            orientation='h',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )
    
    return fig


def create_pca_plot(df, selected_pa):
    """
    Create PCA scatter plot showing all planning areas with the selected one highlighted.
    Labels shown only on hover (except selected area).
    """
    # Check if PCA columns exist
    if 'pca_x' not in df.columns or 'pca_y' not in df.columns:
        return None
    
    fig = go.Figure()
    
    # Plot each archetype (markers only, labels on hover)
    for archetype in ARCHETYPE_COLORS:
        mask = (df['archetype'] == archetype) & (df['planning_area'] != selected_pa)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'pca_x'],
                y=df.loc[mask, 'pca_y'],
                mode='markers',
                name=archetype,
                marker=dict(
                    size=14,
                    color=ARCHETYPE_COLORS[archetype],
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                text=df.loc[mask, 'planning_area'],
                hovertemplate='<b>%{text}</b><br>Score: %{customdata:.1f}<extra></extra>',
                customdata=df.loc[mask, 'Active_Living_Score']
            ))
    
    # Highlight selected area (red star only, no text label)
    selected_row = df[df['planning_area'] == selected_pa]
    if not selected_row.empty:
        fig.add_trace(go.Scatter(
            x=selected_row['pca_x'],
            y=selected_row['pca_y'],
            mode='markers',
            name=f'{selected_pa} (Selected)',
            marker=dict(
                size=22,
                color='red',
                symbol='star',
                line=dict(color='black', width=2)
            ),
            text=[selected_pa],
            hovertemplate='<b>%{text}</b><br>Score: %{customdata:.1f}<extra></extra>',
            customdata=selected_row['Active_Living_Score']
        ))
    
    # Calculate axis ranges with padding
    x_min, x_max = df['pca_x'].min(), df['pca_x'].max()
    y_min, y_max = df['pca_y'].min(), df['pca_y'].max()
    x_padding = (x_max - x_min) * 0.12
    y_padding = (y_max - y_min) * 0.12
    
    fig.update_layout(
        xaxis_title="Green Space & Connectivity →",
        yaxis_title="↑ Urban Density & Services",
        xaxis=dict(
            range=[x_min - x_padding, x_max + x_padding],
            constrain='domain'
        ),
        yaxis=dict(
            range=[y_min - y_padding, y_max + y_padding],
            constrain='domain'
        ),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=30, b=100)
    )
    
    return fig


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("<div style='margin-top: 180px;'></div>", unsafe_allow_html=True)

# Planning Area Selection
selected_pa = st.sidebar.selectbox(
    "📍 Select Planning Area",
    options=sorted(df['planning_area'].unique()),
    index=0
)

# Age Group
selected_age = st.sidebar.selectbox(
    "👤 Your Age Group",
    options=list(AGE_GROUPS.keys()),
    index=0
)
age_group = AGE_GROUPS[selected_age]

# Wellness Interests
selected_lifestyles = st.sidebar.multiselect(
    "💪 Your Wellness Interests",
    options=LIFESTYLE_OPTIONS,
    default=["Jogging/Walking", "Healthy Eating"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
- **Cosine Similarity** for mathematically precise synergy matching
- **User Profile Radar** overlay on town comparison
- **Interactive Map** of selected planning area
- **Gauge Chart** for visual synergy display
""")
st.sidebar.markdown("---")
st.sidebar.markdown("*Built with ❤️ for healthier communities*")

# ============================================================
# MAIN PANEL
# ============================================================
st.markdown("<h1 style='font-size: 3rem;'>🏃 Wellness Explorer</h1>", unsafe_allow_html=True)
st.markdown("##### *Which location best supports your lifestyle and wellness goals?*")
st.markdown("---")

# Get selected town data
town_row = df[df['planning_area'] == selected_pa].iloc[0]
national_avg = df.mean(numeric_only=True)

# Create user vector
user_vector = create_user_vector(selected_lifestyles, age_group)
town_vector = create_town_vector(town_row)

# Compute cosine similarity synergy score
synergy_score = compute_synergy_score_cosine(user_vector, town_vector)

# ============================================================
# SECTION 1: HEADER & SCORE CARDS
# ============================================================
st.header(f"🏘️ {selected_pa}")

col_arch, col_color = st.columns([4, 1])
with col_arch:
    archetype = town_row['archetype']
    st.markdown(f"**Archetype:** {archetype}")
with col_color:
    arch_color = ARCHETYPE_COLORS.get(archetype, "#888888")
    st.markdown(f'<span style="background-color: {arch_color}; color: white; padding: 4px 12px; border-radius: 4px;">{archetype}</span>', unsafe_allow_html=True)

st.markdown("---")

# Score cards with gauge
col_gauge, col_metrics = st.columns([1, 2])

with col_gauge:
    gauge_fig = create_synergy_gauge(synergy_score)
    st.plotly_chart(gauge_fig, key="gauge_chart")

with col_metrics:
    mcol1, mcol2, mcol3 = st.columns(3)
    
    with mcol1:
        st.metric(
            label="Active Living Score",
            value=f"{town_row['Active_Living_Score']:.1f}",
            delta=f"Rank #{int(town_row['rank'])}"
        )
    
    with mcol2:
        st.metric(
            label="Population",
            value=f"{town_row['total_population']:,.0f}"
        )
    
    with mcol3:
        st.metric(
            label="Seniors (65+)",
            value=f"{town_row['pop_aged_65_over_pct']:.1f}%"
        )
    
    # User vector display
    st.markdown("**Your Preference Profile:**")
    pref_cols = st.columns(4)
    categories = ['Active Commute', 'Structured Fitness', 'Healthy Nutrition', 'Preventative Care']
    for i, (cat, val) in enumerate(zip(categories, user_vector)):
        with pref_cols[i]:
            st.markdown(f"*{cat}*")
            st.progress(float(val))
            st.caption(f"{val:.0%}")

# ============================================================
# SECTION 2: MAP & RADAR CHART
# ============================================================
st.markdown("---")
col_map, col_radar = st.columns([1, 1])

with col_map:
    st.subheader("📍 Location Map")
    map_fig = create_planning_area_map(selected_pa, geojson)
    if map_fig:
        st.plotly_chart(map_fig, key="map_chart")
        st.caption(f"🔵 Highlighted area: **{selected_pa}** planning area boundary")
    else:
        st.info("Map visualization requires the planning area GeoJSON file.")

with col_radar:
    st.subheader("📊 Profile Comparison")
    radar_fig = create_radar_chart_with_user(town_row, user_vector, national_avg)
    st.plotly_chart(radar_fig, key="radar_chart")

# ============================================================
# SECTION 3: ARCHETYPE PROFILE
# ============================================================
st.markdown("---")

st.subheader("🏷️ Archetype Profile")
st.markdown(f"**{town_row['archetype']}**")
st.markdown(ARCHETYPE_DESCRIPTIONS.get(town_row['archetype'], ""))

st.markdown("**Category Breakdown:**")
cat_cols = st.columns(4)
with cat_cols[0]:
    st.metric("Active Commute", f"{town_row['cat_active_commute']:.2f}")
with cat_cols[1]:
    st.metric("Structured Fitness", f"{town_row['cat_structured_fitness']:.2f}")
with cat_cols[2]:
    st.metric("Healthy Nutrition", f"{town_row['cat_healthy_nutrition']:.2f}")
with cat_cols[3]:
    st.metric("Preventative Care", f"{town_row['cat_preventative_care']:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("🗺️ Neighbourhood Landscape Map")
st.info(
    f"🗺️ Where Does {selected_pa} Sit Among Singapore's Neighbourhoods?\n\n"
    f"📌 Each dot represents one planning area. The further right, the greener and more connected. "
    f"The higher up, the more urban and service-rich. {selected_pa} is marked with a :red[★]."
)

pca_fig = create_pca_plot(df, selected_pa)
if pca_fig:
    st.plotly_chart(pca_fig, key="pca_chart", use_container_width=True)
else:
    st.info("PCA visualization requires running build_clustering_v2.py first.")

# ============================================================
# SECTION 4: PERSONALISED NUDGES
# ============================================================
st.markdown("---")
st.subheader("💡 Personalised Wellness Nudges")

if selected_lifestyles:
    nudges = generate_nudges(town_row, selected_lifestyles, age_group)
    
    col_str, col_gap = st.columns(2)
    
    with col_str:
        st.markdown("**Strengths**")
        if nudges["strengths"]:
            for s in nudges["strengths"]:
                st.markdown(s)
        else:
            st.markdown("*No standout strengths based on your selection*")
    
    with col_gap:
        st.markdown("**Areas to Consider**")
        if nudges["gaps"]:
            for g in nudges["gaps"]:
                st.markdown(g)
        else:
            st.markdown("*All relevant facilities are adequately available*")
    
    if nudges["adequate"]:
        with st.expander("👍 Adequately Served"):
            for a in nudges["adequate"]:
                st.markdown(a)
else:
    st.info("Select your wellness interests in the sidebar to see personalised recommendations.")

# ============================================================
# SECTION 5: INFRASTRUCTURE SNAPSHOT
# ============================================================
st.markdown("---")
st.subheader("📋 Infrastructure Snapshot")

infra_metrics = [
    ('Park Area (sq m)', 'park_area_sqm'),
    ('PCN Length (km)', 'pcn_length_km'),
    ('Major Sports Facilities', 'count_major_infra'),
    ('School Sports Facilities', 'count_school_facilities'),
    ('Community Gyms', 'count_community_gyms'),
    ('Private Fitness Studios', 'count_private_fitness'),
    ('Healthier Eateries', 'count_healthier_eateries'),
    ('CHAS Clinics', 'count_chas_clinics'),
    ('Eldercare Facilities', 'count_eldercare_facilities')
]

infra_data = []
for label, col in infra_metrics:
    town_val = town_row[col]
    avg_val = national_avg[col]
    ratio = town_val / avg_val if avg_val > 0 else 0
    
    if isinstance(town_val, float):
        if town_val > 1000:
            town_str = f"{town_val:,.0f}"
        else:
            town_str = f"{town_val:.2f}"
    else:
        town_str = str(int(town_val))
    
    status = "🟢" if ratio >= 1.0 else "🟡" if ratio >= 0.5 else "🔴"
    
    infra_data.append({
        'Metric': label,
        'Value': town_str,
        'vs Avg': f"{ratio:.1f}x",
        'Status': status
    })

infra_df = pd.DataFrame(infra_data)
st.dataframe(infra_df, hide_index=True, key="infra_df")

# ============================================================
# SECTION 6: TOP MATCHING TOWNS
# ============================================================
st.markdown("---")
st.subheader("🎯 Top Matching Towns for Your Lifestyle")

if selected_lifestyles:
    st.caption("*Ranked by Cosine Similarity to your preference profile*")
    best_matches = find_best_match_towns(user_vector, selected_pa)
    
    if not best_matches.empty:
        for i, (_, match) in enumerate(best_matches.iterrows(), 1):
            col_match, col_score, col_arch = st.columns([2, 1, 1])
            with col_match:
                st.markdown(f"**{i}. {match['planning_area']}**")
            with col_score:
                st.markdown(f"Synergy: `{match['synergy_score']:.1f}%`")
            with col_arch:
                st.markdown(f"*{match['archetype']}*")
else:
    st.info("Select your wellness interests to find matching towns.")

# ============================================================
# SECTION 7: ALL PLANNING AREAS
# ============================================================
st.markdown("---")
with st.expander("📊 All 42 Planning Areas"):
    st.subheader("Planning Areas Overview")
    
    # Calculate archetype percentages
    archetype_counts = df['archetype'].value_counts()
    total_areas = len(df)
    archetype_pcts = []
    for arch in archetype_counts.index:
        pct = (archetype_counts[arch] / total_areas) * 100
        archetype_pcts.append(f"{pct:.0f}% {arch}")
    archetype_summary = " • ".join(archetype_pcts)
    st.markdown(f"*{archetype_summary}*")
    
    # Table
    display_cols = ['planning_area', 'rank', 'Active_Living_Score', 'archetype',
                    'total_population', 'pop_aged_65_over_pct']
    display_df = df[display_cols].sort_values('rank')
    display_df.columns = ['Planning Area', 'Rank', 'Score', 'Archetype', 'Population', 'Seniors %']
    st.dataframe(display_df, hide_index=True, key="display_df")
    
    # Scatter plot
    st.subheader("Score vs Population Density")
    fig_scatter = px.scatter(
        df,
        x='pop_density',
        y='Active_Living_Score',
        color='archetype',
        hover_name='planning_area',
        size='total_population',
        size_max=50,
        color_discrete_map=ARCHETYPE_COLORS,
        labels={
            'pop_density': 'Population Density (per sq km)',
            'Active_Living_Score': 'Active Living Score',
            'archetype': 'Archetype'
        }
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, key="scatter_chart")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Data sources: URA Master Plan 2019, SportSG, HPB, CHAS, AIC, OpenStreetMap | 
    Singapore Wellness Neighbourhoods Project — February 2026<br>
    Visualisations: Cosine Similarity Synergy • PCA Visualization • Interactive Map • Gauge Charts<br>
    Modelling: K-Means Clustering for Archetypes • Cosine Similarity for Synergy Matching • Rule-Based Nudges</small><br>
    </div>
    """,
    unsafe_allow_html=True
)
