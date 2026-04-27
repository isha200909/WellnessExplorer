# Steps to Build the Singapore Wellness Neighbourhoods App

## Project Overview

This document outlines the end-to-end steps taken to build the **Singapore Wellness Neighbourhoods Streamlit App (V2)** — an interactive tool that profiles Singapore's 42 populated planning areas across health and wellness dimensions, groups them into neighbourhood archetypes, and serves personalised recommendations to users.

The project answers a core question: *Which Singapore planning area best supports your lifestyle and wellness goals?*

---

## Table of Contents

1. [Step 1 — Combining Files from data.gov.sg](#step-1--combining-files-from-datagovsg)
2. [Step 2 — Engineering the Active Living Index](#step-2--engineering-the-active-living-index)
3. [Step 3 — Machine Learning Modelling](#step-3--machine-learning-modelling)
4. [Step 4 — Streamlit App Development](#step-4--streamlit-app-development)

---

## Step 1 — Combining Files from data.gov.sg

**Script:** `build_master_table.py`  
**Output:** `master_table.csv` (55 rows × 15 columns)

### 1.1 Data Sources

All geospatial datasets were downloaded from [data.gov.sg](https://data.gov.sg) and [related government portals](https://beta.data.gov.sg) in **GeoJSON format**. Population data was sourced from SingStat. The following files were used:

| File | Type | Records | Source | Description |
|------|------|---------|--------|-------------|
| `MasterPlan2019PlanningAreaBoundaryNoSea.geojson` | Polygon | 55 | URA / data.gov.sg | Singapore Master Plan 2019 planning area boundaries |
| `PolygonParksandNatureReserves.geojson` | Polygon | 450 | NParks / data.gov.sg | Park and nature reserve boundaries (polygon areas) |
| `ParkConnectorLoop.geojson` | LineString | 883 | NParks / data.gov.sg | Park Connector Network (PCN) path segments |
| `SportSGSportFacilitiesGEOJSON.geojson` | Point | 45 | SportSG / data.gov.sg | Major public sports facilities (stadiums, pools, sport centres) |
| `SportSGDUSSportFacilities.geojson` | Point | 183 | SportSG / data.gov.sg | Dual-Use Scheme (DUS) school sports facilities |
| `GymsSGGEOJSON.geojson` | Point | 159 | ActiveSG / data.gov.sg | Community gyms (Gyms@SG network) |
| `HealthierEateries.geojson` | Point | 1,829 | HPB / data.gov.sg | HPB Healthier Dining Programme eateries |
| `CHASClinics.geojson` | Point | 1,193 | MOH / data.gov.sg | CHAS-subsidised GP clinics |
| `EldercareServices.geojson` | Point | 133 | AIC / data.gov.sg | Eldercare centres and day-care facilities |
| `pop_trends_2025.csv` | CSV | 100,928 | SingStat | Resident population by planning area, subzone, age group, sex |

> **Note:** Private fitness studios (yoga studios, CrossFit gyms, climbing gyms, etc.) were not available on data.gov.sg and were instead fetched at runtime via the **OpenStreetMap Overpass API**, querying tags such as `leisure=fitness_centre`, `amenity=gym`, `sport=yoga/pilates/climbing/crossfit`.

### 1.2 Loading Planning Area Boundaries

The URA Master Plan 2019 GeoJSON was loaded first to establish the 55 planning area polygons. All datasets were **re-projected to EPSG:3414 (SVY21)** — Singapore's national projected coordinate system — so that distances and areas are measured accurately in metres.

```python
pa_gdf = gpd.read_file("MasterPlan2019PlanningAreaBoundaryNoSea.geojson")
pa_gdf = pa_gdf.to_crs("EPSG:3414")
pa_gdf['land_area_sqkm'] = pa_gdf.geometry.area / 1_000_000
```

### 1.3 Spatial Aggregation by Planning Area

Each dataset was spatially joined against the planning area polygons to produce per-planning-area counts or measurements. Two main techniques were used:

**For point datasets** (sports facilities, clinics, eateries, etc.):
- A reusable `count_points_in_pa()` function performed a spatial join (`predicate='within'`) and counted how many points fall within each planning area polygon.

**For polygon datasets** (parks):
- Park polygons were **clipped** to each planning area boundary using `gpd.clip()`, and the clipped areas were summed in square metres. This approach captures the *actual green area* available within each boundary, not just a count of parks.

**For line datasets** (Park Connector Network):
- PCN line segments were clipped to each planning area boundary and lengths were summed, then converted from metres to kilometres.

### 1.4 Deduplication

Two datasets required deduplication to avoid double-counting the same facilities:

- **Gyms@SG vs. SportSG**: Some gyms in the `GymsSGGEOJSON.geojson` file are located inside SportSG complexes already counted under `SportSGSportFacilitiesGEOJSON.geojson`. These were identified by matching postal codes, and duplicates were removed (159 → 145 community gyms).

- **OpenStreetMap vs. Public Facilities**: OSM-sourced private fitness studios were deduplicated against all public gym locations using a **50-metre proximity buffer** — any OSM point within 50m of a known public facility was excluded (~342 → ~308 private studios).

### 1.5 Parsing HTML-Encoded GeoJSON Attributes

Several GeoJSON files (Gyms@SG, CHAS Clinics, Eldercare, DUS) store their attributes inside KML-style HTML tables embedded in a `Description` field, rather than as standard GeoJSON properties. A helper function `parse_html_field()` was written to extract specific fields using regex:

```python
def parse_html_field(html_string, field_name):
    pattern = rf'<th>{field_name}</th>\s*<td>([^<]*)</td>'
    match = re.search(pattern, html_string, re.IGNORECASE)
    return match.group(1).strip() if match else None
```

### 1.6 Population Data Processing

Population data from `pop_trends_2025.csv` (100,928 rows at subzone × age group × sex × dwelling type level) was aggregated to the planning area level:

- **Total population**: Sum of all residents per planning area
- **Seniors (65+)**: Sum of residents in age groups `65_to_69`, `70_to_74`, `75_to_79`, `80_to_84`, `85_to_89`, `90_and_over`
- **Derived metrics**: Population density (per sq km) and percentage aged 65+

### 1.7 Master Table Output

The final `master_table.csv` contains **55 planning areas × 15 columns**:

| Column | Description |
|--------|-------------|
| `planning_area` | Planning area name (uppercase) |
| `park_area_sqm` | Total park area (sq metres) |
| `pcn_length_km` | Park Connector Network length (km) |
| `count_major_infra` | Public sports facilities (SportSG) |
| `count_school_facilities` | DUS school sports facilities |
| `count_community_gyms` | Community gyms (deduplicated) |
| `count_private_fitness` | Private fitness studios (OpenStreetMap) |
| `count_healthier_eateries` | HPB Healthier Dining eateries |
| `count_chas_clinics` | CHAS-subsidised clinics |
| `count_eldercare_facilities` | Eldercare centres |
| `total_population` | Total resident population |
| `pop_aged_65_over` | Residents aged 65+ |
| `pop_aged_65_over_pct` | Percentage aged 65+ |
| `pop_density` | People per sq km |
| `land_area_sqkm` | Land area (sq km) |

---

## Step 2 — Engineering the Active Living Index

**Script:** `build_active_living_index.py`  
**Input:** `master_table.csv`  
**Output:** `normalized_master_table.csv` (42 rows × 30 columns)

### 2.1 Filtering to Populated Planning Areas

Of the 55 planning areas in the master table, 13 have zero or negligible population (e.g., Western Islands, Straits View, Southern Islands). These were filtered out, leaving **42 populated planning areas** for analysis and scoring.

### 2.2 Grouping Infrastructure into Four Wellness Categories

The 9 raw infrastructure variables were grouped into four meaningful wellness categories, reflecting different dimensions of an active and healthy lifestyle:

| Category | Raw Variables Included |
|----------|------------------------|
| **Active Commute** | `pcn_length_km`, `park_area_sqm` |
| **Structured Fitness** | `count_major_infra`, `count_community_gyms`, `count_private_fitness`, `count_school_facilities` |
| **Healthy Nutrition** | `count_healthier_eateries` |
| **Preventative Care** | `count_chas_clinics`, `count_eldercare_facilities` |

### 2.3 Min-Max Normalisation

Each raw variable was independently normalised to a **0–1 scale** using Min-Max normalisation, so that variables with larger absolute values (e.g., `park_area_sqm` in the millions) do not unfairly dominate variables with smaller values (e.g., `count_major_infra` in the tens):

```
norm_x = (x - min) / (max - min)
```

### 2.4 Computing Category Scores

Each category score was computed as the **mean of the normalised variables** within that category:

```
cat_active_commute     = mean(norm_pcn_length_km, norm_park_area_sqm)
cat_structured_fitness = mean(norm_count_major_infra, norm_count_community_gyms,
                               norm_count_private_fitness, norm_count_school_facilities)
cat_healthy_nutrition  = norm_count_healthier_eateries
cat_preventative_care  = mean(norm_count_chas_clinics, norm_count_eldercare_facilities)
```

> **Important:** These four category scores (ranging 0–1) also serve as the **"Town Vector"** for Cosine Similarity matching in the Streamlit app.

### 2.5 Weighted Active Living Score

A composite **Active Living Score** (0–100) was computed as a weighted sum of the four category scores:

```
Active_Living_Score = (
    cat_active_commute     × 0.35 +
    cat_structured_fitness × 0.35 +
    cat_healthy_nutrition  × 0.20 +
    cat_preventative_care  × 0.10
) × 100
```

| Category | Weight | Rationale |
|----------|--------|-----------|
| Active Commute | 35% | Green spaces and connectors are foundational to daily active living |
| Structured Fitness | 35% | Gyms and sports facilities enable deliberate exercise |
| Healthy Nutrition | 20% | Healthy eating options complement active lifestyles |
| Preventative Care | 10% | Clinics and eldercare support long-term health maintenance |

Each of the 42 populated planning areas was then ranked from 1 (highest score) to 42 (lowest).

---

## Step 3 — Machine Learning Modelling

**Script:** `v2/build_clustering_v2.py`  
**Input:** `normalized_master_table.csv`  
**Output:** `v2/normalized_clustering_master_table_v2.csv` + pickled model files

Two machine learning models underpin the app:

1. **Macro Model — K-Means Clustering**: Groups the 42 planning areas into 4 "Neighbourhood Archetypes"
2. **Micro Model — Cosine Similarity (Content-Based Filtering)**: Matches a user's lifestyle preferences to the most suitable planning areas

### 3.1 Feature Engineering — Per-Capita Metrics

Raw infrastructure counts favour larger, more densely populated areas. To cluster towns by their *character* rather than their *size*, 7 per-capita features were engineered:

| Feature | Formula |
|---------|---------|
| `park_area_per_capita` | `park_area_sqm / total_population` |
| `pcn_length_per_10k_pop` | `(pcn_length_km / total_population) × 10,000` |
| `sports_facilities_per_10k_pop` | `(count_major_infra + count_school_facilities) / total_population × 10,000` |
| `all_gyms_per_10k_pop` | `(count_community_gyms + count_private_fitness) / total_population × 10,000` |
| `healthier_eateries_per_10k_pop` | `count_healthier_eateries / total_population × 10,000` |
| `chas_clinics_per_10k_pop` | `count_chas_clinics / total_population × 10,000` |
| `eldercare_per_1k_seniors` | `count_eldercare_facilities / pop_aged_65_over × 1,000` |

Two demographic features were also included as clustering inputs: `pop_density` and `pop_aged_65_over_pct`. This gave a final feature matrix of **9 variables** across 42 planning areas.

> **Note:** `Active_Living_Score` was deliberately excluded from the clustering inputs — it is a derived output metric and including it would circularly influence the clusters.

### 3.2 StandardScaler (Z-Score Normalisation)

K-Means uses Euclidean distance, which means features with larger absolute scales would dominate the clustering. A **StandardScaler** was applied to transform each of the 9 features to mean = 0, standard deviation = 1:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The fitted scaler was saved as `models/scaler.pkl` for reproducibility.

### 3.3 Elbow Method — Choosing K

KMeans was run for K = 1 through K = 10, and the within-cluster sum of squares (inertia) was plotted for each K. The elbow chart (`elbow_plot_v2.png`) showed a clear inflection point at **K = 4**:

- K = 3 over-generalised, lumping 34 diverse planning areas into one cluster
- **K = 4** produced well-separated, interpretable groups
- K = 5+ offered diminishing returns with no meaningful new separation

### 3.4 K-Means Clustering (K = 4)

```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
df['cluster_id'] = kmeans.fit_predict(X_scaled)
```

The trained model was saved as `models/kmeans_model.pkl`. Pickling the model enables:
- Assigning new towns to existing clusters without re-training
- Consistent results across environments
- MLOps best practices for portfolio demonstration

### 3.5 PCA for 2D Visualisation

**Principal Component Analysis (PCA)** was applied to reduce the 9-dimensional feature space to 2 dimensions for interactive visualisation:

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]
```

| Principal Component | Captures | Explained Variance |
|--------------------|----------|-------------------|
| PC1 | Green Space & Connectivity | ~40% |
| PC2 | Urban Density & Services | ~21% |
| **Total** | | **~61%** |

The PCA model was saved as `models/pca_model.pkl` and the 2D scatter plot saved as `pca_archetype_plot_v2.png`.

### 3.6 Archetype Naming

Each cluster was automatically assigned a human-readable name by comparing its centroid feature values against the overall mean of all 42 planning areas. The naming logic uses ratio thresholds:

| Archetype | No. of Towns | Key Distinguishing Characteristics |
|-----------|-------------|-------------------------------------|
| **Mature Active Heartland** | ~26 | High density (1.5× avg), older population (1.2× avg), strong public sports infrastructure and eldercare |
| **Green Haven** | ~4 | Very high park area (7.6× avg), very long PCN (8.9× avg), very low density — nature-rich, residential fringe areas |
| **Urban Wellness Hub** | ~2 | Extremely high private gyms (10.8× avg) and healthier eateries (9.3× avg) — CBD-character areas |
| **Quiet Living Enclave** | ~10 | Low density (0.4× avg), below-average public infrastructure — typically affluent private estate areas |

### 3.7 Town Vectors for Cosine Similarity

The four category scores (`cat_active_commute`, `cat_structured_fitness`, `cat_healthy_nutrition`, `cat_preventative_care`) for each planning area were exported as `town_vectors_v2.csv`. These form the "Town Vector" used for personalized matching in the Streamlit app.

### 3.8 Final Output

All outputs from the clustering step were saved to the `v2/` folder:

| File | Description |
|------|-------------|
| `normalized_clustering_master_table_v2.csv` | Final dataset: 42 planning areas × 41 columns (includes cluster IDs, archetypes, PCA coordinates) |
| `town_vectors_v2.csv` | 4-dimensional category scores per town, for cosine similarity |
| `archetype_rationale_v2.csv` | Summary of each archetype: key features and member towns |
| `elbow_plot_v2.png` | Elbow Method chart used to select K = 4 |
| `pca_archetype_plot_v2.png` | Static PCA scatter plot of all 42 towns coloured by archetype |
| `models/kmeans_model.pkl` | Trained K-Means model |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/pca_model.pkl` | Fitted PCA transformer |
| `models/clustering_config.pkl` | Clustering feature names, K, and random_state |

---

## Step 4 — Streamlit App Development

**Script:** `v2/app_v2.py`  
**Inputs:** `v2/normalized_clustering_master_table_v2.csv`, `MasterPlan2019PlanningAreaBoundaryNoSea.geojson`  
**URL:** `http://localhost:8501`

### 4.1 App Architecture Overview

The app is structured around a **sidebar for user inputs** and a **multi-section main panel** for outputs. The core innovation is replacing simple lookup logic with two ML-backed computations:

| Approach | Description |
|----------|-------------|
| **K-Means Archetype** (Macro) | Each town is pre-assigned an archetype from the clustering step. The app reads and displays this label. |
| **Cosine Similarity Synergy Score** (Micro) | Computed live at runtime: the user's preference vector is compared against each town's category vector to produce a 0–100% match score. |

### 4.2 Sidebar Inputs

The sidebar collects three user inputs:

1. **Planning Area** — Dropdown of all 42 populated planning areas, sorted alphabetically
2. **Age Group** — Four options: Young Adult (18–35), Mid-Career (36–50), Mature Adult (51–64), Senior (65+)
3. **Wellness Interests** — Multi-select from 7 lifestyle options: Jogging/Walking, Cycling, Gym & Fitness, Swimming/Sports, Healthy Eating, Clinics, Eldercare

### 4.3 Cosine Similarity Synergy Score — Core Feature

The Synergy Score answers: *"How well does this planning area match my lifestyle preferences?"*

**Step 1 — Build User Vector**: The user's selections are mapped to a 4-dimensional preference vector using a fixed lifestyle-to-category weight table (e.g., selecting "Jogging/Walking" contributes weight 1.0 to Active Commute and 0.2 to Structured Fitness). The lifestyle vector is then blended 70/30 with an age-based baseline (e.g., seniors have higher weight on Preventative Care):

| Age Group | Active Commute | Structured Fitness | Healthy Nutrition | Preventative Care |
|-----------|:--------------:|:-----------------:|:-----------------:|:-----------------:|
| Young (18–35) | 0.30 | 0.40 | 0.25 | 0.05 |
| Mid-Career (36–50) | 0.30 | 0.30 | 0.25 | 0.15 |
| Mature (51–64) | 0.25 | 0.25 | 0.25 | 0.25 |
| Senior (65+) | 0.20 | 0.15 | 0.25 | 0.40 |

**Step 2 — Build Town Vector**: The town's 4 category scores from `normalized_clustering_master_table_v2.csv` form its vector.

**Step 3 — Compute Cosine Similarity**:

$$\text{Synergy Score} = \cos(\theta) = \frac{\mathbf{U} \cdot \mathbf{T}}{\|\mathbf{U}\| \times \|\mathbf{T}\|} \times 100$$

Cosine Similarity measures the *angle* between two vectors — meaning it captures the proportional *mix* of preferences, not just overall intensity. This is the same algorithm used in production recommendation systems.

### 4.4 Key App Sections

The main panel is divided into seven sections, rendered in sequence:

#### Section 1 — Score Cards & Preference Profile
- **Synergy Score Gauge**: A Plotly indicator gauge that visually shows the 0–100% match score, colour-coded (green ≥ 75%, orange ≥ 50%, red < 50%)
- **Metric Cards**: Active Living Score, total population, and percentage of seniors
- **User Preference Bars**: Progress bars showing the user's 4-dimensional preference profile (Active Commute, Structured Fitness, Healthy Nutrition, Preventative Care)

#### Section 2 — Location Map & Radar Chart
- **Interactive Map**: A Plotly Scattermap rendering all planning area boundaries from the GeoJSON file, with the selected area highlighted in blue and all others in light grey. Centroid markers show planning area names on hover. The map auto-centres on the selected area.
- **3-Layer Radar Chart**: Overlays three profiles on a single spider chart:
  1. **Town Profile** (blue, solid fill) — the selected town's category scores
  2. **User Preferences** (green, dashed) — the user's preference vector
  3. **National Average** (orange, dotted) — mean of all 42 towns
  
  Alignment between the blue and green shapes indicates high synergy.

#### Section 3 — Archetype Profile & PCA View
- **Archetype Description**: Displays the town's assigned archetype (e.g., "Green Haven") with a plain-English description and its 4 category score values
- **Interactive PCA Plot**: A 2D scatter of all 42 towns coloured by archetype, with the selected town highlighted as a red star. Hovering over any point shows the town name and Active Living Score.

#### Section 4 — Personalised Wellness Nudges
Generates tailored recommendations by comparing the town's facility counts against the national average for each metric the user cares about:
- **✅ Strength**: Facility availability ≥ 1.5× national average
- **👍 Adequate**: Facility availability between 0.8× and 1.5× average
- **⚠️ Gap**: Facility availability < 0.8× average — suggests exploring neighbouring areas

Additional nudges are generated based on age group (e.g., senior-specific alerts for eldercare availability) and archetype (e.g., "Green Haven" tips for outdoor activities).

#### Section 5 — Infrastructure Snapshot
A summary table showing raw counts for all 9 facility types, the count relative to the national average, and a traffic-light status indicator (🟢 at or above average, 🟡 slightly below, 🔴 well below).

#### Section 6 — Top Matching Towns
Computes the Cosine Similarity Synergy Score against all other 41 planning areas using the user's preference vector and returns the **Top 5 best-matching towns**, excluding the currently selected one. Each match shows the town name, synergy score, and archetype.

#### Section 7 — All 42 Planning Areas
An expandable panel showing:
- Archetype distribution breakdown (e.g., "62% Mature Active Heartland · 10% Green Haven · …")
- Full ranked table of all 42 planning areas with Active Living Score, archetype, population, and senior percentage
- Scatter plot of Active Living Score vs. population density, with bubble size linked to total population and colour linked to archetype

### 4.5 Running the App

From the `v2/` folder:

```bash
streamlit run app_v2.py
```

The app opens in the browser at `http://localhost:8501`.

---

## Full Pipeline — Execution Order

Run the following scripts in sequence from the project root folder:

```bash
# Step 1: Combine GeoJSON files and population data into master table
python build_master_table.py

# Step 2: Normalise variables and compute the Active Living Index
python build_active_living_index.py

# Step 3 (Optional): Expand aggregate counts into individual facility rows
python build_consolidated_master.py

# Step 4: Navigate to v2 folder, run K-Means clustering and PCA
cd v2
python build_clustering_v2.py

# Step 5: Launch the Streamlit app
streamlit run app_v2.py
```

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| `geopandas` | Reading GeoJSON files, spatial joins (`sjoin`), polygon clipping (`clip`) |
| `pandas` / `numpy` | Data manipulation and aggregation |
| `shapely` | Point geometry creation for OpenStreetMap data |
| `requests` | Live querying of the OpenStreetMap Overpass API |
| `scikit-learn` | `StandardScaler`, `KMeans`, `PCA`, `cosine_similarity` |
| `matplotlib` | Static Elbow Method and PCA plots |
| `streamlit` | Interactive web app framework |
| `plotly` | Radar charts, gauge charts, interactive maps, PCA scatter, score scatter |

---

*Data sources: URA Master Plan 2019, SportSG, NParks, HPB, MOH (CHAS), AIC (Eldercare), SingStat, OpenStreetMap*  
*Singapore Wellness Neighbourhoods Project — February 2026*
