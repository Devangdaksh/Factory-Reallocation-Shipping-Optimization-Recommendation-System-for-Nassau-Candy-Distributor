import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# PAGE CONFIG

st.set_page_config(
    page_title="Nassau Candy | Factory Reallocation & Shipping Optimizer",
    page_icon="🍬",
    layout="wide",
)

DATA_PATHS = [Path("Nassau_Candy_Distributor.csv"), Path("data/Nassau Candy Distributor.csv")]


def resolve_data_path():
    for path in DATA_PATHS:
        if path.exists():
            return path
    return DATA_PATHS[-1]

# REFERENCE DATA
FACTORIES = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176,  -81.088371),
    "Sugar Shack":       (48.119140,  -96.181150),
    "Secret Factory":    (41.446333,  -90.565487),
    "The Other Factory": (35.117500,  -89.971107),
}

CURRENT_ASSIGNMENT = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

# Approximate state / province centroids, used to locate customers since the raw file has no lat/lon fields.
STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130), "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123), "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104), "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141), "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783), "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828), "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278), "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486), "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805), "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101), "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095), "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696), "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353), "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374), "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011), "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051), "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012), "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917), "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755), "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007), "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345), "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434), "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968), "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954903), "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    # Canadian provinces present in the dataset
    "Alberta": (55.0011, -115.0022), "British Columbia": (53.7267, -127.6476),
    "Manitoba": (53.7609, -98.8139), "New Brunswick": (46.5653, -66.4619),
    "Newfoundland and Labrador": (53.1355, -57.6604), "Nova Scotia": (44.6820, -63.7443),
    "Ontario": (51.2538, -85.3232), "Prince Edward Island": (46.5107, -63.4168),
    "Quebec": (52.9399, -73.5491), "Saskatchewan": (52.9399, -106.4509),
}

SHIP_MODE_BASE_DAYS = {"Same Day": 0.5, "First Class": 2.0, "Second Class": 3.5, "Standard Class": 5.0}
SHIP_MODE_SPEED_KM_PER_DAY = {"Same Day": 3000, "First Class": 1200, "Second Class": 700, "Standard Class": 400}

# Assumed freight-cost sensitivity used only to translate distance changes into
# an estimated profit impact for "what-if" reassignment scenarios.
FREIGHT_COST_PER_UNIT_PER_KM = 0.0018

NUM_FEATURES = ["Distance_km", "Units", "Sales"]
CAT_FEATURES = ["Division", "Ship Mode", "Region", "Factory"]


# HELPERS

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in kilometers."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Loading and preparing data...")
def load_data(path):
    df = pd.read_csv(path)
    df["Factory"] = df["Product Name"].map(CURRENT_ASSIGNMENT)
    df = df.dropna(subset=["Factory"]).copy()

    lat = df["State/Province"].map(lambda s: STATE_CENTROIDS.get(s, (np.nan, np.nan))[0])
    lon = df["State/Province"].map(lambda s: STATE_CENTROIDS.get(s, (np.nan, np.nan))[1])
    df["Cust_Lat"], df["Cust_Lon"] = lat, lon
    df = df.dropna(subset=["Cust_Lat", "Cust_Lon"]).copy()

    df["Factory_Lat"] = df["Factory"].map(lambda f: FACTORIES[f][0])
    df["Factory_Lon"] = df["Factory"].map(lambda f: FACTORIES[f][1])
    df["Distance_km"] = haversine_km(df["Factory_Lat"], df["Factory_Lon"], df["Cust_Lat"], df["Cust_Lon"])

    # --- Proxy lead time -----------------------------------------------
    # EDA on this dataset found Ship Date is corrupted (offsets of ~2.5-4.5
    # years vs Order Date), making it unusable as a real lead-time target.
    # A distance + ship-mode based proxy is modeled instead, with light noise,
    # so the predictive/optimization pipeline below has a workable target.
    rng = np.random.default_rng(42)
    base = df["Ship Mode"].map(SHIP_MODE_BASE_DAYS).astype(float)
    speed = df["Ship Mode"].map(SHIP_MODE_SPEED_KM_PER_DAY).astype(float)
    noise = rng.normal(0, 0.6, size=len(df))
    df["Lead_Time_Days"] = (base + df["Distance_km"] / speed + noise).clip(lower=0.3).round(2)

    df["Margin_Pct"] = (df["Gross Profit"] / df["Sales"]).replace([np.inf, -np.inf], np.nan) * 100
    return df


@st.cache_resource(show_spinner="Training predictive models...")
def train_models(df):
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df["Lead_Time_Days"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = ColumnTransformer(
        [("num", StandardScaler(), NUM_FEATURES), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)]
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    fitted, metrics = {}, {}
    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "MAE": float(mean_absolute_error(y_test, preds)),
            "R2": float(r2_score(y_test, preds)),
        }
        fitted[name] = pipe

    best_name = max(metrics, key=lambda n: metrics[n]["R2"])
    return fitted, metrics, best_name


@st.cache_data(show_spinner="Running scenario simulations across factories...")
def simulate_all_factories(df, _model, model_name):
    """For every historical order, predict lead time if it had shipped from
    each candidate factory, then aggregate to product level."""
    rows = []
    for factory, (flat, flon) in FACTORIES.items():
        sim = df[NUM_FEATURES + ["Division", "Ship Mode", "Region", "Product Name", "Gross Profit"]].copy()
        sim["Factory"] = factory
        dist = haversine_km(flat, flon, df["Cust_Lat"].values, df["Cust_Lon"].values)
        sim["Distance_km"] = dist
        pred = _model.predict(sim[NUM_FEATURES + CAT_FEATURES])
        sim["Predicted_Lead_Time"] = pred
        sim["Freight_Delta"] = FREIGHT_COST_PER_UNIT_PER_KM * sim["Units"] * (dist - df["Distance_km"].values)
        sim["Adjusted_Profit"] = sim["Gross Profit"] - sim["Freight_Delta"]
        rows.append(sim)
    full = pd.concat(rows, ignore_index=True)

    agg = (
        full.groupby(["Product Name", "Factory"])
        .agg(
            Avg_Distance_km=("Distance_km", "mean"),
            Avg_Predicted_Lead_Time=("Predicted_Lead_Time", "mean"),
            Total_Adjusted_Profit=("Adjusted_Profit", "sum"),
            Total_Units=("Units", "sum"),
        )
        .reset_index()
    )
    return agg


def build_recommendations(agg, priority_weight):
    """priority_weight: 0 = pure profit, 1 = pure speed."""
    recs = []
    for product, grp in agg.groupby("Product Name"):
        current_factory = CURRENT_ASSIGNMENT[product]
        current = grp[grp["Factory"] == current_factory].iloc[0]

        g = grp.copy()
        g["Speed_Score"] = 1 - (g["Avg_Predicted_Lead_Time"] - g["Avg_Predicted_Lead_Time"].min()) / (
            g["Avg_Predicted_Lead_Time"].max() - g["Avg_Predicted_Lead_Time"].min() + 1e-9
        )
        g["Profit_Score"] = (g["Total_Adjusted_Profit"] - g["Total_Adjusted_Profit"].min()) / (
            g["Total_Adjusted_Profit"].max() - g["Total_Adjusted_Profit"].min() + 1e-9
        )
        g["Combined_Score"] = priority_weight * g["Speed_Score"] + (1 - priority_weight) * g["Profit_Score"]
        best = g.sort_values("Combined_Score", ascending=False).iloc[0]

        lead_reduction_pct = (
            (current["Avg_Predicted_Lead_Time"] - best["Avg_Predicted_Lead_Time"])
            / current["Avg_Predicted_Lead_Time"]
            * 100
        )
        profit_impact = best["Total_Adjusted_Profit"] - current["Total_Adjusted_Profit"]
        profit_impact_pct = profit_impact / (abs(current["Total_Adjusted_Profit"]) + 1e-9) * 100

        recs.append(
            {
                "Product": product,
                "Current Factory": current_factory,
                "Recommended Factory": best["Factory"],
                "Current Lead Time (days)": round(current["Avg_Predicted_Lead_Time"], 2),
                "Recommended Lead Time (days)": round(best["Avg_Predicted_Lead_Time"], 2),
                "Lead Time Reduction (%)": round(lead_reduction_pct, 1),
                "Profit Impact ($)": round(profit_impact, 2),
                "Profit Impact (%)": round(profit_impact_pct, 1),
                "Reassignment Needed": best["Factory"] != current_factory,
            }
        )
    return pd.DataFrame(recs).sort_values("Lead Time Reduction (%)", ascending=False)


def cluster_routes(df, n_clusters=3):
    route = (
        df.groupby(["Region", "Factory"])
        .agg(Avg_Distance_km=("Distance_km", "mean"), Avg_Lead_Time=("Lead_Time_Days", "mean"), Avg_Margin_Pct=("Margin_Pct", "mean"), Orders=("Row ID", "count"))
        .reset_index()
    )
    X = route[["Avg_Distance_km", "Avg_Lead_Time", "Avg_Margin_Pct"]]
    X_scaled = (X - X.mean()) / X.std()
    km = KMeans(n_clusters=min(n_clusters, len(route)), random_state=42, n_init=10)
    route["Cluster"] = km.fit_predict(X_scaled)

    order = route.groupby("Cluster")["Avg_Lead_Time"].mean().sort_values().index
    label_map = {c: lbl for c, lbl in zip(order, ["Fast Routes", "Moderate Routes", "Slow Routes"][: len(order)])}
    route["Cluster Label"] = route["Cluster"].map(label_map)
    return route


# LOAD DATA

try:
    data = load_data(resolve_data_path())
except FileNotFoundError:
    st.warning(
        "Bundled data file not found — please upload the dataset to continue."
    )
    uploaded = st.file_uploader("Upload Nassau Candy Distributor.csv", type="csv")
    if uploaded is None:
        st.stop()
    data = load_data(uploaded)

# SIDEBAR — GLOBAL FILTERS

st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Factory Reallocation & Shipping Optimizer")

product_options = ["All Products"] + sorted(data["Product Name"].unique())
region_options = ["All Regions"] + sorted(data["Region"].unique())
ship_mode_options = ["All Ship Modes"] + sorted(data["Ship Mode"].unique())

sel_product = st.sidebar.selectbox("Product", product_options)
sel_region = st.sidebar.selectbox("Region", region_options)
sel_ship_mode = st.sidebar.selectbox("Ship Mode", ship_mode_options)
priority = st.sidebar.slider(
    "Optimization priority — Profit ⟷ Speed", 0.0, 1.0, 0.5, 0.05,
    help="0 = optimize purely for profit, 1 = optimize purely for lead-time reduction",
)

filtered = data.copy()
if sel_product != "All Products":
    filtered = filtered[filtered["Product Name"] == sel_product]
if sel_region != "All Regions":
    filtered = filtered[filtered["Region"] == sel_region]
if sel_ship_mode != "All Ship Modes":
    filtered = filtered[filtered["Ship Mode"] == sel_ship_mode]

st.sidebar.markdown("---")
st.sidebar.metric("Orders in view", f"{len(filtered):,}")
st.sidebar.metric("Total Sales in view", f"${filtered['Sales'].sum():,.0f}")

# TRAIN MODELS + SIMULATE (on full dataset, filters apply to display only)

fitted_models, metrics, best_model_name = train_models(data)
best_model = fitted_models[best_model_name]
agg_sim = simulate_all_factories(data, best_model, best_model_name)
recommendations = build_recommendations(agg_sim, priority)
route_clusters = cluster_routes(data)

# HEADER
st.title("🏭 Factory Reallocation & Shipping Optimization Recommendation System")
st.caption("Nassau Candy Distributor — decision intelligence for product-to-factory assignment")

with st.expander("⚠️ Data quality note (read before interpreting lead times)", expanded=False):
    st.markdown(
        """
The raw **Ship Date** field in this dataset is unusable as a real shipping-lead-time
signal: it is offset from **Order Date** by roughly **2.5 to 4.5 years** across rows,
with no consistent pattern, and clearly reflects a data entry/export error rather
than real shipment timing.

To keep the modeling, clustering, and recommendation pipeline meaningful, this app
instead trains on a **modeled proxy lead time**, derived from great-circle distance
(factory → customer state/province centroid) and Ship Mode, with light random noise.
Treat all lead-time figures below as **directionally useful estimates for comparing
factory options relative to one another**, not as ground-truth historical shipping
times. Profit-impact figures are similarly built from an assumed freight-cost
sensitivity (\\${:.4f} per unit per km), applied consistently across all scenarios.
        """.format(FREIGHT_COST_PER_UNIT_PER_KM)
    )

# tabs
tabs = st.tabs(
    [
        "📌 Overview",
        "🏭 Factory Optimization Simulator",
        "🔀 What-If Scenario Analysis",
        "📊 Recommendation Dashboard",
        "⚠️ Risk & Impact Panel",
        "🧭 Route & Product Clustering",
        "🤖 Model Performance",
    ]
)

# --------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# --------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{len(data):,}")
    c2.metric("Products", data["Product Name"].nunique())
    c3.metric("Factories", len(FACTORIES))
    c4.metric("Avg Modeled Lead Time", f"{data['Lead_Time_Days'].mean():.2f} days")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            data.groupby("Factory")["Row ID"].count().reset_index(name="Orders"),
            x="Factory", y="Orders", title="Order Volume by Current Factory", color="Factory",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.box(data, x="Region", y="Lead_Time_Days", color="Region", title="Modeled Lead Time by Region")
        st.plotly_chart(fig, use_container_width=True)

    # Simple factory / customer map
    map_rows = [{"Name": k, "Lat": v[0], "Lon": v[1], "Type": "Factory"} for k, v in FACTORIES.items()]
    map_df = pd.DataFrame(map_rows)
    fig = px.scatter_geo(
        map_df, lat="Lat", lon="Lon", text="Name", scope="north america",
        title="Factory Locations", color_discrete_sequence=["#8B0000"],
    )
    fig.update_traces(marker=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 2 — FACTORY OPTIMIZATION SIMULATOR
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Select a product to see predicted performance across every factory")
    sim_product = st.selectbox(
        "Product", sorted(data["Product Name"].unique()),
        index=sorted(data["Product Name"].unique()).index(sel_product) if sel_product != "All Products" else 0,
        key="sim_product",
    )
    prod_sim = agg_sim[agg_sim["Product Name"] == sim_product].copy()
    prod_sim["Is Current"] = prod_sim["Factory"] == CURRENT_ASSIGNMENT[sim_product]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            prod_sim, x="Factory", y="Avg_Predicted_Lead_Time", color="Is Current",
            title=f"Predicted Lead Time by Factory — {sim_product}",
            labels={"Avg_Predicted_Lead_Time": "Predicted Lead Time (days)"},
            color_discrete_map={True: "#2E8B57", False: "#B0B0B0"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            prod_sim, x="Factory", y="Total_Adjusted_Profit", color="Is Current",
            title=f"Estimated Total Profit by Factory — {sim_product}",
            labels={"Total_Adjusted_Profit": "Adjusted Gross Profit ($)"},
            color_discrete_map={True: "#2E8B57", False: "#B0B0B0"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        prod_sim[["Factory", "Avg_Distance_km", "Avg_Predicted_Lead_Time", "Total_Adjusted_Profit", "Is Current"]]
        .sort_values("Avg_Predicted_Lead_Time")
        .style.format({"Avg_Distance_km": "{:.0f} km", "Avg_Predicted_Lead_Time": "{:.2f} days", "Total_Adjusted_Profit": "${:,.0f}"}),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# TAB 3 — WHAT-IF SCENARIO ANALYSIS
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Compare current assignment vs. recommended assignment")
    wi_product = st.selectbox(
        "Product", sorted(data["Product Name"].unique()), key="wi_product",
    )
    row = recommendations[recommendations["Product"] == wi_product].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Factory", row["Current Factory"])
    c2.metric(
        "Recommended Factory", row["Recommended Factory"],
        delta="No change" if not row["Reassignment Needed"] else "Reassignment suggested",
    )
    c3.metric("Lead Time Reduction", f"{row['Lead Time Reduction (%)']:.1f}%")

    c4, c5 = st.columns(2)
    c4.metric("Current → Recommended Lead Time", f"{row['Current Lead Time (days)']:.2f} → {row['Recommended Lead Time (days)']:.2f} days")
    c5.metric("Estimated Profit Impact", f"${row['Profit Impact ($)']:,.0f}", delta=f"{row['Profit Impact (%)']:.1f}%")

    comp = pd.DataFrame(
        {
            "Scenario": ["Current", "Recommended"],
            "Lead Time (days)": [row["Current Lead Time (days)"], row["Recommended Lead Time (days)"]],
        }
    )
    fig = px.bar(comp, x="Scenario", y="Lead Time (days)", color="Scenario", title=f"Lead Time — {wi_product}")
    st.plotly_chart(fig, use_container_width=True)

    if row["Reassignment Needed"]:
        st.success(
            f"Reassigning **{wi_product}** from **{row['Current Factory']}** to "
            f"**{row['Recommended Factory']}** is projected to reduce lead time by "
            f"**{row['Lead Time Reduction (%)']:.1f}%** with an estimated profit impact of "
            f"**${row['Profit Impact ($)']:,.0f}** ({row['Profit Impact (%)']:.1f}%), "
            f"given the current profit/speed priority setting."
        )
    else:
        st.info(f"**{wi_product}** is already at its optimal factory under the current priority setting.")

# --------------------------------------------------------------------------
# TAB 4 — RECOMMENDATION DASHBOARD
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Ranked factory reassignment recommendations")
    only_changes = st.checkbox("Show only products with a suggested reassignment", value=True)
    display_recs = recommendations[recommendations["Reassignment Needed"]] if only_changes else recommendations

    st.dataframe(
        display_recs.style.format(
            {
                "Current Lead Time (days)": "{:.2f}",
                "Recommended Lead Time (days)": "{:.2f}",
                "Lead Time Reduction (%)": "{:.1f}%",
                "Profit Impact ($)": "${:,.0f}",
                "Profit Impact (%)": "{:.1f}%",
            }
        ),
        use_container_width=True,
        height=420,
    )

    coverage = recommendations["Reassignment Needed"].mean() * 100
    st.metric("Recommendation Coverage", f"{coverage:.0f}% of products have a suggested reassignment")

    fig = px.bar(
        recommendations.sort_values("Lead Time Reduction (%)", ascending=True),
        x="Lead Time Reduction (%)", y="Product", orientation="h", color="Reassignment Needed",
        title="Lead Time Reduction Opportunity by Product",
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 5 — RISK & IMPACT PANEL
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Profit impact alerts & high-risk reassignments")

    risk_threshold = st.slider("Flag reassignments with profit impact below (%)", -50, 0, -5)
    high_risk = recommendations[(recommendations["Reassignment Needed"]) & (recommendations["Profit Impact (%)"] < risk_threshold)]
    safe = recommendations[(recommendations["Reassignment Needed"]) & (recommendations["Profit Impact (%)"] >= risk_threshold)]

    c1, c2, c3 = st.columns(3)
    c1.metric("High-Risk Reassignments", len(high_risk))
    c2.metric("Safe Reassignments", len(safe))
    r2_best = metrics[best_model_name]["R2"]
    c3.metric("Scenario Confidence Score (model R²)", f"{r2_best:.2f}")

    if len(high_risk):
        st.warning("The following recommended reassignments show a meaningful projected profit decline:")
        st.dataframe(
            high_risk[["Product", "Current Factory", "Recommended Factory", "Lead Time Reduction (%)", "Profit Impact (%)"]],
            use_container_width=True,
        )
    else:
        st.success("No reassignments currently breach the selected profit-risk threshold.")

    fig = px.scatter(
        recommendations, x="Lead Time Reduction (%)", y="Profit Impact (%)", color="Reassignment Needed",
        hover_data=["Product", "Current Factory", "Recommended Factory"],
        title="Speed Gain vs. Profit Impact — all products",
    )
    fig.add_hline(y=risk_threshold, line_dash="dash", line_color="red", annotation_text="Risk threshold")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 6 — ROUTE & PRODUCT CLUSTERING
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Route performance clusters (Region × Factory)")
    n_clusters = st.slider("Number of clusters", 2, 5, 3)
    clustered = cluster_routes(data, n_clusters)

    fig = px.scatter(
        clustered, x="Avg_Distance_km", y="Avg_Lead_Time", size="Orders", color="Cluster Label",
        hover_data=["Region", "Factory", "Avg_Margin_Pct"],
        title="Route Clusters: Distance vs. Lead Time (bubble size = order volume)",
        labels={"Avg_Distance_km": "Avg Distance (km)", "Avg_Lead_Time": "Avg Lead Time (days)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        clustered.sort_values(["Cluster Label", "Avg_Lead_Time"]).style.format(
            {"Avg_Distance_km": "{:.0f}", "Avg_Lead_Time": "{:.2f}", "Avg_Margin_Pct": "{:.1f}%"}
        ),
        use_container_width=True,
    )
    slow = clustered[clustered["Cluster Label"] == "Slow Routes"]
    if len(slow):
        st.warning(
            "Consistently slow region–factory combinations: "
            + ", ".join(f"{r['Region']} ← {r['Factory']}" for _, r in slow.iterrows())
        )

# --------------------------------------------------------------------------
# TAB 7 — MODEL PERFORMANCE
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Predictive model evaluation")
    st.caption("Target: modeled proxy lead time (see data-quality note above). Models trained on Distance, Units, Sales, Division, Ship Mode, Region, and Factory.")

    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(
        metrics_df.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R2": "{:.3f}"}).highlight_max(subset=["R2"], color="#c6efce"),
        use_container_width=True,
    )
    st.success(f"Best-performing model selected: **{best_model_name}** (highest R²)")

    fig = go.Figure()
    for m in ["RMSE", "MAE"]:
        fig.add_trace(go.Bar(name=m, x=metrics_df["Model"], y=metrics_df[m]))
    fig.update_layout(barmode="group", title="Error Metrics by Model")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Nassau Candy Distributor — Factory Reallocation & Shipping Optimization Recommendation System · Streamlit dashboard")

footer_html = """
<div style='
background: linear-gradient(135deg,#111827,#1f2937);
padding: 30px;
border-radius: 20px;
text-align: center;
border: 1px solid #374151;
margin-top: 30px;
'>

<h2 style='color:#f59e0b;'>
🍬Nassau Candy | Factory Reallocation & Shipping Optimizer
</h2>

<p style='color:#d1d5db; font-size:16px;'>
Advanced Factory Reallocation & Shipping Optimizer Analytics Dashboard
</p>

<div style='margin-top:20px;'>

<a href='https://www.nassaucandy.com/'
target='_blank'
style='
text-decoration:none;
background: linear-gradient(90deg,#f59e0b,#ef4444);
color:white;
padding:12px 20px;
border-radius:12px;
margin:10px;
display:inline-block;
font-weight:bold;
'>
🌐 Official Website
</a>

<a href='https://www.instagram.com/nassau.candy/'
target='_blank'
style='
text-decoration:none;
background: linear-gradient(90deg,#f59e0b,#ef4444);
color:white;
padding:12px 20px;
border-radius:12px;
margin:10px;
display:inline-block;
font-weight:bold;
'>
📸 Instagram
</a>

</div>

<p style='
color:#9ca3af;
margin-top:20px;
font-size:14px;
'>
Designed for Business Intelligence & Retail Analytics
</p>

</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)

st.divider()
st.caption(
    "Built with Streamlit, Pandas & Plotly · Upload your own coffee-shop sales "
    "export (CSV/Excel) in the sidebar, or place a CSV in a local `data/` folder, "
    "to replace the demo dataset."
)