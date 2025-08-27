import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import openpyxl
import base64
import folium
from streamlit_folium import st_folium
from io import BytesIO

# =========================
#  APP STYLING
# =========================
st.set_page_config(
    page_title="Water Usage & Energy Consumption in Africa",
    page_icon="logo/logo.png",
    layout="wide"
)

# Custom CSS for color effects
st.markdown(
    """
    <style>
        /* Background */
        .stApp {
            background-color: #e3f2fd; /* light water blue */
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #fff9c4; /* soft yellow */
        }

        /* Main Titles */
        h1, h2, h3 {
            color: #0d47a1; /* deep water blue */
        }

        /* Subtitles / smaller headings */
        h4, h5, h6, label {
            color: #f9a825; /* golden yellow */
        }

        /* Prediction result box */
        .prediction {
            font-size: 22px;
            font-weight: bold;
            color: #0d47a1; /* navy blue */
            background-color: #fffde7; /* very light yellow */
            padding: 14px;
            border-radius: 12px;
            border: 1px solid #fdd835; /* yellow border */
        }

        /* Buttons */
        button[kind="primary"] {
            background-color: #42a5f5; /* bright water blue */
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        button[kind="primary"]:hover {
            background-color: #fbc02d; /* sunflower yellow */
            color: #0d47a1;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
#  LOAD MODEL & DATA
# =========================
try:
    model = joblib.load('models/solar_rf_model.pkl')
    dummy_encoder = joblib.load('models/dummy_encoder.joblib')
    freq_encoder = joblib.load('models/freq_encoder.pkl')
    allocation_map = joblib.load('models/allocation_map.pkl')
    scaler = joblib.load('models/scaler.pkl')
    pca= joblib.load('models/pca_2d.pkl')
    kmeans = joblib.load('models/kmeans_model.pkl')
    recommender = joblib.load('models/reco_map.pkl')
    metadata = joblib.load('models/preprocessing_metadata.pkl')
    feature_cols = metadata['feature_cols']
    ohe_col = metadata['ohe_col']
    freq_col = metadata['freq_col']
    final_columns = metadata['final_columns']
except Exception as e:
    st.error(f'failed to load model or preprocessing tools: {e}')
    st.stop()

try:
    data = pd.read_csv('Cleaned Data.csv')
except Exception as e:
    st.error(f'failed to load the dataset: {e}')
    st.stop()

# =========================
#  LOGO DISPLAY
# =========================
logo_path = "logo/logo.png"

# Convert logo to Base64
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64(logo_path)

# Main page logo
st.markdown(
    f"""
    <div style="text-align: center; margin-top:0px; margin-bottom:0px; padding:0;">
        <img src="data:image/png;base64,{logo_base64}" width="250" style="display:block; margin:auto; padding:0;">
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar logo
st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-top:0px; margin-bottom:0px; padding:0;">
        <img src="data:image/png;base64,{logo_base64}" width="200" style="display:block; margin:auto; padding:0;">
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
#  APP CONTENT
# =========================
# Custom CSS for centering text and sidebar header
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }

    /* Center align sidebar headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title & Subtitle
st.markdown('<h1 class="centered">Water Usage & Energy Supply App</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="centered">Sustainable Resource Management (Monitor • Analyze • Optimize)</h4>', unsafe_allow_html=True)

# App Description
st.markdown(
    '<p class="centered">This app enables data-driven monitoring and optimization of '
    'water and energy usage to promote efficiency, cost savings, and sustainable decision-making.</p>',
    unsafe_allow_html=True
)

# Sidebar input header
st.sidebar.header("Please Upload Your Data provide your Data Inputs")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel or manually enter the values below", type=["csv", "xlsx"])

# =========================
# Mandatory & Optional Columns Placeholder
# =========================
if uploaded_file is None:
    # Recommended min/max values for features
    feature_ranges = {
        "pop_density": [4.36, 34682.18],
        "safe_ratio": [0.0, 0.033815],
        "unsafe_ratio": [0.0, 0.016959],
        "water_diversity": [0, 7],
        "climate_stress": [-0.000806, 0.469285],
        "compactness": [14.05, 87.93],
        "Distance_to_Center": [0.01, 6.22],
        "Average_Nighttime_mean": [-0.0317, 28.5723],
        "AREA": [11.645, 10357.67],
        "Population": [3459.978, 2181858],
        "Latitude": [4.4579, 13.6867],
        "Longitude": [2.8002, 14.5258]
    }

    df_ranges = pd.DataFrame([
        {"Feature": feature, "Min": values[0], "Max": values[1]}
        for feature, values in feature_ranges.items()
    ])

    # Format numeric columns to 3 decimal places as strings
    df_ranges["Min"] = df_ranges["Min"].map(lambda x: f"{x:,.4f}")
    df_ranges["Max"] = df_ranges["Max"].map(lambda x: f"{x:,.4f}")

    mandatory_columns = [
        "pop_density", "safe_ratio", "unsafe_ratio", "water_diversity",
        "climate_stress", "compactness", "Distance_to_Center",
        "Average_Nighttime_mean", "AREA", "Population", "GROUPED_STATE", "lga"
    ]

    optional_columns = ["Latitude", "Longitude"]

    # --- Mandatory & Optional Columns Collapsible ---
    st.markdown(f"""
    <div style="background-color:#fff8e1; padding:20px; border-radius:12px; border:1px solid #fdd835;">
        <details>
            <summary style="font-weight:bold; cursor:pointer; font-size:18px; color:#f9a825;">
                Mandatory Columns (Click to expand)
            </summary>
            <ul style="margin-top:10px;">
                {''.join([f'<li><code>{col}</code></li>' for col in mandatory_columns])}
            </ul>
        </details>
        <br>
        <details>
            <summary style="font-weight:bold; cursor:pointer; font-size:16px; color:#0d47a1;">
                Optional Columns for Folium Maps
            </summary>
            <ul style="margin-top:5px;">
                {''.join([f'<li><code>{col}</code></li>' for col in optional_columns])}
            </ul>
        </details>
    </div>
    """, unsafe_allow_html=True)

    # --- Recommended Feature Ranges (Collapsible Table) ---
    st.markdown(f"""
    <details>
        <summary style="font-weight:bold; cursor:pointer; font-size:16px; color:#0d47a1;">
            Recommended Value Ranges for Features (Click to expand)
        </summary>
        <div style="margin-top:10px;">
            {df_ranges.to_html(classes='table table-striped', border=0)}
        </div>
    </details>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p>To view valid STATE and LGA options, check the sidebar dropdowns:</p>
    <ul>
        <li><code>STATE</code> → GROUPED_STATE</li>
        <li><code>LGA</code> → lga</li>
    </ul>
    """, unsafe_allow_html=True)

# Options
category_options = {
    "pop_density": [4.36, 34682.18],
    "safe_ratio": [0.0, 0.033815],
    "unsafe_ratio": [0.0, 0.016959],
    "water_diversity": [0.0, 7.0],
    "climate_stress": [-0.000806, 0.469285],
    "compactness": [14.05, 87.93],
    "Distance_to_Center": [0.01, 6.22],
    "Average_Nighttime_mean": [-0.03, 28.57],
    "AREA": [11.64, 10357.67],
    "Population": [3459.97, 2181858.0]
}

lga_options = sorted(data['lga'].unique())
state_options = sorted(data['GROUPED_STATE'].unique())

# Sidebar header note
st.sidebar.markdown(
    "<div class='sidebar-section'>Please note that the only acceptable inputs will be those within the specified ranges or categories.</div>", 
    unsafe_allow_html=True)

# Sidebar inputs
GROUPED_STATE = st.sidebar.selectbox("STATE", state_options)
lga = st.sidebar.selectbox("LGA", lga_options)
pop_density = st.sidebar.number_input("POPULATION DENSITY", min_value=4.36, max_value=34682.19, value=253.11, step=1.0)
safe_ratio = st.sidebar.number_input("SAFE WATER RATIO", min_value=0.0, max_value=0.0338, value=0.000417, step=0.0001, format="%.6f")
unsafe_ratio = st.sidebar.number_input("UNSAFE WATER RATIO", min_value=0.0, max_value=0.0169, value=0.000077, step=0.0001, format="%.6f")
water_diversity = st.sidebar.slider("WATER DIVERSITY", min_value=0, max_value=7, value=3, step=1)
climate_stress = st.sidebar.number_input("CLIMATE STRESS", min_value=-0.0008, max_value=0.4693, value=0.000174, step=0.0001, format="%.6f")
compactness = st.sidebar.number_input("COMPACTNESS", min_value=14.05, max_value=87.93, value=23.99, step=0.1)
Distance_to_Center = st.sidebar.number_input("DISTANCE TO CENTRE", min_value=0.01, max_value=6.22, value=0.69, step=0.01)
Average_Nighttime_mean = st.sidebar.number_input("AVERAGE NIGHT TIME MEAN", min_value=-0.03, max_value=28.57, value=0.0456, step=0.01)
AREA = st.sidebar.number_input("AREA", min_value=11.65, max_value=10357.67, value=680.62, step=1.0)
Population = st.sidebar.number_input("POPULATION", min_value=3459.98, max_value=2181858.0, value=190256.5, step=100.0)

# Optional entry section header
st.sidebar.markdown(
    "<div class='sidebar-section'>Optional entries. If values are inserted, kindly consider ranges.</div>", 
    unsafe_allow_html=True)

# --- Sidebar filters for Latitude & Longitude ---
# Convert to GeoDataFrame if you want to keep .geometry
gdf = gpd.GeoDataFrame(
    data, 
    geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]),
    crs="EPSG:4326"
)

# Now you can safely do:
lat_min, lat_max = st.sidebar.slider(
    "Latitude range",
    float(gdf.geometry.y.min()),
    float(gdf.geometry.y.max()),
    (float(gdf.geometry.y.min()), float(gdf.geometry.y.max())),
    step=0.01
)

lon_min, lon_max = st.sidebar.slider(
    "Longitude range",
    float(gdf.geometry.x.min()),
    float(gdf.geometry.x.max()),
    (float(gdf.geometry.x.min()), float(gdf.geometry.x.max())),
    step=0.01
)

# Apply filters
gdf = gdf[
    (gdf.geometry.x.between(lon_min, lon_max)) &
    (gdf.geometry.y.between(lat_min, lat_max))
]

# ======================
# Build input dataframe
# ======================

# Collect sidebar inputs
input_dict = {
    "pop_density": pop_density,
    "safe_ratio": safe_ratio,
    "unsafe_ratio": unsafe_ratio,
    "water_diversity": water_diversity,
    "climate_stress": climate_stress,
    "compactness": compactness,
    "Distance_to_Center": Distance_to_Center,
    "Average_Nighttime_mean": Average_Nighttime_mean,
    "AREA": AREA,
    "Population": Population,
    "lga": lga,
    "GROUPED_STATE": GROUPED_STATE
}

# Convert to DataFrame
input_df = gpd.GeoDataFrame([input_dict])

# ======================
# Encoding (match training)
# ======================

# One-hot encode GROUPED_STATE
dummies = pd.get_dummies(input_df["GROUPED_STATE"], prefix="state", drop_first=True).astype(int)

# Frequency encode LGA
# You need the mapping from training time
lga_freq = input_df["lga"].value_counts(normalize=True)  # <- use your full dataset (not input)
input_df["lga_encoded"] = input_df["lga"].map(lga_freq)

# Replace NaNs (if unseen LGA at prediction time)
input_df["lga_encoded"].fillna(0, inplace=True)

# Drop raw categorical cols
input_encoded = pd.concat(
    [input_df.drop(columns=["GROUPED_STATE", "lga"]), dummies],
    axis=1
)

# Ensure same feature order as training
for col in final_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[final_columns]


# ======================
# Prediction
# ======================

if st.sidebar.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    st.subheader("Prediction")
    st.write(f"Predicted Solar Tier: **{prediction}**")

# ======================
# INTERACTIVE DATA VISUALIZATION & FILE HANDLING
# ======================

st.set_page_config(page_title="Interactive Visualizations", layout="wide")

st.markdown(
    '<h2 style="text-align:center; color:#0d47a1; margin-bottom:10px;">Interactive Data Visualizations</h2>',
    unsafe_allow_html=True
)

if uploaded_file is not None:
    uploaded_file.seek(0)  # make sure pointer is at the start
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # Optional: check for necessary columns
        necessary_cols = ["Latitude", "Longitude", "Average_Nighttime_mean", "pop_density"]
        missing_cols = [col for col in necessary_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"The uploaded file is missing required columns: {', '.join(missing_cols)}")
        else:
            # ... your visualization / processing code ...

            # ======================
            # Download Option
            # ======================
            st.header("Save Your File")

            file_format = st.radio(
                "Choose format:", 
                ["CSV", "Excel"], 
                key="file_format_radio"
            )

            if file_format == "CSV":
                st.download_button(
                    label="Download Processed CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="processed_data.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            else:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Sheet1")
                st.download_button(
                    label="Download Processed Excel",
                    data=buffer.getvalue(),
                    file_name="processed_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )

    except Exception as e:
        st.warning(f"Could not read the uploaded file. Please check the format and content. Error: {e}")

    # Preview
    st.write("### Preview of Dataset", df.head())

    # ======================
    # Visualization Options
    # ======================
    st.header("Choose Visualization")
    plot_type = st.selectbox(
        "Select plot type:",
        ["Histogram", "Bar Chart", "Scatter Plot", "Correlation Heatmap", "Folium Map"]
    )

    # List of numeric columns excluding Latitude and Longitude
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col not in ["Latitude", "Longitude"]
    ]

    # --- Histogram ---
    if plot_type == "Histogram":
        col = st.selectbox("Select column", numeric_cols)
        bins = st.slider("Number of bins", 5, 50, 20)
        
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    # --- Bar Chart ---
    elif plot_type == "Bar Chart":
        col = st.selectbox("Select categorical column", df.select_dtypes(include=["object"]).columns)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax, color="orange")
        ax.set_title(f"Bar Chart of {col}")
        st.pyplot(fig)

    # --- Scatter Plot with Best Fit Line ---
    elif plot_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)
        
        # Round values for plotting only
        x_vals = df[x_col].round(3)
        y_vals = df[y_col].round(3)
        
        fig, ax = plt.subplots()
        ax.scatter(x_vals, y_vals, alpha=0.6, c="blue", label="Data Points")
        
        # Compute best-fit line
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        best_fit = slope * x_vals + intercept
        ax.plot(x_vals, best_fit, color="red", linewidth=2, label="Best Fit Line")
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        ax.legend()
        
        st.pyplot(fig)

    # --- Correlation Heatmap ---
    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr().round(3), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # --- Folium Map ---
    elif plot_type == "Folium Map":
        # Automatically detect latitude and longitude
        if "Latitude" in df.columns and "Longitude" in df.columns:
            lat_col = "Latitude"
            lon_col = "Longitude"

            # Create GeoDataFrame for consistency
            gdf_map = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                crs="EPSG:4326"
            )

            # Compute Solar_Tier_Label if not present
            if "Solar_Tier_Label" not in gdf_map.columns:
                gdf_map["Solar_Tier"] = (
                    gdf_map["Average_Nighttime_mean"].rank(pct=True) * 0.6 +
                    gdf_map["pop_density"].rank(pct=True) * 0.4
                )
                bins = [0, 0.33, 0.66, 1.0]
                labels = ["Low", "Medium", "High"]
                gdf_map["Solar_Tier_Label"] = pd.cut(gdf_map["Solar_Tier"], bins=bins, labels=labels)

            # Define colors
            tier_colors = {"Low": "red", "Medium": "orange", "High": "green"}

            # Center map
            m = folium.Map(location=[gdf_map[lat_col].mean(), gdf_map[lon_col].mean()], zoom_start=6)

            # Add points
            for _, row in gdf_map.iterrows():
                color = tier_colors.get(row["Solar_Tier_Label"], "blue")
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Solar Tier: {row['Solar_Tier_Label']}"
                ).add_to(m)

            st_map = st_folium(m, width=1200, height=500)

        else:
            st.warning("Latitude and Longitude columns are missing. Cannot create Folium map.")

else:
        st.info("⬅️ Please upload a CSV or Excel file to start exploring.")