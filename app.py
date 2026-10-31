
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Manganese Intelligence",
    page_icon="⛏️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.18);
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "datasets"
    / "SIH26009_Six_Synthetic_Training_Datasets (2)"
)

MODEL_FILE = BASE_DIR / "manganese_model.joblib"

FILES = {
    "Geochemistry": "01_Exploration_Geochemistry_Prospectivity.xlsx",
    "Satellite": "02_Satellite_Environmental_Features.xlsx",
    "Borehole": "03_Borehole_Assay_Intervals.xlsx",
    "Production": "04_Monthly_Production_Shortfall.xlsx",
    "Equipment": "05_Equipment_Downtime_Events.xlsx",
    "Weather": "06_Weather_Operational_Risk.xlsx",
}

GEO_FEATURES = [
    "elevation_m_simulated",
    "manganese_ppm_simulated",
    "iron_pct_simulated",
    "silica_pct_simulated",
    "alumina_pct_simulated",
    "phosphorus_pct_simulated",
    "distance_to_mapped_structure_km_simulated",
    "weathering_index_simulated",
    "sample_method_simulated",
]

TARGET = "prospectivity_label_simulated"

# =====================================================
# 3. DATA LOADING
# =====================================================

@st.cache_data
def load_excel(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        excel = pd.ExcelFile(path)

        sheet = (
            "synthetic_data"
            if "synthetic_data" in excel.sheet_names
            else excel.sheet_names[0]
        )

        return pd.read_excel(path, sheet_name=sheet)

    except Exception as error:
        st.error(f"Error loading {path.name}: {error}")
        return None


@st.cache_data
def load_all_data():
    result = {}

    for name, filename in FILES.items():
        result[name] = load_excel(DATA_DIR / filename)

    return result


@st.cache_resource
def load_model():
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception as error:
            st.warning(f"Model loading error: {error}")

    return None


data = load_all_data()

geology = data["Geochemistry"]
satellite = data["Satellite"]
borehole = data["Borehole"]
production = data["Production"]
equipment = data["Equipment"]
weather = data["Weather"]

model = load_model()

# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

def has_columns(df, columns):
    return (
        df is not None
        and all(column in df.columns for column in columns)
    )


def numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def download_csv(df, filename):
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


def dataset_status():
    records = sum(
        len(df) for df in data.values()
        if df is not None
    )

    available = sum(
        df is not None for df in data.values()
    )

    return records, available


def synthetic_notice():
    st.caption(
        "Research prototype using synthetic/simulated datasets. "
        "Results are not verified geological discoveries or "
        "real-world operational forecasts."
    )


# =====================================================
# 5. SIDEBAR
# =====================================================

st.sidebar.title("⛏️ Manganese Intelligence")

st.sidebar.caption(
    "AI-Powered Mineral Exploration and Mining Analysis"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "AI Exploration",
        "Satellite Analysis",
        "Borehole Analysis",
        "Production Analysis",
        "Equipment Monitoring",
        "Weather Risk Analysis",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Dataset Status")

for name, df in data.items():
    if df is not None:
        st.sidebar.success(f"{name}: Loaded")
    else:
        st.sidebar.error(f"{name}: Missing")

synthetic_notice()

# =====================================================
# 6. OVERVIEW
# =====================================================

if page == "Overview":

    st.title("⛏️ Manganese Intelligence")

    st.subheader(
        "AI-Powered Mineral Exploration and "
        "Mining Operations Analysis"
    )

    st.write(
        "An integrated analytical dashboard for exploring "
        "manganese geochemistry, satellite features, "
        "borehole assays and mining operations."
    )

    total_records, available = dataset_status()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Datasets Available", f"{available}/6")
    c2.metric("Total Records", f"{total_records:,}")
    c3.metric(
        "Geochemical Samples",
        len(geology) if geology is not None else "N/A"
    )
    c4.metric(
        "Satellite Records",
        len(satellite) if satellite is not None else "N/A"
    )

    st.divider()

    st.subheader("Dataset Overview")

    inventory = []

    for name, df in data.items():
        inventory.append({
            "Dataset": name,
            "Status": "Loaded" if df is not None else "Missing",
            "Records": len(df) if df is not None else 0,
            "Columns": len(df.columns) if df is not None else 0
        })

    st.dataframe(
        pd.DataFrame(inventory),
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Manganese Concentration")

        if has_columns(
            geology,
            ["manganese_ppm_simulated"]
        ):

            fig = px.histogram(
                geology,
                x="manganese_ppm_simulated",
                nbins=30,
                title="Geochemical Sample Distribution"
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Geochemistry data unavailable.")

    with right:

        st.subheader("Production Overview")

        if has_columns(
            production,
            [
                "planned_output_tonnes_simulated",
                "actual_output_tonnes_simulated"
            ]
        ):

            planned = numeric(
                production,
                "planned_output_tonnes_simulated"
            ).sum()

            actual = numeric(
                production,
                "actual_output_tonnes_simulated"
            ).sum()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=["Planned", "Actual"],
                y=[planned, actual],
                name="Production"
            ))

            fig.update_layout(
                title="Total Planned vs Actual Production",
                yaxis_title="Tonnes",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Production data unavailable.")

    st.subheader("Project Modules")

    modules = [
        "AI Exploration",
        "Satellite Analysis",
        "Borehole Analysis",
        "Production Analysis",
        "Equipment Monitoring",
        "Weather Risk Analysis"
    ]

    cols = st.columns(3)

    for i, module in enumerate(modules):
        with cols[i % 3]:
            st.markdown(f"**{module}**")

# =====================================================
# 7. AI EXPLORATION
# =====================================================

elif page == "AI Exploration":

    st.title("AI Exploration")

    st.write(
        "Machine-learning-based manganese prospectivity "
        "analysis using simulated geochemical features."
    )

    if geology is None:
        st.error("Geochemistry dataset not found.")

    elif model is None:
        st.error(
            "Trained model not found. Run train_model.py first."
        )

    elif not has_columns(geology, GEO_FEATURES):

        st.error("Required model features are missing.")

    else:

        X = geology[GEO_FEATURES].copy()

        for col in GEO_FEATURES[:-1]:
            X[col] = pd.to_numeric(
                X[col], errors="coerce"
            )

        valid = (
            X[GEO_FEATURES[:-1]].notna().all(axis=1)
            & X["sample_method_simulated"].notna()
        )

        scored = geology.copy()
        scored["prospectivity_score_pct"] = np.nan

        try:

            if valid.any():

                probabilities = model.predict_proba(
                    X.loc[valid]
                )

                positive_index = list(
                    model.classes_
                ).index(1)

                scored.loc[
                    valid,
                    "prospectivity_score_pct"
                ] = (
                    probabilities[:, positive_index] * 100
                )

            threshold = st.slider(
                "Minimum Prospectivity Score (%)",
                0, 100, 50, 5
            )

            filtered = scored[
                scored["prospectivity_score_pct"] >= threshold
            ].copy()

            valid_scores = (
                scored["prospectivity_score_pct"].dropna()
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Samples Scored",
                len(valid_scores)
            )

            c2.metric(
                "Samples Above Threshold",
                len(filtered)
            )

            c3.metric(
                "Mean Prospectivity Score",
                f"{valid_scores.mean():.1f}%"
                if not valid_scores.empty else "N/A"
            )

            st.info(
                "Scores represent model-estimated probabilities "
                "for the positive simulated training label. "
                "They do not establish actual manganese reserves."
            )

            left, right = st.columns(2)

            with left:

                fig = px.histogram(
                    scored.dropna(
                        subset=["prospectivity_score_pct"]
                    ),
                    x="prospectivity_score_pct",
                    nbins=20,
                    title="Prospectivity Score Distribution"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with right:

                fig = px.scatter(
                    scored,
                    x="manganese_ppm_simulated",
                    y="prospectivity_score_pct",
                    title="Manganese Concentration vs Score"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            if has_columns(
                filtered,
                ["latitude_simulated", "longitude_simulated"]
            ):

                map_data = filtered.copy()

                map_data["latitude_simulated"] = numeric(
                    map_data, "latitude_simulated"
                )

                map_data["longitude_simulated"] = numeric(
                    map_data, "longitude_simulated"
                )

                map_data = map_data.dropna(
                    subset=[
                        "latitude_simulated",
                        "longitude_simulated"
                    ]
                )

                if not map_data.empty:

                    fig = px.scatter_geo(
                        map_data,
                        lat="latitude_simulated",
                        lon="longitude_simulated",
                        color="prospectivity_score_pct",
                        hover_data=[
                            "manganese_ppm_simulated",
                            "prospectivity_score_pct"
                        ],
                        title="Simulated Sample Locations",
                        color_continuous_scale="YlOrRd"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            st.subheader("Scored Samples")

            display_cols = [
                col for col in [
                    "sample_id",
                    "manganese_ppm_simulated",
                    "iron_pct_simulated",
                    "prospectivity_score_pct"
                ]
                if col in filtered.columns
            ]

            st.dataframe(
                filtered[display_cols].sort_values(
                    "prospectivity_score_pct",
                    ascending=False
                ),
                use_container_width=True
            )

            download_csv(
                filtered,
                "manganese_prospectivity_scores.csv"
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")

# =====================================================
# 8. SATELLITE ANALYSIS
# =====================================================

elif page == "Satellite Analysis":

    st.title("Satellite & Environmental Analysis")

    if satellite is None:

        st.error("Satellite dataset unavailable.")

    else:

        st.info(
            "Satellite and environmental values are simulated, "
            "not actual Sentinel-2 observations."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Satellite Records", len(satellite))

        c2.metric(
            "Average NDVI",
            f"{numeric(satellite, 'ndvi_simulated').mean():.3f}"
            if "ndvi_simulated" in satellite else "N/A"
        )

        c3.metric(
            "Average NDMI",
            f"{numeric(satellite, 'ndmi_simulated').mean():.3f}"
            if "ndmi_simulated" in satellite else "N/A"
        )

        if "potential_zone_label_simulated" in satellite:

            counts = (
                satellite["potential_zone_label_simulated"]
                .value_counts()
                .reset_index()
            )

            counts.columns = ["Potential Zone", "Records"]

            fig = px.bar(
                counts,
                x="Potential Zone",
                y="Records",
                title="Potential Zone Distribution"
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:

            if has_columns(
                satellite,
                ["ndvi_simulated", "ndmi_simulated"]
            ):

                fig = px.scatter(
                    satellite,
                    x="ndvi_simulated",
                    y="ndmi_simulated",
                    color="potential_zone_label_simulated"
                    if "potential_zone_label_simulated" in satellite
                    else None,
                    title="NDVI vs NDMI"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        with right:

            if has_columns(
                satellite,
                [
                    "bare_soil_index_simulated",
                    "land_surface_temp_c_simulated"
                ]
            ):

                fig = px.scatter(
                    satellite,
                    x="bare_soil_index_simulated",
                    y="land_surface_temp_c_simulated",
                    title="Bare Soil Index vs Surface Temperature"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Spectral and Environmental Statistics")

        st.dataframe(
            satellite.select_dtypes(
                include=np.number
            ).describe().T,
            use_container_width=True
        )

        with st.expander("View Satellite Records"):
            st.dataframe(satellite, use_container_width=True)

        download_csv(
            satellite,
            "satellite_analysis.csv"
        )

# =====================================================
# 9. BOREHOLE ANALYSIS
# =====================================================

elif page == "Borehole Analysis":

    st.title("Borehole Assay Analysis")

    if borehole is None:

        st.error("Borehole dataset unavailable.")

    else:

        st.info(
            "Borehole assay records are simulated, "
            "not verified drilling measurements."
        )

        grade = "manganese_grade_pct_simulated"
        depth = "depth_from_m_simulated"
        recovery = "recovery_pct_simulated"

        c1, c2, c3 = st.columns(3)

        c1.metric("Assay Intervals", len(borehole))

        c2.metric(
            "Average Mn Grade",
            f"{numeric(borehole, grade).mean():.2f}%"
            if grade in borehole else "N/A"
        )

        c3.metric(
            "Maximum Mn Grade",
            f"{numeric(borehole, grade).max():.2f}%"
            if grade in borehole else "N/A"
        )

        left, right = st.columns(2)

        with left:

            if grade in borehole:

                fig = px.histogram(
                    borehole,
                    x=grade,
                    nbins=25,
                    title="Manganese Grade Distribution"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        with right:

            if has_columns(borehole, [depth, grade]):

                fig = px.scatter(
                    borehole,
                    x=depth,
                    y=grade,
                    color="composite_ore_flag_simulated"
                    if "composite_ore_flag_simulated" in borehole
                    else None,
                    title="Manganese Grade by Depth"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        if has_columns(borehole, [grade, recovery]):

            fig = px.scatter(
                borehole,
                x=grade,
                y=recovery,
                title="Manganese Grade vs Recovery"
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Borehole Statistical Summary")

        st.dataframe(
            borehole.select_dtypes(
                include=np.number
            ).describe().T,
            use_container_width=True
        )

        with st.expander("View Borehole Records"):
            st.dataframe(borehole, use_container_width=True)

        download_csv(
            borehole,
            "borehole_analysis.csv"
        )

# =====================================================
# 10. PRODUCTION ANALYSIS
# =====================================================

elif page == "Production Analysis":

    st.title("Production & Shortfall Analysis")

    if production is None:

        st.error("Production dataset unavailable.")

    else:

        date_col = "month_simulated"
        plan_col = "planned_output_tonnes_simulated"
        actual_col = "actual_output_tonnes_simulated"

        p = production.copy()

        p[date_col] = pd.to_datetime(
            p[date_col], errors="coerce"
        )

        p[plan_col] = numeric(p, plan_col)
        p[actual_col] = numeric(p, actual_col)

        p = p.dropna(
            subset=[date_col, plan_col, actual_col]
        )

        if p.empty:

            st.warning("No valid production records.")

        else:

            min_date = p[date_col].min().date()
            max_date = p[date_col].max().date()

            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(date_range, (tuple, list)):
                if len(date_range) == 2:
                    start, end = date_range
                elif len(date_range) == 1:
                    start = end = date_range[0]
                else:
                    start, end = min_date, max_date
            else:
                start = end = date_range

            filtered = p[
                (p[date_col].dt.date >= start)
                & (p[date_col].dt.date <= end)
            ].copy()

            if filtered.empty:

                st.warning("No records in selected date range.")

            else:

                planned = filtered[plan_col].sum()
                actual = filtered[actual_col].sum()

                shortfall = planned - actual

                achievement = (
                    actual / planned * 100
                    if planned > 0 else np.nan
                )

                shortfall_pct = (
                    shortfall / planned * 100
                    if planned > 0 else np.nan
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Planned Production",
                    f"{planned:,.0f} tonnes"
                )

                c2.metric(
                    "Actual Production",
                    f"{actual:,.0f} tonnes"
                )

                c3.metric(
                    "Shortfall",
                    f"{shortfall:,.0f} tonnes"
                )

                c4.metric(
                    "Achievement",
                    f"{achievement:.2f}%"
                    if pd.notna(achievement) else "N/A"
                )

                st.caption(
                    f"Calculated shortfall: {shortfall_pct:.2f}%"
                    if pd.notna(shortfall_pct) else
                    "Shortfall percentage unavailable."
                )

                monthly = (
                    filtered.groupby(date_col, as_index=False)
                    [[plan_col, actual_col]]
                    .sum()
                    .sort_values(date_col)
                )

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=monthly[date_col],
                    y=monthly[plan_col],
                    mode="lines+markers",
                    name="Planned Output"
                ))

                fig.add_trace(go.Scatter(
                    x=monthly[date_col],
                    y=monthly[actual_col],
                    mode="lines+markers",
                    name="Actual Output"
                ))

                fig.update_layout(
                    title="Planned vs Actual Production",
                    xaxis_title="Month",
                    yaxis_title="Tonnes",
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

                monthly["shortfall"] = (
                    monthly[plan_col] - monthly[actual_col]
                )

                fig = px.bar(
                    monthly,
                    x=date_col,
                    y="shortfall",
                    title="Monthly Production Shortfall"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                operational_cols = [
                    col for col in [
                        "rainfall_mm_simulated",
                        "equipment_downtime_hours_simulated",
                        "blasting_delay_hours_simulated",
                        "ore_grade_pct_simulated",
                        "workforce_availability_pct_simulated",
                        "power_availability_pct_simulated"
                    ]
                    if col in filtered.columns
                ]

                if operational_cols:

                    st.subheader("Operational Indicators")

                    st.dataframe(
                        filtered[operational_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .describe().T,
                        use_container_width=True
                    )

                with st.expander("View Production Records"):
                    st.dataframe(filtered, use_container_width=True)

                download_csv(
                    filtered,
                    "production_analysis.csv"
                )

# =====================================================
# 11. EQUIPMENT MONITORING
# =====================================================

elif page == "Equipment Monitoring":

    st.title("Equipment Downtime & Maintenance Monitoring")

    if equipment is None:

        st.error("Equipment dataset unavailable.")

    else:

        downtime = "downtime_hours_simulated"
        cost = "repair_cost_inr_simulated"
        loss = "estimated_output_loss_tonnes_simulated"
        priority = "maintenance_priority_label_simulated"

        c1, c2, c3 = st.columns(3)

        c1.metric("Equipment Events", len(equipment))

        c2.metric(
            "Total Downtime",
            f"{numeric(equipment, downtime).sum():,.1f} hours"
            if downtime in equipment else "N/A"
        )

        c3.metric(
            "Repair Cost",
            f"₹{numeric(equipment, cost).sum():,.0f}"
            if cost in equipment else "N/A"
        )

        left, right = st.columns(2)

        with left:

            if priority in equipment:

                counts = (
                    equipment[priority]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = ["Priority", "Events"]

                fig = px.bar(
                    counts,
                    x="Priority",
                    y="Events",
                    title="Maintenance Priority Distribution"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        with right:

            if has_columns(
                equipment,
                ["equipment_type_simulated", downtime]
            ):

                grouped = (
                    equipment.groupby(
                        "equipment_type_simulated"
                    )[downtime]
                    .sum()
                    .reset_index()
                )

                fig = px.bar(
                    grouped,
                    x="equipment_type_simulated",
                    y=downtime,
                    title="Downtime by Equipment Type"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        if has_columns(equipment, [downtime, loss]):

            fig = px.scatter(
                equipment,
                x=downtime,
                y=loss,
                color=priority if priority in equipment else None,
                title="Downtime vs Estimated Output Loss"
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        if "repeated_failure_30d_label_simulated" in equipment:

            repeated = numeric(
                equipment,
                "repeated_failure_30d_label_simulated"
            ).sum()

            st.metric("Repeated Failure Events", int(repeated))

        with st.expander("View Equipment Records"):
            st.dataframe(equipment, use_container_width=True)

        download_csv(
            equipment,
            "equipment_analysis.csv"
        )

# =====================================================
# 12. WEATHER RISK ANALYSIS
# =====================================================

elif page == "Weather Risk Analysis":

    st.title("Weather & Operational Risk Analysis")

    if weather is None:

        st.error("Weather dataset unavailable.")

    else:

        risk_score = "operational_disruption_risk_score_simulated"
        risk_label = "high_disruption_risk_label_simulated"

        c1, c2, c3 = st.columns(3)

        c1.metric("Weather Records", len(weather))

        c2.metric(
            "Average Risk Score",
            f"{numeric(weather, risk_score).mean():.2f}"
            if risk_score in weather else "N/A"
        )

        c3.metric(
            "High Risk Records",
            int(numeric(weather, risk_label).sum())
            if risk_label in weather else "N/A"
        )

        if risk_label in weather:

            counts = (
                weather[risk_label]
                .value_counts()
                .reset_index()
            )

            counts.columns = ["Risk Label", "Records"]

            fig = px.bar(
                counts,
                x="Risk Label",
                y="Records",
                title="High Disruption Risk Distribution"
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:

            if has_columns(
                weather,
                [
                    "rainfall_mm_simulated",
                    risk_score
                ]
            ):

                fig = px.scatter(
                    weather,
                    x="rainfall_mm_simulated",
                    y=risk_score,
                    color=risk_label if risk_label in weather else None,
                    title="Rainfall vs Operational Risk"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        with right:

            if has_columns(
                weather,
                [
                    "road_accessibility_pct_simulated",
                    risk_score
                ]
            ):

                fig = px.scatter(
                    weather,
                    x="road_accessibility_pct_simulated",
                    y=risk_score,
                    color=risk_label if risk_label in weather else None,
                    title="Road Accessibility vs Risk"
                )

                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        if has_columns(weather, ["date_simulated", risk_score]):

            w = weather.copy()

            w["date_simulated"] = pd.to_datetime(
                w["date_simulated"], errors="coerce"
            )

            w = w.dropna(subset=["date_simulated"])

            daily = (
                w.groupby("date_simulated")[risk_score]
                .mean()
                .reset_index()
            )

            fig = px.line(
                daily,
                x="date_simulated",
                y=risk_score,
                title="Operational Risk Over Time"
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        if "recommended_action_simulated" in weather:

            st.subheader("Recommended Action Labels")

            actions = (
                weather["recommended_action_simulated"]
                .value_counts()
                .reset_index()
            )

            actions.columns = ["Action", "Records"]

            st.dataframe(actions, use_container_width=True)

        with st.expander("View Weather Records"):
            st.dataframe(weather, use_container_width=True)

        download_csv(
            weather,
            "weather_risk_analysis.csv"
        )

# =====================================================
# 13. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("Machine Learning Model Performance")

    if geology is None:

        st.error("Geochemistry dataset unavailable.")

    elif model is None:

        st.error("Run train_model.py to create the model.")

    elif not has_columns(
        geology,
        GEO_FEATURES + [TARGET]
    ):

        st.error("Required features or target are missing.")

    else:

        train_data = geology[
            GEO_FEATURES + [TARGET]
        ].copy()

        for col in GEO_FEATURES[:-1]:

            train_data[col] = pd.to_numeric(
                train_data[col],
                errors="coerce"
            )

        train_data[TARGET] = pd.to_numeric(
            train_data[TARGET],
            errors="coerce"
        )

        train_data = train_data.dropna(
            subset=GEO_FEATURES + [TARGET]
        )

        X = train_data[GEO_FEATURES]
        y = train_data[TARGET].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        try:

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Test Accuracy",
                f"{accuracy * 100:.1f}%"
            )

            c2.metric(
                "Training Samples",
                len(X_train)
            )

            c3.metric(
                "Testing Samples",
                len(X_test)
            )

            st.warning(
                "These metrics evaluate the model on synthetic "
                "data. They do not establish real-world "
                "mineral exploration accuracy."
            )

            st.subheader("Confusion Matrix")

            matrix = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            )

            fig, ax = plt.subplots()

            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Class 0", "Class 1"]
            )

            display.plot(
                ax=ax,
                values_format="d",
                colorbar=False
            )

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0
            )

            st.dataframe(
                pd.DataFrame(report).T.round(3),
                use_container_width=True
            )

            if hasattr(model, "named_steps"):

                classifier = model.named_steps.get(
                    "classifier"
                )

                preprocessor = model.named_steps.get(
                    "preprocessor"
                )

            else:

                classifier = model
                preprocessor = None

            if (
                classifier is not None
                and hasattr(classifier, "feature_importances_")
            ):

                importance = classifier.feature_importances_

                try:

                    feature_names = (
                        preprocessor.get_feature_names_out()
                        if preprocessor is not None
                        else [
                            f"Feature {i+1}"
                            for i in range(len(importance))
                        ]
                    )

                except Exception:

                    feature_names = [
                        f"Feature {i+1}"
                        for i in range(len(importance))
                    ]

                if len(feature_names) == len(importance):

                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importance
                    }).sort_values("Importance")

                    fig = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Random Forest Feature Importance"
                    )

                    fig.update_layout(template="plotly_dark")

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        except Exception as error:

            st.error(f"Model evaluation error: {error}")

# =====================================================
# 14. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("About Manganese Intelligence")

    st.subheader(
        "AI-Powered Mineral Exploration and "
        "Mining Operations Analysis"
    )

    st.write(
        "Manganese Intelligence is a research prototype "
        "that demonstrates machine learning and data "
        "analytics for manganese exploration and mining "
        "operations."
    )

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze manganese geochemical samples.
    - Explore satellite and environmental indicators.
    - Examine borehole assay intervals.
    - Analyze planned and actual production.
    - Monitor equipment downtime.
    - Examine weather-related operational risks.
    - Visualize results through an interactive dashboard.
    """)

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Random Forest
    - Plotly
    - Streamlit
    - Excel
    """)

    st.subheader("Data Limitations")

    st.warning("""
    All six SIH26009 datasets are synthetic/simulated.
    They are not actual mine measurements, surveyed
    coordinates, real satellite observations or verified
    borehole records.

    Results are intended for prototype demonstration
    and software testing only.

    Real-world use requires actual field measurements,
    genuine satellite data, geological validation and
    independent model testing.
    """)

    st.subheader("Future Enhancements")

    st.markdown("""
    - XGBoost model comparison.
    - SHAP explainability.
    - Independent model validation.
    - Real satellite imagery integration.
    - Verified geological spatial layers.
    - Production forecasting.
    """)

    st.caption(
        "Manganese Intelligence | Research Prototype"
    )

# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.caption(
    "Manganese Intelligence | Research and Demonstration "
    "| Synthetic Data Disclaimer Applies"
)
