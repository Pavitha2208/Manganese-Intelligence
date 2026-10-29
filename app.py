
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import joblib


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Manganese Intelligence",
    page_icon="⛏️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 16px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. PROJECT FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DOWNLOADS_DIR = Path.home() / "Downloads"

DATA_DIR = (
    DOWNLOADS_DIR
    / "SIH26009_Six_Synthetic_Training_Datasets (2)"
)

GEO_FILE = DATA_DIR / "01_Exploration_Geochemistry_Prospectivity.xlsx"
SATELLITE_FILE = DATA_DIR / "02_Satellite_Environmental_Features.xlsx"
BOREHOLE_FILE = DATA_DIR / "03_Borehole_Assay_Intervals.xlsx"
PRODUCTION_FILE = DATA_DIR / "04_Monthly_Production_Shortfall.xlsx"
EQUIPMENT_FILE = DATA_DIR / "05_Equipment_Downtime_Events.xlsx"
WEATHER_FILE = DATA_DIR / "06_Weather_Operational_Risk.xlsx"

MODEL_FILE = BASE_DIR / "manganese_model.joblib"


# =========================================================
# 3. CONSTANTS
# =========================================================

TARGET_COLUMN = "prospectivity_label_simulated"

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

NUMERIC_GEO_FEATURES = GEO_FEATURES[:-1]

PRODUCTION_DATE = "month_simulated"
PLANNED_COLUMN = "planned_output_tonnes_simulated"
ACTUAL_COLUMN = "actual_output_tonnes_simulated"
SHORTFALL_COLUMN = "shortfall_tonnes_simulated"
SHORTFALL_PERCENT_COLUMN = "shortfall_pct_simulated"


# =========================================================
# 4. DATA LOADING
# =========================================================

@st.cache_data(show_spinner=False)
def load_excel(file_path):
    """Load an Excel file if it exists."""
    if not file_path.exists():
        return None

    try:
        return pd.read_excel(file_path)
    except Exception as exc:
        st.warning(f"Could not read {file_path.name}: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained classification pipeline."""
    if not MODEL_FILE.exists():
        return None

    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return None


geology = load_excel(GEO_FILE)
satellite = load_excel(SATELLITE_FILE)
borehole = load_excel(BOREHOLE_FILE)
production = load_excel(PRODUCTION_FILE)
equipment = load_excel(EQUIPMENT_FILE)
weather = load_excel(WEATHER_FILE)

model = load_model()


# =========================================================
# 5. GENERAL HELPERS
# =========================================================

def has_columns(data, columns):
    return data is not None and all(
        column in data.columns for column in columns
    )


def number(value, decimals=0):
    """Format a numeric value safely."""
    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def first_existing_column(data, candidates):
    """Find the first matching column name."""
    if data is None:
        return None

    for column in candidates:
        if column in data.columns:
            return column

    return None


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def add_prediction_scores(data):
    """Generate prospectivity scores using the trained model."""
    if model is None:
        return None, "Trained model file was not found."

    if not has_columns(data, GEO_FEATURES):
        return None, "Required model input columns are missing."

    X = data[GEO_FEATURES].copy()

    for column in NUMERIC_GEO_FEATURES:
        X[column] = clean_numeric(X[column])

    valid_rows = X[NUMERIC_GEO_FEATURES].notna().all(axis=1)
    valid_rows &= X["sample_method_simulated"].notna()

    if not valid_rows.any():
        return None, "No complete rows are available for prediction."

    try:
        scores = pd.Series(
            np.nan,
            index=data.index,
            dtype=float,
        )

        probabilities = model.predict_proba(X.loc[valid_rows])

        positive_class_index = list(model.classes_).index(1)

        scores.loc[valid_rows] = (
            probabilities[:, positive_class_index] * 100
        )

        return scores, None

    except Exception as exc:
        return None, f"Prediction could not be generated: {exc}"


def get_production_data():
    """Prepare production records without modifying source values."""
    if production is None:
        return None

    required = [
        PRODUCTION_DATE,
        PLANNED_COLUMN,
        ACTUAL_COLUMN,
    ]

    if not has_columns(production, required):
        return None

    data = production.copy()

    data[PRODUCTION_DATE] = pd.to_datetime(
        data[PRODUCTION_DATE],
        errors="coerce",
    )

    data[PLANNED_COLUMN] = clean_numeric(data[PLANNED_COLUMN])
    data[ACTUAL_COLUMN] = clean_numeric(data[ACTUAL_COLUMN])

    data = data.dropna(
        subset=[PRODUCTION_DATE, PLANNED_COLUMN, ACTUAL_COLUMN]
    )

    data = data.sort_values(PRODUCTION_DATE)

    data["calculated_shortfall_tonnes"] = (
        data[PLANNED_COLUMN] - data[ACTUAL_COLUMN]
    )

    data["calculated_shortfall_pct"] = np.where(
        data[PLANNED_COLUMN] > 0,
        data["calculated_shortfall_tonnes"]
        / data[PLANNED_COLUMN]
        * 100,
        np.nan,
    )

    data["calculated_achievement_pct"] = np.where(
        data[PLANNED_COLUMN] > 0,
        data[ACTUAL_COLUMN] / data[PLANNED_COLUMN] * 100,
        np.nan,
    )

    return data


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("⛏️ Manganese Intelligence")
st.sidebar.caption("AI-powered mineral exploration prototype")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "AI Exploration",
        "Production Analysis",
        "Model Performance",
        "About Project",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Prototype notice: training datasets contain simulated data. "
    "Predictions are exploratory and are not verified mineral discoveries."
)


# =========================================================
# 7. OVERVIEW
# =========================================================

if page == "Overview":

    st.title("Manganese Intelligence")
    st.subheader(
        "AI-Powered Mineral Exploration and Production Analysis"
    )

    st.write(
        "Explore geological sample patterns, model-generated "
        "prospectivity scores, and production performance in one dashboard."
    )

    st.info(
        "This prototype uses simulated training datasets. "
        "Its scores demonstrate the workflow and must not be interpreted "
        "as confirmed manganese reserves or real-world forecasts."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Geochemical Samples",
        number(len(geology)) if geology is not None else "N/A",
    )

    c2.metric(
        "Satellite Records",
        number(len(satellite)) if satellite is not None else "N/A",
    )

    c3.metric(
        "Borehole Records",
        number(len(borehole)) if borehole is not None else "N/A",
    )

    c4.metric(
        "Production Records",
        number(len(production)) if production is not None else "N/A",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Geochemical Sample Distribution")

        if has_columns(geology, ["manganese_ppm_simulated"]):
            fig = px.histogram(
                geology,
                x="manganese_ppm_simulated",
                nbins=30,
                title="Simulated Manganese Concentration",
                labels={
                    "manganese_ppm_simulated":
                    "Manganese concentration (ppm)"
                },
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Geochemical data is unavailable.")

    with right:
        st.subheader("Production Overview")

        prod = get_production_data()

        if prod is not None and not prod.empty:
            monthly = (
                prod.groupby(PRODUCTION_DATE, as_index=False)[
                    [PLANNED_COLUMN, ACTUAL_COLUMN]
                ]
                .sum()
                .sort_values(PRODUCTION_DATE)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=monthly[PRODUCTION_DATE],
                    y=monthly[PLANNED_COLUMN],
                    mode="lines+markers",
                    name="Planned Output",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=monthly[PRODUCTION_DATE],
                    y=monthly[ACTUAL_COLUMN],
                    mode="lines+markers",
                    name="Actual Output",
                )
            )

            fig.update_layout(
                title="Planned vs Actual Production",
                xaxis_title="Month",
                yaxis_title="Production (tonnes)",
                template="plotly_dark",
            )

            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Production data is unavailable.")

    st.subheader("Available Modules")

    a, b, c = st.columns(3)

    a.markdown("**AI Exploration**")
    a.write("View model-generated prospectivity scores.")

    b.markdown("**Production Analysis**")
    b.write("Study planned output, actual output, and shortfall.")

    c.markdown("**Model Performance**")
    c.write("Review classification metrics and feature importance.")


# =========================================================
# 8. AI EXPLORATION
# =========================================================

elif page == "AI Exploration":

    st.title("AI Exploration")
    st.write(
        "Use the trained machine-learning model to explore "
        "prospectivity scores for the available simulated samples."
    )

    if geology is None:
        st.error(
            "Geochemical dataset not found. Check the dataset path "
            "at the top of app.py."
        )

    elif model is None:
        st.error(
            "The trained model was not found. Run train_model.py "
            "to create manganese_model.joblib."
        )

    elif not has_columns(geology, GEO_FEATURES):
        st.error(
            "The geochemical dataset does not contain all required "
            "model input columns."
        )

    else:
        scored_data = geology.copy()
        scores, error = add_prediction_scores(scored_data)

        if error:
            st.error(error)

        else:
            scored_data["prospectivity_score_pct"] = scores

            threshold = st.slider(
                "Minimum prospectivity score (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
            )

            filtered = scored_data[
                scored_data["prospectivity_score_pct"] >= threshold
            ].copy()

            valid_scores = scored_data[
                "prospectivity_score_pct"
            ].dropna()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Samples Scored",
                number(valid_scores.count()),
            )

            c2.metric(
                "Samples Above Threshold",
                number(len(filtered)),
            )

            c3.metric(
                "Mean Score",
                f"{valid_scores.mean():.1f}%"
                if not valid_scores.empty
                else "N/A",
            )

            st.caption(
                "The score is the model's estimated probability for "
                "the positive training label. It is not a measured "
                "probability that an economically mineable deposit exists."
            )

            latitude_column = first_existing_column(
                scored_data,
                [
                    "latitude_simulated",
                    "latitude_deg_simulated",
                    "latitude",
                    "lat",
                    "Latitude",
                ],
            )

            longitude_column = first_existing_column(
                scored_data,
                [
                    "longitude_simulated",
                    "longitude_deg_simulated",
                    "longitude",
                    "lon",
                    "Longitude",
                    "long",
                ],
            )

            st.subheader("Prospectivity Score Distribution")

            fig = px.histogram(
                scored_data.dropna(
                    subset=["prospectivity_score_pct"]
                ),
                x="prospectivity_score_pct",
                nbins=20,
                title="Model-Generated Scores",
                labels={
                    "prospectivity_score_pct": "Score (%)"
                },
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width="stretch")

            st.subheader("Sample Locations")

            if latitude_column and longitude_column:
                map_data = filtered.copy()

                map_data[latitude_column] = clean_numeric(
                    map_data[latitude_column]
                )
                map_data[longitude_column] = clean_numeric(
                    map_data[longitude_column]
                )

                map_data = map_data.dropna(
                    subset=[
                        latitude_column,
                        longitude_column,
                        "prospectivity_score_pct",
                    ]
                )

                if not map_data.empty:
                    fig = px.scatter_geo(
                        map_data,
                        lat=latitude_column,
                        lon=longitude_column,
                        color="prospectivity_score_pct",
                        hover_data=["prospectivity_score_pct"],
                        color_continuous_scale="YlOrRd",
                        title="Filtered Sample Locations",
                        labels={
                            "prospectivity_score_pct": "Score (%)"
                        },
                    )
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info(
                        "No valid coordinates are available for the "
                        "selected threshold."
                    )
            else:
                st.info(
                    "Coordinate columns were not detected. "
                    "The score distribution and sample table are available."
                )

            st.subheader("Scored Samples")

            display_columns = [
                col for col in [
                    "sample_id_simulated",
                    "sample_id",
                    "manganese_ppm_simulated",
                    "iron_pct_simulated",
                    "prospectivity_score_pct",
                ]
                if col in filtered.columns
            ]

            if not display_columns:
                display_columns = [
                    "prospectivity_score_pct"
                ]

            st.dataframe(
                filtered[display_columns].sort_values(
                    "prospectivity_score_pct",
                    ascending=False,
                ),
                width="stretch",
            )

            st.download_button(
                "Download Scored Samples CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="manganese_prospectivity_scores.csv",
                mime="text/csv",
            )


# =========================================================
# 9. PRODUCTION ANALYSIS
# =========================================================

elif page == "Production Analysis":

    st.title("Production Analysis")
    st.write(
        "Compare planned and actual output, calculate production shortfall, "
        "and explore the available operational records."
    )

    prod = get_production_data()

    if prod is None or prod.empty:
        st.error(
            "Production data is missing or does not contain the expected "
            "columns. Check the production Excel path and column names."
        )

    else:
        source_min = prod[PRODUCTION_DATE].min().date()
        source_max = prod[PRODUCTION_DATE].max().date()

        st.caption(
            f"Source data date range: {source_min} to {source_max}"
        )

        date_range = st.date_input(
            "Select date range",
            value=(source_min, source_max),
            min_value=source_min,
            max_value=source_max,
        )

        if isinstance(date_range, (tuple, list)):
            if len(date_range) == 2:
                start_date, end_date = date_range
            elif len(date_range) == 1:
                start_date = end_date = date_range[0]
            else:
                start_date, end_date = source_min, source_max
        else:
            start_date = end_date = date_range

        filtered = prod[
            (prod[PRODUCTION_DATE].dt.date >= start_date)
            & (prod[PRODUCTION_DATE].dt.date <= end_date)
        ].copy()

        if filtered.empty:
            st.warning("No production records fall within this date range.")

        else:
            total_planned = filtered[PLANNED_COLUMN].sum()
            total_actual = filtered[ACTUAL_COLUMN].sum()

            total_shortfall = (
                total_planned - total_actual
            )

            shortfall_pct = (
                total_shortfall / total_planned * 100
                if total_planned > 0
                else np.nan
            )

            achievement_pct = (
                total_actual / total_planned * 100
                if total_planned > 0
                else np.nan
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Planned Production",
                f"{number(total_planned)} tonnes",
            )

            c2.metric(
                "Actual Production",
                f"{number(total_actual)} tonnes",
            )

            c3.metric(
                "Production Shortfall",
                f"{number(total_shortfall)} tonnes",
            )

            c4.metric(
                "Shortfall Percentage",
                f"{shortfall_pct:.2f}%"
                if pd.notna(shortfall_pct)
                else "N/A",
            )

            st.metric(
                "Production Achievement",
                f"{achievement_pct:.2f}%"
                if pd.notna(achievement_pct)
                else "N/A",
                help="Actual production divided by planned production.",
            )

            st.divider()

            st.subheader("Production Trend")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=filtered[PRODUCTION_DATE],
                    y=filtered[PLANNED_COLUMN],
                    mode="lines+markers",
                    name="Planned Output",
                    line=dict(width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=filtered[PRODUCTION_DATE],
                    y=filtered[ACTUAL_COLUMN],
                    mode="lines+markers",
                    name="Actual Output",
                    line=dict(width=2),
                )
            )

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Production (tonnes)",
                template="plotly_dark",
                legend=dict(orientation="h"),
            )

            st.plotly_chart(fig, width="stretch")

            st.subheader("Shortfall by Month")

            monthly = (
                filtered.groupby(PRODUCTION_DATE, as_index=False)
                .agg(
                    planned_output=(PLANNED_COLUMN, "sum"),
                    actual_output=(ACTUAL_COLUMN, "sum"),
                )
            )

            monthly["shortfall_tonnes"] = (
                monthly["planned_output"] - monthly["actual_output"]
            )

            fig = px.bar(
                monthly,
                x=PRODUCTION_DATE,
                y="shortfall_tonnes",
                title="Planned Minus Actual Production",
                labels={
                    PRODUCTION_DATE: "Month",
                    "shortfall_tonnes": "Shortfall (tonnes)",
                },
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width="stretch")

            operational_columns = [
                col for col in [
                    "rainfall_mm_simulated",
                    "equipment_downtime_hours_simulated",
                    "blasting_delay_hours_simulated",
                    "ore_grade_pct_simulated",
                    "workforce_availability_pct_simulated",
                    "power_availability_pct_simulated",
                ]
                if col in filtered.columns
            ]

            if operational_columns:
                st.subheader("Operational Indicators")

                st.dataframe(
                    filtered[operational_columns].describe().T,
                    width="stretch",
                )

            with st.expander("View production records"):
                st.dataframe(filtered, width="stretch")

            st.download_button(
                "Download Filtered Production CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="manganese_production_analysis.csv",
                mime="text/csv",
            )

            st.caption(
                "Shortfall KPIs are calculated from the selected records. "
                "Negative shortfall means actual production exceeded plan. "
                "These records are simulated and are not actual mine output."
            )


# =========================================================
# 10. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("Model Performance")
    st.write(
        "Evaluate the trained classification model using the same "
        "80/20 stratified split and random seed used during training."
    )

    if geology is None:
        st.error("Geochemical training dataset is unavailable.")

    elif model is None:
        st.error(
            "Trained model not found. Run train_model.py first."
        )

    elif not has_columns(geology, GEO_FEATURES + [TARGET_COLUMN]):
        st.error(
            "The dataset is missing model features or the target column."
        )

    else:
        training_data = geology[
            GEO_FEATURES + [TARGET_COLUMN]
        ].copy()

        for column in NUMERIC_GEO_FEATURES:
            training_data[column] = clean_numeric(
                training_data[column]
            )

        training_data[TARGET_COLUMN] = pd.to_numeric(
            training_data[TARGET_COLUMN],
            errors="coerce",
        )

        training_data = training_data.dropna(
            subset=GEO_FEATURES + [TARGET_COLUMN]
        )

        X = training_data[GEO_FEATURES]
        y = training_data[TARGET_COLUMN].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        c1, c2, c3 = st.columns(3)

        c1.metric("Test Accuracy", f"{accuracy * 100:.1f}%")
        c2.metric("Training Samples", number(len(X_train)))
        c3.metric("Testing Samples", number(len(X_test)))

        st.warning(
            "These metrics measure performance on a synthetic dataset. "
            "They do not demonstrate real-world mineral discovery accuracy."
        )

        st.subheader("Confusion Matrix")

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        fig, ax = plt.subplots()

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Class 0", "Class 1"],
        )

        display.plot(
            ax=ax,
            values_format="d",
            colorbar=False,
        )

        ax.set_title("Test Set Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        st.dataframe(
            pd.DataFrame(report).T.round(3),
            width="stretch",
        )

        if hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")

            if classifier is None:
                classifier = model.named_steps.get("model")
        else:
            classifier = model

        if (
            classifier is not None
            and hasattr(classifier, "feature_importances_")
        ):
            st.subheader("Feature Importance")

            importance_values = classifier.feature_importances_

            try:
                preprocessor = model.named_steps.get("preprocessor")

                if preprocessor is not None:
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    feature_names = np.array(
                        [f"Feature {i + 1}" for i in range(
                            len(importance_values)
                        )]
                    )

            except Exception:
                feature_names = np.array(
                    [f"Feature {i + 1}" for i in range(
                        len(importance_values)
                    )]
                )

            if len(feature_names) != len(importance_values):
                feature_names = np.array(
                    [f"Feature {i + 1}" for i in range(
                        len(importance_values)
                    )]
                )

            importance = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": importance_values,
                }
            ).sort_values("Importance", ascending=True)

            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Relative Model Feature Importance",
            )

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Feature importance indicates how the fitted model uses "
            "features; it does not establish geological causation."
        )


# =========================================================
# 11. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("About Manganese Intelligence")

    st.subheader(
        "AI-Powered Mineral Exploration and Production Analysis"
    )

    st.write(
        """
        Manganese Intelligence is a prototype dashboard that demonstrates
        how machine learning and data analysis can support mineral
        exploration workflows and production monitoring.
        """
    )

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Explore geochemical patterns in manganese-related samples.
        - Generate model-based prospectivity scores for available samples.
        - Compare planned and actual production.
        - Quantify production shortfall and achievement.
        - Examine model performance and the relative importance of features.
        - Present results through an interactive dashboard.
        """
    )

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **Python** — data processing and machine learning
        - **Pandas and NumPy** — data analysis
        - **Scikit-learn** — model training and evaluation
        - **Plotly** — interactive charts
        - **Streamlit** — dashboard interface
        - **Excel** — prototype datasets
        """
    )

    st.subheader("Workflow")

    st.markdown(
        """
        1. Load the geochemical and production datasets.
        2. Prepare the geochemical features for the trained model.
        3. Generate prospectivity scores for eligible samples.
        4. Analyze planned output, actual output, and shortfall.
        5. Display model evaluation metrics and interactive visualizations.
        """
    )

    st.subheader("Data Limitations")

    st.warning(
        """
        The bundled exploration, satellite, borehole, production,
        equipment, and weather datasets are synthetic training data.
        Results are intended for prototype demonstration and software
        testing only. They do not confirm manganese deposits, estimate
        verified reserves, or provide validated operational forecasts.
        Real-world use would require reliable field measurements,
        suitable satellite imagery, geological validation, and independent
        testing on representative data.
        """
    )

    st.subheader("Future Enhancements")

    st.markdown(
        """
        - Integrate validated satellite imagery and geological layers.
        - Add explainability methods such as SHAP.
        - Develop and validate production shortfall forecasting.
        - Add spatial visualization using verified coordinates.
        - Evaluate the system with real, independently verified data.
        """
    )

    st.caption("Manganese Intelligence | Research and prototype dashboard")
