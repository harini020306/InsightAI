
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

from modules.model_training import train_model


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="InsightAI | Intelligent Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
}

.metric-card {
    background: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263449;
}

div[data-testid="stMetric"] {
    background-color: #172033;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #263449;
}

.stButton > button {
    background-color: #0284c7;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #0369a1;
    color: white;
}

h1, h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📊 InsightAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Data Analysis & Prediction System</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload your dataset, explore insights, visualize patterns, "
    "compare machine learning models, and generate predictions."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ InsightAI Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "InsightAI helps you analyze datasets, discover patterns, "
    "compare ML models, and generate predictions."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "insightai_model_result" not in st.session_state:
    st.session_state["insightai_model_result"] = None

if "insightai_model_config" not in st.session_state:
    st.session_state["insightai_model_config"] = None


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if uploaded_file is None:

    st.info("👈 Upload a CSV dataset from the sidebar to get started.")

    st.markdown("### 🚀 What can InsightAI do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📁 Data Analysis")
        st.write("Explore datasets and understand their structure.")

    with col2:
        st.markdown("### 📈 Visualization")
        st.write("Discover patterns through interactive charts.")

    with col3:
        st.markdown("### 🤖 Machine Learning")
        st.write("Train models, compare performance, and predict results.")

else:

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded dataset is empty.")
            st.stop()

        if len(df.columns) < 2:
            st.error("Please upload a dataset containing at least two columns.")
            st.stop()

        if not df.columns.is_unique:
            st.error("Your dataset contains duplicate column names. Please rename them.")
            st.stop()

        st.success("Dataset uploaded successfully!")

        # --------------------------------------------------
        # DATASET OVERVIEW
        # --------------------------------------------------

        st.header("📌 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")

        with col2:
            st.metric("Total Columns", f"{df.shape[1]:,}")

        with col3:
            st.metric(
                "Missing Values",
                f"{df.isnull().sum().sum():,}"
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                f"{df.duplicated().sum():,}"
            )

        st.divider()

        # --------------------------------------------------
        # TABS
        # --------------------------------------------------

        tabs = st.tabs([
            "🏠 Overview",
            "📄 Data Preview",
            "🧹 Data Quality",
            "📊 Statistics",
            "📈 Visualizations",
            "🤖 Machine Learning"
        ])

        # --------------------------------------------------
        # TAB 1: OVERVIEW
        # --------------------------------------------------

        with tabs[0]:

            st.subheader("Dataset Overview")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Dataset Information**")
                info_df = pd.DataFrame({
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str).values,
                    "Non-Null Values": df.notnull().sum().values
                })

                st.dataframe(
                    info_df,
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                st.markdown("**Numerical Summary**")

                if len(df.select_dtypes(include=np.number).columns) > 0:
                    st.dataframe(
                        df.describe().T,
                        use_container_width=True
                    )
                else:
                    st.info("No numerical columns found.")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        # --------------------------------------------------
        # TAB 2: DATA PREVIEW
        # --------------------------------------------------

        with tabs[1]:

            st.subheader("📄 Explore Your Dataset")

            st.write("View the complete uploaded dataset.")

            st.dataframe(
                df,
                use_container_width=True,
                height=400
            )

            st.download_button(
                "⬇️ Download Dataset",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="insightai_dataset.csv",
                mime="text/csv"
            )

        # --------------------------------------------------
        # TAB 3: DATA QUALITY
        # --------------------------------------------------

        with tabs[2]:

            st.subheader("🧹 Data Quality Analysis")

            quality_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isnull().sum().values,
                "Missing Percentage": (
                    df.isnull().mean().values * 100
                ).round(2)
            })

            st.dataframe(
                quality_df,
                use_container_width=True,
                hide_index=True
            )

            duplicates = df.duplicated().sum()

            if duplicates > 0:
                st.warning(
                    f"Found {duplicates} duplicate rows."
                )
            else:
                st.success("No duplicate rows found.")

            missing = df.isnull().sum().sum()

            if missing > 0:
                st.warning(
                    f"Found {missing} missing values."
                )
            else:
                st.success("No missing values found.")

            st.markdown("### Missing Values by Column")

            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": df.isnull().sum().values
            })

            missing_df = missing_df[
                missing_df["Missing Values"] > 0
            ]

            if not missing_df.empty:

                fig = px.bar(
                    missing_df,
                    x="Column",
                    y="Missing Values",
                    title="Missing Values Distribution",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info("No missing values to visualize.")

        # --------------------------------------------------
        # TAB 4: STATISTICS
        # --------------------------------------------------

        with tabs[3]:

            st.subheader("📊 Statistical Analysis")

            numeric_df = df.select_dtypes(include=np.number)

            if not numeric_df.empty:

                st.markdown("### Descriptive Statistics")

                st.dataframe(
                    numeric_df.describe().T,
                    use_container_width=True
                )

                st.markdown("### Correlation Matrix")

                corr = numeric_df.corr()

                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info("No numerical columns available for statistics.")

        # --------------------------------------------------
        # TAB 5: VISUALIZATIONS
        # --------------------------------------------------

        with tabs[4]:

            st.subheader("📈 Interactive Data Visualization")

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_columns = df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            if numeric_columns:

                st.markdown("### Histogram")

                selected_numeric = st.selectbox(
                    "Select numerical column",
                    numeric_columns,
                    key="histogram_column"
                )

                fig = px.histogram(
                    df,
                    x=selected_numeric,
                    title=f"Distribution of {selected_numeric}",
                    template="plotly_dark",
                    marginal="box"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.markdown("### Box Plot")

                box_column = st.selectbox(
                    "Select column for box plot",
                    numeric_columns,
                    key="box_column"
                )

                fig = px.box(
                    df,
                    y=box_column,
                    title=f"Box Plot of {box_column}",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            if categorical_columns and numeric_columns:

                st.markdown("### Category-wise Analysis")

                category = st.selectbox(
                    "Select categorical column",
                    categorical_columns,
                    key="category_column"
                )

                metric = st.selectbox(
                    "Select numerical column",
                    numeric_columns,
                    key="metric_column"
                )

                grouped = (
                    df.groupby(category, dropna=False)[metric]
                    .mean()
                    .reset_index()
                )

                fig = px.bar(
                    grouped,
                    x=category,
                    y=metric,
                    title=f"Average {metric} by {category}",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            elif not numeric_columns:
                st.info("No numerical columns available for visualization.")

            if not numeric_columns and not categorical_columns:
                st.info("No supported columns found.")

        # --------------------------------------------------
        # TAB 6: MACHINE LEARNING
        # --------------------------------------------------

        with tabs[5]:

            st.header("🤖 Machine Learning Engine")

            st.write(
                "Train machine learning models, compare their performance, "
                "understand important features, and generate predictions."
            )

            if len(df) < 5:
                st.warning(
                    "Please upload at least 5 rows to train a model."
                )

            else:

                target_column = st.selectbox(
                    "Select Target Column",
                    df.columns.tolist()
                )

                task_type = st.radio(
                    "Select Machine Learning Task",
                    ["Regression", "Classification"],
                    horizontal=True
                )

                test_size = st.slider(
                    "Testing Data Percentage",
                    min_value=20,
                    max_value=40,
                    value=20,
                    step=5
                )

                current_config = (
                    uploaded_file.name,
                    target_column,
                    task_type,
                    test_size
                )

                if (
                    st.session_state["insightai_model_config"]
                    != current_config
                ):
                    st.session_state["insightai_model_result"] = None
                    st.session_state["insightai_model_config"] = None

                if st.button(
                    "🚀 Train Machine Learning Models",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Training and comparing machine learning models..."
                    ):

                        try:

                            result = train_model(
                                df,
                                target_column,
                                task_type,
                                test_size / 100
                            )

                            st.session_state[
                                "insightai_model_result"
                            ] = result

                            st.session_state[
                                "insightai_model_config"
                            ] = current_config

                            st.success(
                                "Machine learning models trained successfully!"
                            )

                        except Exception as e:
                            st.session_state[
                                "insightai_model_result"
                            ] = None

                            st.error(f"Model training failed: {e}")

                result = st.session_state[
                    "insightai_model_result"
                ]

                if (
                    result is not None
                    and st.session_state["insightai_model_config"]
                    == current_config
                ):

                    st.divider()

                    # BEST MODEL

                    st.subheader("🏆 Best Performing Model")

                    st.success(
                        f"Best Performing Model: "
                        f"{result['best_model_name']}"
                    )

                    # MODEL COMPARISON

                    st.subheader("📊 Model Comparison")

                    comparison = result.get("comparison")

                    if comparison is not None and not comparison.empty:

                        st.dataframe(
                            comparison,
                            use_container_width=True,
                            hide_index=True
                        )

                        if task_type == "Classification":

                            fig = px.bar(
                                comparison,
                                x="Model",
                                y="Accuracy",
                                title="Classification Model Accuracy Comparison",
                                text="Accuracy",
                                template="plotly_dark"
                            )

                            fig.update_yaxes(range=[0, 1])

                        else:

                            fig = px.bar(
                                comparison,
                                x="Model",
                                y="RMSE",
                                title="Regression Model RMSE Comparison",
                                text="RMSE",
                                template="plotly_dark"
                            )

                            fig.update_yaxes(rangemode="tozero")

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:
                        st.info("Model comparison results are unavailable.")

                    # MODEL PERFORMANCE

                    st.subheader("📈 Best Model Performance")

                    metrics = result["metrics"]

                    metric_columns = st.columns(len(metrics))

                    for i, (name, value) in enumerate(metrics.items()):

                        with metric_columns[i]:

                            st.metric(
                                name,
                                f"{value:.4f}"
                                if isinstance(value, (int, float, np.number))
                                else str(value)
                            )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Training Rows",
                            result["training_rows"]
                        )

                    with col2:
                        st.metric(
                            "Testing Rows",
                            result["testing_rows"]
                        )

                    # CONFUSION MATRIX

                    if task_type == "Classification":

                        st.subheader("🎯 Confusion Matrix")

                        cm = confusion_matrix(
                            result["actual"],
                            result["predictions"]
                        )

                        fig = px.imshow(
                            cm,
                            text_auto=True,
                            aspect="auto",
                            title="Classification Confusion Matrix",
                            labels={
                                "x": "Predicted Class",
                                "y": "Actual Class",
                                "color": "Count"
                            },
                            template="plotly_dark"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.caption(
                            "Rows represent actual classes; "
                            "columns represent predicted classes."
                        )

                    # REGRESSION PLOT

                    else:

                        st.subheader("📉 Actual vs Predicted Values")

                        prediction_df = pd.DataFrame({
                            "Actual": np.asarray(
                                result["actual"]
                            ).ravel(),
                            "Predicted": np.asarray(
                                result["predictions"]
                            ).ravel()
                        })

                        fig = px.scatter(
                            prediction_df,
                            x="Actual",
                            y="Predicted",
                            title="Actual vs Predicted Values",
                            template="plotly_dark"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    # PERMUTATION FEATURE IMPORTANCE

                    st.subheader("🔍 Permutation Feature Importance")

                    try:

                        scoring = (
                            "accuracy"
                            if task_type == "Classification"
                            else "r2"
                        )

                        importance = permutation_importance(
                            result["model"],
                            result["X_test"],
                            result["y_test"],
                            n_repeats=10,
                            random_state=42,
                            scoring=scoring
                        )

                        importance_df = pd.DataFrame({
                            "Feature": result["X_test"].columns,
                            "Importance": importance.importances_mean
                        }).sort_values(
                            by="Importance",
                            ascending=False
                        )

                        fig = px.bar(
                            importance_df,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title="Feature Importance",
                            template="plotly_dark"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.caption(
                            "Higher positive values indicate that shuffling "
                            "a feature reduces model performance. Negative "
                            "values can occur and should be interpreted carefully."
                        )

                    except Exception as e:

                        st.warning(
                            f"Feature importance unavailable: {e}"
                        )

                    # NEW DATA PREDICTION

                    st.divider()

                    st.subheader("🔮 Make a New Prediction")

                    st.write(
                        "Enter new feature values to generate a prediction."
                    )

                    model = result["model"]

                    feature_columns = result["feature_columns"]

                    with st.form("prediction_form"):

                        input_values = {}

                        for column in feature_columns:

                            if column not in df.columns:
                                continue

                            if pd.api.types.is_numeric_dtype(df[column]):

                                median_value = df[column].median()

                                if pd.isna(median_value):
                                    median_value = 0

                                input_values[column] = st.number_input(
                                    f"{column}",
                                    value=float(median_value)
                                )

                            else:

                                categories = (
                                    df[column]
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist()
                                )

                                categories = sorted(categories)

                                if categories:

                                    input_values[column] = st.selectbox(
                                        f"{column}",
                                        categories
                                    )

                                else:

                                    input_values[column] = st.text_input(
                                        f"{column}"
                                    )

                        predict_button = st.form_submit_button(
                            "🔮 Generate Prediction"
                        )

                    if predict_button:

                        try:

                            input_df = pd.DataFrame(
                                [input_values],
                                columns=feature_columns
                            )

                            prediction = model.predict(input_df)

                            st.success("Prediction generated successfully!")

                            st.metric(
                                "Predicted Result",
                                str(prediction[0])
                            )

                        except Exception as e:

                            st.error(
                                f"Prediction failed: {e}"
                            )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded CSV file: {e}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        Built with Python, Streamlit, Pandas, Plotly & Scikit-learn
        <br>
        InsightAI | Intelligent Data Analysis & Prediction System
    </div>
    """,
    unsafe_allow_html=True
)
