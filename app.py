
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.caption(
    "Machine Learning | Fraud Risk Analysis | Model Evaluation"
)

st.markdown("""
This application compares machine learning models to identify
potentially fraudulent transactions using transaction data.
""")

# ==================================================
# 2. FEATURE DEFINITIONS
# ==================================================
numeric_features = [
    "amt",
    "city_pop",
    "hour",
    "day_of_week",
    "distance_km"
]

categorical_features = [
    "category",
    "gender"
]

features = numeric_features + categorical_features
target = "is_fraud"


# ==================================================
# 3. DATA PREPROCESSING
# ==================================================
def prepare_data(data):
    data = data.copy()

    required_columns = [
        "trans_date_trans_time",
        "amt",
        "city_pop",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "category",
        "gender",
        "is_fraud"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    data["trans_date_trans_time"] = pd.to_datetime(
        data["trans_date_trans_time"],
        errors="coerce"
    )

    data["hour"] = data["trans_date_trans_time"].dt.hour
    data["day_of_week"] = (
        data["trans_date_trans_time"].dt.dayofweek
    )

    # Calculate customer-to-merchant distance
    lat1 = np.radians(
        pd.to_numeric(data["lat"], errors="coerce")
    )
    lon1 = np.radians(
        pd.to_numeric(data["long"], errors="coerce")
    )
    lat2 = np.radians(
        pd.to_numeric(data["merch_lat"], errors="coerce")
    )
    lon2 = np.radians(
        pd.to_numeric(data["merch_long"], errors="coerce")
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    data["distance_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )

    for col in ["amt", "city_pop"]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data[target] = pd.to_numeric(
        data[target], errors="coerce"
    )

    data = data.dropna(subset=[target])
    data = data[data[target].isin([0, 1])]

    for col in categorical_features:
        data[col] = (
            data[col].fillna("Unknown").astype(str)
        )

    data[numeric_features] = data[
        numeric_features
    ].replace([np.inf, -np.inf], np.nan)

    data[target] = data[target].astype(int)

    return data


# ==================================================
# 4. LOAD DATASETS
# ==================================================
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    train = pd.read_csv("fraudTrain.csv")
    test = pd.read_csv("fraudTest.csv")

    train = prepare_data(train)
    test = prepare_data(test)

    return train, test


try:
    train_data, test_data = load_data()

except FileNotFoundError:
    st.error(
        "CSV files not found. Keep fraudTrain.csv and "
        "fraudTest.csv in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load datasets: {error}")
    st.stop()


# ==================================================
# 5. VALIDATE DATA
# ==================================================
if train_data.empty or train_data[target].nunique() != 2:
    st.error(
        "The training dataset must contain both classes: "
        "0 = legitimate and 1 = fraud."
    )
    st.stop()

if test_data.empty or test_data[target].nunique() != 2:
    st.error(
        "The test dataset must contain both classes: "
        "0 = legitimate and 1 = fraud."
    )
    st.stop()


# ==================================================
# 6. SIDEBAR SETTINGS
# ==================================================
st.sidebar.title("⚙️ Model Settings")

training_limit = st.sidebar.selectbox(
    "Training sample size",
    [10000, 25000, 50000, 100000],
    index=2
)

test_limit = st.sidebar.selectbox(
    "Test sample size",
    [10000, 25000, 50000, 100000],
    index=2
)

threshold = st.sidebar.slider(
    "Fraud probability threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

selected_model = st.sidebar.selectbox(
    "Model for new transactions",
    ["Logistic Regression", "Random Forest"]
)

st.sidebar.info(
    "A lower threshold flags more transactions. "
    "A higher threshold flags fewer transactions."
)


# ==================================================
# 7. DATASET OVERVIEW
# ==================================================
st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Training Transactions",
    f"{len(train_data):,}"
)

c2.metric(
    "Test Transactions",
    f"{len(test_data):,}"
)

c3.metric(
    "Fraud Training Records",
    f"{int(train_data[target].sum()):,}"
)

c4.metric(
    "Fraud Percentage",
    f"{train_data[target].mean() * 100:.3f}%"
)

with st.expander("View training data preview"):
    st.dataframe(
        train_data[features + [target]].head(10),
        use_container_width=True
    )

fraud_counts = train_data[target].value_counts().reindex(
    [0, 1], fill_value=0
)

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(
    ["Legitimate", "Fraud"],
    fraud_counts.values
)

ax.set_title("Transaction Class Distribution")
ax.set_ylabel("Number of Transactions")

st.pyplot(fig)
plt.close(fig)


# ==================================================
# 8. SAMPLE THE DATA
# ==================================================
def take_sample(data, limit):
    if len(data) <= limit:
        return data.copy()

    from sklearn.model_selection import train_test_split

    sample, _ = train_test_split(
        data,
        train_size=limit,
        stratify=data[target],
        random_state=42
    )

    return sample.copy()


# ==================================================
# 9. PREPROCESSING PIPELINE
# ==================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ==================================================
# 10. DEFINE MODELS
# ==================================================
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}


# ==================================================
# 11. TRAIN AND COMPARE MODELS
# ==================================================
if st.button("🚀 Train and Compare Models", type="primary"):

    train_sample = take_sample(train_data, training_limit)
    test_sample = take_sample(test_data, test_limit)

    X_train = train_sample[features]
    y_train = train_sample[target]

    X_test = test_sample[features]
    y_test = test_sample[target]

    results = {}

    progress = st.progress(0)
    status = st.empty()

    for index, (name, model) in enumerate(models.items()):

        status.write(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results[name] = {
            "pipeline": pipeline,
            "probabilities": probabilities,
            "predictions": predictions,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            )
        }

        progress.progress((index + 1) / len(models))

    st.session_state["fraud_results"] = results
    st.session_state["fraud_test_labels"] = y_test.reset_index(
        drop=True
    )

    status.success("Both models trained successfully!")


# ==================================================
# 12. MODEL COMPARISON
# ==================================================
if "fraud_results" in st.session_state:

    results = st.session_state["fraud_results"]
    y_eval = st.session_state["fraud_test_labels"]

    st.header("📈 Model Performance Comparison")

    comparison = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-score": result["f1"]
        }
        for name, result in results.items()
    ])

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-score": "{:.2%}"
        }),
        use_container_width=True
    )

    st.bar_chart(
        comparison.set_index("Model")[
            ["Precision", "Recall", "F1-score"]
        ]
    )

    st.caption(
        "Recall measures how many actual fraud cases were detected. "
        "Precision measures how many flagged transactions were fraud."
    )


    # ==================================================
    # 13. STYLED CLASSIFICATION REPORT
    # ==================================================
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #172554, #1D4ED8);
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #3B82F6;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #DBEAFE;
        font-size: 14px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 29px;
        font-weight: bold;
        margin-top: 6px;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0 20px 0;
    }
    .report-table th {
        background-color: #1D4ED8;
        color: white;
        padding: 13px 10px;
        text-align: center;
    }
    .report-table td {
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 12px 10px;
        text-align: center;
        border-bottom: 1px solid #CBD5E1;
    }
    .report-table tr:nth-child(even) td {
        background-color: #DBEAFE;
    }
    .report-table td:first-child {
        font-weight: 600;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    for name, result in results.items():

        st.markdown(f"## 📋 {name} — Classification Report")

        y_pred = result["predictions"]

        report = classification_report(
            y_eval,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            output_dict=True,
            zero_division=0
        )

        metric_values = [
            ("🎯 Accuracy", accuracy_score(y_eval, y_pred)),
            ("🔍 Fraud Precision", report["Fraud"]["precision"]),
            ("🛡️ Fraud Recall", report["Fraud"]["recall"]),
            ("⚖️ Fraud F1-score", report["Fraud"]["f1-score"])
        ]

        columns = st.columns(4)

        for column, (label, value) in zip(columns, metric_values):
            column.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        table_rows = ""

        for class_name in [
            "Legitimate", "Fraud", "macro avg", "weighted avg"
        ]:
            row = report[class_name]

            table_rows += f"""
            <tr>
                <td>{class_name}</td>
                <td>{row['precision']:.2f}</td>
                <td>{row['recall']:.2f}</td>
                <td>{row['f1-score']:.2f}</td>
                <td>{int(row['support']):,}</td>
            </tr>
            """

        st.markdown(
            f"""
            <table class="report-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True
        )

        left, right = st.columns(2)

        with left:
            st.markdown("**Fraud Recall**")
            st.progress(
                float(report["Fraud"]["recall"])
            )
            st.caption(
                "Proportion of actual fraudulent transactions detected."
            )

        with right:
            st.markdown("**Fraud Precision**")
            st.progress(
                float(report["Fraud"]["precision"])
            )
            st.caption(
                "Proportion of flagged transactions that were fraudulent."
            )


    # ==================================================
    # 14. CONFUSION MATRIX
    # ==================================================
    st.header("🔍 Confusion Matrix")

    for name, result in results.items():

        st.subheader(name)

        cm = confusion_matrix(
            y_eval,
            result["predictions"],
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name + " - Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()

        a, b, c, d = st.columns(4)
        a.metric("True Legitimate", f"{tn:,}")
        b.metric("False Alarms", f"{fp:,}")
        c.metric("Missed Fraud", f"{fn:,}")
        d.metric("Detected Fraud", f"{tp:,}")


    # ==================================================
    # 15. THRESHOLD ANALYSIS
    # ==================================================
    st.header("🎚️ Threshold Analysis")

    threshold_rows = []

    for name, result in results.items():

        for cutoff in [0.30, 0.50, 0.70, 0.90]:

            predictions = (
                result["probabilities"] >= cutoff
            ).astype(int)

            threshold_rows.append({
                "Model": name,
                "Threshold": cutoff,
                "Precision": precision_score(
                    y_eval, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_eval, predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y_eval, predictions, zero_division=0
                )
            })

    threshold_df = pd.DataFrame(threshold_rows)

    st.dataframe(
        threshold_df.style.format({
            "Threshold": "{:.2f}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-score": "{:.2%}"
        }),
        use_container_width=True
    )


    # ==================================================
    # 16. NEW TRANSACTION PREDICTION
    # ==================================================
    st.header("💳 Check a New Transaction")

    with st.form("transaction_form"):

        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=250.0
            )

            population = st.number_input(
                "City Population",
                min_value=0,
                value=50000
            )

            hour = st.slider(
                "Transaction Hour",
                min_value=0,
                max_value=23,
                value=2
            )

            day = st.selectbox(
                "Day of Week",
                list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][x]
            )

        with col2:
            distance = st.number_input(
                "Customer-to-Merchant Distance (km)",
                min_value=0.0,
                value=10.0
            )

            categories = sorted(
                train_data["category"].unique().tolist()
            )

            genders = sorted(
                train_data["gender"].unique().tolist()
            )

            category = st.selectbox(
                "Transaction Category",
                categories
            )

            gender = st.selectbox(
                "Gender",
                genders
            )

        submitted = st.form_submit_button(
            "🔎 Predict Transaction Risk"
        )

    if submitted:

        new_transaction = pd.DataFrame([{
            "amt": amount,
            "city_pop": population,
            "hour": hour,
            "day_of_week": day,
            "distance_km": distance,
            "category": category,
            "gender": gender
        }])[features]

        model = results[selected_model]["pipeline"]

        probability = model.predict_proba(
            new_transaction
        )[0][1]

        st.metric(
            "Estimated Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        if probability >= threshold:
            st.warning(
                "⚠️ Flag for further fraud review."
            )
        else:
            st.success(
                "✅ Lower estimated fraud risk at the selected threshold."
            )

        st.caption(
            "This is a model estimate, not proof of fraud."
        )


# ==================================================
# 17. PROJECT LIMITATIONS
# ==================================================
st.divider()
st.header("ℹ️ Project Limitations")

st.write(
    "• The dataset is simulated and may not represent every real-world "
    "fraud pattern."
)

st.write(
    "• Accuracy alone is insufficient for imbalanced fraud data; "
    "precision, recall and F1-score should also be considered."
)

st.write(
    "• Gender is included in this educational prototype. Fairness, "
    "privacy and legal requirements must be assessed before real-world use."
)

st.write(
    "• Threshold analysis on the test set is exploratory. Choose a "
    "final threshold on validation data before final evaluation."
)
