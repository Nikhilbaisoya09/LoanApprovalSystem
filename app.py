import streamlit as st
import pandas as pd
import joblib

from preprocess import preprocess_pipeline
from model import split_data, get_model, train_model, evaluate_model

st.set_page_config(layout="wide")
st.title("🏦 Loan Approval ML Pipeline")

steps = ["1. Data", "2. Preprocess", "3. Split", "4. Model", "5. Train", "6. Evaluate", "7. Predict"]
step = st.radio("Steps", steps, horizontal=True)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None


# -------- STEP 1 --------
if step == steps[0]:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.write(df.head())

        target = st.selectbox("Target Column", df.columns)
        st.session_state.target = target


# -------- STEP 2 --------
elif step == steps[1]:
    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.df

        remove_out = st.checkbox("Remove Outliers")

        X, y, preprocessor, selector = preprocess_pipeline(
            df,
            st.session_state.target,
            remove_out
        )

        st.session_state.X = X
        st.session_state.y = y

        joblib.dump(preprocessor, "preprocessor.pkl")
        joblib.dump(selector, "selector.pkl")

        st.success("✅ Preprocessing Done")


# -------- STEP 3 --------
elif step == steps[2]:
    if "X" not in st.session_state:
        st.warning("Run preprocessing first")
    else:
        X_train, X_test, y_train, y_test = split_data(
            st.session_state.X,
            st.session_state.y
        )

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.success("✅ Data Split Done")


# -------- STEP 4 --------
elif step == steps[3]:
    model_name = st.selectbox(
        "Select Model",
        ["RandomForest", "SVM", "LogisticRegression"]
    )
    st.session_state.model_name = model_name


# -------- STEP 5 --------
elif step == steps[4]:
    if "model_name" not in st.session_state:
        st.warning("Select model first")
    else:
        model = get_model(st.session_state.model_name)

        model, scores = train_model(
            model,
            st.session_state.X_train,
            st.session_state.y_train
        )

        st.session_state.model = model
        joblib.dump(model, "model.pkl")

        st.write("📊 Cross-validation Scores:", scores)
        st.write("📈 Average Score:", scores.mean())

        st.success("✅ Model Trained")


# -------- STEP 6 --------
elif step == steps[5]:
    if "model" not in st.session_state:
        st.warning("Train model first")
    else:
        score = evaluate_model(
            st.session_state.model,
            st.session_state.X_test,
            st.session_state.y_test
        )

        st.write("🎯 Accuracy:", score)


# -------- STEP 7 --------
elif step == steps[6]:
    st.subheader("🔮 Predict Loan Approval")

    if "model" not in st.session_state:
        st.warning("Train model first")
    else:
        df = st.session_state.df
        target = st.session_state.target

        input_data = {}

        for col in df.columns:
            if col == target:
                continue

            # 🔥 Robust numeric detection
            col_data = pd.to_numeric(df[col], errors='coerce')

            if col_data.notna().sum() == len(df[col]):
                # Numeric column
                input_data[col] = st.number_input(
                    col,
                    float(col_data.min()),
                    float(col_data.max()),
                    float(col_data.mean())
                )
            else:
                # Categorical column
                input_data[col] = st.selectbox(
                    col,
                    df[col].astype(str).dropna().unique()
                )

        if st.button("Predict"):
            new_df = pd.DataFrame([input_data])

            preprocessor = joblib.load("preprocessor.pkl")
            selector = joblib.load("selector.pkl")
            model = joblib.load("model.pkl")

            X_processed = preprocessor.transform(new_df)

            from scipy import sparse
            if sparse.issparse(X_processed):
                X_processed = X_processed.toarray()

            X_selected = selector.transform(X_processed)

            pred = model.predict(X_selected)

            # 🔥 Flexible output handling
            if str(pred[0]).lower() in ["1", "yes", "approved"]:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")