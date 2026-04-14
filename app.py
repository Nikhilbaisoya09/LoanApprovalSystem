import streamlit as st
import pandas as pd

from preprocess import preprocess_pipeline
from model import get_model, split_data, train_model, evaluate_model

st.set_page_config(layout="wide")
st.title("🏦 Loan Approval ML Pipeline")

steps = ["1. Data", "2. Preprocess", "3. Split", "4. Model", "5. Train", "6. Evaluate"]
step = st.radio("Steps", steps, horizontal=True)

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
    df = st.session_state.df

    if df is None:
        st.warning("Please upload data first")
    else:
        remove_out = st.checkbox("Remove Outliers")

        X, y = preprocess_pipeline(
            df,
            target=st.session_state.target,
            remove_out=remove_out
        )

        st.session_state.X = X
        st.session_state.y = y

        st.success("Preprocessing Done")


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

        st.success("Data Split Done")


# -------- STEP 4 --------
elif step == steps[3]:
    model_name = st.selectbox(
        "Select Model",
        ["RandomForest", "SVM", "LogisticRegression"]
    )

    st.session_state.model_name = model_name


# -------- STEP 5 --------
elif step == steps[4]:
    if "X_train" not in st.session_state:
        st.warning("Split data first")
    else:
        model = get_model(st.session_state.model_name)

        model, scores = train_model(
            model,
            st.session_state.X_train,
            st.session_state.y_train
        )

        st.session_state.model = model

        st.write("Cross Validation Scores:", scores)
        st.write("Average Score:", scores.mean())


# -------- STEP 6 --------
elif step == steps[5]:
    if "model" not in st.session_state:
        st.warning("Train model first")
    else:
        score = evaluate_model(
            st.session_state.model,
            st.session_state.X_test,
            st.session_state.y_test,
            "Classification"
        )

        st.write("Final Accuracy:", score)