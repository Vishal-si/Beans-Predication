import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide"
)

st.title("🫘 Dry Bean Type Classifier")
st.write("Enter the physical measurements of a bean and get the predicted class.")
st.write("Classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA")

st.sidebar.header("How to use")
st.sidebar.write("Enter the 16 physical measurements.")
st.sidebar.write("The boxes start with the median value from the dataset.")
st.sidebar.write("Click Predict Bean Class to see the result.")

# Same columns are log transformed in beansClassification.py before scaling.
feature_cols = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

skew_cols = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter"
]

defaults = {
    "Area": 44652.0,
    "Perimeter": 794.941,
    "MajorAxisLength": 296.883367,
    "MinorAxisLength": 192.431733,
    "AspectRation": 1.551124,
    "Eccentricity": 0.764441,
    "ConvexArea": 45178.0,
    "EquivDiameter": 238.438026,
    "Extent": 0.759859,
    "Solidity": 0.988283,
    "roundness": 0.883157,
    "Compactness": 0.801277,
    "ShapeFactor1": 0.006645,
    "ShapeFactor2": 0.001694,
    "ShapeFactor3": 0.642044,
    "ShapeFactor4": 0.996386
}


def loadFiles():
    currentPath = os.path.dirname(__file__)
    model = joblib.load(f"{currentPath}//model.pkl")
    scaler = joblib.load(f"{currentPath}//scaler.pkl")
    return model, scaler


try:
    model, scaler = loadFiles()
except Exception as e:
    model = None
    scaler = None
    st.error(f"Model files were not found. Run beansClassification.py first. {e}")


if model is not None and scaler is not None:
    st.header("Bean Measurements")

    inputs = {}
    col1, col2 = st.columns(2)

    for index, col_name in enumerate(feature_cols):
        value = float(defaults[col_name])
        if value > 100:
            step = 1.0
        elif value > 1:
            step = 0.01
        else:
            step = 0.0001

        target_col = col1 if index % 2 == 0 else col2
        with target_col:
            inputs[col_name] = st.number_input(
                col_name,
                min_value=0.0,
                value=value,
                step=step,
                format="%.6f",
                key=col_name
            )

    if st.button("Predict Bean Class"):
        input_df = pd.DataFrame([inputs], columns=feature_cols)

        for col_name in skew_cols:
            input_df[col_name] = np.log1p(input_df[col_name])

        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]

        st.header("Prediction")
        st.metric("Predicted Class", prediction)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled)[0]
            proba_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })
            proba_df = proba_df.sort_values("Probability", ascending=False)

            st.subheader("Class Probabilities")
            st.dataframe(proba_df, width="stretch")
            st.bar_chart(proba_df.set_index("Class"))
