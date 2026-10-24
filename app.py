"""Streamlit inference interface for the artifact exported by Seismic8.ipynb."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACT_PATH = Path(__file__).resolve().parent / 'results' / 'best_model_artifact.pkl'
ORDINAL_ORDER = {
    'seismic': ['a', 'b', 'c', 'd'],
    'seismoacoustic': ['a', 'b', 'c', 'd'],
    'hazard': ['a', 'b', 'c', 'd'],
    'shift': ['N', 'W'],
}

st.set_page_config(page_title='Seismic Bump Prediction', page_icon='⛏️', layout='centered')
st.markdown('''<style>
.block-container {max-width: 900px; padding-top: 2rem;}
.hero {background:linear-gradient(110deg,#10283f,#21567a);color:white;padding:28px;border-radius:16px;margin-bottom:20px;}
.hero h1 {color:white;margin:0 0 8px 0;font-size:2rem;}
.hero p {color:#dbeafe;margin:0;}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Seismic Bump Prediction</h1><p>Research prototype based on the fitted pipeline exported by Seismic8.ipynb.</p></div>', unsafe_allow_html=True)
st.warning('Research demonstration only. Not validated for operational mine-safety decisions.')

@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_PATH)  # Only load a file you generated and trust.

if not ARTIFACT_PATH.is_file():
    st.error('Model artifact not found at results/best_model_artifact.pkl. Run the final save cell in Seismic8.ipynb and place the generated file at this path.')
    st.stop()

try:
    artifact = load_artifact()
    pipeline = artifact['pipeline']
    threshold = float(artifact['threshold'])
    model_name = artifact['model_name']
    feature_columns = list(artifact['feature_columns'])
except (KeyError, TypeError, ValueError, OSError, ImportError, AttributeError) as exc:
    st.error('Could not load the notebook artifact. Verify the file and compatible Python/package versions.')
    st.stop()

st.subheader('Model Information')
a, b, c = st.columns(3)
a.metric('Selected model', str(model_name))
b.metric('Input features', len(feature_columns))
c.metric('Decision threshold', f'{threshold:.2f}')

# The notebook's raw input schema is categorical for the four named columns.
# Other feature columns are numeric. The fitted pipeline handles imputation,
# ordinal encoding and scaling, so do NOT preprocess these inputs again.
with st.form('prediction_form'):
    st.subheader('Enter raw feature values')
    st.caption('Enter measurements using the same units and definitions as the UCI dataset. Numeric fields are not artificially constrained to zero or above.')
    values = {}
    left, right = st.columns(2)
    for i, feature in enumerate(feature_columns):
        with (left if i % 2 == 0 else right):
            if feature in ORDINAL_ORDER:
                values[feature] = st.selectbox(feature, ORDINAL_ORDER[feature])
            else:
                values[feature] = st.number_input(feature, value=0.0, format='%.6f')
    submitted = st.form_submit_button('Predict', type='primary', use_container_width=True)

if submitted:
    raw_input = pd.DataFrame([values], columns=feature_columns)
    try:
        probabilities = pipeline.predict_proba(raw_input)
        classes = list(pipeline.classes_)
        if 1 not in classes:
            st.error('The fitted classifier does not expose class label 1. Check the saved artifact.')
            st.stop()
        score = float(probabilities[0, classes.index(1)])
        if not np.isfinite(score):
            st.error('Model returned a non-finite score.')
            st.stop()
        prediction = int(score >= threshold)
    except Exception:
        st.error('Prediction failed. Check that the artifact, feature types and installed package versions match the notebook.')
    else:
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error('Predicted class: 1 — Hazardous')
        else:
            st.success('Predicted class: 0 — Non-hazardous')
        st.metric('Model score for class 1', f'{score:.4f}')
        st.caption('Classification uses the threshold stored in the notebook artifact. The score is not necessarily a calibrated real-world hazard probability.')
        with st.expander('View submitted input'):
            st.dataframe(raw_input, use_container_width=True, hide_index=True)

with st.expander('Methodology implemented in the notebook'):
    st.write('The notebook compares multiple classifiers using repeated stratified cross-validation on the training split, selects the CV-best model, optionally tunes it, and evaluates the selected fitted pipeline on a held-out test set. The artifact stores the fitted preprocessing-and-classifier pipeline, selected threshold, feature columns, model name, and tuning information.')
with st.expander('Artifact tuning information'):
    st.json(artifact.get('tuning', {}))
with st.expander('Limitations'):
    st.write('This app only performs inference using the saved artifact. It does not rerun cross-validation, hyperparameter tuning, leakage experiments, or test-set evaluation. It is a research prototype and must not be used to make real mine-safety decisions.')
