# Seismic Bump Hazard Prediction

Research implementation and Streamlit inference interface for `Seismic8.ipynb`.

## Notebook implementation

The notebook loads `seismic-bumps.csv` (or the `DATA_PATH` environment variable), removes exact duplicate rows and duplicate-named columns, and drops the constant columns `nbumps6`, `nbumps7`, and `nbumps89` when present. The target column is `class`.

By default, it performs an 80/20 stratified train/test split (`RANDOM_STATE=42`). The preprocessing pipeline imputes categorical variables with the most frequent value, ordinal-encodes categorical features, imputes numerical variables with their median, and standardizes numerical variables. The notebook evaluates candidate classifiers using repeated stratified cross-validation, selects the best by cross-validated average precision (PR-AUC), optionally performs randomized hyperparameter tuning on the selected model, chooses a classification threshold using out-of-fold predictions, and evaluates the final fitted pipeline on the held-out test split.

Candidate methods in the notebook include Logistic Regression, Random Forest, Random Forest with SMOTETomek, LightGBM, Self-Paced Ensemble, BalanceCascade, SMOTEBagging, and Stacking. Additional analyses in the notebook include E1–E6. The selected model and its threshold depend on the executed configuration; the web app reads them from the exported artifact rather than hard-coding them.

## Saved artifact

The final notebook cell exports `results/best_model_artifact.pkl` with these keys:

- `pipeline`: fitted preprocessing + classifier pipeline
- `threshold`: chosen classification threshold
- `model_name`: selected model name
- `feature_columns`: original raw input column order
- `tuning`: hyperparameter tuning metadata
- `versions`: selected library versions recorded by the notebook

The web app loads this file directly and **does not retrain** the model. It uses the same raw feature names, pipeline, and saved threshold. Do not load untrusted `.pkl`/joblib files.

## Run locally

1. Place the UCI data file at `seismic-bumps.csv`, or set `DATA_PATH` to its location.
2. Run `Seismic8.ipynb` in a compatible Python environment to generate the artifact.
3. Confirm that `results/best_model_artifact.pkl` exists alongside `app.py`.
4. Install application dependencies (see `requirements.txt`, if supplied).
5. Run:

   ```bash
   streamlit run app.py
   ```

## Deploy with Streamlit Community Cloud

Push the project to a GitHub repository, including `app.py`, its runtime dependencies, and the trusted artifact at `results/best_model_artifact.pkl` (if permitted and within hosting size limits). In Streamlit Community Cloud, select the repository, `main` branch, and `app.py` entrypoint. Use a compatible Python and package environment for the serialized artifact.

**Before publishing:** review the notebook, dataset, outputs, and artifact for sensitive or restricted material. The `.gitignore` intentionally does not hide `results/` because that directory contains the inference artifact.

## Limitations

The Streamlit app demonstrates predictions from the trained model only. It is not an operational mine-safety warning system. Reported research metrics should be taken from the actual notebook outputs, not recomputed from arbitrary user inputs in the app. The default notebook split is stratified; its optional `temporal` split assumes original row order reflects chronology and requires independent validation of that assumption.
