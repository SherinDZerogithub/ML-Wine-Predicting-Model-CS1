# Wine Quality Predicting Model

End‑to‑end machine learning project to explore, model, and predict red wine quality from physicochemical features. The repository includes data, exploratory notebooks, a saved model artifact, and minimal Python scaffolding to extend into scripts or apps.

## Highlights

- Dataset: UCI Red Wine Quality (1,599 rows, 11 features + quality label)
- Multiple modeling notebooks (Logistic Regression, SVM, KNN, XGBoost, Classification Tree)
- Saved model artifact (.joblib) for quick inference
- Ready-to-use `requirements.txt` for a reproducible Python environment

---

## Project structure

```
wine_quality_predicting_model/
├── Makefile                         # (stub) room for handy commands (train, test, lint)
├── pyproject.toml                   # project metadata (Python >=3.9)
├── README                           # this file
├── requirements.txt                 # Python dependencies
├── data/
│   ├── raw/
│   │   └── winequality-red.csv      # UCI Red Wine Quality dataset
│   └── processed/                   # put cleaned/feature-engineered data here
├── models/
│   └── FC211008-classification-tree-models/
│       └── wine_quality_best_model_random_forest.joblib  # example trained model
├── notebooks/
│   ├── experiment_01.ipynb
│   ├── FC211008_SiyathEpa/classification-tree.ipynb
│   ├── FC211017_ChamodiThennakon/xgboost.ipynb
│   ├── FC211032_VidusahanPerera/logistic-regression.ipynb
│   ├── FC211038_KavindiHewawasam/svm.ipynb
│   └── FC211044_SeriniPuwakgolla/knn.ipynb
├── src/                             # extend with scripts or a package as needed
│   ├── __init__.ipynb               # placeholder (consider converting to .py)
│   ├── test.py                      # placeholder for quick script tests
│   ├── train.py                     # placeholder to move notebook logic into scripts
│   └── validation.py                # placeholder for validation utilities
└── tests/
	├── __init__.py
	└── test.py                      # placeholder for automated tests
```

---

## Dataset

- Source: UCI Machine Learning Repository – Red Wine Quality dataset by Cortez et al. (2009)
- Task: Predict sensory quality score from 11 physicochemical inputs
- File: `data/raw/winequality-red.csv`

Features (inputs):
- fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
  total sulfur dioxide, density, pH, sulphates, alcohol

Target:
- quality (integer 3–8; sometimes binarized in notebooks into good/poor)

---

## Environment setup

Choose one of the following.

### Option A: Conda (recommended)

```
conda create -n wine-ml python=3.10 -y
conda activate wine-ml
pip install -r requirements.txt
```

### Option B: venv + pip

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Working with notebooks

Start JupyterLab and open any notebook under `notebooks/`.

```
jupyter lab
```

Suggested flow in the Logistic Regression notebook:
1. Explore data (class balance, distributions, correlations)
2. Clean data (duplicates, missing values)
3. Feature engineering (e.g., binary target `quality_label`)
4. Train/test split and scaling
5. Train model and evaluate metrics (accuracy, precision/recall, ROC, etc.)

Tips for collaboration:
- Clear outputs before committing to reduce noisy diffs.
- Consider using [nbdime](https://github.com/jupyter/nbdime) for notebook diffs/merges.
- Consider [Jupytext](https://github.com/mwouts/jupytext) to track notebooks as `.py`/`.md` alongside `.ipynb`.

---

## Using the saved model for inference

An example Random Forest model is provided at:
`models/FC211008-classification-tree-models/wine_quality_best_model_random_forest.joblib`

Minimal example to load the model and predict from a single sample:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load(
	"models/FC211008-classification-tree-models/wine_quality_best_model_random_forest.joblib"
)

# Use the same feature order as the dataset (excluding the target column)
cols = [
	"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
	"chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
	"pH", "sulphates", "alcohol"
]

sample = {
	"fixed acidity": 7.4,
	"volatile acidity": 0.70,
	"citric acid": 0.00,
	"residual sugar": 1.9,
	"chlorides": 0.076,
	"free sulfur dioxide": 11.0,
	"total sulfur dioxide": 34.0,
	"density": 0.9978,
	"pH": 3.51,
	"sulphates": 0.56,
	"alcohol": 9.4,
}

X = pd.DataFrame([sample], columns=cols)
pred = model.predict(X)
print("Predicted wine quality:", pred[0])
```

Notes:
- If your model requires scaling or specific preprocessing, persist and load those transformers (e.g., `StandardScaler`) alongside the model and apply them before prediction.

---

## Extending notebooks into scripts

The `src/` folder includes placeholders (`train.py`, `utils.py`, `validation.py`).
Suggested next steps:
- Move data prep and modeling code from notebooks into `src/` modules
- Add a CLI to `train.py` to support reproducible training
- Save artifacts (model, scaler) under `models/` with versioned names

Example CLI sketch for `train.py` (pseudo‑code):

```bash
python -m src.train \
  --data data/raw/winequality-red.csv \
  --target quality \
  --model rf \
  --out models/rf_YYYYMMDD.joblib
```

---

## Testing

The `tests/` directory is present and ready for unit tests (currently placeholders). Recommended stack: `pytest`, `scikit-learn` metrics, and small synthetic fixtures.

---

## Streamlit (optional quick app)

`streamlit` is included in `requirements.txt`. You can spin up a minimal app to demo predictions:

```python
# save as app.py at repo root or under wine_quality_predicting_model/
import joblib
import pandas as pd
import streamlit as st

st.title("Wine Quality Prediction")

cols = [
	"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
	"chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
	"pH", "sulphates", "alcohol"
]

inputs = {c: st.number_input(c, value=0.0) for c in cols}
model = joblib.load(
	"models/FC211008-classification-tree-models/wine_quality_best_model_random_forest.joblib"
)
if st.button("Predict"):
	X = pd.DataFrame([inputs], columns=cols)
	y = model.predict(X)[0]
	st.success(f"Predicted wine quality: {int(y)}")
```

Run:

```
streamlit run app.py
```

---

## Contributing

1. Create a feature branch
2. Work primarily in notebooks or scripts under `src/`
3. Clear notebook outputs before committing
4. Open a PR; ensure CI checks (tests/lint) pass when added

---

## Acknowledgements

- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
  Modeling wine preferences by data mining from physicochemical properties.
  Decision Support Systems, Elsevier, 47(4):547–553, 2009.
- UCI Machine Learning Repository: Wine Quality Data Set

---

## License

Specify your project license here (e.g., MIT). If unsure, add an SPDX license file and update this section.

