# CardioScan AI — Heart Disease Risk Predictor

> An AI-powered web application that predicts heart disease risk from 13 clinical patient metrics, built with scikit-learn, Flask, and a premium dark-mode frontend.

---

## Demo

![CardioScan AI](https://img.shields.io/badge/Status-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey) ![Accuracy](https://img.shields.io/badge/Model%20Accuracy-95.61%25-success)

---

## Project Structure

```text
Heart Disease/
├── detection.ipynb                # EDA, model training, evaluation notebook
├── app.py                         # Flask API server
├── heart.csv                      # Cleveland Heart Disease dataset (1025 rows)
├── best_heart_disease_model.pkl   # Saved best model (Random Forest pipeline)
├── templates/
│   └── index.html                 # Premium dark-mode frontend (CardioScan AI)
└── README.md
```

---

## Model

| Property            | Value                                                    |
|---------------------|----------------------------------------------------------|
| Algorithm           | Random Forest Classifier                                 |
| Preprocessing       | `StandardScaler` (numerical) + `OneHotEncoder` (categorical) |
| Feature Selection   | `SelectKBest` (top 10 of 13 features, `f_classif`)      |
| Hyperparameter Tuning | `GridSearchCV` (5-fold CV, optimizing F1)             |
| Test Accuracy       | **95.61%**                                               |
| Precision (Disease) | 92%                                                      |
| Recall (Disease)    | 100%                                                     |

### Input Features (13 clinical attributes)

| # | Feature    | Description                                         | Type        |
|---|------------|-----------------------------------------------------|-------------|
| 1 | `age`      | Patient age in years                                | Numerical   |
| 2 | `sex`      | Biological sex (1 = Male, 0 = Female)               | Categorical |
| 3 | `cp`       | Chest pain type (0–3)                               | Categorical |
| 4 | `trestbps` | Resting blood pressure (mm Hg)                      | Numerical   |
| 5 | `chol`     | Serum cholesterol (mg/dl)                           | Numerical   |
| 6 | `fbs`      | Fasting blood sugar > 120 mg/dl (1 = True)         | Categorical |
| 7 | `restecg`  | Resting ECG results (0–2)                           | Categorical |
| 8 | `thalach`  | Maximum heart rate achieved                         | Numerical   |
| 9 | `exang`    | Exercise-induced angina (1 = Yes)                   | Categorical |
| 10 | `oldpeak` | ST depression induced by exercise                   | Numerical   |
| 11 | `slope`   | Slope of peak exercise ST segment (0–2)             | Categorical |
| 12 | `ca`      | Number of major vessels colored by fluoroscopy (0–3)| Categorical |
| 13 | `thal`    | Thalassemia type (1 = Normal, 2 = Fixed, 3 = Reversible) | Categorical |

### Output

| Value | Meaning              |
|-------|----------------------|
| `1`   | Heart Disease (High Risk) |
| `0`   | No Heart Disease (Low Risk) |

---

## Running the App

### 1. Install Dependencies

```bash
pip install flask pandas scikit-learn joblib
```

### 2. Start the Flask Server

```bash
python app.py
```

### 3. Open the App

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## API

**`POST /predict`**

Accepts a JSON body with all 13 features and returns a prediction.

**Request:**
```json
{
  "age": 67, "sex": 1, "cp": 0, "trestbps": 160,
  "chol": 295, "fbs": 1, "restecg": 2, "thalach": 108,
  "exang": 1, "oldpeak": 3.4, "slope": 0, "ca": 3, "thal": 3
}
```

**Response:**
```json
{ "prediction": 1 }
```

| `prediction` | Meaning    |
|---|---|
| `1` | High Risk — Heart Disease Indicated |
| `0` | Low Risk — Normal Screening |

---

## High-Risk Test Case

Use this patient profile to verify the model is working correctly — it should always return `prediction: 1` (High Risk):

```python
import pandas as pd, joblib

model = joblib.load('best_heart_disease_model.pkl')

patient = pd.DataFrame([{
    'age': 67, 'sex': 1, 'cp': 0, 'trestbps': 160,
    'chol': 295, 'fbs': 1, 'restecg': 2, 'thalach': 108,
    'exang': 1, 'oldpeak': 3.4, 'slope': 0, 'ca': 3, 'thal': 3
}])

pred  = model.predict(patient)[0]
proba = model.predict_proba(patient)[0]

print(f"Prediction  : {'⚠ HIGH RISK' if pred == 1 else '✓ Low Risk'}")
print(f"Probability : No Disease={proba[0]:.2%}  |  Heart Disease={proba[1]:.2%}")
# Expected → Prediction: ⚠ HIGH RISK | Heart Disease=99.50%
```

---

## Dataset Note

The `heart.csv` file (1025 rows) is the Cleveland Heart Disease dataset from Kaggle. It contains 723 duplicate rows (only 302 unique records) and ships with **inverted target labels** (`0` = disease, `1` = no disease — opposite of clinical convention).

The training pipeline corrects this automatically with `y = 1 - df['target']` before fitting. The CSV file itself is never modified.

---

## Disclaimer

> This tool is for **educational and research purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

## License

MIT License — free to use for research and educational purposes.
