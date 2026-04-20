# 🫀 Heart Disease Prediction Using Machine Learning

An end-to-end machine learning project that predicts whether or not a patient has heart disease based on their clinical parameters. Built using Python, Scikit-Learn, and standard data science libraries.

---

## 📌 Problem Statement

> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

The goal is to build a binary classification model that achieves **≥ 95% accuracy** during the proof of concept stage.

---

## 📁 Project Structure

```
heart-disease-classification/
│
├── end-to-end-heart-disease-classification.ipynb   # Main notebook
├── heart-disease.csv                                # Dataset
└── README.md
```

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Kaggle Mirror:** [Heart Disease Dataset (UCI)](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci/data)
- **Rows:** 303 patients
- **Target:** `0` = No Heart Disease, `1` = Heart Disease

### Feature Dictionary

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (1 = Yes, 0 = No) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment (1–3) |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia (3 = Normal, 6 = Fixed defect, 7 = Reversible defect) |
| `target` | **Heart disease present (1) or not (0)** |

---

## 🔬 Approach

The project follows a structured data science workflow:

1. **Problem Definition** — Frame the clinical prediction task
2. **Data** — Load and understand the UCI heart disease dataset
3. **Evaluation** — Set a target metric (≥ 95% accuracy)
4. **Features** — Explore and understand each clinical attribute
5. **Modelling** — Train, compare, and tune multiple classifiers
6. **Experimentation** — Reflect on results and next steps

---

## 🛠️ Libraries Used

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 🤖 Models Trained

Three classifiers were trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Best baseline performance |
| K-Nearest Neighbors (KNN) | Tuned across k = 1–20 |
| Random Forest Classifier | Ensemble approach |

---

## ⚙️ Hyperparameter Tuning

- **KNN** — Tested `n_neighbors` from 1 to 20, selecting the value with the best test score
- **Logistic Regression** — Tuned `C` parameter using `RandomizedSearchCV` and `GridSearchCV` with 5-fold cross-validation
- **Random Forest** — Tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` using `RandomizedSearchCV`

---

## 📈 Evaluation Metrics

The final tuned Logistic Regression model was evaluated using:

- ROC Curve & AUC Score
- Confusion Matrix
- Classification Report
- Cross-validated Accuracy, Precision, Recall, F1-Score

---

## 🔍 Feature Importance

Feature importance was extracted from the best-performing Logistic Regression model using its coefficients, revealing which clinical parameters contribute most to heart disease prediction.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-classification.git
cd heart-disease-classification
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebook

```bash
jupyter notebook end-to-end-heart-disease-classification.ipynb
```

---

## 🧪 Results Summary

The Logistic Regression classifier with tuned hyperparameters achieved the strongest performance among the three models tested. Cross-validated metrics confirmed consistent results across all folds, validating the model's reliability on unseen data.

---

## 🔭 Future Experiments

- Collect more data to improve model generalisation
- Try gradient-boosted models (XGBoost, CatBoost, LightGBM)
- Apply feature engineering or selection techniques
- Export the final model with `joblib` or `pickle` for deployment

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease), originally collected from the Cleveland Clinic Foundation.
- Project inspired by the structured ML workflow taught in various data science courses.
