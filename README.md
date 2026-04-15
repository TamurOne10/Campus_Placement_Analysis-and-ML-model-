# 🎓 Campus Placement Analysis & ML Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-green?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  A comprehensive data analysis and machine learning project to predict student campus placement outcomes based on academic performance, demographics, and work experience.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Workflow](#-project-workflow)
- [Dataset Description](#-dataset-description)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Future Work](#-future-work)
- [Contact](#-contact)

---

## 🔍 Overview

Campus placement is a critical milestone for students and institutions alike. This project dives deep into placement data to:

- **Identify** the key factors that influence whether a student gets placed
- **Visualize** patterns across gender, specialization, academic scores, and work experience
- **Predict** placement likelihood using supervised machine learning models
- **Generate actionable insights** for students and academic institutions to improve placement rates

---

## 🔄 Project Workflow

```
Raw Data
   │
   ▼
Data Loading & Inspection
   │
   ▼
Data Cleaning & Preprocessing
(Missing values, type conversion, outlier detection via IQR)
   │
   ▼
Exploratory Data Analysis (EDA)
(Distributions, correlations, visualizations)
   │
   ▼
Feature Engineering
   │
   ▼
Machine Learning Model Building
(Classification — Placed / Not Placed)
   │
   ▼
Model Evaluation
(Accuracy, Precision, Recall, F1-Score)
   │
   ▼
Insights & Conclusions
```

---

## 📊 Dataset Description

The dataset contains records of students with the following attributes:

| Feature | Description |
|---|---|
| `gender` | Student's gender (Male / Female) |
| `ssc_p` | Secondary school (10th) percentage |
| `ssc_b` | Board of secondary education (Central / Others) |
| `hsc_p` | Higher secondary (12th) percentage |
| `hsc_b` | Board of higher secondary education |
| `hsc_s` | Specialization in higher secondary (Science / Commerce / Arts) |
| `degree_p` | Undergraduate degree percentage |
| `degree_t` | Type of undergraduate degree |
| `workex` | Prior work experience (Yes / No) |
| `etest_p` | Employability test percentage |
| `specialisation` | MBA specialization (Mkt&HR / Mkt&Fin) |
| `mba_p` | MBA percentage |
| `status` | **Target** — Placement status (Placed / Not Placed) |
| `salary` | Salary offered (available only for placed students) |

### Sample Records

| Gender | SSC % | HSC % | Degree % | Work Exp | Specialization | MBA % | Status |
|--------|-------|-------|----------|----------|----------------|-------|--------|
| Male   | 67.00 | 91.00 | 58.00    | No       | Mkt & HR       | 58.80 | Placed |
| Female | 79.33 | 78.33 | 77.48    | Yes      | Mkt & Fin      | 66.28 | Placed |

---

## 💡 Key Findings

- 📈 **Academic performance** (SSC, HSC, Degree, MBA percentages) are strong predictors of placement
- 💼 **Work experience** significantly increases the probability of placement
- 🎓 Students with **Mkt & Finance** specialization showed slightly higher placement rates
- 👥 **Gender** showed notable patterns in placement rates and salary distribution
- 🔎 **Outlier detection (IQR method)** revealed ~8 high-performing students in HSC scores (e.g., 97.7%), who also tend to attract higher salary offers

---

## 🛠️ Technologies Used

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Environment | `Jupyter Notebook` |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed.

### Installation

```bash
# Clone the repository
git clone https://github.com/TamurOne10/Campus_Placement_Analysis-and-ML-model-.git

# Navigate to the project directory
cd Campus_Placement_Analysis-and-ML-model-

# Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook Campus_Placement.ipynb
```

> **Note:** Place the dataset file (`train.csv`) in the same directory as the notebook before running.


> 🌲 Model Selection
Algorithm: Random Forest Classifier
A Random Forest is an ensemble learning method that builds multiple decision trees during training and merges their predictions through majority voting to produce a more accurate and stable result.
pythonfrom sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
Why Random Forest?
ReasonExplanation🎯 Binary ClassificationTarget variable (status) is Placed / Not Placed — a perfect fit for tree-based classifiers🔢 Mixed Data TypesDataset contains both numerical (SSC %, MBA %) and categorical (gender, work experience) features — Random Forest handles both without needing feature scaling📦 Small DatasetWith ~215 rows, overfitting is a real risk. Random Forest mitigates this through bagging (training each tree on a random data subset)🛡️ Robust to OutliersIQR analysis revealed outliers in HSC % (e.g., 97.7). Unlike distance-based models, Random Forest uses splits and is not skewed by outliers📊 Feature ImportanceProvides built-in feature_importances_ to identify which academic and demographic factors most influenced placement outcomes

---

## 📈 Results

The machine learning model was evaluated using standard classification metrics:

### 🔢 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 92.86% |
| Precision (Class 1 - Placed) | 94% |
| Recall (Class 1 - Placed) | 97% |
| F1-Score (Class 1 - Placed) | 96% |

---

### 📊 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Not Placed (0) | 0.88 | 0.78 | 0.82 | 9 |
| Placed (1) | 0.94 | 0.97 | 0.96 | 33 |

---

### 🧠 Key Observations

- The model achieves a **high overall accuracy of 92.86%**, indicating strong predictive performance.
- **Class 1 (Placed students)** is predicted with very high precision and recall, making the model reliable for identifying successful candidates.
- **Class 0 (Not Placed)** has slightly lower recall (78%), suggesting some misclassification in predicting non-placed students.
- The model performs better on the majority class, which is expected due to class imbalance.

---

> 📌 Full confusion matrix and detailed evaluation are available in the notebook.

> Full results, confusion matrices, and visualizations are available inside the notebook.

---

## 🔮 Future Work

- [ ] Experiment with advanced models (Random Forest, XGBoost, SVM)
- [ ] Hyperparameter tuning using GridSearchCV / RandomizedSearchCV
- [ ] Build an interactive web dashboard (Streamlit / Dash)
- [ ] Incorporate extracurricular activities and soft skills data
- [ ] Deploy the model as a REST API for real-time predictions

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 📬 Contact

**Tamoor Abbas**

[![Email](https://img.shields.io/badge/Email-Tamur110%40gmail.com-red?style=flat-square&logo=gmail)](mailto:Tamur110@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tamoor%20Abbas-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/tamoor-abbas)
[![GitHub](https://img.shields.io/badge/GitHub-TamurOne10-black?style=flat-square&logo=github)](https://github.com/TamurOne10)

---

<p align="center">
  ⭐ If you found this project helpful, please consider giving it a star!
</p>
