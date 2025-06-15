# Fetal-Health-Prediction
This project uses machine learning to classify fetal health as Normal, Suspect, or Pathological based on cardiotocogram (CTG) data. It helps improve early diagnosis of fetal risks using models like Random Forest and SVM for accurate, data-driven predictions.

This project uses machine learning to classify fetal health status based on cardiotocogram (CTG) features. It provides an efficient way to assist healthcare professionals in identifying potential risks to a fetus during pregnancy.

## 📊 Dataset

- **Name**: Fetal Health Classification Dataset
- **Format**: CSV
- **Attributes**:
  - Baseline Value
  - Accelerations
  - Fetal Movement
  - Uterine Contractions
  - Light Decelerations
  - Severe Decelerations
  - Prolongued Decelerations
  - Abnormal Short-Term Variability
  - Mean Value of Short-Term Variability
  - ... (total 21 features)
  - `fetal_health` (target: 1 = Normal, 2 = Suspect, 3 = Pathological)

## 🎯 Objective

To build a machine learning model that can classify fetal health into:
- **1** – Normal
- **2** – Suspect
- **3** – Pathological

## 📌 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

## ⚙️ ML Workflow

1. Data Preprocessing
   - Handling missing values (if any)
   - Normalization
2. Exploratory Data Analysis (EDA)
3. Model Training
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - Support Vector Machine
   - KNN
4. Model Evaluation
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix
   - ROC Curve

## 📈 Results

The best-performing model achieved:
- **Accuracy**: ~98% (Varies based on model)
- **Precision & Recall**: High for Normal and Pathological categories

## 📦 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fetal-health-classification.git
