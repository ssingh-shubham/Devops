# Predict the Introverts from the Extroverts  
**Kaggle Playground Series - Season 5, Episode 7**  

## 📌 Overview  
This project was built for the Kaggle Playground Series competition (Season 5, Episode 7), where the task was to create a machine learning model to **predict whether a person is an Introvert or an Extrovert** based on their social behavior and personality traits.  

The dataset provided by Kaggle contained anonymized features describing individual traits. The challenge involved preprocessing the data, training classification models, tuning hyperparameters, and generating predictions for submission.  

---

## 🎯 Objective  
- To analyze the dataset and identify patterns in personality traits.  
- To build machine learning models that classify individuals as **Introvert (0)** or **Extrovert (1)**.  
- To optimize model performance and generate predictions for submission on Kaggle.  

---

## 🛠️ Tech Stack  

### Languages  
- **Python 3.10+**

### Libraries & Tools  
- **Data Handling & Analysis**: pandas, numpy  
- **Data Visualization**: matplotlib, seaborn  
- **Machine Learning**: scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, etc.)  
- **Model Evaluation**: accuracy_score, confusion_matrix, classification_report  
- **Notebooks**: Jupyter Notebook (`extrointro.ipynb`)  
- **Submission**: CSV file (`submission.csv`) generated for Kaggle  

---

## 📂 Project Structure  
```
├── extrointro.ipynb      # Jupyter Notebook with full analysis & model building
├── submission.csv        # Final predictions submitted to Kaggle
├── README.md             # Project documentation
```

---

## ⚙️ Workflow  
1. **Data Exploration**  
   - Loaded the dataset and performed EDA (checking missing values, distributions, correlations).  
   - Visualized key patterns to understand personality traits.  

2. **Data Preprocessing**  
   - Handled missing values and outliers.  
   - Encoded categorical variables if present.  
   - Normalized/standardized numerical features for certain models.  

3. **Model Building**  
   - Trained multiple models (Logistic Regression, Random Forest, Gradient Boosting, etc.).  
   - Performed hyperparameter tuning using GridSearchCV/RandomizedSearchCV.  
   - Compared results across models.  

4. **Evaluation**  
   - Evaluated models using accuracy, F1-score, and confusion matrix.  
   - Selected the best-performing model for predictions.  

5. **Submission**  
   - Generated final predictions on the test dataset.  
   - Saved results to `submission.csv` for Kaggle.  

---

## 📊 Results  
- Successfully trained multiple models to classify introverts and extroverts.  
- Submitted predictions in Kaggle Playground Competition.  

---

## 🚀 Future Improvements  
- Use advanced ensemble methods (XGBoost, LightGBM, CatBoost) for better accuracy.  
- Perform deeper feature engineering to improve interpretability.  
- Apply cross-validation to enhance model robustness.  
