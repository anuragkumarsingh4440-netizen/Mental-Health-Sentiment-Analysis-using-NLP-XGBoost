#  Mental Health Sentiment Analysis using NLP & XGBoost

##  Project Overview
This project builds an end-to-end NLP system to classify mental health related
text into multiple categories such as Anxiety, Depression, Stress, Suicidal, etc.
An overall sentiment (Positive / Negative) is derived from the predicted category.

The solution follows a complete data science pipeline from text preprocessing
to production-ready deployment.

---

##  Dataset Description
- Source: Public mental health text dataset
- Input: Text statements
- Target: Mental health category (multi-class)
- Classes:
  - Normal
  - Anxiety
  - Depression
  - Stress
  - Suicidal
  - Bipolar
  - Personality disorder

---

##  Tech Stack
- Python
- Pandas, NumPy
- NLTK (text preprocessing)
- Scikit-learn
- XGBoost
- Gradio

---

##  Methodology
1. Text cleaning and normalization
2. Feature extraction using TF-IDF (unigrams & bigrams)
3. Baseline modeling with Logistic Regression
4. Advanced modeling using XGBoost
5. Unified evaluation using weighted F1-score and Cohen’s Kappa
6. Production-ready deployment using Gradio

---

##  Model Comparison Summary

| Model | Test Accuracy | Macro F1 | Production Use |
|-----|--------------|----------|----------------|
| Logistic Regression | ~0.76 | ~0.73 | Baseline |
| XGBoost | ~0.80–0.83 | ~0.77–0.80 |  Final |

XGBoost demonstrated improved performance, particularly on minority classes,
and was selected as the final deployment model.

---

##  Deployment
The trained XGBoost model is deployed using Gradio for real-time inference.

### Run the application:
```bash
python app.py
