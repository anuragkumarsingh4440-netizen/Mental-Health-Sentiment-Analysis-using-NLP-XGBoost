# Mental Health Text Analysis System

This repository contains an end-to-end NLP project focused on analyzing mental health related text and predicting the most relevant mental health category using machine learning. The project is designed as a practical, real-world system rather than a purely academic experiment.

The work demonstrates how to handle noisy text data, class imbalance, model comparison, deployment, and real limitations of NLP models in sensitive domains.

---

## Project Motivation

The objective of this project was to build a system that can take short or long user-written text and classify it into one of the following mental health categories:

Normal  
Anxiety  
Depression  
Stress  
Suicidal  
Bipolar  
Personality disorder  

Along with prediction, the system also provides:
- Prediction confidence
- Class-wise confidence visualization
- A supportive, non-diagnostic suggestion
- A deployable interactive web interface

This project does not aim to provide medical advice. It is built for learning, analysis, and demonstration of NLP and ML engineering skills.

---

## Dataset Overview

The dataset consists of real-world style mental health related statements. It is not based on IMDb reviews or generic sentiment datasets.

Key characteristics:
- Around 52,000 text samples
- Highly imbalanced class distribution
- Mix of very short and very long statements
- Overlapping language across multiple mental health categories

This dataset choice intentionally introduced real challenges such as ambiguity, imbalance, and weak signals in short text.

---

## Project Structure

notebooks/  
01_eda_and_cleaning.ipynb  
02_text_preprocessing.ipynb  
03_class_mapping_and_imbalance.ipynb  
04_vectorization_and_modeling.ipynb  

models/  
tfidf_vectorizer.pkl  
logistic_regression_model.pkl  
final_xgboost_multiclass_model.pkl  
label_encoder.pkl  

app.py  
README.md  

---

## Notebook Walkthrough

### Notebook 01: Exploratory Data Analysis and Cleaning

In this notebook:
- Data types and missing values were checked
- Class distribution was analyzed
- Text length before cleaning was studied
- Initial imbalance was identified

This step helped understand why naive modeling would fail and why careful evaluation metrics are required.

Screenshot placeholder: EDA plots and class distribution

---

### Notebook 02: Text Preprocessing

The following preprocessing steps were applied:
- Lowercasing
- Removing punctuation and numbers
- Stopword removal
- Lemmatization
- Cleaned text length analysis

The goal was to reduce noise while preserving semantic meaning.

Observation:
Text length reduced significantly after cleaning, improving feature consistency for modeling.

Screenshot placeholder: Cleaned text examples and length distribution

---

### Notebook 03: Class Mapping and Imbalance Analysis

Instead of oversampling blindly, the following strategies were used:
- Weighted evaluation metrics
- Class-weight handling in Logistic Regression
- Tree-based robustness in XGBoost
- Cohen’s Kappa for agreement measurement

This notebook focuses on understanding imbalance rather than hiding it.

Screenshot placeholder: Imbalance visualization

---

### Notebook 04: Vectorization and Modeling

TF-IDF was used for vectorization with:
- max_features = 10000
- ngram_range = (1, 2)
- min_df = 5

Two models were trained:
- Logistic Regression as a strong baseline
- XGBoost as the final optimized model

Both models were evaluated on train and test sets using:
- Accuracy
- Precision
- Recall
- F1-score
- Cohen’s Kappa

---

## Model Comparison

Test set performance summary:

Logistic Regression  
Accuracy around 76 percent  
F1-score around 76 percent  
Kappa around 0.69  

XGBoost  
Accuracy around 78 percent  
F1-score around 77 percent  
Kappa around 0.71  

XGBoost consistently outperformed Logistic Regression, especially in handling minority classes.

Screenshot placeholder: Model comparison bar graph
![WhatsApp Image 2025-12-17 at 03 47 40](https://github.com/user-attachments/assets/27f6128a-bfd7-4cd1-8456-5d0f1cfa0a5c)

---

## Real-World Challenge Faced

Short emotional statements such as:
- "I am tired"
- "I hate myself today"

were sometimes predicted as Normal.

Reason:
- TF-IDF struggles with very short text
- The model is conservative by design
- It avoids false positives in sensitive mental health predictions

This behavior is realistic and commonly observed in production systems.

---

## How the Challenge Was Addressed

A hybrid approach was implemented:
- Machine learning model handles general patterns
- Lightweight rule-based logic handles short distress expressions

This approach reflects how real mental health text systems are designed in industry.

The system prioritizes safety and interpretability over aggressive predictions.

---

## Web Application

The project includes a Gradio-based web application that allows:
- Text input
- Real-time prediction
- Confidence percentage display
- Class-wise confidence visualization
- Supportive suggestion output

The application loads trained models from disk and runs inference efficiently.

Screenshot placeholder: Gradio main interface  
Screenshot placeholder: Confidence distribution plot  

---

## Deployment

The application can be launched locally using:

python app.py
<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/59543031-68a3-42b0-9772-4ec276095433" />

---

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/52e163a5-f86a-4124-9050-63bae6f3dc92" />
It can also be deployed on platforms such as Hugging Face Spaces or cloud services with minimal changes.

---

## Limitations and Future Improvements

This system is not 100 percent accurate, and that is clearly acknowledged.

Planned improvements include:
- Using transformer-based embeddings such as BERT or MentalBERT
- Adding risk-level scoring instead of direct class labels
- Improving handling of short, ambiguous text
- Expanding dataset with conversational context

---

## What This Project Demonstrates

- End-to-end NLP pipeline development
- Handling real-world data imperfections
- Model comparison and evaluation
- Ethical awareness in sensitive domains
- Practical deployment skills
- Honest communication of model limitations

---

## Final Note

This project reflects how applied NLP systems are built in practice.
The focus is not only on accuracy, but also on responsibility, interpretability, and continuous improvement.

Feedback and discussions are welcome.
