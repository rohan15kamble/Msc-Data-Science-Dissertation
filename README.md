# Msc-Data-Science-Dissertation
Explainable and reproducible NLP pipeline for depression detection in Reddit posts, comparing TF-IDF baselines, BiLSTM, and SBERT with robust evaluation and domain-shift testing.

# Depression Detection in Reddit Posts (NLP)

This project builds an explainable and reproducible NLP pipeline to detect depression indicators in Reddit posts. 
It compares traditional ML baselines with deep learning and semantic embedding approaches, and evaluates models with
metrics suitable for imbalanced classification.

## Models Implemented
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- BiLSTM (GloVe embeddings, sequence length 200)
- SBERT embeddings + PCA (100D) + Logistic Regression / Linear SVM

## Key Results (Test Set)
- **BiLSTM:** F1 ≈ 0.9436, ROC-AUC ≈ 0.9857
- **TF-IDF + LR:** F1 ≈ 0.9272, ROC-AUC ≈ 0.9778
- **TF-IDF + Linear SVM:** F1 ≈ 0.9268, ROC-AUC ≈ 0.9778
- **SBERT + PCA + Linear SVM:** F1 ≈ 0.9077, ROC-AUC ≈ 0.9651

## Evaluation
In addition to accuracy and F1, the project uses ROC-AUC, PR-AUC, calibration checks, threshold tuning,
and interpretability (coefficients + SHAP for linear models).

## Reproducibility
The notebook contains the full workflow: data preparation, model training, evaluation, and plots.

## Ethical Note
This work is for research/educational purposes and is **not** a clinical diagnostic tool.
