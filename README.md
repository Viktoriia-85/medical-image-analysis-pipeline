# Multi-Class and Binary Classification Pipeline for Lung Disease Diagnosis Based on CT Texture Features

This repository contains the core machine learning pipeline, architectural components, and data engineering logic developed as part of my Master's thesis. The project focuses on the automated differential diagnosis of lung diseases and the identification of specific pulmonary complications using structured CT image descriptors.

## Project Overview

The system processes a real-world clinical dataset to execute two distinct, non-trivial machine learning tasks:
1. **Multi-Class Classification:** Differential diagnosis across 4 clinical categories: **COVID-19, Healthy, Pneumonia, and Long-COVID**.
2. **Binary Classification:** Targeted detection of pulmonary complications — **VLS (Vanishing Lung Syndrome / Idiopathic Bullous Emphysema) vs. non-VLS** — specifically within the acute COVID-19 patient sub-cohort.

The complete pipeline manages a total dataset of **97 patients (26,623 images)** characterized by severe class imbalance and multicenter data variations.

---

## Repository Structure

The codebase is organized into modular, production-ready files demonstrating a complete end-to-end Data Science workflow:

*   **`mask_features.py`**  
    *The Data Engineering and Feature Extraction Layer.* Implements intensity windowing (Hounsfield Unit clamping), 64-level intensity quantization, luma-based mask binarization, and SimpleITK geometry resampling. It builds a structured 36-dimensional feature space based on clinical image texture characteristics: **GLCM, GLDS, GLRLM, and LBP** extracted strictly within automated lung segmentation masks.
*   **`multiclass_classification.py`**  
    *The Multi-Class Modeling Layer.* Implements automated hyperparameter optimization via `GridSearchCV` on slice-level feature data for classical machine learning families: Multinomial Logistic Regression, Random Forest, SVM, KNN, XGBoost, and LightGBM.
*   **`patient_aggregation.py`**  
    *The Clinical Subject Aggregation Layer.* Demonstrates the logic to mathematically collapse thousands of raw slice-level descriptors into stable patient-level vectors using arithmetic mean aggregation (`groupby().mean()`) to prepare data for robust patient-stratified validation.
*   **`binary_classification.py`**  
    *The Advanced Modeling & Anti-Overfitting Layer.* Focuses on the binary classification task within the acute COVID subgroup. Implements a customized training loop featuring an internal `GroupShuffleSplit` for **Early Stopping** in gradient boosting frameworks (XGBoost/LightGBM), followed by a full deterministic refit to control overfitting on specific patient sub-cohorts.

---

## Methodology & Validation Strategy

*   **Robust Preprocessing:** Handles the complete pipeline from raw data cleaning, normalization, and standardization to structured feature extraction.
*   **Leakage Prevention:** To guarantee stability and prevent data leakage, cross-validation boundaries (`StratifiedGroupKFold` and `StratifiedKFold`) are strictly partitioned by **unique patient IDs**, ensuring slices from the same individual never overlap between training and validation sub-cohorts.
*   **Imbalance Handling & Evaluation:** The models are evaluated and interpreted within a rigorous clinical context using a **five-fold cross-validation** setup. Optimization explicitly targets **Accuracy and F1-score (Macro/Weighted)** to ensure stability under significant class imbalance (such as capturing minority classes like Pneumonia, N=5 patients).
*   **Prospective Bias Correction:** Incorporates logic to handle scanner-specific differences (hardware SITE-effects, such as specific hardcoded Hitachi subsets) across multicenter data sources.

---

## Experimental Results Summary

The models were rigorously evaluated on the independent test split, which accounts for 24.74% of the total dataset. To ensure absolute clinical validity, strict patient-level separation was maintained during data splitting to eliminate any risk of data leakage. 

### 1. Multi-Class Classification Task (4 Clinical Categories)
*Evaluated on slice-level texture features.*

| Classification Method | Accuracy | Precision | Recall | Specificity | F1-Score | F1-Macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression** | 0.6349 | 0.5242 | 0.4990 | 0.8642 | 0.6440 | **0.5065** |
| Random Forest | 0.4844 | 0.3300 | 0.3357 | 0.7911 | 0.4756 | 0.3288 |
| SVM | 0.5653 | 0.3916 | 0.4043 | 0.8182 | 0.5443 | 0.3889 |
| KNN | 0.5054 | 0.3068 | 0.3122 | 0.7914 | 0.4760 | 0.3013 |
| XGBoost | 0.5960 | 0.3467 | 0.4088 | 0.8101 | 0.5319 | 0.3368 |
| LightGBM | 0.5853 | 0.3383 | 0.4211 | 0.8020 | 0.5183 | 0.3284 |

### 2. Binary Classification Task (VLS vs. non-VLS inside COVID-19 Cohort)
*Evaluated on slice-level texture features.*

| Classification Method | Accuracy | Precision | Recall | Specificity | F1-Score | F1-Macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7427 | 0.7078 | 0.7001 | 0.7981 | 0.7451 | 0.7034 |
| Random Forest | 0.7841 | 0.6722 | 0.7975 | 0.9619 | 0.7564 | 0.6908 |
| **SVM** | **0.8477** | **0.7758** | **0.8569** | **0.9619** | **0.8331** | **0.8007** |
| KNN | 0.8010 | 0.7635 | 0.7664 | 0.8607 | 0.8005 | 0.7649 |
| XGBoost | 0.7384 | 0.6834 | 0.6908 | 0.8259 | 0.7358 | 0.6867 |
| LightGBM | 0.7358 | 0.6806 | 0.6876 | 0.8235 | 0.7333 | 0.6837 |

---

## Academic Publications

The methodologies and findings implemented in this pipeline have been published and presented across the following peer-reviewed scientific tracks:

1. **"Diagnosis of COVID-19-Associated Cardiopulmonary Pathology From CT Data Using Artificial Intelligence: A Review of Methods and Future Research Directions"**  
   *(Category A professional journal, indexed by Scopus)*
2. **"Efficiency of Machine Learning Algorithms for the Classification of Post-COVID Lung Structure Changes Based on CT Data"**  
   *(Category B professional journal)*
3. **"Comparative Analysis of Machine Learning Algorithms for the Classification of Post-COVID Lung Structure Changes Based on CT Images"**  
   *(proceedings of an international scientific and technical conference)*

---
*Note: Due to strict medical data privacy regulations and non-disclosure agreements regarding patient clinical records, the raw DICOM/CT datasets and final model weights are omitted. This repository serves as a curated showcase of pipeline architecture, algorithmic logic, and coding style.*
