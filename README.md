# Pulmonary Fibrosis Severity Prediction

## 📋 Overview

This project aims to predict the severity of decline in lung function for patients diagnosed with pulmonary fibrosis using machine learning techniques. Pulmonary fibrosis is a chronic lung disease characterized by scarring of the lungs, leading to difficulty in breathing. The severity prediction is based on **CT scan images** of the lungs, **metadata**, and baseline **Forced Vital Capacity (FVC)** measurements obtained from spirometry.

The goal is to assist clinicians and patients in understanding disease progression better, providing early prognosis insights that could influence treatment decisions and clinical trial designs.

## ❓ Problem Statement

Patients diagnosed with pulmonary fibrosis face uncertainty due to the unpredictable nature of the disease progression. Current methods for prognosis are limited, and there is a need for more accurate predictive models using **imaging data** and **clinical parameters**.

## 📊 Dataset

The dataset used in this project is provided by the **Open Source Imaging Consortium (OSIC)**, a collaborative effort involving academia, industry, and philanthropy. It includes **CT scan images** of lungs along with metadata such as age, sex, and baseline FVC measurements.

## ⚙️ Approach

### 1. Data Preprocessing

- **CT Image Processing**: Preprocessing CT scan images to enhance features relevant to lung fibrosis.
- **Feature Engineering**: Extracting relevant features from metadata and spirometry data.

### 2. Model Development

- **Fully Connected Neural Network (FCNN)**: Utilizing a fully connected architecture to predict the severity of pulmonary fibrosis based on **FVC values**. The model processes the input FVC values through a series of linear transformations to produce the final output (severity score or predicted FVC decline).
  
- **FVCTransformer**: Implementing a custom neural network model, the `FVCTransformer`, that takes in FVC input values and applies a two-layer transformation to predict the severity of lung function decline in patients with pulmonary fibrosis. This model leverages dense layers to extract meaningful patterns from the FVC data.


### 3. Evaluation and Validation

- **Cross-Validation**: Assessing model performance through cross-validation techniques to ensure robustness and generalization.
- **Performance Metrics**: Using metrics such as Confidence value to evaluate the model's accuracy in predicting FVC decline.

## 🏆 Results

The **FVCTransformer** model demonstrated promising results in predicting the severity of pulmonary fibrosis, with a **confidence level** of **250 or higher** in forecasting FVC decline. This model provided valuable insights into disease progression, aiding clinicians in early intervention and improving patient management strategies.


## 📂 GitHub Repository

For more details, check out the GitHub repository: [Pulmonary Fibrosis Severity Prediction](https://github.com/UjjawalSah/Pulmonary-Fibrosis-Severity-Prediction)
