# Pulmonary Fibrosis Severity Prediction

## Overview

This project aims to predict the severity of decline in lung function for patients diagnosed with pulmonary fibrosis using machine learning techniques. Pulmonary fibrosis is a chronic lung disease characterized by scarring of the lungs, leading to difficulty in breathing. The severity prediction is based on CT scan images of the lungs, metadata, and baseline Forced Vital Capacity (FVC) measurements obtained from spirometry.

The goal is to assist clinicians and patients in understanding disease progression better, providing early prognosis insights that could influence treatment decisions and clinical trial designs.

## Problem Statement

Patients diagnosed with pulmonary fibrosis face uncertainty due to the unpredictable nature of the disease progression. Current methods for prognosis are limited, and there is a need for more accurate predictive models using imaging data and clinical parameters.

## Dataset

The dataset used in this project is provided by the Open Source Imaging Consortium (OSIC), a collaborative effort involving academia, industry, and philanthropy. It includes CT scan images of lungs along with metadata such as age, sex, and baseline FVC measurements.

## Approach

### 1. Data Preprocessing

- **CT Image Processing**: Preprocessing CT scan images to enhance features relevant to lung fibrosis.
- **Feature Engineering**: Extracting relevant features from metadata and spirometry data.

### 2. Model Development

- **Convolutional Neural Network (CNN)**: Utilizing CNN architecture to analyze CT scan images and extract meaningful features related to lung condition.
- **Residual Networks (ResNet)**: Implementing ResNet to improve model performance and accuracy in severity prediction.

### 3. Evaluation and Validation

- **Cross-Validation**: Assessing model performance through cross-validation techniques to ensure robustness and generalization.
- **Performance Metrics**: Using metrics such as Confidence value to evaluate the model's accuracy in predicting FVC decline.

 

## Results

The CNN model demonstrated promising results with a confidence level of 250 or higher in predicting FVC decline, providing valuable insights into disease progression for early intervention and patient management.

 
