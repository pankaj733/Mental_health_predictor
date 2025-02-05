# Mental Health Condition Prediction

This project aims to develop a multi-class classification model to predict mental health conditions based on user-reported symptoms. The model is trained on publicly available mental health datasets and includes a command-line interface for user interaction.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Model Development](#model-development)
- [LLM Experimentation (Optional)](#llm-experimentation-optional)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict mental health conditions based on symptoms reported by users. The project includes data cleaning, exploratory data analysis (EDA), model training, and a command-line interface for predictions.

## Dataset

The dataset used for this project includes:
- **Mental Health in Tech Survey**
- **Depression and Anxiety Symptoms Dataset**
- **WHO Mental Health Database**

These datasets were preprocessed to ensure consistency and usability.

## Data Preparation

1. **Data Cleaning**: Missing values were handled by dropping rows or filling them with placeholders.
2. **Normalization**: Text data was normalized by converting to lowercase and removing punctuation.
3. **Feature Engineering**: Symptoms and conditions were encoded using Label Encoding.
4. **Feature Selection**: The most impactful features were selected for model training.

## Model Development

Two models were developed and compared:
1. **Random Forest Classifier**
2. **XGBoost Classifier**

The models were evaluated based on accuracy, precision, recall, and F1-score. The best-performing model was selected for predictions.

### Model Training Code

```python
# Example code snippet for training the Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# Mental Health Condition Prediction

This project aims to develop a multi-class classification model to predict mental health conditions based on user-reported symptoms. The model is trained on publicly available mental health datasets and includes a command-line interface for user interaction.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Model Development](#model-development)
- [LLM Experimentation (Optional)](#llm-experimentation-optional)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict mental health conditions based on symptoms reported by users. The project includes data cleaning, exploratory data analysis (EDA), model training, and a command-line interface for predictions.

## Dataset

The dataset used for this project includes:
- **Mental Health in Tech Survey**
- **Depression and Anxiety Symptoms Dataset**
- **WHO Mental Health Database**

These datasets were preprocessed to ensure consistency and usability.

## Data Preparation

1. **Data Cleaning**: Missing values were handled by dropping rows or filling them with placeholders.
2. **Normalization**: Text data was normalized by converting to lowercase and removing punctuation.
3. **Feature Engineering**: Symptoms and conditions were encoded using Label Encoding.
4. **Feature Selection**: The most impactful features were selected for model training.

## Model Development

Two models were developed and compared:
1. **Random Forest Classifier**
2. **XGBoost Classifier**

The models were evaluated based on accuracy, precision, recall, and F1-score. The best-performing model was selected for predictions.

### Model Training Code

```python
# Example code snippet for training the Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
