# Fashion Recommendation Pipeline

This repository contains the complete pipeline for predicting whether a customer will recommend a fashion product based on their review and metadata. The pipeline utilizes structured data and natural language processing to train and evaluate a classification model.

## Project Structure

```
dsnd-pipelines-project/
├── starter/
│   ├── data/
│   │   └── reviews.csv
│   ├── fashion_recommendation_pipeline.pkl         # Final saved pipeline
│   └── starter.ipynb                               # Notebook with code
├── README.md
├── requirements.txt
```

## Introduction

This project builds a machine learning pipeline to classify customer recommendations using logistic regression. The pipeline includes:

- Feature selection
- Stratified train-test split
- Preprocessing for numerical, categorical, and text data
- Model training and evaluation
- Hyperparameter tuning with GridSearchCV
- Model export

## Getting Started

### Installation

```bash
git clone https://github.com/AbdullahAlBurayh/dsnd-pipelines-project.git
cd dsnd-pipelines-project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook starter/starter.ipynb
```

### Inference with Exported Model

```python
import joblib

model = joblib.load("starter/fashion_recommendation_pipeline.pkl")
predictions = model.predict(X_new)
```

## Pipeline Overview

### Preprocessing

- **Numerical**: `Age`, `Positive Feedback Count` scaled with `StandardScaler`
- **Categorical**: Encoded with `OneHotEncoder`
- **Text**: `Review Text` processed with `spaCy` and `TfidfVectorizer`

### Model

- `LogisticRegression` with `class_weight='balanced'`
- Evaluated on training and test data using accuracy, recall, precision, and F1-score
- Best hyperparameters tuned via `GridSearchCV`

### Final Scores

- **Test Accuracy**: 85%
- **F1-Score (Recommended)**: 0.90
- **F1-Score (Not Recommended)**: 0.66

## Built With

- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- spaCy

## License

This project is licensed under the terms of the LICENSE file.