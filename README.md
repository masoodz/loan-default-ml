# 🏦 Loan Default Prediction (Keras + Gradio)

This project builds a binary classification model to predict loan approval based on applicant information. It uses TensorFlow/Keras for model training and Gradio for an interactive UI.

---

## 📁 Project Structure

```
loan-default-ml/
├── app/
│   ├── train.py            # Model training script
│   ├── evaluate.py         # Model evaluation with test split
│   └── gradio_ui.py        # Gradio interface for prediction
├── data/
│   └── train.csv           # Raw training dataset (from Kaggle)
├── models/
│   ├── loan_model.h5       # Trained Keras model
│   ├── preprocessor.pkl    # Saved preprocessing pipeline
│   └── feature_columns.csv # Ordered feature names
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
# 1. Clone the repo

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # (on Windows: venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📥 Get the Data

This project uses the public dataset from [Kaggle - Loan Prediction Problem](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

### Steps:

1. Go to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).
2. Download the dataset ZIP file.
3. Extract it locally.
4. Copy the file `train.csv` into the `data/` folder of this project:
   ```
   cp /path/to/train.csv data/train.csv
   ```

---

## 🚀 Run the Training Script

```bash
python app/train.py
```

This will:

- Preprocess data using a pipeline
- Train a Keras model
- Save the model and transformer to `models/`

---

## 🧪 Evaluate the Model

```bash
python app/evaluate.py
```

This will:

- Load the saved model and transformer
- Evaluate on a validation split
- Print classification report

---

## 🖥️ Run the Gradio UI

```bash
python app/gradio_ui.py
```

This launches a local interactive web app at [http://127.0.0.1:7860](http://127.0.0.1:7860)  
You can enter sample loan applicant info and get a prediction instantly.

---

## 📊 Features Used

- Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area, Credit_History
- Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term

All preprocessing is handled using a `ColumnTransformer` (OneHot + StandardScaler).

---

## 📚 Requirements

Dependencies are managed via `requirements.txt`. Major libraries:

- `tensorflow`
- `pandas`, `numpy`
- `scikit-learn`
- `gradio`
- `joblib`

---

## 🤝 License

MIT License — free to use and modify.

---