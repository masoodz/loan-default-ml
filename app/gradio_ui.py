import gradio as gr
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model('models/loan_model.h5')
preprocessor = joblib.load('models/preprocessor.pkl')

def predict_loan_status(Gender, Married, Dependents, Education, Self_Employed,
                        ApplicantIncome, CoapplicantIncome, LoanAmount,
                        Loan_Amount_Term, Credit_History, Property_Area):

    input_dict = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }

    df = pd.DataFrame([input_dict])

    X_transformed = preprocessor.transform(df)

    prob = model.predict(X_transformed).flatten()[0]
    prediction = "Approved" if prob >= 0.5 else "Rejected"
    return f"{prediction} (Confidence: {prob:.2f})"

demo = gr.Interface(
    fn=predict_loan_status,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender", value="Male"),
        gr.Radio(["Yes", "No"], label="Married", value="Yes"),
        gr.Radio(["0", "1", "2", "3+"], label="Dependents", value="0"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education", value="Graduate"),
        gr.Radio(["Yes", "No"], label="Self Employed", value="No"),
        gr.Number(label="Applicant Income", value=5000),
        gr.Number(label="Co-applicant Income", value=1500),
        gr.Number(label="Loan Amount (in thousands)", value=150),
        gr.Number(label="Loan Term (in days)", value=360),
        gr.Radio(
            [1.0, 0.0],
            label="Credit History",
            value=1.0,
            info="1.0 = Good credit history (no defaults), 0.0 = No or poor credit history"
        ),
        gr.Radio(["Urban", "Rural", "Semiurban"], label="Property Area", value="Urban")
    ],
    outputs="text",
    title="🏦 Loan Approval Predictor",
    description="Enter applicant details to check loan approval likelihood."
)

if __name__ == "__main__":
    demo.launch()
