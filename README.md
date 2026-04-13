# Credit Risk NPA Prediction & Deployment

## Project Overview
Developed an end-to-end machine learning solution to predict the probability of 90-day delinquency (NPA) for credit card portfolios. This project mimics real-world risk assessment workflows utilized in the BFSI sector.

## Key Features
* **Predictive Modeling:** Built using a Random Forest Classifier with a verified AUC-ROC of 0.86.
* **Class Imbalance Handling:** Utilized `class_weight='balanced'` and stratified splitting to ensure high recall for minority default classes.
* **Production Pipeline:** Implemented a Scikit-Learn `Pipeline` to bundle preprocessing (median imputation) and the model into a single serialized object for deployment.
* **Interactive UI:** A Streamlit web application for real-time risk assessment.

## Tech Stack
* **Language:** Python (Pandas, Scikit-Learn)
* **Model:** Random Forest
* **Deployment:** Streamlit
* **Environment Management:** Virtualenv / Git

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app/app.py`