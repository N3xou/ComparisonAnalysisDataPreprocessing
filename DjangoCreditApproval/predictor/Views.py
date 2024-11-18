"""
pip install django
django-admin startproject credit_prediction
cd credit_prediction
python manage.py startapp predictor
"""
INSTALLED_APPS = [
    'predictor',
]


from django.shortcuts import render
import numpy as np
from .Forms import CreditPredictionForm
import joblib


def predict_credit(request):
    prediction = None
    accuracy = None
    confidence = None

    if request.method == 'POST':
        form = CreditPredictionForm(request.POST)
        if form.is_valid():
            # Process form data and run the prediction
            model = joblib.load('Credit_model_DTC.pkl')  # Load your trained model

            # Prepare the data (ensure it is in the same format your model was trained on)
            data = np.array([[
                form.cleaned_data['annual_income'],  # Annual income (AMT_INCOME_TOTAL)
                form.cleaned_data['age'],  # Age in years (DAYS_BIRTH)
                form.cleaned_data['kids'],  # Number of children (CNT_CHILDREN)
                1 if form.cleaned_data['car'] == 'Yes' else 0,  # Owns car (FLAG_OWN_CAR)
                1 if form.cleaned_data['realty'] == 'Yes' else 0,  # Owns property (FLAG_OWN_REALTY)
                form.cleaned_data['income_type'],  # Income category (NAME_INCOME_TYPE)
                form.cleaned_data['education_type'],  # Education level (NAME_EDUCATION_TYPE)
                form.cleaned_data['family_status'],  # Family status (NAME_FAMILY_STATUS)
                form.cleaned_data['housing_type'],  # Housing type (NAME_HOUSING_TYPE)
                form.cleaned_data['age'] * 365,  # Birthday converted to days (DAYS_BIRTH)
                form.cleaned_data['account_duration_years'] * 365,  # How long with bank in days (DAYS_EMPLOYED)
                1 if form.cleaned_data['mobil'] == 'Yes' else 0,  # Has mobile phone (FLAG_MOBIL)
                1 if form.cleaned_data['work_phone'] == 'Yes' else 0,  # Has work phone (FLAG_WORK_PHONE)
                1 if form.cleaned_data['mobil'] == 'Yes' else 0,  # Has phone (FLAG_PHONE)
                1 if form.cleaned_data['email'] == 'Yes' else 0,  # Has email (FLAG_EMAIL)
                form.cleaned_data['occupation'],  # Occupation (OCCUPATION_TYPE)
                form.cleaned_data['family_size'],  # Family size (CNT_FAM_MEMBERS)
                form.cleaned_data['employment_months'] * 30,
                # Duration of employment (DAYS_EMPLOYED) converted to months
            ]])
            # Add other form fields here

            # Predict the result (binary: 1 or 0 for approved/rejected)
            prediction = model.predict(data)[0]

            # Calculate the prediction probability (confidence level)
            confidence = model.predict_proba(data)[0][prediction] * 100  # Convert to percentage

            # Determine loan approval status
            result_text = 'Approved' if prediction == 1 else 'Rejected'

            # Return the result in the template
            return render(request, 'result.html', {'result': result_text, 'confidence': confidence})

    else:
        form = CreditPredictionForm()

    return render(request, 'predict.html', {'form': form})
