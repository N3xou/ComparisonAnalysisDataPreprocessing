"""
pip install django
django-admin startproject credit_prediction
cd credit_prediction
python manage.py startapp predictor
"""
from tensorflow.python.feature_column.feature_column_v2 import numeric_column

INSTALLED_APPS = [
    'predictor',
]
# todo: format data from strings into numbers for the model.

from .models import CreditCardTorch
from django.shortcuts import render
import numpy as np
import pandas as pd
from .Forms import CreditPredictionForm
import joblib
import os
import torch



def predict_credit(request):
    prediction = None
    accuracy = None
    confidence = None
    columns = [
        'Children_count', 'Income', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'Work_phone', 'Phone', 'Email',
        'Family_count', 'Starting_month', 'Gender_M', 'Car_Y', 'Realty_N',
        'Income_type_Commercial associate', 'Income_type_Pensioner', 'Income_type_State servant',
        'Income_type_Student', 'Education_type_Academic degree', 'Education_type_Higher education',
        'Education_type_Incomplete higher', 'Education_type_Lower secondary',
        'Housing_type_Co-op apartment', 'Housing_type_Municipal apartment', 'Housing_type_Office apartment',
        'Housing_type_Rented apartment', 'Housing_type_With parents', 'Occupation_Accountants',
        'Occupation_Cleaning staff', 'Occupation_Cooking staff', 'Occupation_Core staff', 'Occupation_Drivers',
        'Occupation_HR staff', 'Occupation_High skill tech staff', 'Occupation_IT staff',
        'Occupation_Low-skill Laborers',
        'Occupation_Managers', 'Occupation_Medicine staff', 'Occupation_Private service staff',
        'Occupation_Realty agents', 'Occupation_Sales staff', 'Occupation_Secretaries', 'Occupation_Security staff',
        'Occupation_Waiters/barmen staff', 'Family_status_Civil marriage', 'Family_status_Separated',
        'Family_status_Single / not married', 'Family_status_Widow'
    ]

    # Create an empty DataFrame with the given columns
    df = pd.DataFrame(columns=columns)

    model = CreditCardTorch(dim = 46)
    model.load_state_dict(torch.load('credit_card_model.pth'))
    model.eval()
    if request.method == 'POST':
        form = CreditPredictionForm(request.POST)
        if form.is_valid():

            data = np.array([[
                form.cleaned_data['annual_income'],  # Annual income (AMT_INCOME_TOTAL)
                form.cleaned_data['age'],  # Age in years (DAYS_BIRTH)
                form.cleaned_data['kids'],  # Number of children (CNT_CHILDREN)
                True if form.cleaned_data['car'] == 'Yes' else False,  # Owns car (FLAG_OWN_CAR)
                True if form.cleaned_data['realty'] == 'Yes' else False,  # Owns property (FLAG_OWN_REALTY)
                form.cleaned_data['income_type'],  # Income category (NAME_INCOME_TYPE)
                form.cleaned_data['education_type'],  # Education level (NAME_EDUCATION_TYPE)
                form.cleaned_data['family_status'],  # Family status (NAME_FAMILY_STATUS)
                form.cleaned_data['housing_type'],  # Housing type (NAME_HOUSING_TYPE)
                form.cleaned_data['age'] * 365,  # Birthday converted to days (DAYS_BIRTH)
                form.cleaned_data['account_duration_years'] * 365,  # How long with bank in days (DAYS_EMPLOYED)
                True if form.cleaned_data['mobil'] == 'Yes' else False,  # Has mobile phone (FLAG_MOBIL)
                True if form.cleaned_data['work_phone'] == 'Yes' else False,  # Has work phone (FLAG_WORK_PHONE)
                True if form.cleaned_data['mobil'] == 'Yes' else False,  # Has phone (FLAG_PHONE)
                True if form.cleaned_data['email'] == 'Yes' else False,  # Has email (FLAG_EMAIL)
                form.cleaned_data['occupation'],  # Occupation (OCCUPATION_TYPE)
                form.cleaned_data['family_size'],  # Family size (CNT_FAM_MEMBERS)
                form.cleaned_data['employment_months'] * 30,  # Duration of employment (DAYS_EMPLOYED) converted to months
            ]])
            row = {
                'Children_count': 0,
                'Income': 0.0,
                'DAYS_BIRTH': 0,
                'DAYS_EMPLOYED': 0,
                'Work_phone': False,
                'Phone': False,
                'Email': False,
                'Family_count': 0,
                'Starting_month': 1,
                'Gender_M': False,
                'Car_Y': False,
                'Realty_N': False,
                'Income_type_Commercial associate': False,
                'Income_type_Pensioner': False,
                'Income_type_State servant': False,
                'Income_type_Student': False,
                'Education_type_Academic degree': False,
                'Education_type_Higher education': False,
                'Education_type_Incomplete higher': False,
                'Education_type_Lower secondary': False,
                'Housing_type_Co-op apartment': False,
                'Housing_type_Municipal apartment': False,
                'Housing_type_Office apartment': False,
                'Housing_type_Rented apartment': False,
                'Housing_type_With parents': False,
                'Occupation_Accountants': False,
                'Occupation_Cleaning staff': False,
                'Occupation_Cooking staff': False,
                'Occupation_Core staff': False,
                'Occupation_Drivers': False,
                'Occupation_HR staff': False,
                'Occupation_High skill tech staff': False,
                'Occupation_IT staff': False,
                'Occupation_Low-skill Laborers': False,
                'Occupation_Managers': False,
                'Occupation_Medicine staff': False,
                'Occupation_Private service staff': False,
                'Occupation_Realty agents': False,
                'Occupation_Sales staff': False,
                'Occupation_Secretaries': False,
                'Occupation_Security staff': False,
                'Occupation_Waiters/barmen staff': False,
                'Family_status_Civil marriage': False,
                'Family_status_Separated': False,
                'Family_status_Single / not married': False,
                'Family_status_Widow': False,

            }

            df = df._append(row, ignore_index=True)
            # Set the selected income type column to True based on the user's input
            if form.data['income_type'] == 'Working':
                df['Income_type_Working'] = True

            if form.data['income_type'] == 'Commercial Associate':
                df['Income_type_Commercial associate'] = True

            if form.data['income_type'] == 'Pensioner':
                df['Income_type_Pensioner'] = True

            if form.data['income_type'] == 'State Servant':
                df['Income_type_State servant'] = True

            if form.data['income_type'] == 'Unemployed':
                df['Income_type_Unemployed'] = True

            if form.data['income_type'] == 'Student':
                df['Income_type_Student'] = True

            if form.data['income_type'] == 'Other':
                df['Income_type_Other'] = True

            if form.data['education_type'] == 'Academic Degree':
                df['Education_type_Academic degree'] = True

            if form.data['education_type'] == 'Higher education':
                df['Education_type_Higher education'] = True

            if form.data['education_type'] == 'Incomplete higher':
                df['Education_type_Incomplete higher'] = True

            if form.data['education_type'] == 'Lower secondary':
                df['Education_type_Lower secondary'] = True
            if form.data['housing_type'] == 'Co-op apartment':
                df['Housing_type_Co-op apartment'] = True

            if form.data['housing_type'] == 'Municipal apartment':
                df['Housing_type_Municipal apartment'] = True

            if form.data['housing_type'] == 'Office apartment':
                df['Housing_type_Office apartment'] = True

            if form.data['housing_type'] == 'Rented apartment':
                df['Housing_type_Rented apartment'] = True

            if form.data['housing_type'] == 'With parents':
                df['Housing_type_With parents'] = True
            if form.data['occupation'] == 'Accountants':
                df['Occupation_Accountants'] = True

            if form.data['occupation'] == 'Cleaning staff':
                df['Occupation_Cleaning staff'] = True

            if form.data['occupation'] == 'Cooking staff':
                df['Occupation_Cooking staff'] = True

            if form.data['occupation'] == 'Core staff':
                df['Occupation_Core staff'] = True

            if form.data['occupation'] == 'Drivers':
                df['Occupation_Drivers'] = True

            if form.data['occupation'] == 'HR staff':
                df['Occupation_HR staff'] = True

            if form.data['occupation'] == 'High skill tech staff':
                df['Occupation_High skill tech staff'] = True

            if form.data['occupation'] == 'IT staff':
                df['Occupation_IT staff'] = True

            if form.data['occupation'] == 'Low-skill Laborers':
                df['Occupation_Low-skill Laborers'] = True

            if form.data['occupation'] == 'Managers':
                df['Occupation_Managers'] = True

            if form.data['occupation'] == 'Medicine staff':
                df['Occupation_Medicine staff'] = True

            if form.data['occupation'] == 'Private service staff':
                df['Occupation_Private service staff'] = True

            if form.data['occupation'] == 'Realty agents':
                df['Occupation_Realty agents'] = True

            if form.data['occupation'] == 'Sales staff':
                df['Occupation_Sales staff'] = True

            if form.data['occupation'] == 'Secretaries':
                df['Occupation_Secretaries'] = True

            if form.data['occupation'] == 'Security staff':
                df['Occupation_Security staff'] = True

            if form.data['occupation'] == 'Waiters/barmen staff':
                df['Occupation_Waiters/barmen staff'] = True
            if form.data['family_status'] == 'Civil marriage':
                df['Family_status_Civil marriage'] = True

            if form.data['family_status'] == 'Separated':
                df['Family_status_Separated'] = True

            if form.data['family_status'] == 'Single / not married':
                df['Family_status_Single / not married'] = True

            if form.data['family_status'] == 'Widow':
                df['Family_status_Widow'] = True

            print(df.head())


            # Predict the result (binary: 1 or 0 for approved/rejected)
            with torch.no_grad():
                output = model(df)
                prediction = (output >= 0.5).float().item()  # Binary prediction (1 or 0)
                confidence = output.sigmoid().item() * 100  # Get the probability and convert to percentage

            # Determine loan approval status
            result_text = 'Approved' if prediction == 1 else 'Rejected'
            # Return the result in the template
            return render(request, 'result.html', {'result': result_text, 'confidence': confidence})

    else:
        form = CreditPredictionForm()

    return render(request, 'predict.html', {'form': form})
