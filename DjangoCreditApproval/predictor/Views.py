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

columns = [
    'Children_count', 'Income', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'Work_phone', 'Phone', 'Email',
    'Family_count', 'Starting_month', 'Gender_M', 'Car_Y', 'Realty_N',
    'Income_type_Commercial associate', 'Income_type_Pensioner', 'Income_type_State servant',
    'Income_type_Student', 'Education_type_Academic degree', 'Education_type_Higher education',
    'Education_type_Incomplete higher', 'Education_type_Lower secondary',
    'Housing_type_Co-op apartment', 'Housing_type_Municipal apartment', 'Housing_type_Office apartment',
    'Housing_type_Rented apartment', 'Housing_type_With parents', 'Occupation_Accountants',
    'Occupation_Cleaning staff', 'Occupation_Cooking staff', 'Occupation_Core staff', 'Occupation_Drivers',
    'Occupation_HR staff', 'Occupation_High skill tech staff', 'Occupation_IT staff', 'Occupation_Low-skill Laborers',
    'Occupation_Managers', 'Occupation_Medicine staff', 'Occupation_Private service staff',
    'Occupation_Realty agents', 'Occupation_Sales staff', 'Occupation_Secretaries', 'Occupation_Security staff',
    'Occupation_Waiters/barmen staff', 'Family_status_Civil marriage', 'Family_status_Separated',
    'Family_status_Single / not married', 'Family_status_Widow'
]

# Create an empty DataFrame with the given columns
df = pd.DataFrame(columns=columns)

# Set the boolean columns to False
bool_columns = [
    'Work_phone', 'Phone', 'Email', 'Gender_M', 'Car_Y', 'Realty_N', 'Income_type_Commercial associate',
    'Income_type_Pensioner', 'Income_type_State servant', 'Income_type_Student', 'Education_type_Academic degree',
    'Education_type_Higher education', 'Education_type_Incomplete higher', 'Education_type_Lower secondary',
    'Housing_type_Co-op apartment', 'Housing_type_Municipal apartment', 'Housing_type_Office apartment',
    'Housing_type_Rented apartment', 'Housing_type_With parents', 'Occupation_Accountants',
    'Occupation_Cleaning staff', 'Occupation_Cooking staff', 'Occupation_Core staff', 'Occupation_Drivers',
    'Occupation_HR staff', 'Occupation_High skill tech staff', 'Occupation_IT staff', 'Occupation_Low-skill Laborers',
    'Occupation_Managers', 'Occupation_Medicine staff', 'Occupation_Private service staff',
    'Occupation_Realty agents', 'Occupation_Sales staff', 'Occupation_Secretaries', 'Occupation_Security staff',
    'Occupation_Waiters/barmen staff', 'Family_status_Civil marriage', 'Family_status_Separated',
    'Family_status_Single / not married', 'Family_status_Widow'
]

# Set the boolean columns to False (initialization)
df[bool_columns] = False

def predict_credit(request):
    prediction = None
    accuracy = None
    confidence = None

    model = CreditCardTorch(dim = 31)
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
            default_row = {
                'Children_count': 0,  # Default for children count
                'Income': 0.0,  # Default for income
                'DAYS_BIRTH': 0,  # Default for age in days
                'DAYS_EMPLOYED': 0,  # Default for employment days
                'Family_count': 0,  # Default for family count
                'Starting_month': 1  # Default for starting month
            }
            dfs = df.append(default_row, ignore_index=True)
            # Set the selected income type column to True based on the user's input
            if form.cleaned_data['income_type'] == 'Working':
                dfs['Income_type_Working'] = True

            if form.cleaned_data['income_type'] == 'Commercial Associate':
                dfs['Income_type_Commercial associate'] = True

            if form.cleaned_data['income_type'] == 'Pensioner':
                dfs['Income_type_Pensioner'] = True

            if form.cleaned_data['income_type'] == 'State Servant':
                dfs['Income_type_State servant'] = True

            if form.cleaned_data['income_type'] == 'Unemployed':
                dfs['Income_type_Unemployed'] = True

            if form.cleaned_data['income_type'] == 'Student':
                dfs['Income_type_Student'] = True

            if form.cleaned_data['income_type'] == 'Other':
                dfs['Income_type_Other'] = True
            print(dfs.head())
            # Predict the result (binary: 1 or 0 for approved/rejected)
            with torch.no_grad():
                output = model(dfs)
                prediction = (output >= 0.5).float().item()  # Binary prediction (1 or 0)
                confidence = output.sigmoid().item() * 100  # Get the probability and convert to percentage

            # Determine loan approval status
            result_text = 'Approved' if prediction == 1 else 'Rejected'
            # Return the result in the template
            return render(request, 'result.html', {'result': result_text, 'confidence': confidence})

    else:
        form = CreditPredictionForm()

    return render(request, 'predict.html', {'form': form})
