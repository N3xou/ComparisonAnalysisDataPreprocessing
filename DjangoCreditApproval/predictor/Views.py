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

def oneHot(df, feature, rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest], axis=1, inplace=True)
    df = df.join(pos)
    return df
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
                form.cleaned_data['employment_months'] * 30,  # Duration of employment (DAYS_EMPLOYED) converted to months
            ]])
            numeric_columns = [0,1,2,3,4,9,10,11,12,13,14,16,17]
            cat_columns = [5,6,7,8,15]
            data_num = data[:, numeric_columns].astype(float)
            data_cat = data[:, cat_columns]
            print(data_cat.shape)
            data_cat = pd.DataFrame(data_cat.T,columns=['income_type', 'education_type', 'family_status','housing_type', 'occupation'])
            data_num = pd.DataFrame(data_num, columns=[
                'annual_income', 'age', 'kids', 'car', 'realty', 'age_in_days', 'account_duration_days',
                'mobil', 'work_phone', 'phone', 'email', 'family_size', 'employment_duration_days'
            ])

            data = pd.concat([data_num, data_cat], axis=1)

            # Apply one-hot encoding to the categorical columns
            onehot_cols = ['income_type', 'education_type', 'family_status', 'housing_type', 'occupation']
            for col in onehot_cols:
                data = oneHot(data, col)

            # Drop original categorical columns as they are now

            input_df = input_df.drop(columns=onehot_cols)
            for i, value in enumerate(data[0]):
                print(f"Column {i}: {type(value)}")
            data_tensor = torch.tensor(data, dtype=torch.float32)
            # Predict the result (binary: 1 or 0 for approved/rejected)
            with torch.no_grad():
                output = model(data_tensor)
                prediction = (output >= 0.5).float().item()  # Binary prediction (1 or 0)
                confidence = output.sigmoid().item() * 100  # Get the probability and convert to percentage

            # Determine loan approval status
            result_text = 'Approved' if prediction == 1 else 'Rejected'
            # Return the result in the template
            return render(request, 'result.html', {'result': result_text, 'confidence': confidence})

    else:
        form = CreditPredictionForm()

    return render(request, 'predict.html', {'form': form})
