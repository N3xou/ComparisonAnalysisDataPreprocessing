import os
import torch
import numpy as np
from django.shortcuts import render
from Forms import CreditPredictionForm

def predict_credit(request):
    prediction = None
    accuracy = None
    confidence = None
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'predictor', 'credit_card_model.pth')  # Using .pth for the entire model

    # Load the entire model
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode

    if request.method == 'POST':
        form = CreditPredictionForm(request.POST)
        if form.is_valid():
            # Process form data and run the prediction
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

            # Normalize the data here if you used any scaler during training
            # e.g., data = scaler.transform(data)  # If you used StandardScaler

            # Convert the input data to a PyTorch tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)

            # Make the prediction
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

    return render(request, 'predictor/predict.html', {'form': form})
