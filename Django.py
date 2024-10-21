"""
pip install django
django-admin startproject credit_prediction
cd credit_prediction
python manage.py startapp predictor
"""
INSTALLED_APPS = [
    'predictor',
]

from django import forms
from django.shortcuts import render
from .forms import CreditPredictionForm
import joblib

class CreditPredictionForm(forms.Form):
    income = forms.FloatField(label='Income', min_value=0)
    age = forms.IntegerField(label='Age', min_value=18)
    # Add other fields that match your dataset
    # e.g., gender = forms.ChoiceField(choices=[('male', 'Male'), ('female', 'Female')])


def predict_credit(request):
    prediction = None
    accuracy = None

    if request.method == 'POST':
        form = CreditPredictionForm(request.POST)
        if form.is_valid():
            # Process form data and run the prediction
            model = joblib.load('path/to/your_model.pkl')  # Load your model
            data = [[form.cleaned_data['income'], form.cleaned_data['age']]]
            # Include all the input features here
            prediction = model.predict(data)[0]  # Model returns 1 or 0
            accuracy = model.score(data, [prediction])  # Simulated accuracy

            result_text = 'Approved' if prediction == 1 else 'Rejected'
            return render(request, 'result.html', {'result': result_text, 'accuracy': accuracy * 100})

    else:
        form = CreditPredictionForm()

    return render(request, 'predict.html', {'form': form})

