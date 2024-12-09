from django import forms

class CreditPredictionForm(forms.Form):
    # Basic client information
    #client_id = forms.IntegerField(label='Client ID', min_value=1)
    gender = forms.ChoiceField(
        label='Gender',
        choices=[('M', 'Male'), ('F', 'Female')],
        widget=forms.RadioSelect,
    )
    car = forms.ChoiceField(
        label='Owns Car?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    realty = forms.ChoiceField(
        label='Owns Property?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    age = forms.IntegerField(label='Age(years)', min_value=0)
    family_size = forms.IntegerField(label='Family Size', min_value=1)
    kids = forms.IntegerField(label='Number of Children', min_value=0)
    annual_income = forms.FloatField(label='Annual Income in $', min_value=0)
    employment_months = forms.IntegerField(label='Months Employed', min_value=0)

    # Categorical data
    income_type = forms.ChoiceField(
        label='Income Category',
        choices=[
            ('Working', 'Working'),
            ('Commercial Associate', 'Commercial Associate'),
            ('Pensioner', 'Pensioner'),
            ('State Servant', 'State Servant'),
            ('Unemployed', 'Unemployed'),
            ('Student', 'Student'),
            ('Other', 'Other'),
        ]
    )
    education_type = forms.ChoiceField(
        label='Education Level',
        choices=[
            ('Lower Secondary', 'Lower Secondary'),
            ('Secondary / Secondary Special', 'Secondary / Secondary Special'),
            ('Incomplete Higher', 'Incomplete Higher'),
            ('Higher Education', 'Higher Education'),
            ('Academic Degree', 'Academic Degree'),
        ]
    )
    family_status = forms.ChoiceField(
        label='Marital Status',
        choices=[
            ('Single', 'Single'),
            ('Married', 'Married'),
            ('Civil Marriage', 'Civil Marriage'),
            ('Widow', 'Widow'),
            ('Separated', 'Separated'),
            ('Unknown', 'Unknown'),
        ]
    )
    housing_type = forms.ChoiceField(
        label='Way of Living',
        choices=[
            ('House / Apartment', 'House / Apartment'),
            ('Municipal Apartment', 'Municipal Apartment'),
            ('With Parents', 'With Parents'),
            ('Co-op Apartment', 'Co-op Apartment'),
            ('Rented Apartment', 'Rented Apartment'),
            ('Office Apartment', 'Office Apartment'),
        ]
    )

    mobil = forms.ChoiceField(
        label='Has Mobile Phone?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    work_phone = forms.ChoiceField(
        label='Has Work Phone?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    email = forms.ChoiceField(
        label='Has Email?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )

    # Occupation and family size
    occupation = forms.ChoiceField(
        label='Occupation',
        choices=[
            ('Accountants', 'Accountants'),
            ('Cleaning Staff', 'Cleaning Staff'),
            ('Cooking Staff', 'Cooking Staff'),
            ('Core Staff', 'Core Staff'),
            ('Drivers', 'Drivers'),
            ('High Skill Tech Staff', 'High Skill Tech Staff'),
            ('Laborers', 'Laborers'),
            ('Low Skill Laborers', 'Low Skill Laborers'),
            ('Managers', 'Managers'),
            ('Medicine Staff', 'Medicine Staff'),
            ('Other', 'Other'),
            ('Private Service Staff', 'Private Service Staff'),
            ('Realty Agents', 'Realty Agents'),
            ('Sales Staff', 'Sales Staff'),
            ('Secretaries', 'Secretaries'),
            ('Security Staff', 'Security Staff'),
            ('Waiters', 'Waiters'),
        ]
    )
    # Bank account ownership
    account_duration_years = forms.IntegerField(
        label='Bank account owner for: years',
        min_value=0,
        help_text='How long has the person owned an account at the bank?'
    )
