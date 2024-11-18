from django import forms

class CreditPredictionForm(forms.Form):
    # Basic client information
    client_id = forms.IntegerField(label='Client ID', min_value=1)
    gender = forms.ChoiceField(
        label='Gender',
        choices=[('M', 'Male'), ('F', 'Female')],
        widget=forms.RadioSelect,
    )
    owns_car = forms.ChoiceField(
        label='Owns Car?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    owns_realty = forms.ChoiceField(
        label='Owns Property?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    number_of_children = forms.IntegerField(label='Number of Children', min_value=0)
    annual_income = forms.FloatField(label='Annual Income', min_value=0)

    # Categorical data
    income_category = forms.ChoiceField(
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
    education_level = forms.ChoiceField(
        label='Education Level',
        choices=[
            ('Lower Secondary', 'Lower Secondary'),
            ('Secondary / Secondary Special', 'Secondary / Secondary Special'),
            ('Incomplete Higher', 'Incomplete Higher'),
            ('Higher Education', 'Higher Education'),
            ('Academic Degree', 'Academic Degree'),
        ]
    )
    marital_status = forms.ChoiceField(
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

    # Time-related data
    birth_years = forms.IntegerField(label='Age(years)', min_value=0)

    employment_years = forms.IntegerField(label='Years Employed', min_value=0)
    employment_months = forms.IntegerField(label='Months Employed', min_value=0, max_value=11)

    # Flags
    has_mobile = forms.ChoiceField(
        label='Has Mobile Phone?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    has_work_phone = forms.ChoiceField(
        label='Has Work Phone?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    has_phone = forms.ChoiceField(
        label='Has Phone?',
        choices=[('Y', 'Yes'), ('N', 'No')],
        widget=forms.RadioSelect,
    )
    has_email = forms.ChoiceField(
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
    family_size = forms.IntegerField(label='Family Size', min_value=1)

    # Bank account ownership
    account_duration_years = forms.IntegerField(
        label='Years Account Owned',
        min_value=0,
        help_text='How long has the person owned an account at the bank?'
    )
