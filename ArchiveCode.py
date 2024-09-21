# Manual encoding


print(df['FLAG_OWN_CAR'].unique())
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace({'Y': True, 'N': False}).astype(bool)

print(df['FLAG_OWN_REALTY'].unique())
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace({'Y': True, 'N': False}).astype(bool)

print(df['FLAG_WORK_PHONE'].unique())
df['FLAG_WORK_PHONE'] = df['FLAG_WORK_PHONE'].replace({'Y': True, 'N': False}).astype(bool)

print(df['FLAG_PHONE'].unique())
df['FLAG_PHONE'] = df['FLAG_PHONE'].replace({'Y': True, 'N': False}).astype(bool)

print(df['FLAG_EMAIL'].unique())
df['FLAG_EMAIL'] = df['FLAG_EMAIL'].replace({'Y': True, 'N': False}).astype(bool)

print(df.head())
le_gender = LabelEncoder()

df['CODE_GENDER'] = le_gender.fit_transform(df['CODE_GENDER'])
print(f"Unique occupations: {df['OCCUPATION_TYPE'].unique()}")
le_occupations = LabelEncoder()
df['OCCUPATION_TYPE'] = le_occupations.fit_transform(df['OCCUPATION_TYPE'])

le_income = LabelEncoder()
print(df['NAME_INCOME_TYPE'].unique())
df['NAME_INCOME_TYPE'] = le_income.fit_transform(df['NAME_INCOME_TYPE'])

print(df['NAME_EDUCATION_TYPE'].unique())
le_education = LabelEncoder()
df['NAME_EDUCATION_TYPE'] = le_education.fit_transform(df['NAME_EDUCATION_TYPE'])

print(df['NAME_FAMILY_STATUS'].unique())
le_familystatus = LabelEncoder()
df['NAME_FAMILY_STATUS'] = le_familystatus.fit_transform(df['NAME_FAMILY_STATUS'])

print(df['NAME_HOUSING_TYPE'].unique())
le_housing = LabelEncoder()
df['NAME_HOUSING_TYPE'] = le_familystatus.fit_transform(df['NAME_HOUSING_TYPE'])
