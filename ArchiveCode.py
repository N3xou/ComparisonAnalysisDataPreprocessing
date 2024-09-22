# Manual encoding

## one in all encoder
le = LabelEncoder()
for col in df:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col])

# List of flags columns
flag_columns = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

for col in flag_columns:
    print(f"Unique values in {col}: {df_encoded[col].unique()}")
    df_encoded.loc[:,col] = df_encoded[col].replace({'Y': True, 'N': False} ).infer_objects(copy=False).astype(bool)
    print(f"Unique values in {col}: {df_encoded[col].unique()}")


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


######################################### CHILDREN COUNT REMOVE bottom 0.1% occurences

total_occurrences = df['CNT_CHILDREN'].value_counts().sum()
print(f"\nTotal Occurrences: {total_occurrences}")

# Calculate the threshold for 1% of total occurrences
threshold = total_occurrences * 0.001
print(f"Threshold for 0.1% of Total Occurrences: {threshold}")

# Filter the DataFrame to remove labels with occurrences below the threshold
df = df[df['CNT_CHILDREN'].value_counts() >= threshold]

# Display the filtered DataFrame
print("\nFiltered DataFrame (Labels with < 0.1% of Total Removed):")
print(df)
## categorizing age manual
#df.loc[(df['AGE_YEARS'] >= 18) & (df['AGE_YEARS'] <= 28), 'AGE_CATEGORY'] = 'young'
#df.loc[(df['AGE_YEARS'] >= 29) & (df['AGE_YEARS'] <= 55), 'AGE_CATEGORY'] = 'mature'
#df.loc[df['AGE_YEARS'] > 55, 'AGE_CATEGORY'] = 'elder'