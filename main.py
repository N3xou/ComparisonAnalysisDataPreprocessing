# Imports
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Creating dataframe, merging two dataframes into one on ID
applicationRecord = pd.read_csv(r'C:\Users\Yami\PycharmProjects\pythonProject1\application_record.csv')
creditRecord = pd.read_csv(r'C:\Users\Yami\PycharmProjects\pythonProject1\credit_record.csv')
df = pd.merge(applicationRecord, creditRecord, on='ID')

def one_hot(df,feature,rank = 0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest], axis=1, inplace=True)
    df.drop([feature], axis=1, inplace=True)
    df = df.join(pos)
    return df


def get_category(df, col, binsnum, labels, qcut = False, replace = True):
    if replace:
        if pd.qcut:
            # Quantile cut
            df[col] = pd.qcut(df[col], q=binsnum, labels=labels)
        else:
            # Equal-length cut
            df[col] = pd.cut(df[col], bins=binsnum, labels=labels)

        # Convert the column to object type (if necessary)
        df[col] = df[col].astype(object)
    else:
        if qcut:
            localdf = pd.qcut(df[col], q=binsnum, labels=labels)  # quantile cut
        else:
            localdf = pd.cut(df[col], bins=binsnum, labels=labels)  # equal-length cut

        localdf = pd.DataFrame(localdf)
        name = 'cat' + '_' + col
        localdf[name] = localdf[col]
        df = df.join(localdf[name])
        df[name] = df[name].astype(object)
    return df

## feature
creditRecord['dependency'] = None
creditRecord.loc[creditRecord['STATUS'] == '2', 'dependency'] = 'Yes'
creditRecord.loc[creditRecord['STATUS'] == '3', 'dependency'] = 'Yes'
creditRecord.loc[creditRecord['STATUS'] == '4', 'dependency'] = 'Yes'
creditRecord.loc[creditRecord['STATUS'] == '5', 'dependency'] = 'Yes'

# 0 = safe, 1 = flag unsafe
cpunt = creditRecord.groupby('ID').count()
cpunt.loc[cpunt['dependency'] > 0, 'dependency'] = 0
cpunt.loc[cpunt['dependency'] == 0, 'dependency'] = 1
cpunt = cpunt[['dependency']]
df = pd.merge(df, cpunt, how='inner', on='ID')
df['target'] = df['dependency']
df.loc[df['target'] == 'Yes', 'target'] = 1
df.loc[df['target'] == 'No', 'target'] = 0
#print(cpunt['dependency'].value_counts())
#print(cpunt['dependency'].value_counts(normalize=True))

# Looking into the dataframe and preprocessing data

print(df.head())
print(f'Datatypes\n{df.dtypes}')
print(f'Shape{df.shape}')
print(f'Missing data\n{df.isna().sum()}')
msno.matrix(df)
#plt.show()


print(df['FLAG_MOBIL'].unique())

# dropping rows with missing data
for column in df.columns:
    dropped_count = df[column].isna().sum()
    if dropped_count > 0:
        print(f"Dropping {dropped_count} rows due to NaN in {column} column")
df = df.dropna()

# dropping duplicates based on ID

print(f"Dropping duplicates, amount of unique rows: {df['ID'].nunique()}")
df.drop_duplicates('ID', keep='last')
# dropping flag_mobil as all values equal 1
df.drop(columns=['FLAG_MOBIL'], inplace=True)
# bucketing data

plt.figure()
print(df['AMT_INCOME_TOTAL'].unique())
df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].astype(object)
df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL']/10000
df['AMT_INCOME_TOTAL'].plot(kind='hist',bins=40,density=True)
#plt.show()

df = get_category(df,'AMT_INCOME_TOTAL', 3, [ "low", "medium", "high"], qcut=True, replace=False)
print(df['cat_AMT_INCOME_TOTAL'].value_counts())

# Converting negative values to positive the following columns column
print(df['DAYS_BIRTH'].unique())
df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
df['AGE_YEARS'] = (df['DAYS_BIRTH'] / 365).round(0).astype(int)
# categorizing age groups

#plt.figure()
df['AGE_YEARS'].plot(kind='hist',bins = 20,density=True)
#plt.show()
df = get_category(df,'AGE_YEARS', 5, [ "young adult","adult", "mature adult", "middle aged","elderly"], replace=False)

df = df[df['AGE_YEARS'] >= 18] # dropping ages below 18
print("Lowest age per age group")
print(df.loc[df.groupby('cat_AGE_YEARS')['AGE_YEARS'].idxmin()][['cat_AGE_YEARS', 'AGE_YEARS']])
print("Highest age per age group")
print(df.loc[df.groupby('cat_AGE_YEARS')['AGE_YEARS'].idxmax()][['cat_AGE_YEARS', 'AGE_YEARS']])
print(df['cat_AGE_YEARS'].value_counts())


print(df['DAYS_EMPLOYED'].unique())

df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / 365
df['YEARS_EMPLOYED'].plot(kind='hist',bins=20,density=True)
#plt.figure()
#plt.show()
df = get_category(df,'YEARS_EMPLOYED', 5, [ "lowest","low", "medium", "high","highest"], replace=False)
print(df['cat_YEARS_EMPLOYED'].value_counts())


print(df['MONTHS_BALANCE'].unique())
df['MONTHS_BALANCE'] = abs(df['MONTHS_BALANCE'])

print(df['OCCUPATION_TYPE'].unique())
df.loc[(df['OCCUPATION_TYPE'] == 'Waiters/barmen staff') |
(df['OCCUPATION_TYPE'] == 'Cleaning staff') |
(df['OCCUPATION_TYPE'] == 'Cooking staff') |
(df['OCCUPATION_TYPE'] == 'Low-skill Laborers') |
(df['OCCUPATION_TYPE'] == 'Security staff') |
(df['OCCUPATION_TYPE'] == 'Drivers') |
(df['OCCUPATION_TYPE'] == 'Secretaries'), 'OCCUPATION_TYPE'
] = 'Low position job'
df.loc[(df['OCCUPATION_TYPE'] == 'Sales staff') |
(df['OCCUPATION_TYPE'] == 'Accountants') |
(df['OCCUPATION_TYPE'] == 'Laborers') |
(df['OCCUPATION_TYPE'] == 'Core staff') |
(df['OCCUPATION_TYPE'] == 'Private service staff') |
(df['OCCUPATION_TYPE'] == 'Medicine staff') |
(df['OCCUPATION_TYPE'] == 'HR staff') |
(df['OCCUPATION_TYPE'] == 'Realty agents'), 'OCCUPATION_TYPE'
] = 'Medium position job'
df.loc[(df['OCCUPATION_TYPE'] == 'Managers') |
(df['OCCUPATION_TYPE'] == 'High skill tech staff') |
(df['OCCUPATION_TYPE'] == 'IT staff'), 'OCCUPATION_TYPE'
] = 'High position job'
print(df['OCCUPATION_TYPE'].unique())
print(df['NAME_EDUCATION_TYPE'].unique())
####### Ordinal encoding
oe = OrdinalEncoder(categories=[['Low position job','Medium position job','High position job']])
df['OCCUPATION_TYPE'] = oe.fit_transform(df[['OCCUPATION_TYPE']]).astype(int)
print(df['OCCUPATION_TYPE'].unique())
oe = OrdinalEncoder(categories=[['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']])
df['EDUCATION_TYPE'] = oe.fit_transform(df[['NAME_EDUCATION_TYPE']]).astype(int)
print(df['EDUCATION_TYPE'].unique())
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['num_cat_AMT_INCOME_TOTAL'] = oe.fit_transform(df[['cat_AMT_INCOME_TOTAL']]).astype(int)
print(df['cat_AMT_INCOME_TOTAL'].unique())
print(df['num_cat_AMT_INCOME_TOTAL'].unique())
oe = OrdinalEncoder(categories=[[ "lowest","low", "medium", "high","highest"]])
df['num_cat_YEARS_EMPLOYED'] = oe.fit_transform(df[['cat_YEARS_EMPLOYED']]).astype(int)
print(df['cat_YEARS_EMPLOYED'].unique())
print(df['num_cat_YEARS_EMPLOYED'].unique())
######## label encoding (object > numerical values)

#print(df.head())
df_encoded = df.copy()
label_cols = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY']
for col in label_cols:
    le = LabelEncoder()
    print(f"Unique values in {col}: {df_encoded[col].unique()}")
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f"Unique values in {col}: {df_encoded[col].unique()}")

df = df_encoded
######## one hot encoding
onehot_cols = ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'cat_AGE_YEARS']
for col in onehot_cols:
    df = one_hot(df,col)
print(f'Datatypes\n{df.dtypes}')

# decision based on observation of amount of occurences, scaling down 3+ kids into "3" group, and 5+ families into 5
# todo: maybe change the type so it says 3+ instead of 3 for clarity


print(df['CNT_CHILDREN'].value_counts())
df.loc[df['CNT_CHILDREN'] >= 3, 'CNT_CHILDREN'] = 3

print(df['CNT_FAM_MEMBERS'].unique())
print(df['CNT_FAM_MEMBERS'].value_counts())
df.loc[df['CNT_FAM_MEMBERS'] >= 5, 'CNT_FAM_MEMBERS'] = 5

df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype(int)



print(df.head())
print(f'Datatypes\n{df.dtypes}')
print(f'Shape{df.shape}')
print(f'Missing data\n{df.isna().sum()}')


################################################### GRAPHS
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 6))

sns.scatterplot(x='ID', y='CNT_CHILDREN', data=df, ax=ax[0][0], color='orange')
sns.scatterplot(x='ID', y='AMT_INCOME_TOTAL', data=df, ax=ax[0][1], color='orange')
sns.scatterplot(x='ID', y='DAYS_BIRTH', data=df, ax=ax[0][2])
sns.scatterplot(x='ID', y='DAYS_EMPLOYED', data=df, ax=ax[1][0])
sns.scatterplot(x='ID', y='FLAG_WORK_PHONE', data=df, ax=ax[1][2])
sns.scatterplot(x='ID', y='FLAG_PHONE', data=df, ax=ax[2][0])
sns.scatterplot(x='ID', y='FLAG_EMAIL', data=df, ax=ax[2][1])
sns.scatterplot(x='ID', y='CNT_FAM_MEMBERS', data=df, ax=ax[2][2], color='orange')

#plt.show()

q_hi = df['CNT_CHILDREN'].quantile(0.999)
q_low = df['CNT_CHILDREN'].quantile(0.001)
df = df[(df['CNT_CHILDREN'] > q_low) & (df['CNT_CHILDREN'] < q_hi)]

q_hi = df['AMT_INCOME_TOTAL'].quantile(0.999)
q_low = df['AMT_INCOME_TOTAL'].quantile(0.001)
df = df[(df['AMT_INCOME_TOTAL'] > q_low) & (df['AMT_INCOME_TOTAL'] < q_hi)]

q_hi = df['CNT_FAM_MEMBERS'].quantile(0.999)
q_low = df['CNT_FAM_MEMBERS'].quantile(0.001)
df = df[(df['CNT_FAM_MEMBERS'] > q_low) & (df['CNT_FAM_MEMBERS'] < q_hi)]

# status 1 = bad, status 0 = good . status=fraud
df['STATUS'] = df['STATUS'].replace({'C': 0, 'X': 0})
df['STATUS'] = df['STATUS'].astype('int')
df['STATUS'] = df['STATUS'].apply(lambda x: 1 if x >= 2 else 0)

print(df['STATUS'].value_counts(normalize=True))

# drop or leave for the future ?
df.drop(columns='FLAG_PHONE', axis=1)


# calculting WOE and IV

def iv_woe(data, target, bins=10, show_woe=False):
    #Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    #Extract Column Names
    cols = data.columns

    for i in cols[~cols.isin([target])]:
        if (data[i].dtype.kind in 'bifc') and (
                len(np.unique(data[i])) > 10):  # bifc =  boolean, integer, float, or complex.
            binned_x = pd.qcut(data[i], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[i], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events'] / d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events'] - d['% of Events'])
        d.insert(loc=0, column='Variable', value=i)
        print("Information value of " + i + " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [i], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)
        if show_woe == True:
            print(d)
    return newDF, woeDF


iv_woe(df, 'dependency', 10, True)


# what is months balance needed for ?

# Working with https://www.kaggle.com/code/rikdifos/credit-card-approval-prediction-using-ml/notebook
# 22/09/2024
# todo: test different bins for ages and salary
# todo: make label columns create new ones for meaningful things such as age
# todo: hot encoding income type, grouping up occupation type