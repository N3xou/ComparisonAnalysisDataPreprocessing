# Imports
import pandas as pd
import numpy as np
import missingno as msno
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Creating dataframe, merging two dataframes into one on ID
applicationRecord = pd.read_csv(r'C:\Users\Yami\PycharmProjects\pythonProject1\application_record.csv')
creditRecord = pd.read_csv(r'C:\Users\Yami\PycharmProjects\pythonProject1\credit_record.csv')
credit_agg = creditRecord.groupby('ID').agg({
    'MONTHS_BALANCE': 'min',  # Earliest month balance with max(worst) status
    'STATUS': 'max'
}).reset_index()

df = pd.merge(applicationRecord, credit_agg, how='left', on='ID')
df.rename(columns = {'CODE_GENDER':'Gender', 'FLAG_OWN_CAR' : 'Car', 'FLAG_OWN_REALTY' : 'Realty', 'CNT_CHILDREN' : 'Children_count',
                     'AMT_INCOME_TOTAL' : 'Income', 'NAME_INCOME_TYPE' : 'Income_type', 'NAME_EDUCATION_TYPE':'Education_type',
                     'NAME_FAMILY_STATUS' : 'Family_status', 'NAME_HOUSING_TYPE':'Housing_type','FLAG_MOBIL' : 'Mobile',
                     'FLAG_WORK_PHONE' : 'Work_phone', 'FLAG_PHONE' : 'Phone', 'FLAG_EMAIL' : 'Email', 'OCCUPATION_TYPE': 'Occupation',
                     'CNT_FAM_MEMBERS' : 'Family_count', 'MONTHS_BALANCE' : 'Starting_month'}, inplace=True)
def oneHot(df, feature, rank = 0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest], axis=1, inplace=True)
    #df.drop([feature], axis=1, inplace=True)
    df = df.join(pos)
    return df

def getCategory(df, col, binsnum, labels, qcut = False, replace = True):
    if replace:
        if qcut:
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
def ivWoe(data, target, bins=10, show_woe=False):

    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns

    for i in cols[~cols.isin([target])]:
        #if (data[i].dtype.kind in 'bifc') and (
        #        len(np.unique(data[i])) > 10):  # bifc =  boolean, integer, float, or complex.
        #    binned_x = pd.qcut(data[i], bins, duplicates='drop')
        #    d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        #else:
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
## feature engineering
print(creditRecord['STATUS'].unique())

creditRecord['dependency'] = None
creditRecord.loc[creditRecord['STATUS'] == '2', 'dependency'] = 'Yes'
creditRecord.loc[creditRecord['STATUS'] == '3', 'dependency'] = 'Yes'
creditRecord.loc[creditRecord['STATUS'] == '4', 'dependency'] = 'Yes'
creditRecord.loc[creditRecord['STATUS'] == '5', 'dependency'] = 'Yes'

print(creditRecord['dependency'].unique())


# 0 = safe, 1 = flag unsafe (to approve credit)


cpunt = creditRecord.groupby('ID').count()
#print(df.shape)
#print(cpunt.shape)

print(cpunt['dependency'].unique())

cpunt.loc[cpunt['dependency'] > 0, 'dependency'] = 1
cpunt = cpunt[['dependency']] #reducing cpunt to 1 column
df = pd.merge(df, cpunt, how='inner', on='ID')
df['target'] = df['dependency']
df = df.drop(columns=['dependency'])

#df.loc[df['target'] == 'Yes', 'target'] = 1
#df.loc[df['target'] == 'No', 'target'] = 0
print(cpunt['dependency'].value_counts())
print(df['target'].value_counts())
print(cpunt['dependency'].value_counts(normalize=True))
print(df['target'].value_counts(normalize=True))

print(df.shape)
print(cpunt.shape)
print(df['target'].isnull().sum())

# Looking into the dataframe and preprocessing data

print(df.head())
print(f'Datatypes\n{df.dtypes}')
print(f'Shape{df.shape}')
print(f'Missing data\n{df.isna().sum()}')
msno.matrix(df)
#plt.show()

print(df['Mobile'].unique())

# dropping rows with missing data

for column in df.columns:
    dropped_count = df[column].isna().sum()
    if dropped_count > 0:
        print(f"Dropping {dropped_count} rows due to NaN in {column} column")
df = df.dropna()

# dropping duplicates based on ID

print(f"Dropping duplicates, amount of unique rows: {df['ID'].nunique()}")
df.drop_duplicates('ID', keep='last')

# dropping Mobile> as all values equal 1

df.drop(columns=['Mobile'], inplace=True)

# bucketing data

#plt.figure()
print(df['Income'].unique())
df['Income'] = df['Income'].astype(object)
df['Income'] = df['Income']/10000
df['Income'].plot(kind='hist',bins=40,density=True)
#plt.show()

df = getCategory(df, 'Income', 5, ["lowest", "low", "medium", "high", "highest"], qcut=True, replace=False)
print(df['cat_Income'].value_counts())

# Converting negative values to positive the following columns column

print(df['DAYS_BIRTH'].unique())
df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
df['Age'] = (df['DAYS_BIRTH'] / 365).round(0).astype(int)

# categorizing age groups

#plt.figure()
df['Age'].plot(kind='hist',bins = 20,density=True)
#plt.show()
df = getCategory(df, 'Age', 3, ["young adult", "mature", "elder"], qcut = True, replace=False)
#"young adult","elderly"
df = df[df['Age'] >= 18] # dropping ages below 18
print("Lowest age per age group")
print(df.loc[df.groupby('cat_Age')['Age'].idxmin()][['cat_Age', 'Age']])
print("Highest age per age group")
print(df.loc[df.groupby('cat_Age')['Age'].idxmax()][['cat_Age', 'Age']])
print(df['cat_Age'].value_counts())

print(df['DAYS_EMPLOYED'].unique())
df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
df['Employment_years'] = df['DAYS_EMPLOYED'] / 365
#plt.figure()
df['Employment_years'].plot(kind='hist',bins=20,density=True)
#plt.show()
df = getCategory(df, 'Employment_years', 5, ["lowest", "low", "medium", "high", "highest"], replace=False)
print(df['cat_Employment_years'].value_counts())

print(df['Starting_month'].unique())
df['Starting_month'] = abs(df['Starting_month'])

print(df['Occupation'].unique())
df.loc[(df['Occupation'] == 'Waiters/barmen staff') |
(df['Occupation'] == 'Cleaning staff') |
(df['Occupation'] == 'Cooking staff') |
(df['Occupation'] == 'Low-skill Laborers') |
(df['Occupation'] == 'Security staff') |
(df['Occupation'] == 'Drivers') |
(df['Occupation'] == 'Secretaries'), 'Occupation'
] = 'Low position job'
df.loc[(df['Occupation'] == 'Sales staff') |
(df['Occupation'] == 'Accountants') |
(df['Occupation'] == 'Laborers') |
(df['Occupation'] == 'Core staff') |
(df['Occupation'] == 'Private service staff') |
(df['Occupation'] == 'Medicine staff') |
(df['Occupation'] == 'HR staff') |
(df['Occupation'] == 'Realty agents'), 'Occupation'
] = 'Medium position job'
df.loc[(df['Occupation'] == 'Managers') |
(df['Occupation'] == 'High skill tech staff') |
(df['Occupation'] == 'IT staff'), 'Occupation'
] = 'High position job'
print(df['Occupation'].unique())
print(df['Education_type'].unique())



#df.loc[(df['Income_type'] == 'Student') | (df['Income_type'] == 'Pensioner'), 'Income_type'] = 'State servant'

# Ordinal encoding

oe = OrdinalEncoder(categories=[['Low position job','Medium position job','High position job']])
df['Occupation'] = oe.fit_transform(df[['Occupation']]).astype(int)
print(df['Occupation'].unique())
oe = OrdinalEncoder(categories=[['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']])
df['Education_type'] = oe.fit_transform(df[['Education_type']]).astype(int)
print(df['Education_type'].unique())
#oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
#df['num_cat_Income'] = oe.fit_transform(df[['cat_Income']]).astype(int)
print(df['cat_Income'].unique())
#print(df['num_cat_Income'].unique())
oe = OrdinalEncoder(categories=[[ "lowest","low", "medium", "high","highest"]])
df['num_cat_Employment_years'] = oe.fit_transform(df[['cat_Employment_years']]).astype(int)
print(df['cat_Employment_years'].unique())
print(df['num_cat_Employment_years'].unique())

# Label encoding

print(df.head())
df_encoded = df.copy()
label_cols = ['Gender','Car','Realty']
for col in label_cols:
    le = LabelEncoder()
    print(f"Unique values in {col}: {df_encoded[col].unique()}")
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f"Unique values in {col}: {df_encoded[col].unique()}")
df = df_encoded

######## one hot encoding

onehot_cols = ['Income_type','Family_status', 'Housing_type', 'cat_Age']
for col in onehot_cols:
    df = oneHot(df, col)
print(f'Datatypes\n{df.dtypes}')

# decision based on observation of amount of occurences, scaling down 3+ kids into "3" group, and 5+ families into 5
# maybe change the type so it says 3+ instead of 3 for clarity

print(df['Children_count'].value_counts())
df.loc[df['Children_count'] >= 3, 'Children_count'] = 3

print(df['Family_count'].value_counts())
df.loc[df['Family_count'] >= 5, 'Family_count'] = 5

df['Family_count'] = df['Family_count'].astype(int)
print(df['Family_count'].value_counts())

print(df.head())
print(f'Datatypes\n{df.dtypes}')
print(f'Shape{df.shape}')
print(f'Missing data\n{df.isna().sum()}')

################################################### GRAPHS

# calculting WOE and IV

print(df.shape)

df_for_iv = df[['Car','Gender', 'Realty', 'Children_count', 'cat_Income', 'Education_type', 'num_cat_Employment_years',
'Work_phone', 'Phone', 'Email', 'Occupation', 'Family_count', 'Income_type',
'Family_status', 'Housing_type', 'cat_Age','target']]
#, 'Starting_monthStarting_month'
ivWoe(df_for_iv, 'target', show_woe=True)

X = df_for_iv.drop(columns = ['target'])
Y = df_for_iv('target')

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.25, random_state=1)
X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train)



# Working with https://www.kaggle.com/code/rikdifos/credit-card-approval-prediction-using-ml/notebook
# 23/09/2024

# todo: work on bucketing the Income_type. currently there are 5 , student and pensioner only have 10 records each . possibly merge with one of
# todo: the other two or drop

# iv woe further work
# test different bins for ages and salary