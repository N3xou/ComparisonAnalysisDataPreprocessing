# Imports
import pandas as pd
import numpy as np
import missingno as msno
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from pathlib import Path
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


# Creating dataframe, merging two dataframes into one on ID
path = Path(r'C:\Users\Yami\PycharmProjects\pythonProject1')
applicationRecord = pd.read_csv(path / 'application_record.csv')
creditRecord = pd.read_csv(path / 'credit_record.csv')

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
            localdf = pd.qcut(df[col], q=binsnum, labels=labels)
        else:
            localdf = pd.cut(df[col], bins=binsnum, labels=labels)
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

def fitModel(model, name, adjustment = 0.3, show_matrix = True, show_roc = False,show_precision_recall = False):
    model.fit(X_train_smote,y_train_smote)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_proba_adj = (y_pred_proba > adjustment).astype(int)
    print('Dokładność dla {} wynosi {:.5}'.format(name,accuracy_score(y_test, y_pred_proba_adj)))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_proba_adj)))
    conf_matrix = confusion_matrix(y_test, y_pred_proba_adj, normalize='true')
    print(pd.DataFrame(conf_matrix))
    print(f"Tablica kontyngencji dla {name}:\n", conf_matrix)
    class_report = classification_report(y_test, y_pred)
    print(f"\nRaport klasyfikacji dla {name}:\n", class_report)
    if show_matrix:
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True)
        plt.title(f"Macierz pomyłek dla modelu {name}")
        plt.ylabel("Wartość rzeczywista")
        plt.xlabel("Wartość przewidywana")
        plt.show()
    if show_roc:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{name} (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Wskaźnik fałszywych pozytywów')
        plt.ylabel('Wskaźnik prawdziwych pozytywów')
        plt.title('Krzywa charakterystyki odbiornika (ROC)')
        plt.legend(loc="lower right")
        plt.show()
    if show_precision_recall:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label='Krzywa precyzji i czułości')
        plt.xlabel('Czułość')
        plt.ylabel('Precyzja')
        plt.title('Krzywa precyzji i czułości')
        plt.show()
    print('-------Success-------')

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
print(df.shape)
print(cpunt.shape)

print(cpunt['dependency'].unique())

cpunt.loc[cpunt['dependency'] > 0, 'dependency'] = 1
cpunt = cpunt[['dependency']] #reducing cpunt to 1 column
df = pd.merge(df, cpunt, how='inner', on='ID')
df['target'] = df['dependency']
df = df.drop(columns=['dependency'])

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

msno.matrix(df,figsize=(18,9))
plt.title('Macierz brakujących danych', fontsize=14)
plt.xlabel('Kolumny', fontsize=12)
plt.ylabel('Dane', fontsize=12)
plt.show()

print(df['Mobile'].unique())

# dropping rows with missing data

for column in df.columns:
    dropped_count = df[column].isna().sum()
    if dropped_count > 0:
        print(f"Dropping {dropped_count} rows due to NaN in {column} column")
df = df.dropna()

print(f"Dropping duplicates, amount of unique rows: {df['ID'].nunique()}")
df.drop_duplicates('ID', keep='last')

# dropping Mobile because all values equal 1

df.drop(columns=['Mobile'], inplace=True)

# bucketing data

plt.figure()
print(df['Income'].unique())
df['Income'] = df['Income'].astype(object)
df['Income'] = df['Income']/10000
df['Income'].plot(kind='hist',bins=40)
plt.xlabel('Zarobki roczne podane w dziesiątkach tysięcy',fontsize='12')
plt.ylabel('Liczba klientów', fontsize='12')


df = getCategory(df, 'Income', 4,  ["low", "medium", "high", 'highest'], qcut=True, replace=False)
print(df['cat_Income'].value_counts())

print(df['DAYS_BIRTH'].unique())
df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
df['Age'] = (df['DAYS_BIRTH'] / 365.25).round(0).astype(int)

plt.figure()
df['Age'].plot(kind='hist',bins = 20)
plt.xlabel('Wiek',fontsize='12')
plt.ylabel('Liczba klientów', fontsize='12')
df = getCategory(df, 'Age', 3, ["young adult", "mature", "elder"], qcut = True, replace=False)

print("Lowest age per age group")
print(df.loc[df.groupby('cat_Age')['Age'].idxmin()][['cat_Age', 'Age']])
print("Highest age per age group")
print(df.loc[df.groupby('cat_Age')['Age'].idxmax()][['cat_Age', 'Age']])
print(df['cat_Age'].value_counts())

print(df['DAYS_EMPLOYED'].unique())
df.loc[(df['DAYS_EMPLOYED'] > 0), 'DAYS_EMPLOYED'] = 0
df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
df['Employment_years'] = df['DAYS_EMPLOYED'] / 365.25
plt.figure()
df['Employment_years'].plot(kind='hist',bins=30)
plt.xlabel('Ilość lat w pracy w obecnej firmie',fontsize='12')
plt.ylabel('Liczba klientów', fontsize='12')
plt.show()
df = getCategory(df, 'Employment_years', 5, ["lowest", "low", "medium", "high", "highest"], qcut = True, replace=False)
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

oe = OrdinalEncoder(categories=[['High position job', 'Medium position job', 'Low position job']])
df['Occupation'] = oe.fit_transform(df[['Occupation']]).astype(int)
print(df['Occupation'].unique())

oe = OrdinalEncoder(categories=[['Academic degree', 'Higher education', 'Incomplete higher', 'Secondary / secondary special', 'Lower secondary']])
df['Education_type'] = oe.fit_transform(df[['Education_type']]).astype(int)
print(df['Education_type'].unique())

oe = OrdinalEncoder(categories=[['highest', 'high', 'medium', 'low', 'lowest']])
df['num_cat_Income'] = oe.fit_transform(df[['cat_Income']]).astype(int)
print(df['num_cat_Income'].unique())

oe = OrdinalEncoder(categories=[['highest', 'high', 'medium', 'low', 'lowest']])
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

# one hot encoding

onehot_cols = ['Income_type','Family_status', 'Housing_type', 'cat_Age']
for col in onehot_cols:
    df = oneHot(df, col)
print(f'Datatypes\n{df.dtypes}')

# decision based on observation of amount of occurences, scaling down 3+ kids into "3" group, and 5+ families into 5

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

###### GRAPHS

# calculting WOE and IV

print(df.shape)

df_for_iv = df[['Car','Gender', 'Realty', 'Children_count', 'cat_Income', 'Education_type', 'num_cat_Employment_years',
'Work_phone', 'Phone', 'Email', 'Occupation', 'Family_count', 'Income_type',
'Family_status', 'Housing_type', 'cat_Age','target']]

ivWoe(df_for_iv, 'target', show_woe=True)

# data for ML
X = df.drop(columns = ['target', 'Employment_years', 'cat_Age', 'Age', 'STATUS', 'DAYS_EMPLOYED', 'DAYS_BIRTH',
                       'Housing_type', 'Family_status', 'Income_type', 'ID', 'Income', 'cat_Income','cat_Employment_years', 'Starting_month'])
print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
print(X.dtypes)
Y = df['target']

# Modeling

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.25, random_state=1)

X_train_smote, y_train_smote = SMOTE(random_state=1 ).fit_resample(X_train, y_train)

# LogisticRegression

modelReg = LogisticRegression(solver='liblinear', random_state=1, class_weight='balanced',C=0.1)
fitModel(modelReg,'Regresja Logistyczna',0.26, show_roc=True,show_precision_recall=True)
feature_coef = pd.Series(modelReg.coef_[0], index=X_train.columns).abs().sort_values(ascending=False)
print('Coefficients for Logistic Regression')
print(feature_coef)

# improving the model (GRID SEARCH DO NOT DELETE)

#param_grid = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}
#grid = GridSearchCV(LogisticRegression(random_state=1, max_iter=3000), param_grid, scoring='f1', cv =5 )
#grid.fit(X_train_smote, y_train_smote)
#best_params = grid.best_params_
#best_score = grid.best_score_
#print(f'Best parameters: {best_params}')
#print(f'Best score: {best_score}')

#y_grid_proba = grid.predict_proba(X_test)[:, 1]
#y_grid = (y_grid_proba > 0.5).astype(int)

#precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
#for i, val in enumerate(recall):
#    if val >= 0.1:  # or your desired recall level
#        print(f"Threshold: {thresholds[i]}, Precision: {precision[i]}")
# visualization

#conf_matrix2 = confusion_matrix(y_test, y_grid)
#conf_matrix_normalized2 = conf_matrix2.astype('float') / conf_matrix2.sum(axis=1)[:, np.newaxis]


# decision tree
modelDTC = DecisionTreeClassifier(max_depth=15,
                               min_samples_split=8,
                               random_state=1)
fitModel(modelDTC,'Drzewo decyzyjne', 0.21, show_roc=True,show_precision_recall=True)

# inspecting importances values for DecisionTree
importancesDTC = modelDTC.feature_importances_
feature_names = X_train.columns
print('Importances for DTC')
print(sorted(zip(importancesDTC, feature_names), reverse=True))

# random forest

modelRFC = RandomForestClassifier(n_estimators=250,
                              max_depth=10,
                              min_samples_leaf=16
                              )
fitModel(modelRFC,'Las losowy', show_roc=True,show_precision_recall=True)

importancesRFC = modelRFC.feature_importances_
feature_names = X_train.columns
print('Importances for RFC')
print(sorted(zip(importancesRFC, feature_names), reverse=True))

# SVM

modelSVM = svm.SVC(C = 0.8, kernel='linear')
fitModel(modelSVM,'Maszyna wektorów nośnych', show_roc=True,show_precision_recall=True)




# todo: feature importances  for svm,randomforest
# todo: model optimalization, accuracy/recall is too low

# todo: needs work - iv/woe values overall seem low
