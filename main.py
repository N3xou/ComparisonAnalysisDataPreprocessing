# Imports
import pandas as pd
import numpy as np
import missingno as msno
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from pathlib import Path
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

import joblib

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# Creating dataframe, merging two dataframes into one on ID
path = Path(r'C:\Users\Yami\PycharmProjects\PracaInz')
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

def categorize(df, col, binsnum, labels, qcut = False, replace = True):
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


def fitModel(model, name,x, y,  adjustment = 0.3, show_matrix = True, show_roc = False,show_precision_recall = False):

    model.fit(x,y)
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


print("Sample data:\n", df.head())


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

df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])

df.loc[(df['DAYS_EMPLOYED'] > 0), 'DAYS_EMPLOYED'] = 0
df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])

print(df['Starting_month'].unique())
df['Starting_month'] = abs(df['Starting_month'])
# buckets

df['Income'] = scaler.fit_transform(df[['Income']])
df = categorize(df, 'Income', 4,  ["low", "medium", "high", 'highest'], qcut=True, replace=True)

df.loc[df['Children_count'] >= 3, 'Children_count'] = 3
df.loc[df['Family_count'] >= 5, 'Family_count'] = 5

onehot_cols = ['Gender', 'Car', 'Realty', 'Income_type', 'Education_type', 'Housing_type', 'Occupation','Family_status', 'Income']
for col in onehot_cols:
    df = oneHot(df, col)
print(f'Datatypes\n{df.dtypes}')

###### GRAPHS

# calculting WOE and IV

print(df.shape)

df_for_iv = df.drop(columns = ['STATUS', 'ID'])

ivWoe(df_for_iv, 'target', show_woe=True)

# data for ML
X = df.drop(columns = ['target', 'STATUS', 'ID','Gender', 'Car', 'Realty', 'Income_type', 'Education_type', 'Housing_type', 'Occupation','Family_status',
                       'Income'])

print(X.dtypes)
Y = df['target']

# Modeling

print(X.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.25, random_state=1)


#X_train_smote, y_train_smote = SMOTE(random_state=1 ).fit_resample(X_train, y_train)

Y = Y.astype('int')
X_balance,Y_balance = SMOTE().fit_resample(X,Y)
X_balance = pd.DataFrame(X_balance, columns = X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_balance,Y_balance,
                                                    stratify=Y_balance, test_size=0.3,
                                                    random_state = 10086)
# LogisticRegression

modelReg = LogisticRegression(solver='liblinear', random_state=1, class_weight='balanced',C=0.1)
fitModel(modelReg,'Regresja Logistyczna', X_train, y_train, 0.26, show_roc=True,show_precision_recall=True)

feature_coef = pd.Series(modelReg.coef_[0], index=X_train.columns).abs().sort_values(ascending=False)
print('Coefficients for Logistic Regression')
print(feature_coef)

# decision tree
modelDTC = DecisionTreeClassifier(max_depth=15,
                         min_samples_split=8,
                               random_state=1)
fitModel(modelDTC,'Drzewo decyzyjne',X_train, y_train, 0.21, show_roc=True,show_precision_recall=True)


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

fitModel(modelRFC,'Las losowy',X_train, y_train, show_roc=True,show_precision_recall=True)


importancesRFC = modelRFC.feature_importances_
feature_names = X_train.columns
print('Importances for RFC')
print(sorted(zip(importancesRFC, feature_names), reverse=True))

# SVM


#modelSVM = svm.SVC(C = 0.8, kernel='linear', probability=True)
#fitModel(modelSVM,'Maszyna wektorów nośnych',X_train, y_train, show_roc=True,show_precision_recall=True)

#feature_coef_svm = pd.Series(modelSVM.coef_[0], index=X_train.columns).abs().sort_values(ascending=False)

#print('Coefficients for SVM (absolute values, sorted):')
#print(feature_coef_svm)


#joblib.dump(modelDTC,'Credit_model_DTC.pkl')


# PyTorch implementation

import torch
import torch.nn as nn
import torch.optim as optim


class CreditCardTorch(nn.Module):
    def __init__(self,dim):
        super(CreditCardTorch, self).__init__()
        self.fc1 = nn.Linear(dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


model = CreditCardTorch(X_train.shape[1])

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_labels = (y_pred >= 0.5).float()  # Threshold at 0.5

    # Convert tensors to numpy arrays for confusion matrix
    y_test_np = y_test_tensor.numpy()
    y_pred_np = y_pred_labels.numpy()

    # Generate the confusion matrix

    conf_matrix = confusion_matrix(y_test_np, y_pred_np)
    tn, fp, fn, tp = conf_matrix.ravel()
    print("Confusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    # Calculate accuracy, precision, recall, F1-score
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])

modelSVM = svm.SVC(C = 0.8, kernel='linear', probability=True)
fitModel(modelSVM,'Maszyna wektorów nośnych', show_roc=True,show_precision_recall=True)
fitModel(modelSVM,'Maszyna wektorów nośnych',X_train,y_train, show_roc=True,show_precision_recall=True)


    # Set titles and labels
    plt.title(f"Macierz pomyłek dla modelu biblioteki Pytorch")
    plt.ylabel("Wartość rzeczywista")
    plt.xlabel("Wartość przewidywana")

    # Show the heatmap
    plt.show()


