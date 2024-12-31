# Imports
import pandas as pd
import time
import pickle
import numpy as np
import missingno as msno
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from pathlib import Path
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler,LabelEncoder, OrdinalEncoder
scaler = StandardScaler()
import time

cat = 0

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
def oneHot(df, feature, rank = 0,drop = False):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest], axis=1, inplace=True)
    if drop:
        df.drop([feature], axis=1, inplace=True)
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


def fitModel(model, name,x, y, X_test, y_test,  adjustment = 0.5, show_matrix = True, show_roc = False,show_precision_recall = False):
    start_time = time.time()
    model.fit(x, y)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_proba_adj = (y_pred_proba > adjustment).astype(int)
    accuracy = accuracy_score(y_test, y_pred_proba_adj)
    print('Dokładność dla {} wynosi {:.5}'.format(name,accuracy))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_proba_adj)))
    conf_matrix = confusion_matrix(y_test, y_pred_proba_adj)
    print(pd.DataFrame(conf_matrix))
    print(f"Tablica kontyngencji dla {name}:\n", conf_matrix)
    class_report = classification_report(y_test, y_pred)
    print(f"\nRaport klasyfikacji dla {name}:\n", class_report)
    if show_matrix:
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True,fmt='d', cmap='Blues', cbar=True)
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
        plt.title(f'Krzywa charakterystyki odbiornika (ROC) dla modelu {name}')
        plt.legend(loc="lower right")
        plt.show()
    if show_precision_recall:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label='Krzywa precyzji i czułości')
        plt.xlabel('Czułość')
        plt.ylabel('Precyzja')
        plt.title(f'Krzywa precyzji i czułości dla modelu {name}')
        plt.show()
    print('-------Success-------')
    return accuracy,training_time

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

df['Work_phone'] = df['Work_phone'].astype(bool)
df['Phone'] = df['Phone'].astype(bool)
df['Email'] = df['Email'].astype(bool)
df['Family_count'] = df['Family_count'].astype(int)
df['Starting_month'] = df['Starting_month'].astype(int)
#df['Income'] = scaler.fit_transform(df[['Income']])
print('CHILDREN COUNT AND FAMILY ------------')
print(df['Children_count'].value_counts())
print(df['Family_count'].value_counts())

if (cat == 1):
    df = categorize(df, 'Income', 4,  ["low", "medium", "high", 'highest'], qcut=True, replace=True)
    df = categorize(df, 'Age', 3, ["young adult", "mature", "elder"], qcut=True, replace=False)
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
    df.loc[df['Children_count'] >= 3, 'Children_count'] = 3
    df.loc[df['Family_count'] >= 5, 'Family_count'] = 5


onehot_cols = ['Gender', 'Car', 'Realty', 'Income_type', 'Education_type', 'Housing_type', 'Occupation','Family_status']
for col in onehot_cols:
    df = oneHot(df, col)
print(f'Datatypes\n{df.dtypes}')

#df_encoded = df.copy()
#label_cols = ['Gender','Car','Realty']
#for col in label_cols:
#    le = LabelEncoder()
#    print(f"Unique values in {col}: {df_encoded[col].unique()}")
#    df_encoded[col] = le.fit_transform(df_encoded[col])
#    print(f"Unique values in {col}: {df_encoded[col].unique()}")
#df = df_encoded

#oe = OrdinalEncoder(categories=[['highest', 'high', 'medium', 'low', 'lowest']])
#df['num_cat_Employment_years'] = oe.fit_transform(df[['cat_Employment_years']]).astype(int)

###### GRAPHS

# calculting WOE and IV

print(df.shape)

df_for_iv = df.drop(columns = ['STATUS', 'ID'])

ivWoe(df_for_iv, 'target', show_woe=True)

# data for ML
X = df.drop(columns = ['target', 'STATUS', 'ID','Gender', 'Car', 'Realty', 'Income_type', 'Education_type', 'Housing_type', 'Occupation','Family_status'])

print(X.dtypes)
Y = df['target']
print('-------------------------------------')
print('-------------------------------------')
print('-------------------------------------')
# Modeling

print(X.shape)

Y = Y.astype('int')
X_balance,Y_balance = SMOTE().fit_resample(X,Y)
X_balance = pd.DataFrame(X_balance, columns = X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_balance,Y_balance,
                                                    stratify=Y_balance, test_size=0.3,
                                                    random_state = 10086)

scores = []
# LogisticRegression

modelReg = LogisticRegression(solver='liblinear', random_state=1, class_weight='balanced',C=0.1)
LR_accuracy, LR_time = fitModel(modelReg,'Regresja logistyczna', X_train, y_train,X_test, y_test, 0.26)#, show_roc=True,show_precision_recall=True)
scores.append(('Regresja logistyczna',LR_accuracy,LR_time))
feature_coef = pd.Series(modelReg.coef_[0], index=X_train.columns).abs().sort_values(ascending=False)
print('Coefficients for Logistic Regression')
print(feature_coef)

# decision tree
modelDTC = DecisionTreeClassifier(max_depth=15,
                         min_samples_split=8,
                               random_state=1)
DTC_accuracy, DTC_time = fitModel(modelDTC,'Drzewo decyzyjne',X_train, y_train,X_test, y_test, 0.21)#, show_roc=True,show_precision_recall=True)
scores.append(('Drzewo decyzyjne',DTC_accuracy,DTC_time))

# inspecting importances values for DecisionTree
importancesDTC = modelDTC.feature_importances_
feature_names = X_train.columns
print('Importances for DTC')
print(sorted(zip(importancesDTC, feature_names), reverse=True))

# random forest
modelRFC = RandomForestClassifier(n_estimators=250,
                              max_depth=10,
                              min_samples_leaf=16)
RFC_accuracy, RFC_time = fitModel(modelRFC,'Las losowy',X_train, y_train,X_test, y_test )#,show_roc=True,show_precision_recall=True)
scores.append(('Las losowy',RFC_accuracy,RFC_time))

importancesRFC = modelRFC.feature_importances_
feature_names = X_train.columns
print('Importances for RFC')
print(sorted(zip(importancesRFC, feature_names), reverse=True))

# SVM
#modelSVM = svm.SVC(C = 0.001, kernel='linear', probability=True)
#SVM_accuracy , SVM_time = fitModel(modelSVM,'Maszyna wektorów nośnych',X_train, y_train,X_test, y_test, show_roc=True,show_precision_recall=True)
#scores.append(('Maszyna wektorów nośnych',SVM_accuracy,SVM_time))
#feature_coef_svm = pd.Series(modelSVM.coef_[0], index=X_train.columns).abs().sort_values(ascending=False)

#print('Coefficients for SVM (absolute values, sorted):')
#print(feature_coef_svm)


#joblib.dump(modelDTC,'Credit_model_DTC.pkl')


# PyTorch implementation

import torch
import torch.nn as nn
import torch.optim as optim


class TimeAccuracyTracker:
    def __init__(self, max_duration=10, interval=0.5):
        self.max_duration = max_duration
        self.interval = interval
        self.times = []
        self.accuracies = []

    def track(self, epoch, accuracy):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= len(self.times) * self.interval:
            self.times.append(elapsed_time)
            self.accuracies.append(accuracy)

    def start(self):
        self.start_time = time.time()

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
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
epochs = 5000
max_training_time = 10
time_tracker = TimeAccuracyTracker(max_duration=10, interval=0.5)
time_tracker.start()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    elapsed_time = time.time() - start_time
    if elapsed_time > max_training_time:
        print(f"Stopping training after {elapsed_time} seconds at epoch {epoch + 1}.")
        scores.append(('Sieci neuronowe PyTorch',accuracy,elapsed_time))
        break

    with torch.no_grad():
        outputs_labels = (outputs >= 0.5).float()
        accuracy = (outputs_labels == y_train_tensor).float().mean().item()

    time_tracker.track(epoch, accuracy)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# Evaluate the model

model.eval()

# Saving the state dictionary
torch.save(model.state_dict(), 'DjangoCreditApproval/credit_card_model.pth')
#torch.save(model, 'DjangoCreditApproval/predictor/credit_card_model.pth')
#print("Model saved to credit_card_model.pth.")

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_proba = y_pred.numpy()
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
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['0', '1'],
                yticklabels=['0', '1'])

    plt.title(f"Macierz pomyłek dla sieci neuronowych biblioteki Pytorch")
    plt.ylabel("Wartość rzeczywista")
    plt.xlabel("Wartość przewidywana")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test_np, y_pred_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label='Krzywa precyzji i czułości')
    plt.xlabel('Czułość')
    plt.ylabel('Precyzja')
    plt.title('Krzywa precyzji i czułości dla sieci neuronowych biblioteki Pytorch')
    plt.show()
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Wskażnik fałszywych pozytywów')
    plt.ylabel('Wskażnik prawdziwych pozytywów')
    plt.title('Krzywa charakterystyki odbiornika (ROC) dla sieci neuronowych biblioteki Pytorch')
    plt.legend(loc="lower right")
    plt.show()





# Tensorflow

import tensorflow as tf

# Custom callback to track time and accuracy every 0.5 seconds
class TimeAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_duration=10, interval=0.5):
        super(TimeAccuracyCallback, self).__init__()
        self.max_duration = max_duration
        self.interval = interval
        self.times = []
        self.accuracies = []

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        # Store accuracy and time every interval seconds
        if elapsed_time >= len(self.times) * self.interval:
            self.times.append(elapsed_time)
            self.accuracies.append(logs['accuracy'])
        if elapsed_time > self.max_duration:
            scores.append(('Sieci neuronowe TensorFlow',logs['accuracy'],elapsed_time))
            self.model.stop_training = True
def CreditCardTensor():

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # Checking the learning time to better compare with pytorch
    time_callback = TimeAccuracyCallback(max_duration=10, interval=0.5)
    start_time = time.time()

    model.fit(X_train_tensor, y_train_tensor, epochs=5000, batch_size=32, verbose=0, callbacks = [time_callback])

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    # Evaluate the model
    start_time = time.time()

    y_pred = model.predict(X_test_tensor)
    y_pred_labels = (y_pred >= 0.5).astype(int)

    end_time = time.time()
    prediction_time = end_time - start_time
    print(f"Prediction time: {prediction_time:.2f} seconds")
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test_tensor.numpy(), y_pred_labels)
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

    # Normalize the confusion matrix for plotting
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plotting confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['0', '1'],
                yticklabels=['0', '1'])

    plt.title(f"Macierz pomyłek dla sieci neuronowych biblioteki TensorFlow")
    plt.ylabel("Wartość rzeczywista")
    plt.xlabel("Wartość przewidywana")

    plt.show()

    # Calculate precision, recall, and thresholds for Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test_tensor.numpy(), y_pred)

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Krzywa precyzji i czułości')
    plt.xlabel('Czułość')
    plt.ylabel('Precyzja')
    plt.title('Krzywa precyzji i czułości dla sieci neuronowych biblioteki TensorFlow')
    plt.show()
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Wskażnik fałszywych pozytywów')
    plt.ylabel('Wskażnik prawdziwych pozytywów')
    plt.title('Krzywa charakterystyki odbiornika (ROC) dla sieci neuronowych biblioteki TensorFlow')
    plt.legend(loc="lower right")
    plt.show()
    return time_callback

time_callback = CreditCardTensor()

# plot time/acc

plt.figure(figsize=(8, 6))
plt.plot(time_tracker.times, time_tracker.accuracies, marker='o', linestyle='-', color='r', label='Pytorch')
plt.plot(time_callback.times, time_callback.accuracies, marker='o', linestyle='-', color='b' , label='Tensor')
plt.title('Accuracy vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# all models learning/timew
model_names, accuracies, times = zip(*scores)

fig, ax1 = plt.subplots(figsize=(10, 6))
bars_accuracy = ax1.bar(model_names, accuracies, color='royalblue', label='Celność (%)')
ax2 = ax1.twinx()
ax2.plot(model_names, times, color='red', marker='o', linestyle='-', label='Czas uczenia (s)')
ax1.set_xlabel('Czas uczenia (s)', fontsize=12)
ax1.set_ylabel('Celność (%)', fontsize=12)

# range for the (accuracy) between 0 and 1
ax1.set_ylim(0, 1)

for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.02, f'{v:.2f}', color='black', ha='center', va='center', fontsize=10, fontweight='bold')

for i, v in enumerate(times):
    ax2.text(i, v + 0.2, f'{v:.2f}', color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Porównanie modeli: Celność i czas uczenia', fontsize=14)
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9), frameon=False)
plt.tight_layout()

plt.show()
