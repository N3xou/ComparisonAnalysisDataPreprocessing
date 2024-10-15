import pandas as pd
import numpy as np

df = pd.read_csv('application_record.csv')
df2 = pd.read_csv('credit_record.csv')

def augment_data(df, num_copies=100):
    new_data = []

    for i in range(num_copies):
        df_copy = df.copy()

        for column in df_copy.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, 0.05, size=df_copy[column].shape)
            df_copy[column] += noise


        for column in df_copy.select_dtypes(include=[object]).columns:
            unique_vals = df_copy[column].unique()
            mask = np.random.rand(len(df_copy)) > 0.9
            random_values = np.random.choice(unique_vals, size=mask.sum())
            df_copy.loc[mask, column] = random_values

        new_data.append(df_copy)
        print(i)

    augmented_df = pd.concat(new_data, ignore_index=True)
    return augmented_df


#augmented_df = augment_data(df, num_copies=5)
augmented_df2 = augment_data(df2, num_copies=5)

#augmented_df.to_csv('application_record_augmented.csv', index=False)
augmented_df2.to_csv('credit_record_augmented.csv', index=False)
#print("Nowy zbiór danych ma rozmiar:", augmented_df.shape)
print("Nowy zbiór danych ma rozmiar:", augmented_df2.shape)
