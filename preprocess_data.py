import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from joblib import dump

SEED = 0

# loading dataset 
df = pd.read_csv("dataset/210619monatszahlenjuni2021monatszahlen2106verkehrsunfaelle.csv")

# dropping last 3 columns
df = df.iloc[:, :-3]

# renaming columns from German to English
# for easy access
df.columns = [
    "Category",
    "Accident_type",
    "Year", 
    "Month",
    "Value",
    "Prev_Year_Value"
]

# removing records with monthly total
df = df.loc[df['Month'] != 'Summe']

# extracting data other than 2000
# Year 2000 values fed as Prev_Year_Value to 2001
df = df.loc[df['Year'] != 2000]

# extracting data of 2021 
df_2021 = df.loc[df['Year'] == 2021]

# extracting data other than 2021 
df = df.loc[df['Year'] != 2021]

# sorting data wrt Year and Month 
df = df.sort_values(by=['Year', 'Month'])

# removing Year and Month column
df.drop(labels=['Year', 'Month'], axis=1, inplace=True);

# printing dataset stats
print("DATASET STATS:")
print(f"\t# of samples: {df.shape[0]}")

null_sample = df[df.isnull().any(axis=1)]
print(f"\t# of samples containing null: {null_sample.shape[0]}")
print() # \n
# seperating dataset features and labels
X = df[["Category", "Accident_type", "Prev_Year_Value"]]
Y = df["Value"]

# # splitting to train set - 80% train set
X_train_orig, X_valid_test, Y_train_orig, Y_valid_test= train_test_split(
        X.values, Y.values, 
        test_size=0.2, 
        shuffle=False, 
        random_state=SEED
    )

# # further splitting valid_test into seperate validation and test sets 
# # 50% each
X_valid_orig, X_test_orig, Y_valid_orig, Y_test_orig = train_test_split(
        X_valid_test, Y_valid_test,
        test_size=0.5, 
        shuffle=False, 
        random_state=None
    )

# reshaping label vectors 
Y_train_orig = Y_train_orig.reshape(-1, 1)
Y_valid_orig = Y_valid_orig.reshape(-1, 1)
Y_test_orig = Y_test_orig.reshape(-1, 1)

# defining preprocessing tranforms
tranforms = [
    ('encoder', OneHotEncoder(), [0, 1]),
    ('scaler', MinMaxScaler(), [2]),
]

ft = ColumnTransformer(transformers=tranforms)  # feature transform
lt = MinMaxScaler()                             # label transform

# fitting tranforms to training data 
ft.fit(X_train_orig)
lt.fit(Y_train_orig)

# appplying tranforms to data splits
X_train = ft.transform(X_train_orig) 
X_valid = ft.transform(X_valid_orig) 
X_test = ft.transform(X_test_orig) 

Y_train = lt.transform(Y_train_orig)
Y_valid = lt.transform(Y_valid_orig)
Y_test = lt.transform(Y_test_orig)

# # printing shapes of final sets 
print("DATA SPLIT SHAPES:")
print(f"\tX_train: {X_train.shape} - Y_train: {Y_train.shape}")
print(f"\tY_valid: {X_valid.shape} - Y_valid: {Y_valid.shape}")
print(f"\tX_test: {X_test.shape} - Y_test: {Y_test.shape}")

# saving tranformed data and transforms
dump({
    "train": [X_train, Y_train],
    "valid": [X_valid, Y_valid],
    "test": [X_test, Y_test],
    "orig": Y_test_orig,
}, open('dataset/transformed_dataset.pkl', 'wb'))

dump(ft, open('assets/feature_tranform.pkl', 'wb'))
dump(lt, open('assets/label_tranform.pkl', 'wb'))

# saving df_2021 as a csv 
df_2021.to_csv('dataset/forecast_data.csv')