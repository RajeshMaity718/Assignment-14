import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv("dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==================================================
# PART 1 : FEATURE ENGINEERING
# ==================================================

# ==================================================
# Task 1 : Creating New Features
# ==================================================

# Example Feature Engineering
# Modify according to your dataset columns

if 'Quantity' in df.columns and 'Price' in df.columns:

    df['Total_Value'] = (
        df['Quantity'] * df['Price']
    )

    df['Price_Per_Unit'] = (
        df['Price'] / df['Quantity']
    )

print("\nFeature Engineered Dataset:")
print(df.head())

# ==================================================
# Task 2 : Date & Text Features
# ==================================================

# Date Feature Extraction

if 'Date' in df.columns:

    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    print("\nDate Features Created")

else:
    print("\nNo Date Column Available")

# Text Feature Extraction

text_columns = df.select_dtypes(
    include='object'
).columns

if len(text_columns) > 0:

    text_col = text_columns[0]

    df[f'{text_col}_length'] = (
        df[text_col]
        .astype(str)
        .apply(len)
    )

    print("\nText Length Feature Created")

else:
    print("\nNo Text Column Available")

# ==================================================
# PART 2 : FEATURE ENCODING
# ==================================================

# ==================================================
# Task 3 : One-Hot Encoding
# ==================================================

categorical_columns = df.select_dtypes(
    include='object'
).columns

encoded_df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

print("\nOne Hot Encoded Data:")
print(encoded_df.head())

# ==================================================
# Task 4 : Column Transformer
# ==================================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include='object'
).columns.tolist()

# Choose Target Column
target_column = numerical_columns[-1]

if target_column in numerical_columns:
    numerical_columns.remove(target_column)

X = df.drop(columns=[target_column])

y = df[target_column]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            categorical_columns
        ),
        (
            'num',
            'passthrough',
            numerical_columns
        )
    ]
)

X_transformed = preprocessor.fit_transform(X)

print("\nColumn Transformer Applied")
print(X_transformed.shape)

# ==================================================
# PART 3 : FEATURE SCALING
# ==================================================

# ==================================================
# Task 5 : StandardScaler
# ==================================================

scaler = StandardScaler()

scaled_standard = scaler.fit_transform(
    X[numerical_columns]
)

print("\nStandard Scaled Data:")
print(scaled_standard[:5])

print("""
Explanation:
Mean becomes approximately 0
Standard deviation becomes approximately 1
""")

# ==================================================
# Task 6 : MinMaxScaler
# ==================================================

minmax_scaler = MinMaxScaler()

scaled_minmax = minmax_scaler.fit_transform(
    X[numerical_columns]
)

print("\nMinMax Scaled Data:")
print(scaled_minmax[:5])

print("""
Explanation:
All values are scaled between 0 and 1
""")

# ==================================================
# PART 4 : BUILDING ML PIPELINE
# ==================================================

# ==================================================
# Task 7 : Preprocessing Pipeline
# ==================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            'scaler',
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            'encoder',
            OneHotEncoder(
                handle_unknown='ignore'
            )
        )
    ]
)

preprocessor_pipeline = ColumnTransformer(
    transformers=[
        (
            'num',
            numeric_pipeline,
            numerical_columns
        ),
        (
            'cat',
            categorical_pipeline,
            categorical_columns
        )
    ]
)

print("\nPreprocessing Pipeline Created")

# ==================================================
# Task 8 : Full Scikit-Learn Pipeline
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

full_pipeline = Pipeline(
    steps=[
        (
            'preprocessing',
            preprocessor_pipeline
        ),
        (
            'model',
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

full_pipeline.fit(
    X_train,
    y_train
)

predictions = full_pipeline.predict(
    X_test
)

print("\nPredictions:")
print(predictions[:10])

# ==================================================
# Task 9 : Pipeline Benefits
# ==================================================

print("")

#1. Why Pipelines Are Important?

#Pipelines automate preprocessing and model
#training steps in a single workflow.

-----------------------------------------------

#2. What Problems Do Pipelines Solve?

#Prevents data leakage.

#Reduces manual work.

#Makes code reusable.

#Ensures same preprocessing on train and test data.

------------------------------------------------

#3. Manual Preprocessing vs Pipeline

#Manual:
#- More coding
#- Error prone
#- Difficult to maintain

#Pipeline:
#- Automated
#- Reusable
#- Cleaner code
#- Easy deployment

------------------------------------------------
