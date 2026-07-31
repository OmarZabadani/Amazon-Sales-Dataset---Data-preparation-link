import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("/content/amazon.csv")


print("Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)

print(data.head())


print("\nDataset Information:")
data.info()

print("\nDuplicates Before:")
print(data.duplicated().sum())


# Remove duplicates

data = data.drop_duplicates()


print("\nDuplicates After:")
print(data.duplicated().sum())


print("\nMissing Values:")
print(data.isnull().sum())


# Remove missing values

data = data.dropna()


print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Discounted Price

data["discounted_price"] = (
    data["discounted_price"]
    .str.replace("₹", "")
    .str.replace(",", "")
    .astype(float)
)


# Actual Price

data["actual_price"] = (
    data["actual_price"]
    .str.replace("₹", "")
    .str.replace(",", "")
    .astype(float)
)


# Discount Percentage

data["discount_percentage"] = (
    data["discount_percentage"]
    .str.replace("%", "")
    .astype(float)
)


# Rating

data["rating"] = pd.to_numeric(
    data["rating"],
    errors="coerce"
)


# Rating Count

data["rating_count"] = (
    data["rating_count"]
    .str.replace(",", "")
)

data["rating_count"] = pd.to_numeric(
    data["rating_count"],
    errors="coerce"
)


# Fill any new missing numerical values

for col in [
    "rating",
    "rating_count"
]:
    data[col].fillna(
        data[col].median(),
        inplace=True
    )

ml_data = data[
    [
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "rating",
        "rating_count",
        "category"
    ]
].copy()


print("\nSelected Features:")
print(ml_data.head())

encoder = LabelEncoder()

ml_data["category"] = encoder.fit_transform(
    ml_data["category"]
)


print("\nAfter Encoding:")
print(ml_data.head())

scaler = MinMaxScaler()


scaled_data = scaler.fit_transform(
    ml_data
)


scaled_data = pd.DataFrame(
    scaled_data,
    columns=ml_data.columns
)
print("\nAfter Normalization:")
print(scaled_data.head())


data["rating_class"] = pd.cut(
    data["rating"],
    bins=[0,3,4,5],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)
X = scaled_data
y = data["rating_class"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


print("\nPreprocessing Completed Successfully!")
