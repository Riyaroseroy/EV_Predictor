import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Load & Prepare Data ---
@st.cache_data
def load_data():
    data = pd.read_csv("Electric_Vehicle_Population_Data.csv")
    
    # Fill missing categorical with mode
    cat_cols = ['County', 'City', 'Electric Utility', 'Postal Code', 'Legislative District', '2020 Census Tract', 'Vehicle Location']
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0])

    # Fill missing numeric with mean
    num_cols = ['Electric Range', 'Base MSRP']
    for col in num_cols:
        data[col] = data[col].fillna(data[col].mean())

    # Drop unwanted columns
    data.drop(columns=['VIN (1-10)', 'DOL Vehicle ID'], inplace=True, errors='ignore')

    # Encode categorical features
    label_cols = data.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in label_cols:
        data[col] = le.fit_transform(data[col])

    # Remove outliers
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]

    for col in ['Electric Range', 'Base MSRP']:
        data = remove_outliers_iqr(data, col)

    return data

data = load_data()

# Features and target
X = data.drop(columns=['Electric Range', '2020 Census Tract'], errors='ignore')
y = data['Electric Range']

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# --- Streamlit UI ---
st.title("🔋 EV Range Predictor")
st.write("Enter EV details below to estimate the electric range:")

user_input = {}
for col in X.columns:
    dtype = X[col].dtype
    if dtype == 'int64' or dtype == 'float64':
        val = st.number_input(f"{col}", value=float(X[col].median()))
    else:
        val = st.text_input(f"{col}", value=str(X[col].mode()[0]))
    user_input[col] = val

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Ensure all input types match training data
for col in input_df.columns:
    if X[col].dtype == 'int64':
        input_df[col] = input_df[col].astype(int)
    elif X[col].dtype == 'float64':
        input_df[col] = input_df[col].astype(float)

# Prediction
if st.button("Predict Range"):
    prediction = model.predict(input_df)[0]
    lower = prediction * 0.90
    upper = prediction * 1.10
    st.success(f"Predicted EV Range: {lower:.2f} – {upper:.2f} miles")
