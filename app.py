import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("laptop_data_cleaned.csv")

# Load trained model and columns
model = pickle.load(open("laptop_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("💻 AI Laptop Advisor")

st.write("Enter laptop specifications to estimate price and get recommendations.")

# ---------------- USER INPUT ----------------

company = st.selectbox("Company", ["Apple","Asus","Dell","HP","Lenovo"])

type_name = st.selectbox(
    "Laptop Type",
    ["Gaming","Notebook","Ultrabook","Workstation"]
)

ram = st.slider("RAM (GB)", 4, 64, 8)

weight = st.slider("Weight (kg)", 1.0, 3.5, 2.0)

touchscreen = st.selectbox("Touchscreen", ["No","Yes"])

ips = st.selectbox("IPS Display", ["No","Yes"])

ssd = st.selectbox("SSD (GB)", [0,128,256,512,1024])

cpu = st.selectbox(
    "CPU Brand",
    ["Intel Core i3","Intel Core i5","Intel Core i7"]
)

gpu = st.selectbox("GPU Brand", ["Intel","Nvidia"])

os = st.selectbox("Operating System", ["Windows","Others"])

# ---------------- PREDICTION ----------------

if st.button("Predict Price"):

    # Create empty input dataframe
    data = pd.DataFrame([[0]*len(columns)], columns=columns)

    # Numerical inputs
    data["Ram"] = ram
    data["Weight"] = weight
    data["SSD"] = ssd
    data["TouchScreen"] = 1 if touchscreen == "Yes" else 0
    data["Ips"] = 1 if ips == "Yes" else 0

    # Hidden/default values
    if "Ppi" in data.columns:
        data["Ppi"] = 140

    if "HDD" in data.columns:
        data["HDD"] = 0

    # One-hot encoding
    if "Company_" + company in data.columns:
        data["Company_" + company] = 1

    if "TypeName_" + type_name in data.columns:
        data["TypeName_" + type_name] = 1

    if "Cpu_brand_" + cpu in data.columns:
        data["Cpu_brand_" + cpu] = 1

    if "Gpu_brand_" + gpu in data.columns:
        data["Gpu_brand_" + gpu] = 1

    if "Os_" + os in data.columns:
        data["Os_" + os] = 1

    # -------- PRICE PREDICTION --------

    pred_price = np.exp(model.predict(data)[0])

    st.success(f"💰 Estimated Laptop Price: ₹{int(pred_price):,}")

    # -------- ADVISOR SYSTEM --------

    df_copy = df.copy()

    # Convert dataset log prices to real prices
    df_copy["Price"] = np.exp(df_copy["Price"])

    # Find laptops near predicted price
    budget_range = df_copy[
        (df_copy["Price"] >= pred_price * 0.7) &
        (df_copy["Price"] <= pred_price * 1.3)
    ].copy()

    # If no laptops found → show closest ones
    if budget_range.empty:
        df_copy["price_diff"] = abs(df_copy["Price"] - pred_price)
        recommendations = df_copy.sort_values("price_diff").head(10)
    else:
        budget_range["price_diff"] = abs(budget_range["Price"] - pred_price)
        recommendations = budget_range.sort_values("price_diff").head(10)

    st.subheader(" : ) Recommended Laptops Near This Price")

    st.dataframe(
        recommendations[["Company","TypeName","Ram","Cpu_brand","SSD","Price"]]
    )