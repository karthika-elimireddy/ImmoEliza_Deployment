import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import category_encoders as ce
import joblib
from price_model import PriceModel


df = pd.read_csv("data/dataset.csv")
df["locality"] = df["locality"].fillna("Unknown").astype(str)
localities = sorted(df["locality"].str.lower().unique())
buildingstates = [
    "AS_NEW",
    "JUST_RENOVATED",
    "GOOD",
    "TO_BE_DONE_UP",
    "TO_RENOVATE",
    "TO_RESTORE",
]

# Mapping building states to numerical values
state_to_value = {
    "AS_NEW": 6,
    "JUST_RENOVATED": 5,
    "GOOD": 4,
    "TO_BE_DONE_UP": 3,
    "TO_RENOVATE": 2,
    "TO_RESTORE": 1,
}
years = list(range(2024, 1929, -1))
# Streamlit UI
st.title("Property Price Estimation")

# Form for user input
with st.form(key="prediction_form"):
    property_type = st.radio(
        "Property Type",
        options=[1, 0],
        format_func=lambda x: "HOUSE" if x == 1 else "APARTMENT",
    )
    locality = st.selectbox("Select Locality", options=localities)
    mobib_score = st.number_input(
        "Mobility Score", min_value=0.0, max_value=10.0, step=0.1
    )
    livingArea = st.number_input("Living Area (sqm)", min_value=0.0, step=1.0)
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
    toilets = st.number_input("Number of Toilets", min_value=0, step=1)
    facades = st.number_input("Number of Facades", min_value=0, step=1)
    fireplace = st.radio(
        "Fireplace Available",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    pool = st.radio(
        "Swimming Pool Available",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    buildingState = st.selectbox("Building Condition", options=buildingstates)
    buildingState = state_to_value[buildingState]
    constructionYear = st.selectbox("Year of Construction", options=years)
    cadastralIncome = st.number_input("Cadastral Income", min_value=0.0, step=1.0)
    submit_button = st.form_submit_button(label="Predict Price")

# Prediction
if submit_button:
    try:
        # input data
        input_data = {
            "locality": locality,
            "mobib_score": mobib_score,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "cadastralIncome": cadastralIncome,
            "livingArea": livingArea,
            "buildingState": buildingState,
            "constructionYear": constructionYear,
            "facades": facades,
            "fireplace": fireplace,
            "toilets": toilets,
            "pool": pool,
        }

        # Creating the instance of the model.
        model = PriceModel()
        # preprocess the input data
        preprocessed_data = model.preprocess_data(input_data)
        # Predict price
        predicted_price = model.predict(preprocessed_data)
        # Display the result
        st.success(f"The estimated price of the property is: €{predicted_price:,.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
