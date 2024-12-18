import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import category_encoders as ce
import joblib
from price_model import PriceModel
from PIL import Image

df = pd.read_csv("data/dataset.csv")
df["locality"] = df["locality"].fillna("Unknown").astype(str)
localities=["Select Locality"]
localities.extend(sorted(df["locality"].str.lower().unique()))
buildingstates = ["Select",
    "AS_NEW",
    "JUST_RENOVATED",
    "GOOD",
    "TO_BE_DONE_UP",
    "TO_RENOVATE",
    "TO_RESTORE",
]

# Mapping building states to numerical values
state_to_value = {
    "Select":0,
    "AS_NEW": 6,
    "JUST_RENOVATED": 5,
    "GOOD": 4,
    "TO_BE_DONE_UP": 3,
    "TO_RENOVATE": 2,
    "TO_RESTORE": 1,
}
years = list(range(2024, 1929, -1))
# Initialize session state to track the page
if 'page' not in st.session_state:
    st.session_state.clear()
    st.session_state.page = 1

# Streamlit UI
image = Image.open("logo/logo.jpg")
resized_image = image.resize((100, 90))
st.image(resized_image, use_container_width=False)

# Change the color of the header
st.markdown("<h2 style='color: orange;'>Immo Eliza Property Price Estimation</h2>", unsafe_allow_html=True)
with st.form(key='prediction_form'):
        st.markdown("<h2 style='color: black;'>Property Information</h2>",unsafe_allow_html=True)
        st.markdown("<p style='color: black;'>Fields marked with * are required.</p>",unsafe_allow_html=True)
        locality = st.selectbox("Select Locality *", options=localities)
        property_type = st.radio(
            "Property Type",
            options=[1, 0],
            format_func=lambda x: "HOUSE" if x == 1 else "APARTMENT"
        )
        #st.markdown("<h2 style='color: black;'>Additional Information</h2>",unsafe_allow_html=True)
        livingArea = st.number_input("Living Area (sqm) *", min_value=0.0, step=1.0,help="ℹ️ The living area refers to the part of a property that is habitable and used for day-to-day living. It is the portion of the home that is enclosed and usable for living purposes")
        bedrooms = st.number_input("Number of Bedrooms *", min_value=0, step=1)
        bathrooms = st.number_input("Number of Bathrooms *", min_value=0, step=1)
        toilets = st.number_input("Number of Toilets *", min_value=0, step=1)
        fireplace = st.radio(
            "Fireplace Available",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        facades = st.number_input("Number of Facades *", min_value=0, step=1)
        buildingState = st.selectbox("Building Condition *", options=buildingstates)
        buildingState = state_to_value[buildingState]
        constructionYear = st.selectbox("Year of Construction", options=years)
        st.markdown("<h2 style='color: black;'>Additional Information (Optional)</h2>",unsafe_allow_html=True)
        # pool = st.radio(
        #     "Swimming Pool Available",
        #     options=[0, 1],
        #     format_func=lambda x: "Yes" if x == 1 else "No",
        # )
        mobib_score = st.number_input(
            "Mobility Score ", min_value=0.0, max_value=10.0, step=0.1,help="ℹ️ Choose a mobib score number between 0,0 and 10,0."
        )
        
        cadastralIncome = st.number_input("Cadastral Income", min_value=0.0, step=1.0,help="ℹ️ This income can include rent, leasing income, or any other financial benefit derived from the ownership or use of the property as determined by its cadastral assessment.")
        estimate_button = st.form_submit_button(label="Estimate Price")
        
        if estimate_button:
            if locality == 'Select Locality' and facades <=0 and livingArea < 20 and  bedrooms <= 0 and toilets <= 0 and buildingState == 0:
                st.error("Please fill in all the required fields")
            elif locality == 'Select Locality':
                st.error("Please select Locality.")
            elif livingArea < 20:
                st.error("Please specify a valid living area (minimum 20sqm).")
            elif bedrooms <= 0:
                st.error("Please specify a valid number of bedrooms.")
            elif bathrooms <= 0:
                st.error("Please specify a valid number of bathrooms.")
            elif toilets <= 0:
                st.error("Please specify a valid number of toilets.")
            elif facades <= 0:
                st.error("Please specify the number of facades.")
            elif buildingState == 0:
                st.error("Please specify the state of the building.")
            else:
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
                        "pool":0
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

   
        
       
        
        
        
        
    

