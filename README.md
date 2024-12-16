# ImmoEliza_Deployment

## Table of Contents
- [Project Overview](#project_overview)
- [Prerequisites](#Prerequisites)
- [Usage](#Usage)
- [Structure](#Structure)
- [Contributors](#Contributors)

# Project Overview
- Predict the price of a property located in any locality in Belgium using a saved model from the previous project.

 
# Prerequisites
## Technologies
- Pandas
- streamlit
- pickle
- joblib
- Scikit-learn
- LinearRegression(ElasticNet)
- Python 3.11.3 

Make sure you have the following:
- requirements.txt --- install using the command pip install -r requirements.txt


# Usage

This script will:
1. Takes the property data from the user through the UI created using streamlit.
2. Perform Data preprocessing.
3. Predict the price the property.
4. Displays the Estimated price in the UI.

--- To execute the project: 
         streamlit run app.py

# Structure
The project has the following core components:

1. data: is a directory contains data files
    - dataset.csv
2. pickle: is a directory with pkl files
    - price_model.pkl
3. price_model: is a directory with py file
    - price_model.py
4. app.py  # prediction app file
5. requirements.txt  #contains list of dependencies for the project.
6. .gitignore

## App UserInterface
![Immo Eliza Property Price Estimation](https://immoeliza.streamlit.app)


## Contributors
![Karthika](https://github.com/karthika-elimireddy)
