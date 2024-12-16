import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
import category_encoders as ce
import pickle
from typing import Dict
from pandas import DataFrame
import joblib


class PriceModel:
    def __init__(self):
        """Initialize the prediction model"""
        self.model = ElasticNet(max_iter=10000, alpha=1.0, l1_ratio=1.0)

    def preprocess_data(self, input_data: DataFrame) -> DataFrame:
        """
        Preprocess the user data to feed the model to predict.
        Args:
        - input_data: DataFrame containing the input data.

        Returns:
        - input_data: Processed data.
        """

        # Load necessary model
        file_path = "pickle/price_model.pkl"
        data = joblib.load(file_path)
        self.model = data["model"]
        scaler = data["scaler"]
        pca = data["pca"]
        poly = data["poly"]
        target_Encoder = data["targetEncoder"]
        features = data["features"]

        # Preprocess input data
        input_data = pd.DataFrame([input_data])
        input_data["locality"] = target_Encoder.transform(input_data["locality"])
        input_data = scaler.transform(input_data)
        input_data = pca.transform(input_data)
        input_data = poly.transform(input_data)

        return input_data

    def predict(self, new_data: Dict) -> float:
        """
        Predict the price for a new property.
        Args:
        - new_data: A dictionary of feature values.
        Returns:
        - Predicted price.
        """

        predicted_price = self.model.predict(new_data)[0]

        return predicted_price
