# Cardiff House Price Prediction

This project was developed as part of an MSc dissertation and focuses on predicting residential property prices in Cardiff using machine learning.

The project combines property transaction data with structural, spatial and temporal features to produce house price estimates. Two main machine-learning models are explored: **Random Forest** and **XGBoost**.

## Project Features

The prediction models use information including:

* Property type
* Postcode and location
* Latitude and longitude
* Property floor area
* Number of habitable rooms
* Distance to nearby schools
* Distance to public transport
* Distance to greenspace
* Crime levels within the surrounding area
* Flood-risk indicators
* Housing market changes over time using the UK House Price Index (UK HPI)

Two model configurations are developed:

* **Model A** – Uses transaction, location, spatial and temporal features without EPC property characteristics.
* **Model B** – Extends Model A by including EPC-derived floor area and number of habitable rooms.

## Machine Learning

The project compares tree-based machine-learning approaches including:

* Random Forest
* XGBoost

Model performance is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R²
* Cross-validation

SHAP and permutation importance are also used to examine how different features influence model predictions.

## Prediction Tool

The trained model can be used through `prediction.py`.

Run the script from the terminal:

```bash
python prediction.py
```

The user is asked to enter property information such as postcode, prediction date and property type.

The script then:

1. Finds the property location.
2. Calculates or retrieves the required spatial features.
3. Prepares the input for the trained model.
4. Predicts the property value.
5. Adjusts the prediction using the relevant House Price Index.
6. Displays the estimated property price and contributing factors.

## Data Sources

The project uses publicly available UK datasets, including:

* HM Land Registry Price Paid Data
* UK House Price Index
* Energy Performance Certificate data
* Geographic postcode data
* Crime data
* Transport, school and greenspace datasets
* Flood-risk spatial datasets

## Purpose

The aim of the project is not only to predict property prices, but also to provide a more transparent explanation of the factors considered by the model when producing a prediction.
