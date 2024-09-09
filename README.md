# House-Price-Predictor

This project focuses on predicting house prices using a **Random Forest** model. The model is trained and evaluated using the Root Mean Squared Error (RMSE) metric.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The project implements a Random Forest model to predict house prices based on various features such as the number of rooms, area, and location.

## Installation
To run this project, you need to install the following dependencies:
pip install numpy pandas scikit-learn jupyter

## Dataset
The dataset contains various features of houses (e.g., size, number of rooms, location) and their corresponding prices. It is split into training and testing sets.

# Load dataset
data = pd.read_csv('data.csv')  


## Modeling
random_forest = RandomForestRegressor()
random_forest.fit(X_train_prepared, y_train)

## Evaluation
We use the Root Mean Squared Error (RMSE) to evaluate the model's performance.

## Results
After training and evaluating the Random Forest model, the following RMSE was obtained:

Random Forest RMSE: 3.004

## Contributing
Feel free to submit issues or pull requests for improvements.

This project is licensed under the MIT License - see the LICENSE file for details.
