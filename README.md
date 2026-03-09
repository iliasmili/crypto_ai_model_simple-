# Crypto AI Model (Simple Baseline)

This project is a simple machine learning experiment that attempts to predict whether the **Bitcoin price is more likely to move up or down in the next period** using historical market data.

The goal of this project is not to build a perfect trading model, but to demonstrate a **baseline machine learning pipeline for financial time series**.

## Project Overview

The model uses basic market features such as:

- Moving Average (MA5)
- Moving Average (MA10)
- Trading Volume

These features are used to train a **Logistic Regression classifier** that predicts the next price direction.

## Machine Learning Pipeline

The project follows a basic ML workflow:

1. **Data Collection**
   - Historical BTC data is downloaded using Yahoo Finance.

2. **Feature Engineering**
   - Moving averages are calculated.
   - Volume data is included as a market activity signal.

3. **Model Training**
   - A Logistic Regression model is trained using the engineered features.

4. **Evaluation**
   - The model is tested on unseen data to estimate predictive accuracy.

5. **Prediction**
   - The model outputs whether the next price movement is more likely **UP or DOWN**.

## Model Performance

The current model achieves accuracy close to **random (~50%)**.

This is expected when using only a few simple features in highly noisy financial markets such as cryptocurrency.

Rather than being a failure, this result highlights an important concept:

> Financial markets are difficult to predict without strong feature engineering and deeper modeling techniques.

## How the Model Can Be Improved

Several improvements can significantly enhance the model:

### Better Features
- RSI 
- MACD
- Bollinger Bands
- Momentum indicators
- Volatility metrics

### More Advanced Models
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Networks

### Better Validation
- Walk forward validation
- Time series cross-validation

### More Market Signals
- Order flow
- Market structure
- Multiple timeframe indicators

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- yFinance

## Project Structure

