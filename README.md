# Bitcoin Price Prediction using CNN-LSTM, Attention, GARCH, and Random Forest:
# Project Overview:

This project focuses on predicting the next-day closing price of Bitcoin (BTC/USD) using a hybrid deep learning and machine learning pipeline. The goal is to capture both time-series dependencies and market volatility for more accurate forecasts.

To achieve this, the model integrates:

CNN-LSTM with Attention: Captures sequential patterns and enhances feature importance recognition.

GARCH(1,1): Models financial market volatility, adding risk-awareness to the dataset.

Random Forest Regressor: Provides a complementary machine learning prediction.

Blended Model (Ensemble): Combines deep learning and random forest outputs with weighted averaging for improved accuracy.

# Workflow:
1. Data Collection & Preprocessing

Dataset: Bitcoin historical daily OHLC data (open, high, low, close) from Bitstamp Exchange.

Extracted log returns to capture percentage changes in prices.

Modeled volatility using GARCH(1,1) and added it as a new feature.

Scaled data using RobustScaler to handle outliers.

2. Sequence Preparation

Created 60-day look-back windows (sequences) for model input.

Target variable: Next-day closing price.

3. Model Design

CNN Layer: Extracts short-term price movement features.

LSTM Layer: Captures long-term sequential dependencies.

Attention Layer: Focuses on the most important time steps.

Dropout & BatchNorm: Prevents overfitting and stabilizes training.

Random Forest Regressor: Provides a non-deep-learning comparison baseline.

4. Ensemble Prediction

Predictions from CNN-LSTM-Attention and Random Forest are combined using a weighted average (alpha=0.6).

Final blended prediction improves robustness.

5. Output

Predicts next-day BTC/USD closing price.

Provides a line chart comparing:

Actual Prices

LSTM Predictions

Random Forest Predictions

Blended Predictions

# Results & Visualization:

The ensemble prediction generally tracks the actual Bitcoin prices more smoothly than individual models.

Final output example:

🔮 Predicted BTC/USD Closing Price for YYYY-MM-DD: $XXXXX.XX


Visualization: Compares actual vs predicted values for testing data.

# Key Highlights:

✔ Hybrid model combining deep learning and traditional ML
✔ Attention-enhanced CNN-LSTM for sequential financial data
✔ GARCH volatility feature improves robustness
✔ Blended output provides balanced and stable predictions
✔ Ready-to-use prediction pipeline with visualization

# Tech Stack:

Python

TensorFlow / Keras (LSTM, Attention, CNN)

Scikit-learn (Random Forest, preprocessing)

ARCH package (GARCH modeling)

Matplotlib & Seaborn (visualization)
