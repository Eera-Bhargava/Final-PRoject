# Final-PRoject
# Stock Price Trend Prediction using LSTM

This project uses an LSTM (Long Short-Term Memory) neural network to predict future stock prices based on historical stock data. Additional financial indicators like the Moving Average (MA20) and Relative Strength Index (RSI) are also included for deeper analysis.

---

## Project Objective

* Forecast the stock's closing price using time-series data
* Use an LSTM model for capturing temporal dependencies
* Compare predicted prices with actual prices
* Analyze stock using technical indicators: MA20 and RSI

---

## Libraries Used

```python
import yfinance as yf                
import pandas as pd                 
import numpy as np                
import matplotlib.pyplot as plt     
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```

---

## Dataset

* **Source**: Yahoo Finance (`yfinance`)
* **Example Stock**: Apple Inc. (**AAPL**)
* **Time Period**: `2019-01-01` to `2024-12-31`

---

## Project Steps Overview

### 1. Load Stock Data

```python
data = yf.download('AAPL', start='2019-01-01', end='2024-12-31')
```

### 2. Visualize the Closing Price

Line plot of historical closing prices.

### 3. Normalize the Data

```python
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
```

Scaling helps the LSTM model converge faster and more accurately.

### 4. Create Sequences for LSTM

Using a window size of 60 days to predict the next day.

```python
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])
```

### 5. Train-Test Split

80% data for training, 20% for testing.

### 6. Build the LSTM Model

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))
```

### 7. Compile and Train

```python
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)
```

### 8. Predict and Inverse Scale

Predictions are made and then converted back to original price scale.

### 9. Plot Actual vs Predicted

Visual comparison of model performance using matplotlib.

---

## 📉 Technical Indicators

### MA20 - Moving Average

```python
data['MA20'] = data['Close'].rolling(window=20).mean()
```

### RSI - Relative Strength Index

Measures momentum of price movements over a 14-day window.

---

## Visualizations

*  Historical vs Predicted Price
* Close Price with MA20
* RSI Indicator (with overbought/oversold zones)

---

---
