# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from seaborn import sns  # Removed unused import

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from keras.models import Sequential
from keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings("ignore")

# === Load Dataset ===
df = pd.read_csv('./data/MicrosoftStock.csv')

print("Dataset Shape:", df.shape)
print(df.head())

# =======================
# Step 7: Data Preprocessing
# =======================

# 1. Convert 'Date' column to datetime
# Check which column exists for date
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
elif 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
else:
    raise ValueError("No 'Date' or 'date' column found in the dataset.")

# Sort by date index
df.sort_index(inplace=True)


# 4. Handle missing values (interpolation)
df = df.interpolate(method='linear')

# Check again
print("\nAfter preprocessing:")
print(df.info())

print("Available columns:", df.columns.tolist())  # Add this before line 52

# Clean column names (remove whitespace)
df.rename(columns=lambda x: x.strip(), inplace=True)

# Rename common alternatives to 'Close'
if 'close' in df.columns:
    df.rename(columns={'close': 'Close'}, inplace=True)
elif 'Closing Price' in df.columns:
    df.rename(columns={'Closing Price': 'Close'}, inplace=True)
elif 'CLOSE' in df.columns:
    df.rename(columns={'CLOSE': 'Close'}, inplace=True)

# Final check
if 'Close' not in df.columns:
    raise ValueError("❌ The dataset does not contain a 'Close' column.")


# 5. Simple Moving Average (20 days)
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# 6. Exponential Moving Average (20 days)
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# 7. Bollinger Bands (20-day window)
df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(20).std()
df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(20).std()

# 8. RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Show the dataframe with new features
print("\nFinal Data with Indicators:")
print(df.tail())

# Save the clean dataset with indicators
df.to_csv('./outputs/preprocessed_data.csv')

plt.figure(figsize=(14, 6))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.title("Microsoft Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.plot(df['SMA_20'], label='SMA 20', color='orange')
plt.plot(df['EMA_20'], label='EMA 20', color='green')
plt.title("Close Price vs SMA & EMA")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.plot(df['BB_upper'], label='Upper Band', linestyle='--', color='red')
plt.plot(df['BB_lower'], label='Lower Band', linestyle='--', color='purple')
plt.fill_between(df.index, df['BB_lower'], df['BB_upper'], color='lightgray', alpha=0.3)
plt.title("Bollinger Bands")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(df['RSI'], label='RSI', color='brown')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title("RSI (Relative Strength Index)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.grid(True)
plt.show()
# We’ll focus on these features
features = ['Close', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'RSI']

# Drop rows with NaNs (due to rolling calculations)
df_model = df[features].dropna()

print("Model data shape:", df_model.shape)
print(df_model.tail())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_model)

# Convert to DataFrame to keep column names
scaled_df = pd.DataFrame(scaled_data, columns=features, index=df_model.index)

print("Scaled data sample:\n", scaled_df.head())

# Sequence length
LOOKBACK = 60

X = []
y = []

for i in range(LOOKBACK, len(scaled_df)):
    X.append(scaled_df.iloc[i-LOOKBACK:i].values)
    y.append(scaled_df.iloc[i]['Close'])  # Predict next day's close

X = np.array(X)
y = np.array(y)

print("Input shape for LSTM (X):", X.shape)  # (samples, 60, features)
print("Output shape (y):", y.shape)

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Use only the last row of each 60-day sequence as a single sample (non-sequential)
X_flat = scaled_df.values
X_tabular = X_flat[LOOKBACK:]  # Skip first 60 rows (used in LSTM sequences)
y_tabular = scaled_df['Close'].values[LOOKBACK:]

# Train-test split for tabular models
split_tabular = int(len(X_tabular) * 0.8)
X_train_tab, X_test_tab = X_tabular[:split_tabular], X_tabular[split_tabular:]
y_train_tab, y_test_tab = y_tabular[:split_tabular], y_tabular[split_tabular:]

lr = LinearRegression()
lr.fit(X_train_tab, y_train_tab)

y_pred_lr = lr.predict(X_test_tab)

# Evaluation
print("\n📊 Linear Regression Results")
print("MAE:", mean_absolute_error(y_test_tab, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test_tab, y_pred_lr)))
print("R²:", r2_score(y_test_tab, y_pred_lr))

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_tab, y_train_tab)

y_pred_rf = rf.predict(X_test_tab)

print("\n📊 Random Forest Results")
print("MAE:", mean_absolute_error(y_test_tab, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test_tab, y_pred_rf)))
print("R²:", r2_score(y_test_tab, y_pred_rf))

xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb.fit(X_train_tab, y_train_tab)

y_pred_xgb = xgb.predict(X_test_tab)

print("\n📊 XGBoost Results")
print("MAE:", mean_absolute_error(y_test_tab, y_pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_test_tab, y_pred_xgb)))
print("R²:", r2_score(y_test_tab, y_pred_xgb))

plt.figure(figsize=(14, 6))
plt.plot(y_test_tab, label='Actual', color='blue')
plt.plot(y_pred_lr, label='Linear Regression', linestyle='--')
plt.plot(y_pred_rf, label='Random Forest', linestyle='--')
plt.plot(y_pred_xgb, label='XGBoost', linestyle='--')
plt.title("Traditional ML Model Predictions vs Actual")
plt.legend()
plt.show()

# Input shape: (samples, timesteps=60, features=6)
model = Sequential()

# LSTM layers
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=32))

# Output layer
model.add(Dense(1))  # Predicting 'Close' price

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
model.summary()

# Fit the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("LSTM Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()

# Predict on test set
y_pred_lstm = model.predict(X_test)

# Convert predictions back to original scale
close_index = df_model.columns.get_loc('Close')  # Index of 'Close' in original data
y_pred_actual = scaler.inverse_transform(np.hstack((y_pred_lstm, X_test[:, -1, 1:])))[:, close_index]
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), X_test[:, -1, 1:])))[:, close_index]

# Evaluation metrics
print("\n📊 LSTM Model Results")
print("MAE:", mean_absolute_error(y_test_actual, y_pred_actual))
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)))
print("R²:", r2_score(y_test_actual, y_pred_actual))

plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Price', color='blue')
plt.plot(y_pred_actual, label='Predicted Price (LSTM)', linestyle='--', color='orange')
plt.title("LSTM Prediction vs Actual Close Prices")
plt.xlabel("Time Step")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Use the last 60 days of data from the scaled DataFrame
last_60_days = scaled_df[-60:].values
future_input = last_60_days.copy()  # Preserve original input

# Store predictions
future_predictions = []

# Predict 30 future days iteratively
for _ in range(30):
    input_seq = future_input[-60:].reshape(1, 60, scaled_df.shape[1])  # (1, 60, features)
    
    pred_scaled = model.predict(input_seq, verbose=0)[0][0]  # Scaled close price
    
    # Create dummy row to match full feature count (we only predicted 'Close')
    dummy_row = np.zeros(scaled_df.shape[1])
    dummy_row[0] = pred_scaled  # Set predicted Close value

    future_input = np.vstack((future_input, dummy_row))  # Append new row
    future_predictions.append(pred_scaled)

# Prepare array with dummy features to inverse transform
future_full = np.zeros((30, scaled_df.shape[1]))
future_full[:, 0] = future_predictions  # Only Close is predicted

# Inverse transform
future_close_actual = scaler.inverse_transform(future_full)[:, 0]

# Create future dates
last_date = df_model.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_model['Close'].iloc[-100:], label='Recent Prices', color='blue')
plt.plot(future_dates, future_close_actual, label='Future Predictions', color='green', linestyle='--')
plt.title("Next 30 Days Microsoft Stock Price Prediction (LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
# Paste your main.py content here

# Save the trained LSTM model
model.save('./models/lstm_model.h5')

