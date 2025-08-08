# LSTM-Bitcoin-Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.dates as mdates
from matplotlib.animation import FFMpegWriter
import matplotlib
import os

# Use non-interactive backend for saving
matplotlib.use('Agg')

# Settings
ticker = 'BTC-USD'
interval = '15m'
prediction_window = 60
update_interval = 30000
prediction_steps = 5

# Function to get live data
def get_live_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data[['Close']]

# Prepare data
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Future prediction
def forecast_future(model, last_sequence, steps):
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], pred)
        
    return np.array(predictions)

# Create figure and axes
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs[:2, 0])
ax2 = fig.add_subplot(gs[2, 0])

fig.suptitle('Bitcoin Live Analysis with LSTM Network', fontsize=18, fontweight='bold', color='gold')

# Initialize plot lines
live_line, = ax1.plot([], [], 'lime', linewidth=2, label='Actual Price')
pred_line, = ax1.plot([], [], 'cyan', linestyle='--', linewidth=2, label='Prediction')
future_pred_line, = ax1.plot([], [], 'magenta', marker='o', markersize=6, linewidth=2, label='Future Prediction')
ax1.legend(loc='upper right', fontsize=12)

# Price chart settings
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
ax1.set_title('Live Price Chart and Predictions', fontsize=14, pad=15, color='white')

# Error histogram
ax2.set_ylabel('Prediction Error (%)', fontsize=12)
ax2.set_xlabel('Time', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 10)
ax2.set_title('Prediction Error History', fontsize=14, pad=15, color='white')

# Get initial data
live_data = get_live_data()
scaled_data, scaler = prepare_data(live_data.values)

# Create sequences
if len(scaled_data) > prediction_window:
    X, y = create_sequences(scaled_data, prediction_window)
    X = X.reshape(X.shape[0], X.shape[1], 1)
else:
    X, y = np.array([]), np.array([])

# Build model
model = build_lstm_model((prediction_window, 1))
if len(X) > 0:
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

# Initialize animation variables
timestamps = live_data.index[-len(y):].tolist() if len(y) > 0 else []
actual_prices = []
predicted_prices = []
future_pred_prices = []
errors = []
error_timestamps = []
last_update_time = datetime.now()

# Create animation writer
writer = FFMpegWriter(fps=3, bitrate=1800)

# Open the video file
with writer.saving(fig, "bitcoin_analysis.mp4", dpi=100):
    # Simulate 100 frames (adjust as needed)
    for frame in range(100):
        try:
            # Get new data (simulated for saving)
            new_data = get_live_data()
            
            # Process if new data exists
            if not new_data.empty and new_data.index[-1] > live_data.index[-1]:
                live_data = new_data
                scaled_data, scaler = prepare_data(live_data.values)
                
                # Create sequences
                if len(scaled_data) > prediction_window:
                    X, y = create_sequences(scaled_data, prediction_window)
                    X = X.reshape(X.shape[0], X.shape[1], 1)
                    timestamps = live_data.index[-len(y):].tolist()
                    
                    # Train model
                    model.fit(X, y, epochs=5, batch_size=8, verbose=0)
                    last_update_time = datetime.now()
            
            # Make predictions
            if len(X) > 0:
                # Current predictions
                current_pred = model.predict(X, verbose=0).flatten()
                
                # Future predictions
                last_sequence = X[-1].flatten()
                future_pred = forecast_future(model, last_sequence, prediction_steps)
                
                # Convert to actual prices
                current_pred_prices = scaler.inverse_transform(current_pred.reshape(-1, 1)).flatten()
                future_pred_prices = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
                actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
                
                # Future timestamps
                last_timestamp = timestamps[-1]
                future_timestamps = [last_timestamp + timedelta(minutes=15*i) for i in range(1, prediction_steps+1)]
                
                # Calculate error
                last_actual_price = actual_prices[-1]
                last_pred_price = current_pred_prices[-1]
                error = abs((last_actual_price - last_pred_price) / last_actual_price) * 100
                
                # Store data for display
                errors.append(error)
                error_timestamps.append(last_timestamp)
                
                # Limit history
                if len(errors) > 20:
                    errors.pop(0)
                    error_timestamps.pop(0)
                
                # Update price chart
                live_line.set_data(timestamps, actual_prices)
                pred_line.set_data(timestamps, current_pred_prices)
                future_pred_line.set_data(future_timestamps, future_pred_prices)
                
                # Set axis limits
                all_prices = np.concatenate([actual_prices, current_pred_prices, future_pred_prices])
                min_price = np.min(all_prices) * 0.98
                max_price = np.max(all_prices) * 1.02
                ax1.set_ylim(min_price, max_price)
                
                all_timestamps = timestamps + future_timestamps
                ax1.set_xlim(all_timestamps[0], all_timestamps[-1])
                
                # Update error histogram
                ax2.clear()
                ax2.bar([ts.strftime('%H:%M') for ts in error_timestamps], errors, color='orange')
                ax2.set_ylabel('Prediction Error (%)', fontsize=12)
                ax2.set_xlabel('Time', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, max(errors) * 1.2 if errors else 10)
                ax2.tick_params(axis='x', rotation=45)
                
                # Display live info
                last_price = actual_prices[-1]
                ax1.set_title(
                    f'Last Price: ${last_price:.2f} | Error: {error:.2f}% | Last Update: {last_update_time.strftime("%H:%M:%S")}',
                    fontsize=14, color='white'
                )
            
            # Draw and save frame
            fig.canvas.draw()
            writer.grab_frame()
            
            # Print progress
            print(f"Processed frame {frame+1}/100")
            
            # Simulate update interval
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")
            continue

print("Video saved successfully: bitcoin_analysis.mp4")
