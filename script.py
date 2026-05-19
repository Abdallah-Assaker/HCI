import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
try:
    df = pd.read_csv('accelerometer.csv')
except FileNotFoundError:
    print("Error: 'accelerometer.csv' not found.")
    exit()

# Compute magnitude (x, y, z → single value)
df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

# Config
window_size = 50
noise_floor = 1.1

# Initialize labels
df['status'] = ""

# Collect window stats
window_avgs = []
window_starts = []

for i in range(0, len(df), window_size):
    window_indices = range(i, min(i + window_size, len(df)))
    avg_motion = df['magnitude'].iloc[window_indices].mean()
    window_avgs.append(avg_motion)
    window_starts.append(i)

# Detect motion windows
motion_windows = [avg for avg in window_avgs if avg >= noise_floor]
session_is_stationary = len(motion_windows) == 0

if session_is_stationary:
    # No motion
    df['status'] = "Stationary"
    walking_threshold = noise_floor
    running_threshold = noise_floor
else:
    # Learn thresholds from data
    walking_threshold = np.percentile(motion_windows, 33)
    running_threshold = np.percentile(motion_windows, 66)

    # Keep thresholds above noise
    walking_threshold = max(walking_threshold, noise_floor)
    running_threshold = max(running_threshold, walking_threshold)

    # Classify each window
    for avg_motion, start_idx in zip(window_avgs, window_starts):
        window_indices = range(start_idx, min(start_idx + window_size, len(df)))

        if avg_motion < noise_floor:
            current_status = "Stationary"
        elif avg_motion < walking_threshold:
            current_status = "Stationary"
        elif avg_motion < running_threshold:
            current_status = "Walking"
        else:
            current_status = "Running/Active"

        df.loc[window_indices, 'status'] = current_status

# Plot results
plt.figure(figsize=(12, 6))

colors = {
    'Stationary': 'green',
    'Walking': 'orange',
    'Running/Active': 'red'
}

for status, group in df.groupby('status'):
    plt.scatter(
        group['seconds_elapsed'],
        group['magnitude'],
        label=status,
        s=15,
        color=colors.get(status, 'blue')
    )

# Draw thresholds
if session_is_stationary:
    plt.axhline(y=noise_floor, linestyle='--', alpha=0.5, label='Noise Floor')
else:
    plt.axhline(y=walking_threshold, linestyle='--', alpha=0.5, label='Walking Threshold')
    plt.axhline(y=running_threshold, linestyle='--', alpha=0.5, label='Running Threshold')

# Final touches
plt.title('Human Activity Recognition')
plt.xlabel('Time (Seconds)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, alpha=0.3)

# Print dominant activity
dominant_activity = df['status'].mode()[0]
print(f"Detected Activity: {dominant_activity}")

plt.show()