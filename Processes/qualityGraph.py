import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("performance_log.csv")

# Normalize timestamps to start at 0 seconds
df['Time_s'] = (df['Timestamp_ms'] - df['Timestamp_ms'].iloc[0]) / 1000.0

# Replace -1 (tracking lost) with NaN so it creates gaps in the line graph instead of dropping to 0
df['Reprojection_Error'] = df['Reprojection_Error'].replace(-1.0, np.nan)

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Confidence on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Confidence', color=color)
ax1.plot(df['Time_s'], df['Confidence'], color=color, label='Confidence')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-0.1, 1.1)

# Create a secondary y-axis for Reprojection Error
ax2 = ax1.twinx()  
color = 'tab:orange'
ax2.set_ylabel('Reprojection Error (Pixels)', color=color)
ax2.plot(df['Time_s'], df['Reprojection_Error'], color=color, alpha=0.7, label='Reprojection Error')
ax2.tick_params(axis='y', labelcolor=color)

# Safely calculate the max error with a fallback
max_error = df['Reprojection_Error'].max()
if pd.isna(max_error):
    max_error = 10.0  # Default fallback if all reprojection errors are NaN
ax2.set_ylim(0, max_error * 1.2) # Scale slightly above max error

# Highlight periods where tracking was lost (Tracking_Valid == 0)
lost_tracking = df[df['Tracking_Valid'] == 0]
ax1.scatter(lost_tracking['Time_s'], lost_tracking['Confidence'], color='red', zorder=5, label='Tracking Lost')

fig.suptitle('System Performance: Confidence and Reprojection Error Over Time')
fig.tight_layout()
plt.grid(True)
plt.show()