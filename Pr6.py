import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv("noise_data.csv")

# Convert time to datetime format
data["Start date/time in UTC"] = pd.to_datetime(data["Start date/time in UTC"])

# Analyze noise levels in different locations
mean_noise_by_location = data.groupby(["Latitude", "Longitude"])['Mean volume (dBA)'].mean().reset_index()

# Plot noise levels by coordinates
plt.figure(figsize=(10, 6))

# Create scatter plot
scatter = plt.scatter(mean_noise_by_location["Longitude"],
                      mean_noise_by_location["Latitude"],
                      c=mean_noise_by_location['Mean volume (dBA)'],
                      s=100,  # Size of the points
                      cmap="coolwarm", alpha=0.75, edgecolor="k")

# Add colorbar
plt.colorbar(scatter, label="Average Noise Level (dBA)")

plt.title("Noise Levels at Different Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
