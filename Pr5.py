import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Генеруємо тестові дані
np.random.seed(42)
data = np.random.randn(200, 2) * 0.5
outliers = np.random.uniform(low=-3, high=3, size=(10, 2))
data = np.vstack([data, outliers])

# Перетворюємо у DataFrame
df = pd.DataFrame(data, columns=["Feature1", "Feature2"])

# Створюємо та навчаємо Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso_forest.fit_predict(df[["Feature1", "Feature2"]])

# Візуалізуємо результати
plt.figure(figsize=(8, 6))
plt.scatter(df["Feature1"], df["Feature2"], c=df["Anomaly"], cmap="coolwarm", edgecolors='k')
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Isolation Forest - Виявлення аномалій")
plt.colorbar(label="Аномалія (-1: так, 1: ні)")
plt.show()