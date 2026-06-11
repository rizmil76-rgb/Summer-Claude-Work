import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("/Users/milesrizor/Downloads/diamonds.csv")

# --- Encode ordinal categoricals ---
# cut, color, clarity have a natural order, so ordinal encoding preserves
# that ranking rather than treating all categories as equally distant.
cut_order    = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order  = ["J", "I", "H", "G", "F", "E", "D"]          # J=worst, D=best
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

enc = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
df[["cut", "color", "clarity"]] = enc.fit_transform(df[["cut", "color", "clarity"]])

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1,
                                   max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: ${rmse:,.0f}")

# --- Plot predicted vs actual ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred, alpha=0.15, s=5, color="steelblue")
lims = [0, max(y_test.max(), y_pred.max()) * 1.02]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
ax.set_xlabel("Actual price ($)")
ax.set_ylabel("Predicted price ($)")
ax.set_title(f"Gradient Boosting — Predicted vs Actual Price\nRMSE = ${rmse:,.0f}")
ax.legend()
plt.tight_layout()
plt.savefig("/Users/milesrizor/summer-claude/predicted_vs_actual.png", dpi=150)
print("Plot saved to predicted_vs_actual.png")
