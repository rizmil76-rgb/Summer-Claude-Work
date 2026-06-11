import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

st.title("Diamond Price Predictor")
st.write("Trains a Gradient Boosting model on `diamonds.csv` and shows results.")

@st.cache_data
def load_and_train():
    df = pd.read_csv("/Users/milesrizor/Downloads/diamonds.csv")

    cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order   = ["J", "I", "H", "G", "F", "E", "D"]
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
    return y_test, y_pred, rmse, model, X.columns.tolist()

with st.spinner("Training model…"):
    y_test, y_pred, rmse, model, feature_names = load_and_train()

st.metric("RMSE", f"${rmse:,.0f}")

# --- Predicted vs Actual plot ---
st.subheader("Predicted vs Actual Price")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred, alpha=0.15, s=5, color="steelblue")
lims = [0, max(y_test.max(), float(np.max(y_pred))) * 1.02]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
ax.set_xlabel("Actual price ($)")
ax.set_ylabel("Predicted price ($)")
ax.legend()
st.pyplot(fig)

# --- Feature importance ---
st.subheader("Feature Importances")
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
fig2, ax2 = plt.subplots(figsize=(6, 4))
importances.plot.barh(ax=ax2, color="steelblue")
ax2.set_xlabel("Importance")
st.pyplot(fig2)
