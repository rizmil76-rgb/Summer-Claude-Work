import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/milesrizor/Downloads/apple_stock.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ── Target ──────────────────────────────────────────────────────────────────
# Will TOMORROW's close be higher than TODAY's close?
# shift(-1) looks one row forward — only assigned using today's close vs tomorrow's,
# no future prices are used as features.
df["target"] = (df["AAPL.Close"].shift(-1) > df["AAPL.Close"]).astype(int)

# Verify the dataset's 'direction' column is NOT the same thing (would be leakage).
# If it were the next-day direction it would correlate perfectly with our target.
df["direction_encoded"] = (df["direction"] == "Increasing").astype(int)
corr = df["direction_encoded"].corr(df["target"])
print(f"Correlation of 'direction' col with next-day target: {corr:.3f}")
# If ~0 → direction is NOT tomorrow's label (safe to use as feature or ignore)
# If ~1 → direction IS tomorrow's label → must never be a feature

# ── Features (only today's information) ─────────────────────────────────────
close = df["AAPL.Close"]

df["return_1d"]  = close.pct_change(1)
df["return_3d"]  = close.pct_change(3)
df["return_5d"]  = close.pct_change(5)
df["return_10d"] = close.pct_change(10)

df["volatility_5d"]  = df["return_1d"].rolling(5).std()
df["volatility_10d"] = df["return_1d"].rolling(10).std()

df["day_range"]  = (df["AAPL.High"] - df["AAPL.Low"]) / close          # intraday range
df["gap"]        = (df["AAPL.Open"] - close.shift(1)) / close.shift(1) # overnight gap
df["close_vs_open"] = (close - df["AAPL.Open"]) / df["AAPL.Open"]      # intraday drift

df["ma_5"]  = close.rolling(5).mean()
df["ma_20"] = close.rolling(20).mean()
df["ma_cross"] = (df["ma_5"] - df["ma_20"]) / df["ma_20"]              # MA crossover signal

# RSI (14-day)
delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# Bollinger band position — already in file, computed from historical window (safe)
df["bb_position"] = (close - df["dn"]) / (df["up"] - df["dn"])

df["volume_ratio"] = df["AAPL.Volume"] / df["AAPL.Volume"].rolling(10).mean()

# Drop 'direction' — correlation check above will tell us if it leaks
FEATURE_COLS = [
    "return_1d", "return_3d", "return_5d", "return_10d",
    "volatility_5d", "volatility_10d",
    "day_range", "gap", "close_vs_open",
    "ma_cross", "rsi_14", "bb_position", "volume_ratio",
]

df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

X = df[FEATURE_COLS].values
y = df["target"].values

# ── Time-based split (NEVER random — that would leak future into training) ──
# Train on first 80%, test on last 20%.
split = int(len(df) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {df['Date'].iloc[0].date()} → {df['Date'].iloc[split-1].date()} ({split} days)")
print(f"Test:  {df['Date'].iloc[split].date()} → {df['Date'].iloc[-1].date()} ({len(X_test)} days)")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Metrics ──────────────────────────────────────────────────────────────────
model_acc    = accuracy_score(y_test, y_pred)
baseline_acc = y_test.mean()   # always predicting "up" = fraction of up days
print(f"\nModel accuracy:    {model_acc*100:.1f}%")
print(f"Always-up baseline:{baseline_acc*100:.1f}%")
print(f"Lift over baseline:{(model_acc - baseline_acc)*100:.1f} pp")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

# ── Feature importances ───────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4])

# Left: accuracy comparison bar chart
ax1 = fig.add_subplot(gs[0])
bars = ax1.bar(["Always 'Up'\n(baseline)", "This model"],
               [baseline_acc * 100, model_acc * 100],
               color=["#aac4e0", "#2a6fa8"], width=0.5)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Model vs Baseline Accuracy")
for bar, val in zip(bars, [baseline_acc * 100, model_acc * 100]):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
             ha="center", fontweight="bold")
ax1.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Random chance")
ax1.legend(fontsize=8)

# Right: feature importances
ax2 = fig.add_subplot(gs[1])
importances.plot.barh(ax=ax2, color="#2a6fa8")
ax2.set_xlabel("Importance")
ax2.set_title("Feature Importances")

plt.tight_layout()
plt.savefig("/Users/milesrizor/summer-claude/stock_results.png", dpi=150)
print("\nPlot saved.")
