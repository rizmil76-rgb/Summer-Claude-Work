import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

STOCKS = {
    "AAPL": ("/Users/milesrizor/Downloads/apple_stock.csv", "AAPL"),
    "MSFT": ("/Users/milesrizor/Downloads/msft_stock.csv", "MSFT"),
    "NVDA": ("/Users/milesrizor/Downloads/nvda_stock.csv", "NVDA"),
}

def build_features(df, ticker):
    close  = df[f"{ticker}.Close"]
    open_  = df[f"{ticker}.Open"]
    high   = df[f"{ticker}.High"]
    low    = df[f"{ticker}.Low"]
    vol    = df[f"{ticker}.Volume"]

    df = df.copy()
    df["return_1d"]     = close.pct_change(1)
    df["return_3d"]     = close.pct_change(3)
    df["return_5d"]     = close.pct_change(5)
    df["return_10d"]    = close.pct_change(10)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"]= df["return_1d"].rolling(10).std()
    df["day_range"]     = (high - low) / close
    df["gap"]           = (open_ - close.shift(1)) / close.shift(1)
    df["close_vs_open"] = (close - open_) / open_
    df["ma_5"]          = close.rolling(5).mean()
    df["ma_20"]         = close.rolling(20).mean()
    df["ma_cross"]      = (df["ma_5"] - df["ma_20"]) / df["ma_20"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"]        = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["bb_position"]   = (close - df["dn"]) / (df["up"] - df["dn"])
    df["volume_ratio"]  = vol / vol.rolling(10).mean()
    return df

FEATURE_COLS = [
    "return_1d", "return_3d", "return_5d", "return_10d",
    "volatility_5d", "volatility_10d", "day_range", "gap",
    "close_vs_open", "ma_cross", "rsi_14", "bb_position", "volume_ratio",
]

results = {}

for name, (path, ticker) in STOCKS.items():
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["target"] = (df[f"{ticker}.Close"].shift(-1) > df[f"{ticker}.Close"]).astype(int)
    df = build_features(df, ticker)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    X, y = df[FEATURE_COLS].values, df["target"].values
    split = int(len(df) * 0.80)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                        learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_acc    = accuracy_score(y_test, y_pred)
    baseline_acc = y_test.mean()
    lift         = model_acc - baseline_acc

    results[name] = {
        "n_test":    len(y_test),
        "baseline":  baseline_acc,
        "model":     model_acc,
        "lift":      lift,
        "importances": pd.Series(model.feature_importances_, index=FEATURE_COLS),
    }
    print(f"{name}: baseline={baseline_acc*100:.1f}%  model={model_acc*100:.1f}%  lift={lift*100:+.1f}pp  (n={len(y_test)})")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n┌─────────┬────────────┬────────────┬──────────┐")
print("│ Stock   │ Always-Up  │ Model Acc  │ Lift     │")
print("├─────────┼────────────┼────────────┼──────────┤")
for name, r in results.items():
    flag = " ◄ suspiciously good?" if r["lift"] > 0.08 else ""
    print(f"│ {name:<7} │ {r['baseline']*100:>8.1f}%  │ {r['model']*100:>8.1f}%  │ {r['lift']*100:>+6.1f}pp │{flag}")
print("└─────────┴────────────┴────────────┴──────────┘")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

for ax, (name, r) in zip(axes, results.items()):
    vals  = [r["baseline"] * 100, r["model"] * 100]
    cols  = ["#aac4e0", "#2a6fa8"]
    bars  = ax.bar(["Always Up\n(baseline)", "Model"], vals, color=cols, width=0.5)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(name)
    ax.set_ylabel("Accuracy (%)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Up/Down Prediction: Model vs Always-Up Baseline", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("/Users/milesrizor/summer-claude/multi_stock_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved.")
