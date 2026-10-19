"""
Forecasting helpers for the Streamlit dashboard.
Same logic as forecasting_model_sklearn.ipynb, packaged as functions.
Data is SYNTHETIC (see generate_data.ipynb).
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["hour", "day_of_week", "is_weekend", "is_holiday", "temperature", "rain",
            "branch_code", "lag_7d", "lag_1d", "roll_mean_7d"]
TEST_DAYS = 14
BUSINESS_HOUR_ORDER = list(range(12, 24)) + [0, 1, 2]


def load_data(path="sales_data.csv"):
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values(["branch", "datetime"]).reset_index(drop=True)


def get_branch_codes(df):
    return {name: i for i, name in enumerate(sorted(df["branch"].unique()))}


def build_features(df):
    """Add lag / rolling features (same as the notebook). Returns rows that have all features."""
    df = df.copy()

    def add_lag(frame, days, name):
        lagged = frame[["branch", "datetime", "orders"]].copy()
        lagged["datetime"] = lagged["datetime"] + pd.Timedelta(days=days)
        lagged = lagged.rename(columns={"orders": name})
        return frame.merge(lagged, on=["branch", "datetime"], how="left")

    df = add_lag(df, 7, "lag_7d")
    df = add_lag(df, 1, "lag_1d")

    df = df.sort_values(["branch", "hour", "datetime"]).reset_index(drop=True)
    df["roll_mean_7d"] = (
        df.groupby(["branch", "hour"])["orders"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    )
    df = df.sort_values(["datetime", "branch"]).reset_index(drop=True)

    codes = get_branch_codes(df)
    df["branch_code"] = df["branch"].map(codes)
    return df.dropna(subset=["lag_7d", "lag_1d", "roll_mean_7d"]).reset_index(drop=True)


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    )


def train_model(feat_df):
    model = make_model()
    model.fit(feat_df[FEATURES], feat_df["orders"])
    return model


def evaluate(feat_df):
    """Train on everything before the last 14 days, test on the last 14 days."""
    cutoff = feat_df["datetime"].max() - pd.Timedelta(days=TEST_DAYS)
    train = feat_df[feat_df["datetime"] <= cutoff]
    test = feat_df[feat_df["datetime"] > cutoff]
    model = make_model()
    model.fit(train[FEATURES], train["orders"])
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    y = test["orders"].values
    base = test["lag_7d"].values
    return {
        "baseline_mae": mean_absolute_error(y, base),
        "model_mae": mean_absolute_error(y, pred),
        "baseline_wape": np.abs(y - base).sum() / y.sum() * 100,
        "model_wape": np.abs(y - pred).sum() / y.sum() * 100,
        "test_start": test["datetime"].min(),
        "test_end": test["datetime"].max(),
    }


def _hour_label(ts):
    return ts.strftime("%I %p").lstrip("0")


def forecast_next_day(raw_df, model, branch, temperature, rain, is_holiday, orders_per_staff=8):
    """Forecast every opening hour of the next business day for one branch."""
    codes = get_branch_codes(raw_df)
    bdf = raw_df[raw_df["branch"] == branch]
    hist = bdf.set_index("datetime")["orders"].sort_index()

    last_ts = raw_df["datetime"].max()
    next_biz = (last_ts - pd.Timedelta(hours=3)).normalize() + pd.Timedelta(days=1)
    biz_dow = next_biz.dayofweek
    avg_bill = bdf["sales"].sum() / max(bdf["orders"].sum(), 1)

    open_hours = [h for h in BUSINESS_HOUR_ORDER if h in set(bdf["hour"].unique())]

    rows = []
    for h in open_hours:
        ts = next_biz + pd.Timedelta(hours=h) if h >= 12 else next_biz + pd.Timedelta(days=1, hours=h)

        same_hour = hist[(hist.index.hour == h) & (hist.index < ts)].tail(7)
        roll = float(same_hour.mean()) if len(same_hour) else float(hist.mean())

        lag7 = hist.get(ts - pd.Timedelta(days=7), np.nan)
        if pd.isna(lag7):
            lag7 = roll

        # most recent same-hour value at least 1 day back (falls back to older days if needed)
        lag1 = np.nan
        for k in range(1, 8):
            v = hist.get(ts - pd.Timedelta(days=k), np.nan)
            if not pd.isna(v):
                lag1 = v
                break
        if pd.isna(lag1):
            lag1 = roll

        rows.append({
            "datetime": ts,
            "hour": h,
            "day_of_week": biz_dow,
            "is_weekend": int(biz_dow in (4, 5)),
            "is_holiday": int(is_holiday),
            "temperature": float(temperature),
            "rain": int(rain),
            "branch_code": codes[branch],
            "lag_7d": float(lag7),
            "lag_1d": float(lag1),
            "roll_mean_7d": roll,
        })

    out = pd.DataFrame(rows)
    out["predicted_orders"] = np.clip(model.predict(out[FEATURES]), 0, None)
    out["hour_label"] = out["datetime"].apply(_hour_label)
    out["est_sales"] = out["predicted_orders"] * avg_bill
    out["suggested_staff"] = out["predicted_orders"].apply(
        lambda o: max(2, math.ceil(o / orders_per_staff))
    )
    return out, next_biz
