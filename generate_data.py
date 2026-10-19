"""
Synthetic hourly sales data for a 5-branch burger chain (demo only).
NOTE: This data is FAKE. Patterns are made up so a forecasting model has
something to learn. Mention this clearly in your README.

Run:  python generate_data.py
Out:  sales_data.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed -> same data every run

# ---------- Settings you can edit ----------
START = "2026-07-01"
END = "2026-10-17"
HOLIDAYS = ["2026-08-14", "2026-09-06"]  # add more dates as "YYYY-MM-DD"

# branch: (popularity multiplier, opening hours as list of hours 0-23)
LATE_HOURS = list(range(12, 24)) + [0, 1, 2]   # 12 PM - 3 AM
MALL_HOURS = list(range(12, 23))               # 12 PM - 11 PM
BRANCHES = {
    "Johar Town": (1.00, LATE_HOURS),
    "DHA Phase 4": (0.95, LATE_HOURS),
    "DHA Phase 6": (0.85, LATE_HOURS),
    "Bahria Town": (0.70, LATE_HOURS),
    "Emporium Mall": (0.80, MALL_HOURS),
}

# Average orders in an hour (before multipliers): lunch bump + big night peak
HOUR_BASE = {12: 6, 13: 9, 14: 10, 15: 7, 16: 6, 17: 8, 18: 11, 19: 15,
             20: 20, 21: 24, 22: 25, 23: 21, 0: 15, 1: 9, 2: 4}


def make_weather(dates):
    """Fake daily temperature (C) and rain flag for Lahore-like weather."""
    n = len(dates)
    day_of_year = dates.dayofyear.values
    # hot in July, cooling toward October
    temp = 36 - (day_of_year - 182) * 0.08 + rng.normal(0, 1.5, n)
    rain_prob = np.where(dates.month.isin([7, 8, 9]), 0.25, 0.05)
    rain = (rng.random(n) < rain_prob).astype(int)
    return pd.DataFrame({"date": dates.date, "temperature": temp.round(1), "rain": rain})


def main():
    days = pd.date_range(START, END, freq="D")
    weather = make_weather(days).set_index("date")
    holidays = set(pd.to_datetime(HOLIDAYS).date)

    rows = []
    for ts in pd.date_range(START, pd.Timestamp(END) + pd.Timedelta(hours=23), freq="h"):
        hour = ts.hour
        # after-midnight hours belong to the previous business day's weather/holiday
        biz_date = (ts - pd.Timedelta(hours=3)).date() if hour <= 2 else ts.date()
        if biz_date not in weather.index:
            continue
        w = weather.loc[biz_date]
        dow = (pd.Timestamp(biz_date)).dayofweek  # Mon=0 ... Sun=6
        is_weekend = int(dow in (4, 5))           # Fri & Sat evenings are busiest
        is_holiday = int(biz_date in holidays)

        for branch, (pop, open_hours) in BRANCHES.items():
            if hour not in open_hours:
                continue
            lam = HOUR_BASE[hour] * pop
            lam *= 1.25 if is_weekend else 1.0
            lam *= 1.30 if is_holiday else 1.0
            lam *= 1.15 if w["rain"] == 1 else 1.0          # rain -> more delivery orders
            lam *= 1.0 + max(w["temperature"] - 33, 0) * 0.01  # very hot -> slightly more cold drinks
            orders = int(rng.poisson(lam))
            avg_bill = float(np.clip(rng.normal(1500, 250), 700, 2500))
            rows.append({
                "datetime": ts,
                "branch": branch,
                "hour": hour,
                "day_of_week": dow,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "temperature": w["temperature"],
                "rain": int(w["rain"]),
                "orders": orders,
                "sales": round(orders * avg_bill),
            })

    df = pd.DataFrame(rows).sort_values(["datetime", "branch"]).reset_index(drop=True)
    df.to_csv("sales_data.csv", index=False)

    print(f"Saved sales_data.csv with {len(df):,} rows")
    print(df.head(8).to_string(index=False))
    print("\nTotal orders per branch:")
    print(df.groupby("branch")["orders"].sum().sort_values(ascending=False))
    print("\nAverage orders by hour (all branches):")
    print(df.groupby("hour")["orders"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
