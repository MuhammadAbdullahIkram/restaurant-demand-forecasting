"""
Branch demand forecast dashboard (Streamlit).
Run:  streamlit run app.py
Needs sales_data.csv in the same folder (create it with generate_data.ipynb).
DATA IS SYNTHETIC: this is a demo, not real Johnny & Jugnu data.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import forecast_utils as fu

st.set_page_config(page_title="Branch Demand Forecast (Demo)", page_icon="🍔", layout="wide")


@st.cache_data
def get_data():
    return fu.load_data("sales_data.csv")


@st.cache_data
def get_features(raw):
    return fu.build_features(raw)


@st.cache_resource
def get_model(_feat):
    return fu.train_model(_feat)


@st.cache_data
def get_eval(_feat):
    return fu.evaluate(_feat)


try:
    raw = get_data()
except FileNotFoundError:
    st.error("sales_data.csv not found. Run generate_data.ipynb first and keep the CSV next to app.py.")
    st.stop()

feat = get_features(raw)
model = get_model(feat)

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
branches = sorted(raw["branch"].unique())
branch = st.sidebar.selectbox("Branch", branches)

last_temp = float(raw.sort_values("datetime")["temperature"].iloc[-1])
temperature = st.sidebar.slider("Expected temperature (°C)", 15.0, 45.0, round(last_temp, 0))
rain = st.sidebar.checkbox("Rain expected", value=False)
is_holiday = st.sidebar.checkbox("Public holiday", value=False)
orders_per_staff = st.sidebar.slider(
    "Orders one kitchen staff can handle per hour (assumption)", 4, 15, 8
)
st.sidebar.caption("Weather inputs would come from a weather forecast in a real setup.")

# ---------------- Header ----------------
st.title("🍔 Branch Demand Forecast")
st.warning("Demo with SYNTHETIC data. Numbers are made up to show the approach, not real sales.")

out, next_day = fu.forecast_next_day(raw, model, branch, temperature, rain, is_holiday, orders_per_staff)
st.subheader(f"{branch}: forecast for {next_day.strftime('%A, %d %B %Y')}")

peak = out.loc[out["predicted_orders"].idxmax()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected orders", f"{out['predicted_orders'].sum():.0f}")
c2.metric("Estimated sales (PKR)", f"{out['est_sales'].sum():,.0f}")
c3.metric("Peak hour", f"{peak['hour_label']}")
c4.metric("Max staff needed", f"{int(out['suggested_staff'].max())}")

tab1, tab2, tab3 = st.tabs(["Hourly forecast", "All branches", "Model accuracy"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e4572e" if h == peak["hour_label"] else "#f2a65a" for h in out["hour_label"]]
    ax.bar(out["hour_label"], out["predicted_orders"], color=colors)
    ax.set_ylabel("Predicted orders")
    ax.set_title("Predicted orders per hour (peak hour in red)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    table = out[["hour_label", "predicted_orders", "est_sales", "suggested_staff"]].rename(columns={
        "hour_label": "Hour", "predicted_orders": "Predicted orders",
        "est_sales": "Est. sales (PKR)", "suggested_staff": "Suggested kitchen staff",
    })
    table["Predicted orders"] = table["Predicted orders"].round(1)
    table["Est. sales (PKR)"] = table["Est. sales (PKR)"].round(0).astype(int)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        "Staff suggestion = predicted orders / orders per staff (sidebar), minimum 2. "
        "It is a simple rule to show the idea; a manager decides the final roster."
    )

with tab2:
    rows = []
    for b in branches:
        o, _ = fu.forecast_next_day(raw, model, b, temperature, rain, is_holiday, orders_per_staff)
        rows.append({"Branch": b, "Expected orders": round(o["predicted_orders"].sum()),
                     "Est. sales (PKR)": round(o["est_sales"].sum()),
                     "Peak hour": o.loc[o["predicted_orders"].idxmax(), "hour_label"]})
    comp = pd.DataFrame(rows).sort_values("Expected orders", ascending=False)
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.bar_chart(comp.set_index("Branch")["Expected orders"])

with tab3:
    ev = get_eval(feat)
    st.write(
        f"Tested on the last {fu.TEST_DAYS} days of data "
        f"({ev['test_start']:%d %b} to {ev['test_end']:%d %b %Y}), "
        "model trained only on the days before."
    )
    acc = pd.DataFrame({
        "Model": ["Baseline (same hour last week)", "Gradient boosting"],
        "MAE (orders/hour)": [round(ev["baseline_mae"], 2), round(ev["model_mae"], 2)],
        "WAPE %": [round(ev["baseline_wape"], 1), round(ev["model_wape"], 1)],
    })
    st.dataframe(acc, hide_index=True, use_container_width=True)
    st.caption("Data is synthetic with random noise, so some error can never be removed. "
               "Re-check these numbers with real POS data.")
