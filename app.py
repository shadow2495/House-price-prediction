"""
House Price Prediction — Streamlit Frontend
Run: streamlit run app.py
Make sure FastAPI is running on port 8000 first.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

st.set_page_config(page_title="House Price Predictor",
                   page_icon="🏠", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.markdown("""
<style>
  .price-box {
    background: linear-gradient(135deg, #185FA5 0%, #1D9E75 100%);
    border-radius: 14px; padding: 28px 24px; text-align: center; color: white;
  }
  .price-main { font-size: 42px; font-weight: 700; letter-spacing: -1px; }
  .price-sub  { font-size: 15px; opacity: 0.85; margin-top: 6px; }
  .ci-box {
    background: #F0F4F8; border-radius: 10px;
    padding: 14px 18px; text-align: center; margin-top: 12px;
  }
  .ci-label { font-size: 12px; color: #666; }
  .ci-val   { font-size: 18px; font-weight: 600; color: #185FA5; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 House Price Predictor")
st.caption("Powered by XGBoost · Ames Housing Dataset · FastAPI + Streamlit")
st.divider()

col_form, col_result = st.columns([1.3, 1], gap="large")

with col_form:
    st.subheader("🔢 Enter House Details")
    c1, c2 = st.columns(2)
    with c1:
        overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 7)
        gr_liv_area  = st.number_input("Living Area (sq ft)", 300, 6000, 1710, step=50)
        total_bsmt   = st.number_input("Basement Area (sq ft)", 0, 3000, 856, step=50)
        first_flr    = st.number_input("1st Floor Area (sq ft)", 300, 4000, 856, step=50)
        year_built   = st.number_input("Year Built", 1870, 2024, 2003)
    with c2:
        garage_cars  = st.selectbox("Garage Cars", [0, 1, 2, 3, 4], index=2)
        garage_area  = st.number_input("Garage Area (sq ft)", 0, 1500, 548, step=20)
        full_bath    = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
        tot_rooms    = st.number_input("Total Rooms (above grade)", 2, 15, 8)
        year_remod   = st.number_input("Remodel Year", 1870, 2024, 2003)
    c3, c4 = st.columns(2)
    with c3:
        lot_area   = st.number_input("Lot Area (sq ft)", 1000, 50000, 8450, step=500)
        fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=0)
    with c4:
        yr_sold = st.selectbox("Year Sold", list(range(2006, 2025)), index=4)

    st.write("")
    predict_btn = st.button("🔍 Predict Price", use_container_width=True, type="primary")

with col_result:
    st.subheader("📊 Prediction Result")
    if predict_btn:
        payload = {
            "OverallQual": overall_qual, "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,  "TotalBsmtSF": total_bsmt,
            "FirstFlrSF": first_flr,    "FullBath": full_bath,
            "TotRmsAbvGrd": tot_rooms,  "YearBuilt": year_built,
            "YearRemodAdd": year_remod, "LotArea": lot_area,
            "Fireplaces": fireplaces,   "GarageArea": garage_area,
            "YrSold": yr_sold,
        }
        with st.spinner("Running prediction..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                data   = resp.json()
                price  = data['predicted_price']
                ci_low = data['confidence_low']
                ci_hi  = data['confidence_high']

                st.markdown(f"""
                <div class="price-box">
                  <div class="price-main">{data['predicted_price_fmt']}</div>
                  <div class="price-sub">Predicted Sale Price</div>
                </div>
                <div class="ci-box">
                  <div class="ci-label">90% Confidence Interval</div>
                  <div class="ci-val">${ci_low:,.0f} – ${ci_hi:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                fig, ax = plt.subplots(figsize=(6, 2.5))
                pr = np.linspace(50_000, 800_000, 300)
                mu, sig = price, price * 0.12
                dist = np.exp(-0.5 * ((pr - mu) / sig) ** 2)
                ax.fill_between(pr, dist, alpha=0.18, color='#185FA5')
                ax.plot(pr, dist, color='#185FA5', linewidth=1.8)
                ax.axvline(price,  color='#1D9E75', linewidth=2,
                           label=f'Predicted: ${price:,.0f}')
                ax.axvline(ci_low, color='#D85A30', linewidth=1.2,
                           linestyle='--', alpha=0.7)
                ax.axvline(ci_hi,  color='#D85A30', linewidth=1.2,
                           linestyle='--', alpha=0.7,
                           label=f'CI: ${ci_low:,.0f} – ${ci_hi:,.0f}')
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
                ax.set_yticks([])
                ax.legend(fontsize=8)
                for sp in ['top','right','left']:
                    ax.spines[sp].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                with st.expander("📋 Input summary"):
                    st.dataframe(
                        pd.DataFrame(payload.items(), columns=["Feature","Value"]),
                        use_container_width=True, hide_index=True)

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to API.\n\n"
                         "Run: `uvicorn src.api:app --reload --port 8000`")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Fill in house details and click **Predict Price**.")
        st.markdown("**📌 Typical Ames price ranges**")
        st.dataframe(pd.DataFrame({
            "Segment":      ["Budget","Mid-range","Premium","Luxury"],
            "Price Range":  ["< $120K","$120K–$200K","$200K–$300K","> $300K"],
            "Quality (1-10)":["3–5","5–7","7–8","9–10"],
        }), use_container_width=True, hide_index=True)

st.divider()
st.caption("Stack: Python · Scikit-learn · XGBoost · MLflow · FastAPI · Streamlit · Docker")
