# power_app.py
import streamlit as st
from statsmodels.stats.power import TTestIndPower

st.title("Statistical Power Calculator")

effect_size = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5)
alpha = st.number_input("Alpha (significance level)", 0.001, 0.5, 0.05)
power = st.slider("Desired Power", 0.5, 0.99, 0.8)
test_type = st.selectbox("Test Type", ["Two-sample t-test"])

analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)

st.write(f"Required sample size per group: **{sample_size:.2f}**")
