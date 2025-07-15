# streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm

# --- Power Calculation Functions ---
def power_two_sample_ttest(effect_size, n1, n2, alpha=0.05, alternative='two-sided'):
    df = n1 + n2 - 2
    ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))

    t_crit = stats.t.ppf(1 - alpha / 2, df) if alternative == 'two-sided' else stats.t.ppf(1 - alpha, df)

    if alternative == 'two-sided':
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    elif alternative == 'larger':
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    elif alternative == 'smaller':
        power = stats.nct.cdf(-t_crit, df, ncp)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
    
    return power

def plot_power_curve_with_distributions(effect_size, n1, n2, alpha=0.05, alternative='two-sided', normal_approx_threshold=350):
    df = n1 + n2 - 2
    ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))

    use_normal_approx = df > normal_approx_threshold

    # x-axis range
    x = np.linspace(-6, 6 + ncp, 1000)

    # Null distribution (central t)
    null_pdf = stats.t.pdf(x, df) if not use_normal_approx else stats.norm.pdf(x, 0, 1)

    # Alternative distribution
    alt_pdf = stats.nct.pdf(x, df, ncp) if not use_normal_approx else stats.norm.pdf(x, ncp, 1)

    # Critical values
    if alternative == 'two-sided':
        t_crit_low = stats.t.ppf(alpha / 2, df) if not use_normal_approx else stats.norm.ppf(alpha / 2)
        t_crit_high = stats.t.ppf(1 - alpha / 2, df) if not use_normal_approx else stats.norm.ppf(1 - alpha / 2)
    else:
        t_crit = stats.t.ppf(1 - alpha, df) if not use_normal_approx else stats.norm.ppf(1 - alpha)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, null_pdf, label='Null Distribution', color='blue')
    plt.plot(x, alt_pdf, label='Alternative Distribution', color='orange')

    # Shading rejection regions and power regions
    if alternative == 'two-sided':
        plt.fill_between(x, 0, null_pdf, where=(x <= t_crit_low) | (x >= t_crit_high), color='blue', alpha=0.3, label='Rejection Region (α/2)')
        plt.fill_between(x, 0, alt_pdf, where=(x <= t_crit_low) | (x >= t_crit_high), color='red', alpha=0.3, label='Power')
        plt.axvline(t_crit_low, color='black', linestyle='--', label='-t critical')
        plt.axvline(t_crit_high, color='black', linestyle='--', label='t critical')
    elif alternative == 'larger':
        plt.fill_between(x, 0, null_pdf, where=(x >= t_crit), color='blue', alpha=0.3, label='Rejection Region (α)')
        plt.fill_between(x, 0, alt_pdf, where=(x >= t_crit), color='red', alpha=0.3, label='Power')
        plt.axvline(t_crit, color='black', linestyle='--', label='t critical')
    elif alternative == 'smaller':
        plt.fill_between(x, 0, null_pdf, where=(x <= -t_crit), color='blue', alpha=0.3, label='Rejection Region (α)')
        plt.fill_between(x, 0, alt_pdf, where=(x <= -t_crit), color='red', alpha=0.3, label='Power')
        plt.axvline(-t_crit, color='black', linestyle='--', label='t critical')

    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Two-Sample t-Test Power Curve\nEffect Size={effect_size}, α={alpha}, n1={n1}, n2={n2}')
    plt.xlabel('t-value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())


# --- Sample Size Solver ---
def sample_size_two_sample_ttest(effect_size, power_target=0.8, alpha=0.05,
                                 alternative='two-sided', allocation_ratio=2.0):
    
    z_alpha = norm.ppf(1 - alpha / 2) if alternative == 'two-sided' else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power_target)

    # n1 is the size of the smaller (or reference) group
    n1 = ((z_alpha + z_beta) ** 2 * (1 + allocation_ratio)) / (effect_size ** 2)
    n2 = n1 / allocation_ratio

    return np.ceil(n1), np.ceil(n2)

def plot_power_curve(effect_size, alpha=0.05, power_target=0.8, alternative='two-sided', allocation_ratio=1.0, max_n1=200):
    n1_values = np.arange(2, max_n1 + 1, 1)
    power_values = []

    for n1 in n1_values:
        n2 = n1 / allocation_ratio
        power = power_two_sample_ttest(effect_size, n1, n2, alpha=alpha, alternative=alternative)
        power_values.append(power)

    plt.figure(figsize=(10, 6))
    plt.plot(n1_values, power_values, label=f'Power', )
    plt.axhline(y=power_target, color='red', linestyle='--', label=f'Target Power = {power_target}')
    plt.xlabel('Sample Size of Group 1')
    plt.ylabel('Power')
    plt.title('Power vs. Sample Size (Group 1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())


# --- Effect Size Solver ---
def effect_size_two_sample_ttest(power_target, n_per_group, alpha=0.05, alternative='two-sided'):
    """
    Solve for the minimum detectable effect size (Cohen's d) given power, sample size, and alpha.
    """
    def power_diff(d):
        return power_two_sample_ttest(d, n_per_group, alpha=alpha, alternative=alternative) - power_target

    return brentq(power_diff, a=1e-6, b=5.0, xtol=1e-4)


# --- Streamlit Interface ---
st.title("Two Sample t-Test Power Calculator")

# Sidebar input options
st.sidebar.header("Input Parameters")
effect_size = round(st.sidebar.number_input("Effect Size (Cohen's d)", min_value=-2.0, max_value=2.0, value=0.5, step=0.01, format="%.2f"), 4)
alpha = round(st.sidebar.number_input("Significance Level (alpha)", min_value=0.001, max_value=0.3, value=0.05, step=0.01, format="%.2f"), 4)
alternative = st.sidebar.selectbox("Test Type", options=['two-sided', 'larger', 'smaller'])

# Power or sample size mode
mode = st.radio("What do you want to calculate?", options=["Power", "Sample Size", "Effect Size"])

if mode == "Power":
    n1 = round(st.number_input("Group 1 Sample Size", min_value=2, value=50), 4)
    n2 = round(st.number_input("Group 2 Sample Size", min_value=2, value=50), 4)
    power = power_two_sample_ttest(effect_size, n1, n2, alpha, alternative)
    st.success(f"Estimated Power: **{power:.3f}**")

    if st.checkbox("Show Distribution Plot", value=True):
        plot_power_curve_with_distributions(effect_size, n1, n2, alpha, alternative)

elif mode == "Sample Size":
    power_target = round(st.number_input("Desired Power", min_value=0.01, max_value=0.99, value=0.8, step=0.01), 4)
    sample_size_ratio = round(st.number_input("Ratio of Group 1 Size to Group 2 Size", min_value=0.01, max_value=1000.0, value=1.0, step=0.1), 4)
    try:
        n1_required, n2_required = sample_size_two_sample_ttest(effect_size, power_target, alpha, alternative, sample_size_ratio)
        st.success(f"Required Sample Sizes:\n- Group 1: **{int(n1_required)}**\n- Group 2: **{int(n2_required)}**")
    except Exception as e:
        st.error(f"Error computing sample size: {e}")

    if st.checkbox("Show Power Curve", value=True):
        max_n1 = st.number_input("Max Sample Size to Plot", min_value=2.0, max_value=100000.0, value=200.0, step=1.0)
        plot_power_curve(effect_size, alpha=alpha, power_target=power_target, alternative=alternative, allocation_ratio=sample_size_ratio, max_n1=max_n1)

elif mode == "Effect Size":
    n_per_group = st.number_input("Sample Size per Group", min_value=2, value=50)
    power_target = st.slider("Desired Power", min_value=0.01, max_value=0.99, value=0.8, step=0.01)
    effect_size = effect_size_two_sample_ttest(power_target, n_per_group, alpha, alternative)
    st.success(f"Effect Size: **{effect_size:.3f}**")